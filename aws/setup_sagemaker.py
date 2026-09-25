#!/usr/bin/env python3
"""
One-command AWS provisioning for the Business Entity Resolution pipeline.

Reads every parameter from ``aws/aws_config.yaml`` (the single config file)
and:

  1. Creates (or reuses) an S3 bucket.
  2. Uploads the raw challenge dataset and the ``business_entity_resolution``
     source tree to S3.
  3. Creates (or reuses) an IAM role for the SageMaker Notebook Instance.
  4. Renders ``lifecycle_config.sh.template`` with this run's bucket/prefixes
     and registers it as a SageMaker Lifecycle Configuration.
  5. Creates (or starts) a CPU-only SageMaker Notebook Instance running that
     lifecycle configuration, which syncs the code+data and installs
     ``requirements.txt`` on every start.

This script only calls AWS control-plane APIs (S3/IAM/SageMaker) -- it does
not run any model training or data processing itself, and it never creates
a SageMaker training job or endpoint (per the challenge's compute rules;
this pipeline is designed to run entirely inside a Notebook Instance / plain
scripts on it).

Usage:
    pip install -r aws/requirements.txt        # boto3 + pyyaml, once
    aws configure                                # if not already done
    python aws/setup_sagemaker.py --config aws/aws_config.yaml up
    python aws/setup_sagemaker.py --config aws/aws_config.yaml status
    python aws/setup_sagemaker.py --config aws/aws_config.yaml destroy
"""
from __future__ import annotations

import argparse
import base64
import fnmatch
import json
import sys
import time
from pathlib import Path

import boto3
import yaml
from botocore.exceptions import ClientError

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../AmazonML


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def get_session(cfg: dict) -> boto3.Session:
    profile = cfg["aws"].get("profile")
    return boto3.Session(profile_name=profile, region_name=cfg["aws"]["region"])


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------
def ensure_bucket(session: boto3.Session, cfg: dict) -> None:
    s3 = session.client("s3")
    bucket = cfg["s3"]["bucket_name"]
    region = cfg["aws"]["region"]
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"[s3] bucket already exists: {bucket}")
        return
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchBucket"):
            raise
    print(f"[s3] creating bucket: {bucket} in {region}")
    kwargs = {"Bucket": bucket}
    if region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket(**kwargs)
    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True,
        },
    )


def _iter_files(root: Path, exclude_dirs: set[str]):
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in exclude_dirs for part in path.relative_to(root).parts[:-1]):
            continue
        if fnmatch.fnmatch(path.name, "*.pyc"):
            continue
        yield path


def upload_dir(session: boto3.Session, cfg: dict, local_dir: Path, prefix: str, exclude_dirs: set[str] = frozenset()) -> int:
    s3 = session.client("s3")
    bucket = cfg["s3"]["bucket_name"]
    n = 0
    for path in _iter_files(local_dir, exclude_dirs):
        rel = path.relative_to(local_dir).as_posix()
        key = f"{prefix}/{rel}"
        s3.upload_file(str(path), bucket, key)
        n += 1
    return n


def upload_dataset_and_code(session: boto3.Session, cfg: dict) -> None:
    dataset_dir = REPO_ROOT / cfg["local_paths"]["dataset_dir"]
    code_dir = REPO_ROOT / cfg["local_paths"]["code_dir"]
    print(f"[s3] uploading dataset from {dataset_dir} ...")
    n_data = upload_dir(session, cfg, dataset_dir, cfg["s3"]["data_prefix"])
    print(f"[s3]   uploaded {n_data} dataset files")
    print(f"[s3] uploading code from {code_dir} ...")
    n_code = upload_dir(session, cfg, code_dir, cfg["s3"]["code_prefix"], set(cfg["upload"]["exclude_dirs"]))
    print(f"[s3]   uploaded {n_code} code files")


# ---------------------------------------------------------------------------
# IAM
# ---------------------------------------------------------------------------
TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"}, "Action": "sts:AssumeRole"}],
}


def ensure_role(session: boto3.Session, cfg: dict) -> str:
    if cfg["iam"].get("role_arn"):
        print(f"[iam] reusing existing role: {cfg['iam']['role_arn']}")
        return cfg["iam"]["role_arn"]

    iam = session.client("iam")
    role_name = cfg["iam"]["role_name"]
    try:
        resp = iam.get_role(RoleName=role_name)
        print(f"[iam] role already exists: {role_name}")
        return resp["Role"]["Arn"]
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise

    print(f"[iam] creating role: {role_name}")
    resp = iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(TRUST_POLICY))
    for arn in cfg["iam"]["managed_policy_arns"]:
        iam.attach_role_policy(RoleName=role_name, PolicyArn=arn)

    bucket = cfg["s3"]["bucket_name"]
    s3_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
            }
        ],
    }
    iam.put_role_policy(RoleName=role_name, PolicyName="EntityResolutionS3Access", PolicyDocument=json.dumps(s3_policy))
    print("[iam] waiting for IAM role propagation ...")
    time.sleep(10)
    return resp["Role"]["Arn"]


# ---------------------------------------------------------------------------
# SageMaker: lifecycle config + notebook instance
# ---------------------------------------------------------------------------
def render_lifecycle_script(cfg: dict) -> str:
    template = (Path(__file__).parent / "lifecycle_config.sh.template").read_text()
    return (
        template.replace("{{BUCKET}}", cfg["s3"]["bucket_name"])
        .replace("{{DATA_PREFIX}}", cfg["s3"]["data_prefix"])
        .replace("{{CODE_PREFIX}}", cfg["s3"]["code_prefix"])
    )


def ensure_lifecycle_config(session: boto3.Session, cfg: dict) -> str:
    sm = session.client("sagemaker")
    name = cfg["sagemaker"]["lifecycle_config_name"]
    script = render_lifecycle_script(cfg)
    encoded = base64.b64encode(script.encode()).decode()
    on_start = [{"Content": encoded}]
    try:
        sm.describe_notebook_instance_lifecycle_config(NotebookInstanceLifecycleConfigName=name)
        print(f"[sagemaker] updating lifecycle config: {name}")
        sm.update_notebook_instance_lifecycle_config(NotebookInstanceLifecycleConfigName=name, OnStart=on_start)
    except ClientError as e:
        if "does not exist" not in str(e) and e.response.get("Error", {}).get("Code") != "ValidationException":
            raise
        print(f"[sagemaker] creating lifecycle config: {name}")
        sm.create_notebook_instance_lifecycle_config(NotebookInstanceLifecycleConfigName=name, OnStart=on_start)
    return name


def ensure_notebook_instance(session: boto3.Session, cfg: dict, role_arn: str, lifecycle_name: str) -> None:
    sm = session.client("sagemaker")
    name = cfg["sagemaker"]["notebook_instance_name"]
    try:
        desc = sm.describe_notebook_instance(NotebookInstanceName=name)
        status = desc["NotebookInstanceStatus"]
        print(f"[sagemaker] notebook instance already exists: {name} (status={status})")
        if status == "Stopped":
            print("[sagemaker] starting it ...")
            sm.start_notebook_instance(NotebookInstanceName=name)
        return
    except ClientError as e:
        if e.response["Error"]["Code"] != "ValidationException":
            raise

    print(f"[sagemaker] creating notebook instance: {name} ({cfg['sagemaker']['instance_type']})")
    sm.create_notebook_instance(
        NotebookInstanceName=name,
        InstanceType=cfg["sagemaker"]["instance_type"],
        RoleArn=role_arn,
        VolumeSizeInGB=cfg["sagemaker"]["volume_size_gb"],
        LifecycleConfigName=lifecycle_name,
        RootAccess=cfg["sagemaker"].get("root_access", "Enabled"),
    )


def wait_for_in_service(session: boto3.Session, cfg: dict, timeout_s: int = 900) -> None:
    sm = session.client("sagemaker")
    name = cfg["sagemaker"]["notebook_instance_name"]
    print("[sagemaker] waiting for InService ...", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        status = sm.describe_notebook_instance(NotebookInstanceName=name)["NotebookInstanceStatus"]
        if status == "InService":
            print(" done.")
            return
        if status == "Failed":
            raise RuntimeError("Notebook instance entered Failed state; check the AWS console for details.")
        print(".", end="", flush=True)
        time.sleep(15)
    print("\n[sagemaker] timed out waiting for InService; check the console.")


def print_status(session: boto3.Session, cfg: dict) -> None:
    sm = session.client("sagemaker")
    name = cfg["sagemaker"]["notebook_instance_name"]
    try:
        desc = sm.describe_notebook_instance(NotebookInstanceName=name)
        print(f"NotebookInstanceStatus: {desc['NotebookInstanceStatus']}")
        print(f"InstanceType: {desc['InstanceType']}")
        print(f"Url: https://{desc.get('Url', '')}")
    except ClientError as e:
        print(f"No notebook instance named {name} found ({e.response['Error']['Code']}).")


def destroy(session: boto3.Session, cfg: dict, delete_bucket: bool = False) -> None:
    sm = session.client("sagemaker")
    name = cfg["sagemaker"]["notebook_instance_name"]
    try:
        status = sm.describe_notebook_instance(NotebookInstanceName=name)["NotebookInstanceStatus"]
        if status not in ("Stopped", "Failed"):
            print(f"[sagemaker] stopping {name} ...")
            sm.stop_notebook_instance(NotebookInstanceName=name)
            while sm.describe_notebook_instance(NotebookInstanceName=name)["NotebookInstanceStatus"] != "Stopped":
                time.sleep(10)
        print(f"[sagemaker] deleting {name} ...")
        sm.delete_notebook_instance(NotebookInstanceName=name)
    except ClientError as e:
        print(f"[sagemaker] {e.response['Error']['Code']}: nothing to delete")

    if delete_bucket:
        s3 = session.client("s3")
        bucket = cfg["s3"]["bucket_name"]
        print(f"[s3] emptying and deleting bucket {bucket} ...")
        paginator = s3.get_paginator("list_object_versions")
        for page in paginator.paginate(Bucket=bucket):
            objs = [{"Key": v["Key"], "VersionId": v["VersionId"]} for v in page.get("Versions", [])]
            if objs:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": objs})
        s3.delete_bucket(Bucket=bucket)
    print("[destroy] IAM role and lifecycle config are left in place (cheap, reusable); delete manually if desired.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["up", "upload-only", "status", "destroy"])
    parser.add_argument("--config", default=str(Path(__file__).parent / "aws_config.yaml"))
    parser.add_argument("--wait", action="store_true", help="Block until the notebook instance is InService.")
    parser.add_argument("--delete-bucket", action="store_true", help="With `destroy`, also empty+delete the S3 bucket.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    session = get_session(cfg)

    if args.action == "status":
        print_status(session, cfg)
        return
    if args.action == "destroy":
        destroy(session, cfg, delete_bucket=args.delete_bucket)
        return

    ensure_bucket(session, cfg)
    upload_dataset_and_code(session, cfg)
    if args.action == "upload-only":
        print("[done] upload-only complete.")
        return

    role_arn = ensure_role(session, cfg)
    lifecycle_name = ensure_lifecycle_config(session, cfg)
    ensure_notebook_instance(session, cfg, role_arn, lifecycle_name)
    if args.wait:
        wait_for_in_service(session, cfg)
    print_status(session, cfg)
    print(
        "\nNext steps:\n"
        "  1. Open the notebook instance from the SageMaker console (or the Url above).\n"
        "  2. It boots into /home/ec2-user/SageMaker/amazon_ml_challenge with the code\n"
        "     and dataset already synced by the lifecycle script.\n"
        "  3. Open a terminal in Jupyter and run, e.g.:\n"
        "       cd business_entity_resolution\n"
        "       python -m src.data_loader          # build parquet caches\n"
        "       python -m src.train --sample-size 0  --chunk-size 20000\n"
        "       python -m src.inference --split test --n-jobs-normalize 16\n"
        "       python -m src.validation\n"
    )


if __name__ == "__main__":
    main()
