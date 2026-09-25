# Running this pipeline on AWS SageMaker

This project was developed on a 4-core / 12GB-RAM Windows laptop, which is
enough to write and correctness-test the pipeline (see
`cache/smoke_test*.py`) but not to comfortably run it at the challenge's
real scale (~2.2M Source-1 / ~5M Source-2 / ~5.3M Source-3 rows in train;
~1.7M / ~4.9M / ~5.1M in test). Every heavy step -- full-table
normalization, TF-IDF vectorization, feature engineering over millions of
candidate pairs, and model training/threshold search -- is meant to run on
a SageMaker **Notebook Instance** instead. This folder makes that a
one-command setup, driven by a single config file.

We deliberately do **not** use SageMaker Training Jobs or Endpoints: the
challenge asks for CPU-only, efficient, self-contained processing, and a
Notebook Instance's terminal is sufficient (and cheaper/simpler) for
running `src/train.py` and `src/inference.py` as plain scripts.

## 1. Prerequisites

* An AWS account with permission to create S3 buckets, an IAM role, and a
  SageMaker Notebook Instance (or an existing role ARN you can pass in).
* AWS CLI credentials configured locally (`aws configure`), or a named
  profile.
* Python 3.9+ locally, with `pip install -r aws/requirements.txt`
  (installs `boto3` + `PyYAML` -- only needed to run the provisioning
  script itself, not the ML pipeline).

## 2. Edit the one config file

Open `aws/aws_config.yaml` and set at minimum:

* `s3.bucket_name` -- must be globally unique (e.g. add your initials/date).
* `aws.region` -- pick a region close to you / where SageMaker is available.
* `sagemaker.instance_type` -- `ml.m5.4xlarge` (16 vCPU / 64GB RAM) is the
  default and comfortably fits normalizing + vectorizing the full Source-2/3
  tables in memory. `ml.m5.2xlarge` (8 vCPU / 32GB) is a cheaper option for
  smoke-testing at a larger-than-laptop sample size before committing to a
  full run.

Everything else (S3 prefixes, IAM role name, lifecycle config name, volume
size) has a sensible default and rarely needs changing.

## 3. Provision

```bash
pip install -r aws/requirements.txt
python aws/setup_sagemaker.py up --config aws/aws_config.yaml --wait
```

This will:

1. Create the S3 bucket (skipped if it already exists).
2. Upload `student_resource/dataset/` (the raw TSVs) and
   `business_entity_resolution/` (all source code, excluding
   `cache/`, `data/`, `models/`) to S3.
3. Create an IAM role scoped to SageMaker + that one S3 bucket (or reuse
   `iam.role_arn` if you set one).
4. Register a Lifecycle Configuration (`aws/lifecycle_config.sh.template`)
   that, on every instance start, syncs the code+data from S3 into
   `/home/ec2-user/SageMaker/amazon_ml_challenge/` and
   `pip install`s the pinned `requirements.txt`.
5. Create (or start, if already stopped) the Notebook Instance.

When it finishes it prints the instance status and a Jupyter URL.

Re-running `up` after editing local code is exactly how you push updates:
it re-uploads the code to S3, and the lifecycle script re-syncs it the next
time the instance starts (or stop/start it once to force a re-sync
immediately: `aws sagemaker stop-notebook-instance` /
`start-notebook-instance`, or just re-run `setup_sagemaker.py up`).

## 4. Run the pipeline on the instance

Open the Notebook Instance in Jupyter, open a **terminal**, then:

```bash
cd /home/ec2-user/SageMaker/amazon_ml_challenge/business_entity_resolution

# 1. Build parquet caches from the raw TSVs (fast, streaming, low memory)
python -m src.data_loader

# 2. Phase 1 data audit (writes experiments/data_audit_report.json)
python cache/audit_scratch.py   # or run notebooks/01_data_audit.ipynb

# 3. Train + compare models + pick a threshold (writes experiments/results.csv
#    and models/final_model.joblib, idf_tables.joblib, threshold.json).
#    --sample-size 0 means "use every TRAIN Source-1 entity" (full scale).
#    Start smaller (e.g. 200000) to sanity check timing before going full scale.
python -m src.train --sample-size 0 --chunk-size 20000

# 4. Full-scale test inference (writes ../../output/matching_results.tsv
#    and candidate_pairs.tsv at the repo root, per the submission layout).
#    --n-jobs-normalize should match the instance's vCPU count.
python -m src.inference --split test --n-jobs-normalize 16 --chunk-size 25000

# 5. Validate the output with the OFFICIAL validator
python -m src.validation
```

Each step logs timestamps and RSS memory (`utils.timer` / `utils.log_mem`)
so you can see exactly where time/memory goes and re-tune
`config.S1_CHUNK_SIZE` / instance size if needed.

## 5. Cost control

* **Stop the instance when you're not using it** -- a Notebook Instance
  bills by the hour while `InService`; it does not bill while `Stopped`
  (only its EBS volume does, which is small/cheap).
  ```bash
  python aws/setup_sagemaker.py status --config aws/aws_config.yaml
  aws sagemaker stop-notebook-instance --notebook-instance-name amazon-ml-challenge-notebook
  ```
* To tear everything down (notebook instance; optionally the S3 bucket):
  ```bash
  python aws/setup_sagemaker.py destroy --config aws/aws_config.yaml            # keeps S3 data
  python aws/setup_sagemaker.py destroy --config aws/aws_config.yaml --delete-bucket
  ```
  The IAM role and lifecycle configuration are left in place (free, and
  reusable next time) -- delete them manually from the console/CLI if you
  want a fully clean account.

## 6. Getting results back off the instance

The Jupyter file browser can download individual files, or from the
instance terminal push the final outputs back to S3 for easy retrieval:

```bash
aws s3 cp /home/ec2-user/SageMaker/amazon_ml_challenge/output/matching_results.tsv \
    s3://<your-bucket>/entity-resolution/artifacts/matching_results.tsv
aws s3 cp /home/ec2-user/SageMaker/amazon_ml_challenge/output/candidate_pairs.tsv \
    s3://<your-bucket>/entity-resolution/artifacts/candidate_pairs.tsv
```
then `aws s3 cp` them back down locally.
