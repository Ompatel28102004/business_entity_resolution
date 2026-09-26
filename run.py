#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Business Entity Resolution
Master CLI runner.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src import config as static_config
from src.config_loader import load_config_with_overrides, get
from src.utils import log, log_mem, print_system_info, timer, write_json

def stage_audit(cfg: dict) -> None:
    log("=" * 60)
    log("STAGE: audit")
    log("=" * 60)
    from src.data_loader import SOURCE_COLUMNS, GT_COLUMNS
    checks = {
        "train_source1": static_config.TRAIN_SOURCE1,
        "train_source2": static_config.TRAIN_SOURCE2,
        "train_source3": static_config.TRAIN_SOURCE3,
        "train_gt":      static_config.TRAIN_GROUND_TRUTH,
        "test_source1":  static_config.TEST_SOURCE1,
        "test_source2":  static_config.TEST_SOURCE2,
        "test_source3":  static_config.TEST_SOURCE3,
    }
    import pyarrow.csv as pv_csv
    import pyarrow as pa
    all_ok = True
    results = {}
    for name, path in checks.items():
        if not path.exists():
            log(f"  MISSING  {name}: {path}")
            all_ok = False
            results[name] = {"status": "MISSING", "path": str(path)}
            continue
        size_mb = path.stat().st_size / 1e6
        ro = pv_csv.ReadOptions(block_size=64 << 20)
        po = pv_csv.ParseOptions(delimiter="\t")
        cols = SOURCE_COLUMNS if "gt" not in name else GT_COLUMNS
        co = pv_csv.ConvertOptions(column_types={c: pa.string() for c in cols}, include_columns=cols[:1])
        reader = pv_csv.open_csv(str(path), read_options=ro, parse_options=po, convert_options=co)
        n_rows = sum(batch.num_rows for batch in reader)
        log(f"  OK  {name:20s}  {n_rows:>10,} rows  {size_mb:>7.1f} MB  {path.name}")
        results[name] = {"status": "OK", "rows": n_rows, "size_mb": round(size_mb, 1)}
    
    report_path = static_config.EXPERIMENTS_DIR / "audit_report.json"
    static_config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(report_path, results)
    log(f"Audit report written -> {report_path}")
    if not all_ok:
        log("ERROR: One or more required data files are missing.")
        sys.exit(1)

def stage_prepare(cfg: dict) -> None:
    log("=" * 60)
    log("STAGE: prepare")
    log("=" * 60)
    from src.data_loader import build_all_caches
    with timer("build all parquet caches"):
        build_all_caches()
    log_mem("after prepare")

def stage_train(cfg: dict, sample_size: int | None = None) -> dict:
    log("=" * 60)
    log("STAGE: train")
    log("=" * 60)
    from src.train import run_training
    ss = sample_size if sample_size is not None else get(cfg, "sampling.train_sample_size", 0)
    val_fraction = get(cfg, "validation.validation_fraction", 0.25)
    seed = get(cfg, "sampling.random_seed", 42)
    chunk_size = get(cfg, "runtime.chunk_size", 2000)
    bg_mul = get(cfg, "sampling.background_multiplier", 20)
    use_bg = get(cfg, "sampling.use_background_pool", True)

    summary = run_training(
        sample_size=int(ss),
        val_fraction=float(val_fraction),
        seed=int(seed),
        chunk_size=int(chunk_size),
        background_multiplier=int(bg_mul),
        use_background_pool=bool(use_bg),
    )
    return summary

def stage_infer(cfg: dict) -> None:
    log("=" * 60)
    log("STAGE: infer")
    log("=" * 60)
    import joblib
    from src.inference import run_pipeline
    from src.model import SklearnModelWrapper

    model_path = static_config.MODELS_DIR / "final_model.joblib"
    idf_path   = static_config.MODELS_DIR / "idf_tables.joblib"
    thr_path   = static_config.MODELS_DIR / "threshold.json"

    if not model_path.exists():
        log(f"ERROR: {model_path} not found. Run --stage train first.")
        sys.exit(1)

    model_wrapper = SklearnModelWrapper.load(model_path)
    idf_tables    = joblib.load(idf_path)
    threshold     = json.loads(thr_path.read_text())["threshold"]

    chunk_size = get(cfg, "runtime.chunk_size", 25000)
    output_dir = static_config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loaded model from {model_path}, threshold={threshold}")
    with timer("test-set inference"):
        run_pipeline(
            split="test",
            model_wrapper=model_wrapper,
            name_idf=idf_tables["name_idf"],
            addr_idf=idf_tables["addr_idf"],
            threshold=threshold,
            output_dir=output_dir,
            chunk_size=int(chunk_size),
            write_outputs=True,
        )
    log(f"Outputs written -> {output_dir / 'matching_results.tsv'}")
    log(f"                -> {output_dir / 'candidate_pairs.tsv'}")

def stage_validate(cfg: dict) -> None:
    log("=" * 60)
    log("STAGE: validate")
    log("=" * 60)
    from src.validation import run_official_validator
    matching_path  = static_config.OUTPUT_DIR / "matching_results.tsv"
    candidate_path = static_config.OUTPUT_DIR / "candidate_pairs.tsv"

    if not matching_path.exists() or not candidate_path.exists():
        log("ERROR: Submission files not found. Run --stage infer first.")
        sys.exit(1)

    rc = run_official_validator(
        matching_path=matching_path,
        candidate_path=candidate_path,
        test_dir=static_config.RAW_TEST_DIR,
        check_ids=True,
    )
    if rc == 0:
        log("Official Validator PASS!")
    else:
        log(f"Official Validator returned exit code {rc}")
    sys.exit(rc)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Pipeline Runner")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--stage", choices=["audit", "prepare", "train", "validate", "infer", "all"], default="all")
    p.add_argument("--train-sample-size", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=None)
    return p

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = HERE / config_path

    overrides = {}
    if args.train_sample_size is not None:
        overrides["sampling.train_sample_size"] = args.train_sample_size
    if args.chunk_size is not None:
        overrides["runtime.chunk_size"] = args.chunk_size

    cfg = load_config_with_overrides(config_path, overrides)

    log("=" * 60)
    log("Amazon ML Challenge 2026 — Business Entity Resolution")
    log("=" * 60)
    print_system_info()

    for d in (static_config.DATA_DIR, static_config.EXPERIMENTS_DIR, static_config.MODELS_DIR, static_config.OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    sample_size = get(cfg, "sampling.train_sample_size", 0)
    if args.train_sample_size is not None:
        sample_size = args.train_sample_size

    if args.stage == "audit":
        stage_audit(cfg)
    elif args.stage == "prepare":
        stage_prepare(cfg)
    elif args.stage == "train":
        stage_train(cfg, sample_size=sample_size)
    elif args.stage == "infer":
        stage_infer(cfg)
    elif args.stage == "validate":
        stage_validate(cfg)
    elif args.stage == "all":
        stage_audit(cfg)
        stage_prepare(cfg)
        stage_train(cfg, sample_size=sample_size)
        stage_infer(cfg)
        stage_validate(cfg)

if __name__ == "__main__":
    main()

