#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Business Entity Resolution
Master entry-point script.
"""
from __future__ import annotations

import argparse
import json
import os
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
    log("STAGE: audit")
    from src.data_loader import SOURCE_COLUMNS, GT_COLUMNS
    checks = {
        "train_source1":    static_config.TRAIN_SOURCE1,
        "train_source2":    static_config.TRAIN_SOURCE2,
        "train_source3":    static_config.TRAIN_SOURCE3,
        "train_gt":         static_config.TRAIN_GROUND_TRUTH,
        "test_source1":     static_config.TEST_SOURCE1,
        "test_source2":     static_config.TEST_SOURCE2,
        "test_source3":     static_config.TEST_SOURCE3,
    }
    import pyarrow.csv as pv_csv
    import pyarrow as pa
    all_ok = True
    for name, path in checks.items():
        if not path.exists():
            log(f"  MISSING  {name}: {path}")
            all_ok = False
            continue
        ro = pv_csv.ReadOptions(block_size=64 << 20)
        po = pv_csv.ParseOptions(delimiter="\t")
        cols = SOURCE_COLUMNS if "gt" not in name else GT_COLUMNS
        co = pv_csv.ConvertOptions(column_types={c: pa.string() for c in cols}, include_columns=cols[:1])
        reader = pv_csv.open_csv(str(path), read_options=ro, parse_options=po, convert_options=co)
        n_rows = sum(batch.num_rows for batch in reader)
        log(f"  OK  {name:20s}  {n_rows:>10,} rows")
    if not all_ok:
        sys.exit(1)

def stage_prepare(cfg: dict) -> None:
    log("STAGE: prepare")
    from src.data_loader import build_all_caches
    build_all_caches()

def stage_train(cfg: dict, sample_size: int | None = None) -> dict:
    log("STAGE: train")
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
    log("STAGE: infer")
    import joblib
    from src.inference import run_pipeline
    from src.model import SklearnModelWrapper

    model_path = static_config.MODELS_DIR / "final_model.joblib"
    idf_path   = static_config.MODELS_DIR / "idf_tables.joblib"
    thr_path   = static_config.MODELS_DIR / "threshold.json"

    model_wrapper = SklearnModelWrapper.load(model_path)
    idf_tables    = joblib.load(idf_path)
    threshold = json.loads(thr_path.read_text())["threshold"]

    chunk_size = get(cfg, "runtime.chunk_size", 25000)
    run_pipeline(
        split="test",
        model_wrapper=model_wrapper,
        name_idf=idf_tables["name_idf"],
        addr_idf=idf_tables["addr_idf"],
        threshold=threshold,
        output_dir=static_config.OUTPUT_DIR,
        chunk_size=int(chunk_size),
        write_outputs=True,
    )

def stage_validate(cfg: dict) -> None:
    log("STAGE: validate")
    from src.validation import run_official_validator
    rc = run_official_validator(
        matching_path=static_config.OUTPUT_DIR / "matching_results.tsv",
        candidate_path=static_config.OUTPUT_DIR / "candidate_pairs.tsv",
        test_dir=static_config.RAW_TEST_DIR,
        check_ids=True,
    )
    sys.exit(rc)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--stage", choices=["audit", "prepare", "train", "validate", "infer", "all"], default="all")
    p.add_argument("--train-sample-size", type=int, default=None)
    return p

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    
    config_path = Path(args.config)
    overrides = {}
    if args.train_sample_size is not None:
        overrides["sampling.train_sample_size"] = args.train_sample_size
    cfg = load_config_with_overrides(config_path, overrides)
    
    for d in (static_config.DATA_DIR, static_config.EXPERIMENTS_DIR, static_config.MODELS_DIR, static_config.OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
        
    sample_size = get(cfg, "sampling.train_sample_size", 0)
    if args.train_sample_size is not None:
        sample_size = args.train_sample_size

    if args.stage in ["audit", "all"]: stage_audit(cfg)
    if args.stage in ["prepare", "all"]: stage_prepare(cfg)
    if args.stage in ["train", "all"]: stage_train(cfg, sample_size=sample_size)
    if args.stage in ["infer", "all"]: stage_infer(cfg)
    if args.stage in ["validate", "all"]: stage_validate(cfg)

if __name__ == "__main__":
    main()
