"""
Loads ``config.yaml`` (the single source of truth for tunable pipeline
settings) and applies command-line overrides on top of it.

This module is intentionally separate from ``src/config.py``: ``config.py``
holds fixed, code-level constants (repo-relative paths, the legal-suffix
vocabulary, etc.) that rarely change, while ``config.yaml`` /
``config_loader.py`` hold the *runtime-tunable* knobs a user running on a
different machine (a laptop vs. an EC2 instance with a different core/RAM
budget) reasonably wants to change without editing source code.

Precedence: CLI override > config.yaml > built-in defaults in ``config.py``.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import yaml

from . import config as static_config

DEFAULT_CONFIG_PATH = static_config.PROJECT_DIR / "config.yaml"


def load_yaml_config(path: Optional[Path] = None) -> dict:
    """Load ``config.yaml`` into a plain nested dict. Missing file -> empty dict (all defaults)."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def get(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    """Fetch a nested value by dotted path, e.g. ``get(cfg, "sampling.train_sample_size", 0)``."""
    node = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node is None else node


def set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set a nested value by dotted path, creating intermediate dicts as needed."""
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def apply_path_overrides(cfg: dict) -> None:
    """Patch ``src.config`` path constants in place for every non-null ``paths.*`` entry.

    Every other module does ``from . import config`` and reads
    ``config.DATA_DIR`` / ``config.MODELS_DIR`` / etc. at call time (not at
    import time), so reassigning these attributes here -- before any stage
    runs -- transparently redirects the whole pipeline to different
    directories (e.g. a large EBS data volume mounted at ``/mnt/data`` on
    EC2) without touching a single line of ``src/data_loader.py`` or
    ``src/train.py``.
    """
    mapping = {
        "raw_data_dir": "RAW_DATA_DIR",
        "data_dir": "DATA_DIR",
        "models_dir": "MODELS_DIR",
        "experiments_dir": "EXPERIMENTS_DIR",
        "output_dir": "OUTPUT_DIR",
    }
    paths_cfg = cfg.get("paths") or {}
    for yaml_key, attr in mapping.items():
        value = paths_cfg.get(yaml_key)
        if value:
            setattr(static_config, attr, Path(value))
    # Derived sub-paths that depend on DATA_DIR / RAW_DATA_DIR must be re-derived
    # if their parent was overridden.
    if paths_cfg.get("data_dir"):
        static_config.TRAIN_CACHE_DIR = static_config.DATA_DIR / "train"
        static_config.TEST_CACHE_DIR = static_config.DATA_DIR / "test"
    if paths_cfg.get("raw_data_dir"):
        static_config.RAW_TRAIN_DIR = static_config.RAW_DATA_DIR / "train"
        static_config.RAW_TEST_DIR = static_config.RAW_DATA_DIR / "test"
        static_config.TRAIN_SOURCE1 = static_config.RAW_TRAIN_DIR / "train_source1.tsv"
        static_config.TRAIN_SOURCE2 = static_config.RAW_TRAIN_DIR / "train_source2.tsv"
        static_config.TRAIN_SOURCE3 = static_config.RAW_TRAIN_DIR / "train_source3.tsv"
        static_config.TRAIN_GROUND_TRUTH = static_config.RAW_TRAIN_DIR / "train_ground_truth.tsv"
        static_config.TEST_SOURCE1 = static_config.RAW_TEST_DIR / "test_source1.tsv"
        static_config.TEST_SOURCE2 = static_config.RAW_TEST_DIR / "test_source2.tsv"
        static_config.TEST_SOURCE3 = static_config.RAW_TEST_DIR / "test_source3.tsv"


def artifacts_dir(cfg: dict) -> Path:
    """Resume-state / stage-manifest directory (defaults under the project root)."""
    override = get(cfg, "paths.artifacts_dir")
    if override:
        return Path(override)
    return static_config.PROJECT_DIR / "artifacts"


def resolve_n_jobs(cfg: dict) -> int:
    """Resolve ``runtime.n_jobs: -1`` to the actual detected CPU core count."""
    import os

    n_jobs = get(cfg, "runtime.n_jobs", 1)
    if n_jobs is None or n_jobs <= 0:
        return os.cpu_count() or 1
    return int(n_jobs)


def load_config_with_overrides(config_path: Optional[Path], overrides: dict) -> dict:
    """Load ``config.yaml`` and apply a flat {dotted_key: value} dict of CLI overrides.

    ``overrides`` values of ``None`` are ignored (meaning "not provided on
    the CLI, defer to config.yaml"). Returns the merged config dict and
    applies path overrides as a side effect.
    """
    cfg = load_yaml_config(config_path)
    cfg = copy.deepcopy(cfg)
    for dotted_key, value in overrides.items():
        if value is not None:
            set_nested(cfg, dotted_key, value)
    apply_path_overrides(cfg)
    return cfg
