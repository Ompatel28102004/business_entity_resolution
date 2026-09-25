"""
Small shared utilities: timing, memory reporting, logging helpers.

Kept dependency-light (stdlib + psutil-if-available) since these helpers are
called from performance-sensitive, memory-constrained code paths.
"""
from __future__ import annotations

import json
import os
import time
import uuid
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def log(msg: str) -> None:
    """Print a timestamped progress message and flush immediately.

    Using explicit flush matters when running inside notebooks / redirected
    terminals so progress is visible while long-running cells execute.
    """
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@contextmanager
def timer(label: str) -> Iterator[None]:
    """Context manager that logs the wall-clock runtime of a code block.

    Example:
        with timer("blocking"):
            candidates = build_candidates(...)
    """
    t0 = time.perf_counter()
    log(f"START  {label}")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log(f"DONE   {label}  ({dt:.2f}s)")


def mem_mb() -> float:
    """Return current process resident memory in MB (0.0 if psutil missing).

    Used to sanity-check that memory-bounded / chunked processing is actually
    staying within budget on this 4-core, ~12GB-RAM CPU-only machine.
    """
    try:
        import psutil  # local import: optional dependency

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return 0.0


def log_mem(label: str = "") -> None:
    """Log current process memory usage, tagged with an optional label."""
    log(f"MEM {label}: {mem_mb():.0f} MB")


def chunked(seq, size: int):
    """Yield ``seq`` in consecutive chunks of at most ``size`` elements.

    Works for anything sliceable (lists, numpy arrays, pandas Index/Series).
    """
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def eprint(*args, **kwargs) -> None:
    """Print to stderr (used for warnings that shouldn't pollute stdout)."""
    print(*args, file=sys.stderr, **kwargs)


def parallel_map_df(df, func, n_jobs: int = 1, chunk_size: int | None = None):
    """Apply ``func(df_chunk) -> DataFrame`` over row-chunks of ``df`` in parallel.

    This is the workhorse used to scale the (embarrassingly parallel, pure)
    row-wise normalization / feature functions from a 4-core laptop up to a
    many-vCPU SageMaker/EC2 instance: string-heavy pandas ``.str`` operations
    run single-threaded under the GIL, so on a bigger machine the only way to
    use the extra cores is explicit process-level parallelism.

    ``n_jobs=1`` (the default) runs in-process with no multiprocessing
    overhead -- safe for small/dev inputs and for environments where
    spawning subprocesses is undesirable (e.g. inside some notebook kernels).

    Set ``n_jobs>1`` (e.g. to the vCPU count of the machine) for full-scale runs.
    """
    import pandas as pd

    n = len(df)
    if n_jobs <= 1 or n == 0:
        return func(df)

    if chunk_size is None:
        chunk_size = max(1, -(-n // n_jobs))  # ceil division: n_jobs roughly-equal chunks

    chunks = [df.iloc[i : i + chunk_size] for i in range(0, n, chunk_size)]

    from concurrent.futures import ProcessPoolExecutor

    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for out in pool.map(func, chunks):
            results.append(out)
    return pd.concat(results, axis=0, ignore_index=False)


def get_system_info() -> dict:
    """Detect CPU cores, total/available RAM, and free disk space.

    Printed at program startup (``run.py``) so every run's log records what
    hardware it actually executed on -- essential for judging whether a
    given ``--train-sample-size`` / full-scale run is realistic on the
    current machine (a laptop today, an EC2 instance in production), rather
    than assuming a fixed 16/32/64GB RAM box.
    """
    info = {
        "cpu_count_logical": os.cpu_count() or 1,
        "total_ram_gb": None,
        "available_ram_gb": None,
        "disk_free_gb": None,
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["total_ram_gb"] = round(vm.total / 1e9, 2)
        info["available_ram_gb"] = round(vm.available / 1e9, 2)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False) or info["cpu_count_logical"]
    except Exception:
        pass
    try:
        import shutil

        usage = shutil.disk_usage(str(Path(__file__).resolve().parents[2]))
        info["disk_free_gb"] = round(usage.free / 1e9, 2)
        info["disk_total_gb"] = round(usage.total / 1e9, 2)
    except Exception:
        pass
    return info


def print_system_info(info: dict | None = None) -> dict:
    """Log the output of ``get_system_info`` in a human-readable form."""
    info = info or get_system_info()
    log(
        "system: "
        f"{info['cpu_count_logical']} logical CPUs, "
        f"RAM total={info.get('total_ram_gb')}GB available={info.get('available_ram_gb')}GB, "
        f"disk free={info.get('disk_free_gb')}GB"
    )
    if info.get("total_ram_gb") is not None and info["total_ram_gb"] < 8:
        log("  WARNING: less than 8GB total RAM detected -- use small --train-sample-size values and small chunk sizes.")
    return info


def new_run_id() -> str:
    """A short, sortable, unique run identifier: <UTC timestamp>_<4 hex chars>."""
    return f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:4]}"


def read_json(path: Path, default=None):
    """Read a JSON file, returning ``default`` if it doesn't exist or fails to parse."""
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, obj) -> None:
    """Write ``obj`` as pretty-printed JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
