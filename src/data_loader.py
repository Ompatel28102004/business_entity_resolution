"""
Data loading and parquet caching.

The raw challenge files are large (up to ~510MB / ~5.3M rows per TSV) and the
development machine has only ~12GB RAM (often <4GB free). To keep every stage
of the pipeline memory-safe we:

1. Convert each raw TSV to a snappy-compressed parquet file exactly once,
   streaming the conversion in batches with pyarrow (never materialising the
   whole file as a single Python/pandas object).
2. From then on, everything reads the parquet cache with column pruning
   (`pd.read_parquet(..., columns=[...])`), which is both far faster and far
   lighter than re-parsing the TSV.

Ground-truth matches are also cached as an "exploded" parquet (one row per
(source1_entity_id, matched_entity_id) pair) since that is the shape needed
for label generation and recall measurement.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv_csv
import pyarrow.parquet as pq

from . import config
from .utils import log, timer

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def _cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name}.parquet"


def convert_tsv_to_parquet(
    tsv_path: Path,
    parquet_path: Path,
    schema_columns: Iterable[str],
    batch_size: int = 250_000,
    force: bool = False,
) -> Path:
    """Stream-convert a large tab-separated file into a parquet cache.

    Uses ``pyarrow.csv.open_csv`` to read the TSV in bounded-size record
    batches and a ``ParquetWriter`` to append them, so peak memory is
    proportional to ``batch_size`` rather than to the whole file. This is
    what makes it safe to "load" multi-GB / multi-million-row TSVs on a
    machine with only a few GB of free RAM.
    """
    if parquet_path.exists() and not force:
        return parquet_path
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    read_options = pv_csv.ReadOptions(block_size=64 << 20)  # 64MB blocks
    parse_options = pv_csv.ParseOptions(delimiter="\t")
    convert_options = pv_csv.ConvertOptions(
        column_types={c: pa.string() for c in schema_columns},
        strings_can_be_null=False,
    )

    with timer(f"convert {tsv_path.name} -> parquet"):
        reader = pv_csv.open_csv(
            str(tsv_path),
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        )
        writer: Optional[pq.ParquetWriter] = None
        n_rows = 0
        tmp_path = parquet_path.with_suffix(".tmp")
        try:
            for batch in reader:
                table = pa.Table.from_batches([batch])
                # Keep only the expected columns, in a fixed order.
                table = table.select(list(schema_columns))
                if writer is None:
                    writer = pq.ParquetWriter(str(tmp_path), table.schema, compression="snappy")
                writer.write_table(table)
                n_rows += table.num_rows
        finally:
            if writer is not None:
                writer.close()
        os.replace(tmp_path, parquet_path)
        log(f"  wrote {n_rows:,} rows -> {parquet_path}")
    return parquet_path


def ensure_source_cache(split: str, source: str) -> Path:
    """Ensure a parquet cache exists for one (split, source) file; return its path.

    ``split`` is "train" or "test"; ``source`` is "source1"/"source2"/"source3".
    """
    raw_map = {
        ("train", "source1"): config.TRAIN_SOURCE1,
        ("train", "source2"): config.TRAIN_SOURCE2,
        ("train", "source3"): config.TRAIN_SOURCE3,
        ("test", "source1"): config.TEST_SOURCE1,
        ("test", "source2"): config.TEST_SOURCE2,
        ("test", "source3"): config.TEST_SOURCE3,
    }
    cache_dir = config.TRAIN_CACHE_DIR if split == "train" else config.TEST_CACHE_DIR
    tsv_path = raw_map[(split, source)]
    parquet_path = _cache_path(cache_dir, source)
    return convert_tsv_to_parquet(tsv_path, parquet_path, SOURCE_COLUMNS)


def load_source(
    split: str, source: str, columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """Load a source table (source1/2/3, train/test) from its parquet cache.

    Only requested ``columns`` are read off disk (columnar pruning), which
    matters a lot at this scale (multi-million-row tables).
    """
    path = ensure_source_cache(split, source)
    cols = columns or SOURCE_COLUMNS
    df = pd.read_parquet(path, columns=cols)
    return df


def load_source_ids(split: str, source: str) -> pd.Series:
    """Cheaply load ONLY the entity_id column for a source table.

    Column-pruned parquet reads make this fast and light even for the
    ~5M-row Source-2/Source-3 tables (a single string column, not the full
    row) -- used to sample/select Source-1 entities and to build background
    pools without ever materialising full name/address text for rows we
    won't use.
    """
    return load_source(split, source, columns=["entity_id"])["entity_id"]


def load_source_subset(
    split: str, source: str, entity_ids, columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """Load only the rows whose ``entity_id`` is in ``entity_ids`` (predicate pushdown).

    Uses PyArrow's filtered ``read_table`` so memory is proportional to the
    (small) result size, not to the full multi-million-row source file --
    this is what makes small ``--train-sample-size`` runs fast: instead of
    normalizing the entire ~5M-row Source-2/Source-3 tables, only the
    hundreds/thousands of rows actually needed for a sampled training pool
    (true matches + a background sample, see ``train.build_training_pool``)
    are ever read off disk or normalized.

    Falls back to a full load + pandas-side filter if the id set is so large
    that pushdown filtering would not be worth it (e.g. > 60% of the table).
    """
    path = ensure_source_cache(split, source)
    cols = columns or SOURCE_COLUMNS
    ids = list(entity_ids) if not isinstance(entity_ids, (list, set)) else (entity_ids if isinstance(entity_ids, list) else list(entity_ids))
    if not ids:
        return pd.DataFrame(columns=cols)

    table = pq.read_table(str(path), columns=cols, filters=[("entity_id", "in", ids)])
    return table.to_pandas()


def ensure_ground_truth_cache(force: bool = False) -> Path:
    """Ensure a parquet cache exists for the raw (non-exploded) ground truth."""
    parquet_path = _cache_path(config.TRAIN_CACHE_DIR, "ground_truth")
    return convert_tsv_to_parquet(
        config.TRAIN_GROUND_TRUTH, parquet_path, GT_COLUMNS, force=force
    )


def load_ground_truth() -> pd.DataFrame:
    """Load the raw ground truth: one row per Source-1 entity, comma-joined matches."""
    path = ensure_ground_truth_cache()
    return pd.read_parquet(path, columns=GT_COLUMNS)


def ensure_ground_truth_exploded_cache(force: bool = False) -> Path:
    """Build (once) a long-format ground truth: one row per matched pair.

    Streams the raw ground-truth parquet in row-group batches, splits
    ``matched_entity_ids`` on commas, and writes an exploded parquet with
    columns (source1_entity_id, matched_entity_id). Empty-match rows produce
    no exploded rows (they are singletons; handled separately by whoever
    needs the full S1 id universe).
    """
    out_path = _cache_path(config.TRAIN_CACHE_DIR, "ground_truth_exploded")
    if out_path.exists() and not force:
        return out_path

    raw_path = ensure_ground_truth_cache()
    with timer("explode ground truth into pairs"):
        pf = pq.ParquetFile(str(raw_path))
        writer: Optional[pq.ParquetWriter] = None
        tmp_path = out_path.with_suffix(".tmp")
        n_pairs = 0
        try:
            for batch in pf.iter_batches(batch_size=200_000):
                df = batch.to_pandas()
                df = df[df["matched_entity_ids"].str.len() > 0]
                if df.empty:
                    continue
                exploded = df.assign(
                    matched_entity_id=df["matched_entity_ids"].str.split(",")
                ).explode("matched_entity_id")[["source1_entity_id", "matched_entity_id"]]
                table = pa.Table.from_pandas(exploded, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(tmp_path), table.schema, compression="snappy")
                writer.write_table(table)
                n_pairs += table.num_rows
        finally:
            if writer is not None:
                writer.close()
        os.replace(tmp_path, out_path)
        log(f"  exploded ground truth -> {n_pairs:,} true pairs")
    return out_path


def load_ground_truth_exploded() -> pd.DataFrame:
    """Load the long-format (one row per true match pair) ground truth."""
    path = ensure_ground_truth_exploded_cache()
    return pd.read_parquet(path)


def ensure_normalized_cache(split: str, source: str, force: bool = False, n_jobs: int = 1) -> Path:
    """Ensure a parquet cache exists with every normalized representation for one table.

    Computes ``normalization.add_all_normalizations`` once and caches the
    result (including list-typed columns such as ``name_tokens``, which
    round-trip through parquet fine via the pyarrow engine). ``n_jobs`` is
    forwarded to a process-parallel implementation for full-scale runs on a
    many-vCPU machine (see ``utils.parallel_map_df``); it defaults to 1
    in-process for small/dev inputs.
    """
    from . import normalization  # local import: avoids a cycle at module import time
    from .utils import parallel_map_df

    cache_dir = config.TRAIN_CACHE_DIR if split == "train" else config.TEST_CACHE_DIR
    out_path = _cache_path(cache_dir, f"{source}_normalized")
    if out_path.exists() and not force:
        return out_path

    raw = load_source(split, source)
    with timer(f"normalize {split}/{source} (n_jobs={n_jobs})"):
        normalized = parallel_map_df(raw, normalization.add_all_normalizations, n_jobs=n_jobs)
    normalized.to_parquet(out_path, index=False)
    log(f"  wrote normalized cache -> {out_path}")
    return out_path


def load_normalized_source(
    split: str, source: str, n_jobs: int = 1, columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """Load a normalized table, optionally reading only selected parquet columns."""
    path = ensure_normalized_cache(split, source, n_jobs=n_jobs)
    return pd.read_parquet(path, columns=columns)


def build_all_caches() -> None:
    """Convenience: materialise every parquet cache once (train + test)."""
    for split, sources in (("train", ("source1", "source2", "source3")), ("test", ("source1", "source2", "source3"))):
        for src in sources:
            ensure_source_cache(split, src)
    ensure_ground_truth_cache()
    ensure_ground_truth_exploded_cache()


if __name__ == "__main__":
    build_all_caches()
