"""Persistent SQLite indexes for test-set hash blocking.

The raw S2/S3 TSVs are normalized in bounded Arrow batches. The resulting
records and rule-key postings live on disk; each S1 batch performs indexed
lookups instead of joining against every row in the other source.
"""
from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv_csv

from . import blocking, config
from .data_loader import SOURCE_COLUMNS
from .utils import log, timer

INDEX_VERSION = 1
RARE_TOKEN_LIMIT = 50
COMPACT_PREFIX_LENGTH = 6
RECORD_COLUMNS = [
    "entity_id", "country_norm", "name_norm", "name_alnum", "name_core",
    "name_compact", "name_tokens", "name_sorted", "name_numbers",
    "address_norm", "address_alnum", "address_tokens", "address_numbers",
    "address_postal", "address_first_number", "name_first_token",
]
JSON_COLUMNS = {"name_tokens", "name_numbers", "address_tokens", "address_numbers"}
PAIR_RECORD_COLUMNS = [
    "entity_id", "country_norm", "name_norm", "name_alnum", "name_core",
    "name_compact", "name_tokens", "name_sorted", "name_numbers",
    "address_alnum", "address_tokens", "address_numbers", "address_postal",
]
RULE_NAMES = [fn.__name__.replace("block_", "") for fn in blocking.ALL_HASH_RULES] + ["rare_name_token"]


def _items(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _text(value) -> str:
    return "" if value is None else str(value)


def _manifest_for_sources(source_paths: Mapping[str, Path], max_block_size: int) -> dict:
    files = {}
    for source, path in sorted(source_paths.items()):
        stat = path.stat()
        files[source] = {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return {
        "index_version": INDEX_VERSION,
        "max_block_size": max_block_size,
        "rare_token_limit": RARE_TOKEN_LIMIT,
        "compact_prefix_length": COMPACT_PREFIX_LENGTH,
        "legal_suffixes": sorted(config.LEGAL_SUFFIXES),
        "sources": files,
    }


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        DROP TABLE IF EXISTS metadata;
        DROP TABLE IF EXISTS records;
        DROP TABLE IF EXISTS postings;
        DROP TABLE IF EXISTS token_counts;
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE records (
            source TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            entity_id TEXT NOT NULL,
            country_norm TEXT NOT NULL,
            name_norm TEXT NOT NULL,
            name_alnum TEXT NOT NULL,
            name_core TEXT NOT NULL,
            name_compact TEXT NOT NULL,
            name_tokens TEXT NOT NULL,
            name_sorted TEXT NOT NULL,
            name_numbers TEXT NOT NULL,
            address_norm TEXT NOT NULL,
            address_alnum TEXT NOT NULL,
            address_tokens TEXT NOT NULL,
            address_numbers TEXT NOT NULL,
            address_postal TEXT NOT NULL,
            address_first_number TEXT NOT NULL,
            name_first_token TEXT NOT NULL,
            PRIMARY KEY (source, entity_id),
            UNIQUE (source, ordinal)
        );
        CREATE TABLE postings (
            source TEXT NOT NULL,
            rule TEXT NOT NULL,
            key TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            PRIMARY KEY (source, rule, key, entity_id)
        ) WITHOUT ROWID;
        CREATE TABLE token_counts (
            source TEXT NOT NULL,
            token TEXT NOT NULL,
            occurrences INTEGER NOT NULL,
            PRIMARY KEY (source, token)
        ) WITHOUT ROWID;
        CREATE INDEX records_by_ordinal ON records(source, ordinal);
        """
    )


def _insert_normalized_batch(
    connection: sqlite3.Connection,
    source: str,
    start_ordinal: int,
    normalized: pd.DataFrame,
    max_block_size: int,
) -> int:
    del max_block_size  # block-size filtering is applied on query, matching _capped_merge.
    record_values = []
    posting_values = []
    token_counts: Counter[str] = Counter()
    columns = normalized.columns.get_indexer(RECORD_COLUMNS)

    for offset, row in enumerate(normalized.itertuples(index=False, name=None)):
        record = {name: row[position] for name, position in zip(RECORD_COLUMNS, columns)}
        entity_id = _text(record["entity_id"])
        record_values.append(
            (
                source, start_ordinal + offset, entity_id,
                _text(record["country_norm"]), _text(record["name_norm"]),
                _text(record["name_alnum"]), _text(record["name_core"]),
                _text(record["name_compact"]),
                json.dumps(_items(record["name_tokens"]), separators=(",", ":")),
                _text(record["name_sorted"]),
                json.dumps(_items(record["name_numbers"]), separators=(",", ":")),
                _text(record["address_norm"]), _text(record["address_alnum"]),
                json.dumps(_items(record["address_tokens"]), separators=(",", ":")),
                json.dumps(_items(record["address_numbers"]), separators=(",", ":")),
                _text(record["address_postal"]), _text(record["address_first_number"]),
                _text(record["name_first_token"]),
            )
        )

        name_tokens = _items(record["name_tokens"])
        keys = (
            ("exact_name", _text(record["name_norm"])),
            ("exact_address", _text(record["address_norm"])),
            (
                "name_first_token_country",
                f"{_text(record['name_first_token'])}|{_text(record['country_norm'])}",
            ),
            ("name_first_two_tokens", " ".join(name_tokens[:2]) if len(name_tokens) >= 2 else ""),
            (
                "address_number_country",
                f"{_text(record['address_first_number'])}|{_text(record['country_norm'])}",
            ),
            ("name_compact_prefix", _text(record["name_compact"])[:COMPACT_PREFIX_LENGTH]),
        )
        posting_values.extend((source, rule, key, entity_id) for rule, key in keys)

        valid_tokens = [_text(token) for token in name_tokens if token is not None and len(_text(token)) >= 3]
        token_counts.update(valid_tokens)
        posting_values.extend(
            (source, "rare_name_token", token, entity_id)
            for token in set(valid_tokens)
        )

    connection.executemany(
        "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", record_values,
    )
    connection.executemany(
        "INSERT OR IGNORE INTO postings VALUES (?,?,?,?)", posting_values,
    )
    connection.executemany(
        """INSERT INTO token_counts(source, token, occurrences) VALUES (?,?,?)
           ON CONFLICT(source, token) DO UPDATE SET occurrences=occurrences+excluded.occurrences""",
        ((source, token, count) for token, count in token_counts.items()),
    )
    connection.commit()
    return len(normalized)


class SQLiteBlockingIndex:
    """Disk-backed candidate index for Source-2/Source-3 test records."""

    def __init__(self, path: Path, max_block_size: int = config.MAX_BLOCK_SIZE):
        self.path = Path(path)
        self.max_block_size = max_block_size
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute("PRAGMA cache_size=-32768")
        self.connection.execute("PRAGMA mmap_size=268435456")
        self.connection.row_factory = sqlite3.Row

    @classmethod
    def build_from_normalized_frames(
        cls,
        path: Path,
        source_frames: Mapping[str, pd.DataFrame],
        max_block_size: int = config.MAX_BLOCK_SIZE,
    ) -> "SQLiteBlockingIndex":
        """Build an index from small, already-normalized frames (used by smoke tests)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        connection = sqlite3.connect(str(temporary))
        try:
            _create_schema(connection)
            for source, frame in source_frames.items():
                _insert_normalized_batch(connection, source, 0, frame, max_block_size)
            connection.execute(
                "INSERT INTO metadata(key,value) VALUES('manifest',?)",
                (json.dumps({"index_version": INDEX_VERSION, "synthetic": True}),),
            )
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, path)
        return cls(path, max_block_size)

    @classmethod
    def build_from_tsvs(
        cls,
        path: Path,
        source_paths: Mapping[str, Path],
        manifest: dict,
        max_block_size: int = config.MAX_BLOCK_SIZE,
    ) -> "SQLiteBlockingIndex":
        """Stream raw TSV batches through the shared normalizer into SQLite."""
        from . import normalization

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        connection = sqlite3.connect(str(temporary))
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        try:
            _create_schema(connection)
            for source, source_path in source_paths.items():
                with timer(f"build test blocking index for {source}"):
                    reader = pv_csv.open_csv(
                        str(source_path),
                        read_options=pv_csv.ReadOptions(block_size=8 << 20),
                        parse_options=pv_csv.ParseOptions(delimiter="\t"),
                        convert_options=pv_csv.ConvertOptions(
                            column_types={column: pa.string() for column in SOURCE_COLUMNS},
                            strings_can_be_null=True,
                        ),
                    )
                    ordinal = 0
                    for batch in reader:
                        raw = batch.to_pandas()
                        normalized = normalization.add_all_normalizations(raw)
                        ordinal += _insert_normalized_batch(
                            connection, source, ordinal, normalized, max_block_size,
                        )
                        if ordinal and ordinal % 100_000 < len(normalized):
                            log(f"  indexed {source}: {ordinal:,} rows")
                        del normalized, raw
            connection.execute(
                "INSERT INTO metadata(key,value) VALUES('manifest',?)",
                (json.dumps(manifest, sort_keys=True),),
            )
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, path)
        return cls(path, max_block_size)

    @classmethod
    def ensure_test_index(cls, max_block_size: int = config.MAX_BLOCK_SIZE) -> "SQLiteBlockingIndex":
        source_paths = {
            "source2": config.TEST_SOURCE2,
            "source3": config.TEST_SOURCE3,
        }
        manifest = _manifest_for_sources(source_paths, max_block_size)
        path = config.CACHE_DIR / "test_blocking_index.sqlite3"
        if path.exists():
            try:
                connection = sqlite3.connect(str(path))
                row = connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
                connection.close()
                if row and json.loads(row[0]) == manifest:
                    log(f"Reusing test blocking index: {path}")
                    return cls(path, max_block_size)
            except sqlite3.Error:
                pass
        log(f"Building reusable test blocking index: {path}")
        return cls.build_from_tsvs(path, source_paths, manifest, max_block_size)

    def close(self) -> None:
        self.connection.close()

    def candidates_for_chunk(self, s1_chunk: pd.DataFrame, source: str) -> pd.DataFrame:
        """Query all seven rules for a chunk with the existing merge-cap semantics."""
        by_rule_key: dict[str, dict[str, list[str]]] = {rule: defaultdict(list) for rule in RULE_NAMES}
        left_counts: dict[str, Counter[str]] = {rule: Counter() for rule in RULE_NAMES}

        for row in s1_chunk.itertuples(index=False):
            entity_id = _text(getattr(row, "entity_id"))
            tokens = _items(getattr(row, "name_tokens"))
            keys = (
                ("exact_name", _text(getattr(row, "name_norm"))),
                ("exact_address", _text(getattr(row, "address_norm"))),
                (
                    "name_first_token_country",
                    f"{_text(getattr(row, 'name_first_token'))}|{_text(getattr(row, 'country_norm'))}",
                ),
                ("name_first_two_tokens", " ".join(tokens[:2]) if len(tokens) >= 2 else ""),
                (
                    "address_number_country",
                    f"{_text(getattr(row, 'address_first_number'))}|{_text(getattr(row, 'country_norm'))}",
                ),
                ("name_compact_prefix", _text(getattr(row, "name_compact"))[:COMPACT_PREFIX_LENGTH]),
            )
            for rule, key in keys:
                by_rule_key[rule][key].append(entity_id)
                left_counts[rule][key] += 1
            valid_tokens = [_text(token) for token in tokens if token is not None and len(_text(token)) >= 3]
            left_counts["rare_name_token"].update(valid_tokens)
            for token in set(valid_tokens):
                by_rule_key["rare_name_token"][token].append(entity_id)

        pairs_by_rule: dict[str, set[tuple[str, str]]] = {rule: set() for rule in RULE_NAMES}
        for rule in RULE_NAMES:
            for key, s1_ids in by_rule_key[rule].items():
                if not key or left_counts[rule][key] > self.max_block_size:
                    continue
                if rule == "rare_name_token":
                    count_row = self.connection.execute(
                        "SELECT occurrences FROM token_counts WHERE source=? AND token=?",
                        (source, key),
                    ).fetchone()
                    if count_row is None or count_row[0] > min(RARE_TOKEN_LIMIT, self.max_block_size):
                        continue
                rows = self.connection.execute(
                    "SELECT entity_id FROM postings WHERE source=? AND rule=? AND key=? LIMIT ?",
                    (source, rule, key, self.max_block_size + 1),
                ).fetchall()
                if len(rows) > self.max_block_size:
                    continue
                unique_s1_ids = set(s1_ids)
                for row in rows:
                    pairs_by_rule[rule].update((s1_id, row[0]) for s1_id in unique_s1_ids)

        records = [
            {"entity_id_s1": s1_id, "entity_id_other": other_id, "rule": rule}
            for rule in RULE_NAMES
            for s1_id, other_id in pairs_by_rule[rule]
        ]
        return pd.DataFrame(records, columns=["entity_id_s1", "entity_id_other", "rule"])

    def fetch_records(self, source: str, entity_ids) -> pd.DataFrame:
        """Fetch only normalized feature records for candidate IDs."""
        ids = list(dict.fromkeys(_text(entity_id) for entity_id in entity_ids))
        if not ids:
            return pd.DataFrame(columns=PAIR_RECORD_COLUMNS)
        columns = ",".join(PAIR_RECORD_COLUMNS)
        rows = []
        for start in range(0, len(ids), 500):
            batch_ids = ids[start : start + 500]
            placeholders = ",".join("?" for _ in batch_ids)
            rows.extend(
                self.connection.execute(
                    f"SELECT {columns} FROM records WHERE source=? AND entity_id IN ({placeholders})",
                    (source, *batch_ids),
                ).fetchall()
            )
        result = pd.DataFrame.from_records(rows, columns=PAIR_RECORD_COLUMNS)
        for column in JSON_COLUMNS:
            if column in result:
                result[column] = result[column].map(json.loads)
        return result


def ensure_test_blocking_index(max_block_size: int = config.MAX_BLOCK_SIZE) -> SQLiteBlockingIndex:
    return SQLiteBlockingIndex.ensure_test_index(max_block_size)
