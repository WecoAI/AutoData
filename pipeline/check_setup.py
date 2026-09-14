"""Validate an AutoData installation without starting training."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


FEATURES = [
    "doc_tokens",
    "doc_chars",
    *(f"distinct_{n}gram_bpe" for n in range(1, 6)),
    "avg_distinct_ngram_bpe",
    "logppl_qwen",
    "topic_id",
    "format_id",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"setup check failed: {message}")


def main() -> None:
    names = [
        "AUTODATA_NANOCHAT_ROOT",
        "AUTODATA_POOL_DIR",
        "AUTODATA_VAL_SHARD",
        "AUTODATA_FEATURES_DIR",
    ]
    missing = [name for name in names if not os.environ.get(name)]
    require(not missing, f"source .env; missing {', '.join(missing)}")

    nanochat = Path(os.environ["AUTODATA_NANOCHAT_ROOT"])
    pool = Path(os.environ["AUTODATA_POOL_DIR"])
    val = Path(os.environ["AUTODATA_VAL_SHARD"])
    features = Path(os.environ["AUTODATA_FEATURES_DIR"])

    require((nanochat / ".venv/bin/python").is_file(), "nanochat environment is missing")
    cache = Path(os.environ.get("NANOCHAT_CACHE_DIR", Path.home() / ".cache/nanochat"))
    require((cache / "tokenizer").is_dir(), "nanochat tokenizer is missing")
    require(pool.is_dir(), f"pool directory not found: {pool}")
    require(val.is_file(), f"validation shard not found: {val}")
    require(features.is_dir(), f"feature directory not found: {features}")

    shards = sorted(pool.glob("shard_*.parquet"))
    train_shards = [path for path in shards if path.resolve() != val.resolve()]
    require(train_shards, f"no training shards found in {pool}")
    columns = pq.ParquetFile(train_shards[0]).schema_arrow.names
    require("text" in columns, f"{train_shards[0].name} has no 'text' column")

    lengths = {}
    for name in FEATURES:
        path = features / f"{name}.npy"
        require(path.is_file(), f"missing feature: {path}")
        array = np.load(path, mmap_mode="r")
        require(array.ndim == 1, f"feature must be one-dimensional: {path}")
        lengths[name] = len(array)
    require(len(set(lengths.values())) == 1, f"feature lengths differ: {lengths}")
    n_documents = next(iter(lengths.values()))
    require(n_documents > 0, "feature arrays are empty")

    offsets_path = features / "shard_offsets.json"
    require(offsets_path.is_file(), f"missing {offsets_path}")
    offsets = json.loads(offsets_path.read_text())
    require(offsets, "shard_offsets.json is empty")
    require(
        max(int(row["offset"]) + int(row["count"]) for row in offsets) == n_documents,
        "shard_offsets.json does not match the feature arrays",
    )

    selector_path = Path("search/data_select_template.py")
    spec = importlib.util.spec_from_file_location("autodata_setup_selector", selector_path)
    require(spec is not None and spec.loader is not None, f"cannot load {selector_path}")
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    budget = min(1000, n_documents)
    first = selector.select_docs(budget=budget, seed=42)
    second = selector.select_docs(budget=budget, seed=42)
    require(first.dtype == np.int64, f"selector returned {first.dtype}, expected int64")
    require(first.shape == (budget,), f"selector returned {first.shape}, expected ({budget},)")
    require(np.array_equal(first, second), "selector is not deterministic")
    require(np.all(first[:-1] < first[1:]), "selector IDs are not sorted and unique")
    require(first[0] >= 0 and first[-1] < n_documents, "selector IDs are out of range")

    print(
        f"setup OK: {len(train_shards):,} training shards, "
        f"{n_documents:,} feature rows, selector contract valid"
    )


if __name__ == "__main__":
    main()
