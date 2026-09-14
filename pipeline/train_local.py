"""Run an AutoData training job on user-selected local GPUs.

The caller provides a directory of materialized training parquets. This module
creates the directory layout nanochat expects, links the shared tokenizer and
evaluation assets, launches nanochat, and returns the final metrics.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import time
from pathlib import Path


# Keep these settings aligned with pipeline/modal_train.py. They reproduce the
# published runs; callers can override the memory-sensitive values.
SIZE_RECIPE = {
    "d8": dict(depth=8, ratio=10, device_batch_size=16, max_seq_len=1024,
               total_batch_size=131072, timeout_s=5400, fp8=True),
    "d12": dict(depth=12, ratio=10, device_batch_size=32, max_seq_len=2048,
                total_batch_size=524288, timeout_s=7200, fp8=True),
    "d16": dict(depth=16, ratio=10, device_batch_size=32, max_seq_len=None,
                total_batch_size=None, timeout_s=10800, fp8=True),
    "d20": dict(depth=20, ratio=10, device_batch_size=32, max_seq_len=None,
                total_batch_size=None, timeout_s=10800, fp8=True),
    "d24": dict(depth=24, ratio=8, device_batch_size=16, max_seq_len=None,
                total_batch_size=None, timeout_s=14400, fp8=True),
}


def parse_gpus(value: str) -> list[int]:
    """Parse a comma-separated list such as ``0,1,2,3``."""
    try:
        gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"invalid GPU list {value!r}; expected e.g. 0,1,2,3") from exc
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError(f"invalid GPU list {value!r}; GPU IDs must be unique non-negative integers")
    return gpus


def _replace_link(link: Path, target: Path) -> None:
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"refusing to replace existing path: {link}")
    link.symlink_to(target)


def _prepare_nanochat_base(
    base_dir: Path,
    train_dir: Path,
    val_shard: Path,
    cache_dir: Path,
) -> None:
    """Build an isolated nanochat base directory without copying the corpus."""
    tokenizer = cache_dir / "tokenizer"
    if not tokenizer.exists():
        raise FileNotFoundError(
            f"nanochat tokenizer not found at {tokenizer}; run `python -m scripts.tok_train` "
            "in the nanochat environment"
        )
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "base_checkpoints").mkdir(exist_ok=True)

    data_dir = base_dir / "base_data_climbmix"
    data_dir.mkdir(exist_ok=True)
    existing = list(data_dir.iterdir())
    if existing:
        raise FileExistsError(
            f"training data view is not empty: {data_dir}; choose a new --key or remove it"
        )

    parquets = sorted(train_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"no parquet files found in {train_dir}")
    for index, parquet in enumerate(parquets):
        (data_dir / f"shard_{index:05d}.parquet").symlink_to(parquet.resolve())
    # nanochat treats the lexicographically last shard as validation data.
    (data_dir / f"shard_{len(parquets):05d}.parquet").symlink_to(val_shard.resolve())

    _replace_link(base_dir / "tokenizer", tokenizer.resolve())
    # CORE downloads its bundle automatically when it is absent. Reuse a
    # shared copy when the user has already prepared one.
    for name in ("eval_bundle", "annotation"):
        target = cache_dir / name
        if target.exists():
            _replace_link(base_dir / name, target.resolve())


def _training_command(
    nanochat_root: Path,
    size: str,
    run_tag: str,
    n_gpus: int,
    ratio: int | None,
    fp8: bool | None,
    device_batch_size: int | None,
    total_batch_size: int | None,
    max_seq_len: int | None,
) -> tuple[list[str], int]:
    recipe = SIZE_RECIPE[size]
    python = nanochat_root / ".venv" / "bin" / "python"
    torchrun = nanochat_root / ".venv" / "bin" / "torchrun"
    if not python.exists():
        raise FileNotFoundError(f"nanochat Python not found at {python}; run `uv sync` in nanochat")

    if n_gpus == 1:
        launcher = [str(python), "-m", "scripts.base_train"]
    else:
        if not torchrun.exists():
            raise FileNotFoundError(f"torchrun not found at {torchrun}; run `uv sync` in nanochat")
        launcher = [
            str(torchrun), "--standalone", f"--nproc_per_node={n_gpus}",
            "-m", "scripts.base_train", "--",
        ]

    args = [
        f"--depth={recipe['depth']}",
        f"--device-batch-size={device_batch_size or recipe['device_batch_size']}",
        f"--target-param-data-ratio={ratio or recipe['ratio']}",
        f"--run={run_tag}",
        f"--model-tag={run_tag}",
        "--eval-every=999999",
        "--eval-tokens=2097152",
        "--core-metric-every=999999",
        "--sample-every=-1",
        "--save-every=-1",
    ]
    use_fp8 = recipe["fp8"] if fp8 is None else fp8
    if use_fp8:
        args.append("--fp8")
    resolved_max_seq = max_seq_len or recipe.get("max_seq_len")
    resolved_total_batch = total_batch_size or recipe.get("total_batch_size")
    if resolved_max_seq:
        args.append(f"--max-seq-len={resolved_max_seq}")
    if resolved_total_batch:
        args.append(f"--total-batch-size={resolved_total_batch}")
    return launcher + args, int(recipe["timeout_s"])


def train_local(
    *,
    key: str,
    size: str,
    parquet_key: str,
    train_seed: int,
    train_dir: Path,
    val_shard: Path,
    nanochat_root: Path,
    output_root: Path,
    gpus: list[int],
    ratio: int | None = None,
    fp8: bool | None = None,
    device_batch_size: int | None = None,
    total_batch_size: int | None = None,
    max_seq_len: int | None = None,
    timeout_hours: float | None = None,
) -> dict:
    """Run one local training job and return its final metrics."""
    if size not in SIZE_RECIPE:
        raise ValueError(f"unknown size {size!r}; choose from {sorted(SIZE_RECIPE)}")
    if not train_dir.is_dir():
        raise FileNotFoundError(f"materialized selection not found: {train_dir}")
    if not val_shard.is_file():
        raise FileNotFoundError(f"validation shard not found: {val_shard}")

    nanochat_root = nanochat_root.resolve()
    train_dir = train_dir.resolve()
    val_shard = val_shard.resolve()
    output_root = output_root.resolve()
    run_tag = f"{key}_{size}_s{train_seed}"
    run_dir = output_root / run_tag
    base_dir = run_dir / "nanochat"
    log_path = run_dir / "train.log"
    cmd, timeout_s = _training_command(
        nanochat_root, size, run_tag, len(gpus), ratio, fp8,
        device_batch_size, total_batch_size, max_seq_len,
    )
    if run_dir.exists():
        raise FileExistsError(
            f"local run already exists: {run_dir}; use a new --key to avoid overwriting it"
        )

    cache_dir = Path(
        os.environ.get("NANOCHAT_CACHE_DIR", str(Path.home() / ".cache" / "nanochat"))
    )
    _prepare_nanochat_base(base_dir, train_dir, val_shard, cache_dir)
    if timeout_hours is not None:
        timeout_s = int(timeout_hours * 3600)
    else:
        # Published timeouts assume eight GPUs.
        timeout_s *= math.ceil(8 / len(gpus))

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
        "NANOCHAT_BASE_DIR": str(base_dir),
        "NANOCHAT_SEED": str(train_seed),
        "OMP_NUM_THREADS": "1",
        "WANDB_MODE": "disabled",
    })
    print(f"[local] {run_tag}: GPUs {gpus}; log {log_path}", flush=True)
    started = time.time()
    try:
        with log_path.open("w") as log:
            result = subprocess.run(
                cmd,
                cwd=nanochat_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        exit_code = -1

    text = log_path.read_text()
    bpbs = re.findall(r"Validation bpb:\s*([\d.]+)", text)
    cores = re.findall(r"CORE metric:\s*([\d.]+)", text)
    result = {
        "backend": "local",
        "size": size,
        "parquet_key": parquet_key,
        "train_seed": train_seed,
        "gpus": gpus,
        "val_bpb": float(bpbs[-1]) if bpbs else None,
        "core": float(cores[-1]) if cores else None,
        "exit": exit_code,
        "wall_min": (time.time() - started) / 60,
        "log_path": str(log_path),
        "checkpoint_dir": str(base_dir / "base_checkpoints" / run_tag),
    }
    print(
        f"[local] {run_tag}: exit={result['exit']} wall={result['wall_min']:.1f}m "
        f"bpb={result['val_bpb']} CORE={result['core']}",
        flush=True,
    )
    return result
