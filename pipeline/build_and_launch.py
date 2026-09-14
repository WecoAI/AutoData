"""Build a recipe and train it locally or on Modal.

Given a path to a `data_select_*.py` (a recipe — either from `recipes/` or
a fresh WECO step), this script:

  1. Loads + calls select_docs(BUDGET, seed=42) → 14.37M doc indices
  2. Optionally sub-samples for d8/d12 (Option-B variance protocol)
  3. Materialises the parquet sets locally
  4. Trains on user-selected local GPUs (default), or uploads to Modal
  5. Saves metrics or Modal call handles as JSON

Usage:
  python -m pipeline.build_and_launch \\
      --selector recipes/autodata_core_gpt55_step105.py \\
      --key core_gpt55_step105 \\
      --sizes d12 d24 \\
      --train-seeds 42 43 44 \\
      --gpus 0,1,2,3,4,5,6,7
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.train_local import parse_gpus, train_local

POOL_DIR     = Path(os.environ["AUTODATA_POOL_DIR"])
SCRATCH_VOL  = os.environ.get("AUTODATA_SCRATCH_VOL", "autodata-scratch")
MODAL_APP    = os.environ.get("AUTODATA_MODAL_APP",   "autodata-train")
LOCAL_RUNS_DIR = Path(os.environ.get("AUTODATA_LOCAL_RUNS_DIR", "results/local_runs"))

STAGE_DIR    = Path(os.environ.get("AUTODATA_STAGE_DIR", "/dev/shm/autodata_stage"))
INDEX_DIR    = STAGE_DIR / "indices"
PARQUET_DIR  = STAGE_DIR / "parquets"
N_OUT_BUCKETS = 50

# Sub-sample sizes per recipe size; d16/d20/d24 use full selection.
SUBSAMPLE = {"d8": 880_000, "d12": 2_310_000}


def load_selector(path: Path):
    """Import the selector .py and return (module, BUDGET)."""
    spec = importlib.util.spec_from_file_location(f"sel_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.pop(spec.name, None)
    spec.loader.exec_module(mod)
    return mod, int(mod.BUDGET)


def build_indices(mod, budget: int, sub_seeds_d8: list, sub_seeds_d12: list, key: str) -> dict:
    """Cache full + sub-sample indices to INDEX_DIR. Returns {key: np.ndarray}."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    full_path = INDEX_DIR / f"{key}_full__index.npy"
    if full_path.exists():
        full_idx = np.load(full_path)
        print(f"  cached {key}_full: {full_idx.shape[0]:,}")
    else:
        print(f"[select] running select_docs(budget={budget:,}, seed=42)...")
        t0 = time.time()
        full_idx = mod.select_docs(budget=budget, seed=42)
        np.save(full_path, full_idx)
        print(f"  built {key}_full: {full_idx.shape[0]:,} in {time.time()-t0:.1f}s")
    indices = {f"{key}_full": full_idx}
    for ss in sub_seeds_d8:
        sub_key = f"{key}_d8sub{ss}"
        sp = INDEX_DIR / f"{sub_key}__index.npy"
        if sp.exists():
            indices[sub_key] = np.load(sp)
        else:
            rng = np.random.default_rng(2025 + ss)
            sub = np.sort(rng.choice(full_idx, size=SUBSAMPLE["d8"], replace=False)).astype(np.int64)
            np.save(sp, sub); indices[sub_key] = sub
            print(f"  built  {sub_key}: {sub.shape[0]:,}")
    for ss in sub_seeds_d12:
        sub_key = f"{key}_d12sub{ss}"
        sp = INDEX_DIR / f"{sub_key}__index.npy"
        if sp.exists():
            indices[sub_key] = np.load(sp)
        else:
            rng = np.random.default_rng(2025 + ss)
            sub = np.sort(rng.choice(full_idx, size=SUBSAMPLE["d12"], replace=False)).astype(np.int64)
            np.save(sp, sub); indices[sub_key] = sub
            print(f"  built  {sub_key}: {sub.shape[0]:,}")
    return indices


def materialize_all(indices: dict):
    train_files = sorted(p for p in os.listdir(POOL_DIR)
                         if p.startswith("shard_") and p.endswith(".parquet")
                         and not p.endswith("_06542.parquet"))
    n_shards = len(train_files)
    counts = [0] * n_shards
    with ThreadPoolExecutor(max_workers=32) as ex:
        for i, n in ex.map(lambda i: (i, pq.ParquetFile(POOL_DIR / train_files[i]).metadata.num_rows),
                            range(n_shards)):
            counts[i] = n
    offsets = np.zeros(n_shards + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    src_buckets = {k: {} for k in indices}
    for k, idx in indices.items():
        sf = np.clip(np.searchsorted(offsets[1:], idx, side="right"), 0, n_shards - 1)
        for shard_i in np.unique(sf):
            mask = sf == shard_i
            src_buckets[k][int(shard_i)] = (idx[mask] - offsets[shard_i]).astype(np.int64)
    src_to_out = np.asarray([(i * N_OUT_BUCKETS) // n_shards for i in range(n_shards)])

    for k in indices:
        d = PARQUET_DIR / k
        d.mkdir(parents=True, exist_ok=True)
        for p in d.iterdir():
            if p.is_file(): p.unlink()

    print(f"\n[materialize] {len(indices)} keys × {N_OUT_BUCKETS} buckets")
    t_phase = time.time()
    for out_b in range(N_OUT_BUCKETS):
        contributing = [i for i in range(n_shards) if src_to_out[i] == out_b]
        per_key = {k: [] for k in indices}
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = [ex.submit(lambda i=sh: (i, pq.read_table(POOL_DIR / train_files[i]))) for sh in contributing]
            for fut in as_completed(futs):
                shard_i, tbl = fut.result()
                for k in indices:
                    if shard_i in src_buckets[k]:
                        per_key[k].append((shard_i, tbl.take(pa.array(src_buckets[k][shard_i]))))
        for k in indices:
            tables = per_key[k]
            if not tables: continue
            tables.sort(key=lambda x: x[0])
            big = pa.concat_tables([t for _, t in tables])
            pq.write_table(big, PARQUET_DIR / k / f"bucket_{out_b:03d}.parquet",
                           compression="snappy", row_group_size=1024)
        rate = (out_b + 1) / (time.time() - t_phase)
        eta = (N_OUT_BUCKETS - out_b - 1) / rate / 60 if rate > 0 else 0
        if (out_b + 1) % 5 == 0 or out_b == N_OUT_BUCKETS - 1:
            print(f"  bucket {out_b+1}/{N_OUT_BUCKETS}  rate={rate:.2f}/s eta~{eta:.1f}m", flush=True)
    print(f"[materialize] DONE wall {(time.time()-t_phase)/60:.1f}m")


def upload_to_modal(indices: dict):
    import modal

    vol = modal.Volume.from_name(SCRATCH_VOL)
    print(f"\n[upload] {len(indices)} keys → modal volume {SCRATCH_VOL}")
    def _upload_one(k):
        files = sorted((PARQUET_DIR / k).iterdir())
        sz = sum(f.stat().st_size for f in files) / 1e6
        t0 = time.time()
        with vol.batch_upload(force=True) as batch:
            for f in files:
                batch.put_file(str(f), f"main_baselines/{k}/parquets/{f.name}")
        return k, sz, (time.time()-t0)/60
    with ThreadPoolExecutor(max_workers=min(8, len(indices))) as ex:
        for k, sz, dt in ex.map(_upload_one, list(indices)):
            print(f"  [{k}] {sz:.0f} MB in {dt:.1f}m")


def spawn_modal_trainings(key: str, sizes: list, train_seeds: list, sub_seeds_d8: list,
                          sub_seeds_d12: list, parallel: bool, ratio: int = None):
    import modal

    train_row = modal.Function.from_name(MODAL_APP, "train_row")
    handles = []
    extra = {"ratio": ratio} if ratio else {}

    def _jobs_for_size(size):
        if size == "d8":   return [{"size":"d8",  "parquet_key": f"{key}_d8sub{ss}",  "train_seed": 42, **extra} for ss in sub_seeds_d8]
        if size == "d12":  return [{"size":"d12", "parquet_key": f"{key}_d12sub{ss}", "train_seed": 42, **extra} for ss in sub_seeds_d12]
        # d16 / d20 / d24 → full
        return [{"size": size, "parquet_key": f"{key}_full", "train_seed": ts, **extra} for ts in train_seeds]

    if parallel:  # one container per (size, seed) — fastest
        for size in sizes:
            for j in _jobs_for_size(size):
                row = f"{key}_{size}_s{j['train_seed']}"
                if "sub" in j["parquet_key"]:
                    row = f"{key}_{size}_{j['parquet_key'].split('_')[-1]}"
                fc = train_row.spawn(row, [j])
                handles.append({"row_name": row, "call_id": fc.object_id, "jobs": [j]})
                print(f"  [{row}] spawned {fc.object_id}")
    else:  # one container per size — packs all seeds serially
        for size in sizes:
            jobs = _jobs_for_size(size)
            row = f"{key}_{size}"
            fc = train_row.spawn(row, jobs)
            handles.append({"row_name": row, "call_id": fc.object_id, "jobs": jobs})
            print(f"  [{row}] spawned {fc.object_id}  ({len(jobs)} jobs serial)")
    return handles


def local_jobs(key: str, size: str, train_seeds: list, sub_seeds_d8: list,
               sub_seeds_d12: list) -> list[dict]:
    if size == "d8":
        return [
            {"run_key": f"{key}_d8sub{seed}", "parquet_key": f"{key}_d8sub{seed}",
             "train_seed": 42}
            for seed in sub_seeds_d8
        ]
    if size == "d12":
        return [
            {"run_key": f"{key}_d12sub{seed}", "parquet_key": f"{key}_d12sub{seed}",
             "train_seed": 42}
            for seed in sub_seeds_d12
        ]
    return [
        {"run_key": key, "parquet_key": f"{key}_full", "train_seed": seed}
        for seed in train_seeds
    ]


def run_local_trainings(args, gpus: list[int]) -> list[dict]:
    nanochat_root = Path(os.environ["AUTODATA_NANOCHAT_ROOT"])
    val_shard = Path(os.environ["AUTODATA_VAL_SHARD"])
    results = []
    for size in args.sizes:
        jobs = local_jobs(
            args.key, size, args.train_seeds, args.sub_seeds_d8, args.sub_seeds_d12
        )
        for job in jobs:
            results.append(train_local(
                key=job["run_key"],
                size=size,
                parquet_key=job["parquet_key"],
                train_seed=job["train_seed"],
                train_dir=PARQUET_DIR / job["parquet_key"],
                val_shard=val_shard,
                nanochat_root=nanochat_root,
                output_root=Path(args.local_runs_dir),
                gpus=gpus,
                ratio=args.ratio,
                fp8=args.fp8,
                device_batch_size=args.device_batch_size,
                total_batch_size=args.total_batch_size,
                max_seq_len=args.max_seq_len,
                timeout_hours=args.timeout_hours,
            ))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selector", required=True, help="path to a select_docs .py")
    p.add_argument("--key", required=True, help="short identifier used in parquet paths and log names")
    p.add_argument("--sizes", nargs="+", default=["d24"], choices=["d8","d12","d16","d20","d24"])
    p.add_argument("--train-seeds", nargs="+", type=int, default=[42, 43, 44],
                   help="for full-selection trainings (d16/d20/d24)")
    p.add_argument("--sub-seeds-d8", nargs="+", type=int, default=[42, 43, 44],
                   help="for d8 sub-sample selection variance")
    p.add_argument("--sub-seeds-d12", nargs="+", type=int, default=[42, 43, 44],
                   help="for d12 sub-sample selection variance")
    p.add_argument("--parallel", action="store_true",
                   help="Modal only: one container per (size, seed)")
    p.add_argument("--backend", choices=["local", "modal"],
                   default=os.environ.get("AUTODATA_TRAIN_BACKEND", "local"),
                   help="training backend (default: local)")
    p.add_argument("--gpus", default=os.environ.get(
        "AUTODATA_TRAIN_GPUS", "0,1,2,3,4,5,6,7"),
        help="local GPU IDs, comma-separated")
    p.add_argument("--local-runs-dir", default=str(LOCAL_RUNS_DIR),
                   help="local logs and checkpoints directory")
    p.add_argument("--out", "--out-handles", dest="out", default=None,
                   help="JSON output path (default: pipeline/<key>_<backend>.json)")
    p.add_argument("--ratio", type=int, default=None,
                   help="nanochat --target-param-data-ratio: training tokens per "
                        "scaling-parameter. Default: the per-size value in "
                        "the training recipe (r=10 for d8-d20, r=8 for d24)")
    fp8 = p.add_mutually_exclusive_group()
    fp8.add_argument("--fp8", dest="fp8", action="store_const", const=True,
                     help="local only: force FP8 training")
    fp8.add_argument("--no-fp8", dest="fp8", action="store_const", const=False,
                     help="local only: disable FP8")
    p.set_defaults(fp8=None)
    p.add_argument("--device-batch-size", type=int, default=None,
                   help="local only: override the recipe's per-GPU batch size")
    p.add_argument("--total-batch-size", type=int, default=None,
                   help="local only: override the recipe's total token batch size")
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="local only: override the recipe's maximum sequence length")
    p.add_argument("--timeout-hours", type=float, default=None,
                   help="local per-job timeout; default scales with the GPU count")
    p.add_argument("--budget", type=int, default=None,
                   help="documents to select (default: the recipe's own BUDGET). May be "
                        "lowered to train on less data, but never raised above the budget "
                        "the recipe was searched under.")
    args = p.parse_args()

    if args.backend == "local" and args.parallel:
        p.error("--parallel is only supported by the Modal backend; local jobs run serially")
    if args.backend == "modal":
        local_only = {
            "fp8": args.fp8,
            "device-batch-size": args.device_batch_size,
            "total-batch-size": args.total_batch_size,
            "max-seq-len": args.max_seq_len,
            "timeout-hours": args.timeout_hours,
        }
        used = [f"--{name}" for name, value in local_only.items() if value is not None]
        if used:
            p.error(f"{', '.join(used)} only applies to the local backend")
    for name in ("ratio", "device_batch_size", "total_batch_size", "max_seq_len", "timeout_hours"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    try:
        gpus = parse_gpus(args.gpus) if args.backend == "local" else []
    except ValueError as exc:
        p.error(str(exc))

    mod, recipe_budget = load_selector(Path(args.selector))
    budget = args.budget or recipe_budget

    # The selector was optimised against `recipe_budget`; its thresholds and keep
    # rates are calibrated to that regime. Selecting MORE than it was searched for
    # applies the rule outside its validated range, so refuse rather than silently
    # extrapolate.
    if budget > recipe_budget:
        raise SystemExit(
            f"--budget {budget:,} exceeds the recipe's searched budget "
            f"{recipe_budget:,} ({Path(args.selector).name}). The instructions that "
            f"produced this recipe declare the budget; selecting beyond it runs the "
            f"rule outside the regime it was validated in. Lower --budget, or re-run "
            f"the search with a larger budget declared in search/instructions/.")

    # Sub-samples are drawn WITHOUT replacement from the selection, so the budget
    # has to be at least as large as the sub-sample each requested size needs.
    for size in ("d8", "d12"):
        if size in args.sizes and budget < SUBSAMPLE[size]:
            raise SystemExit(
                f"--budget {budget:,} is smaller than the {size} sub-sample "
                f"({SUBSAMPLE[size]:,} docs). Drop {size} from --sizes or raise --budget.")

    if budget != recipe_budget:
        print(f"[budget] {budget:,} docs (recipe declares {recipe_budget:,})")
    indices = build_indices(mod, budget,
                            args.sub_seeds_d8 if "d8" in args.sizes else [],
                            args.sub_seeds_d12 if "d12" in args.sizes else [],
                            args.key)

    materialize_all(indices)

    if args.ratio:
        print(f"[ratio] --target-param-data-ratio={args.ratio}")
    if args.backend == "modal":
        upload_to_modal(indices)
        print(f"\n[launch] spawning Modal containers (parallel={args.parallel})")
        records = spawn_modal_trainings(
            args.key, args.sizes, args.train_seeds,
            args.sub_seeds_d8, args.sub_seeds_d12, args.parallel,
            ratio=args.ratio,
        )
    else:
        print(f"\n[launch] training locally on GPUs {gpus}; jobs run serially")
        records = run_local_trainings(args, gpus)

    out = Path(args.out or f"pipeline/{args.key}_{args.backend}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2))
    print(f"\nresults → {out}")


if __name__ == "__main__":
    main()
