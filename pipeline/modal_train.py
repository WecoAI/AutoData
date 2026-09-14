"""Modal app — H100:8 DDP training for nanochat at d8/d12/d16/d20/d24.

Exposes `train_row(row_name, jobs)` where each job is
`{size, parquet_key, train_seed}`. Jobs run serially inside one container;
spawn multiple containers in parallel (via `train_row.spawn(...)` from your
builder script) for parallel trainings — up to your Modal H100:8 quota.

Volumes (create once via `modal volume create <name>`):
  AUTODATA_SCRATCH_VOL    holds pre-materialized parquets (selections) +
                          train_logs/ output. Default: 'autodata-scratch'
  AUTODATA_ARCHIVE_VOL    holds the val shard parquet that every training
                          uses for val_bpb. Default: 'autodata-archive'

Deploy after edits:
  modal deploy pipeline/modal_train.py
"""
import os
import modal

APP_NAME      = os.environ.get("AUTODATA_MODAL_APP",   "autodata-train")
SCRATCH_VOL   = os.environ.get("AUTODATA_SCRATCH_VOL", "autodata-scratch")
ARCHIVE_VOL   = os.environ.get("AUTODATA_ARCHIVE_VOL", "autodata-archive")
# Pools for external corpora (FineWeb-Edu / DCLM / Dolma / DeMix), built by
# pipeline/modal_corpora.py. Mounted at /corpora; jobs reach it by passing
# scratch_root="/corpora".
CORPORA_VOL   = os.environ.get("AUTODATA_CORPORA_VOL", "autodata-corpora")

# Container image: pin to nanochat's training requirements.
# Adapt the add_local_dir calls below to point at YOUR nanochat checkout
# (set AUTODATA_NANOCHAT_ROOT in .env).
NANOCHAT_ROOT = os.environ.get(
    "AUTODATA_NANOCHAT_ROOT",
    os.path.join(os.path.expanduser("~"), "nanochat"),
)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.5.1", "numpy>=1.26", "pyarrow", "tiktoken", "pyyaml",
        "regex", "wandb", "transformers", "rustbpe", "kernels>=0.11.7",
        "scipy", "tabulate", "zstandard", "psutil",
    )
    .add_local_dir(os.path.join(NANOCHAT_ROOT, "nanochat"), remote_path="/root/nanochat/nanochat")
    .add_local_dir(os.path.join(NANOCHAT_ROOT, "scripts"),  remote_path="/root/nanochat/scripts")
    .add_local_dir(os.path.join(NANOCHAT_ROOT, "tasks"),    remote_path="/root/nanochat/tasks")
    .add_local_dir(os.path.join(os.path.expanduser("~"), ".cache/nanochat/tokenizer"),
                   remote_path="/root/.cache/nanochat/tokenizer")
    .add_local_dir(os.path.join(os.path.expanduser("~"), ".cache/nanochat/eval_bundle"),
                   remote_path="/root/.cache/nanochat/eval_bundle")
)

# create_if_missing=False on purpose — see the same note in
# pipeline/modal_search_eval.py. Both volumes must already hold data (staged
# selections; the val shard), so auto-creation can only ever yield an EMPTY
# volume whose failure surfaces inside an H100:8 container. The real names come
# from .env and do NOT match the defaults above, so an unsourced deploy must
# fail here rather than train against nothing.
scratch = modal.Volume.from_name(SCRATCH_VOL, create_if_missing=False)
archive = modal.Volume.from_name(ARCHIVE_VOL, create_if_missing=False)
corpora = modal.Volume.from_name(CORPORA_VOL, create_if_missing=True)
app = modal.App(APP_NAME, image=image)

# Per-size training recipe — H100:8 DDP, FP8.
# `ratio` is nanochat's --target-param-data-ratio (tokens per scaling-param,
# not per total-param). See eval/recompute_training_config.py for the param
# breakdown that explains why r=8 at d24 produces 5.84B training tokens.
SIZE_RECIPE = {
    "d8":  dict(depth=8,  ratio=10, device_batch_size=16, max_seq_len=1024,
                total_batch_size=131072, timeout_s=5400, fp8=True),
    "d12": dict(depth=12, ratio=10, device_batch_size=32, max_seq_len=2048,
                total_batch_size=524288, timeout_s=7200, fp8=True),
    "d16": dict(depth=16, ratio=10, device_batch_size=32, max_seq_len=None,
                total_batch_size=None, timeout_s=10800, fp8=True),
    "d20": dict(depth=20, ratio=10, device_batch_size=32, max_seq_len=None,
                total_batch_size=None, timeout_s=10800, fp8=True),
    "d24": dict(depth=24, ratio=8,  device_batch_size=16, max_seq_len=None,
                total_batch_size=None, timeout_s=14400, fp8=True),
    # ~2B model: depth 28 (params ~ depth^3 from d24=1.3B -> ~2.05B). Smaller
    # device batch than d24 for the larger model; nanochat derives lr/total-batch
    # from depth when unset. Generous timeout for an 8xH100 DDP run.
    "d28": dict(depth=28, ratio=8,  device_batch_size=8,  max_seq_len=None,
                total_batch_size=None, timeout_s=28800, fp8=True),
}


@app.function(
    gpu="H100:8",
    # 24 h (Modal's max). The old 30000 s (8.3 h) covered a single d28 run, but
    # jobs in one call run SERIALLY: a 4-corpus d24 sweep is ~12.7 h wall, so the
    # container would have been killed part-way through job 3. Per-job limits
    # still come from SIZE_RECIPE[size]["timeout_s"].
    timeout=86400,
    memory=131072,
    volumes={"/scratch": scratch, "/archive": archive, "/corpora": corpora},
)
def train_row(row_name: str, jobs: list) -> list:
    """Run `jobs` serially in one H100:8 container. Returns a list of result dicts.

    Args:
      row_name: human-readable container label, used in log filenames
      jobs: list of dicts {size, parquet_key, train_seed}
        - size: one of 'd8' / 'd12' / 'd16' / 'd20' / 'd24'
        - parquet_key: directory key on the scratch volume holding the training
          parquets (under main_baselines/<parquet_key>/parquets/bucket_*.parquet)
        - train_seed: int (passed as NANOCHAT_SEED env var)

    Returns:
      [{size, parquet_key, train_seed, val_bpb, core, exit, wall_min, log_path}, ...]
    """
    import re, shutil, subprocess, time

    os.chdir("/root/nanochat")
    os.environ["HOME"] = "/root"
    os.environ["NANOCHAT_BASE_DIR"] = "/root/.cache/nanochat"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["WANDB_MODE"] = "disabled"

    results = []
    for job in jobs:
        size, parquet_key, ts = job["size"], job["parquet_key"], int(job["train_seed"])
        # Optional per-job overrides:
        #   core_every  — CORE eval cadence in steps. Default 999999 = final step
        #                 only (what every published run used). Set e.g. 1000 to
        #                 get a (tokens, CORE) curve across training.
        #   val_shard   — path to a val parquet, for corpora whose held-out set
        #                 is not ClimbMix's. Default keeps /archive/val_shard.parquet.
        #   scratch_root — volume dir holding <parquet_key>/parquets (default
        #                 main_baselines, i.e. the ClimbMix staging area).
        #   ratio       — nanochat's --target-param-data-ratio, i.e. how many
        #                 training tokens per scaling-parameter. Defaults to the
        #                 per-size value in SIZE_RECIPE. Raising it demands more
        #                 tokens than the default run; the selection must be large
        #                 enough to supply them or nanochat will re-read data.
        core_every  = int(job.get("core_every", 999999))
        val_shard   = job.get("val_shard") or "/archive/val_shard.parquet"
        scratch_root = job.get("scratch_root", "/scratch/main_baselines")
        if size not in SIZE_RECIPE:
            raise ValueError(f"Unknown size: {size}. Expected one of {list(SIZE_RECIPE)}.")
        recipe = SIZE_RECIPE[size]
        ratio = int(job.get("ratio") or recipe["ratio"])
        os.environ["NANOCHAT_SEED"] = str(ts)

        DATA_DIR = "/root/.cache/nanochat/base_data_climbmix"
        os.makedirs(DATA_DIR, exist_ok=True)
        for p in os.listdir(DATA_DIR):
            try: os.unlink(os.path.join(DATA_DIR, p))
            except IsADirectoryError: shutil.rmtree(os.path.join(DATA_DIR, p))

        scratch.reload()
        corpora.reload()
        src_dir = f"{scratch_root}/{parquet_key}/parquets"
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(f"scratch dir {src_dir} not found — upload first")
        parquets = sorted(p for p in os.listdir(src_dir) if p.endswith(".parquet"))
        if not parquets:
            raise FileNotFoundError(f"no parquets in {src_dir}")
        for i, p in enumerate(parquets):
            os.symlink(f"{src_dir}/{p}", f"{DATA_DIR}/shard_{i:05d}.parquet")
        # Held-out val shard copied from archive
        val_src = val_shard
        if not os.path.exists(val_src):
            raise FileNotFoundError(f"val shard not found at {val_src} — upload to archive volume")
        shutil.copy(val_src, f"{DATA_DIR}/shard_{len(parquets):05d}.parquet")

        log_name = f"{row_name}_{size}_{parquet_key}_s{ts}.log"
        log_path = f"/scratch/train_logs/{log_name}"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        cmd = [
            "torchrun", "--standalone", "--nproc_per_node=8",
            "-m", "scripts.base_train", "--",
            f"--depth={recipe['depth']}",
            f"--device-batch-size={recipe['device_batch_size']}",
            f"--target-param-data-ratio={ratio}",
            "--fp8" if recipe["fp8"] else "",
            f"--run={row_name}_{size}_s{ts}",
            f"--model-tag={row_name}_{size}_s{ts}",
            "--eval-every=999999", "--sample-every=-1", "--save-every=-1",
            "--eval-tokens=2097152", f"--core-metric-every={core_every}",
        ]
        if recipe.get("max_seq_len"):
            cmd.append(f"--max-seq-len={recipe['max_seq_len']}")
        if recipe.get("total_batch_size"):
            cmd.append(f"--total-batch-size={recipe['total_batch_size']}")
        cmd = [c for c in cmd if c]
        print(f"[{row_name}] {size} s{ts} on {parquet_key} (r={ratio}): launching torchrun", flush=True)

        t1 = time.time()
        with open(log_path, "w") as fh:
            res = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=recipe["timeout_s"])
        wall_min = (time.time() - t1) / 60
        out = open(log_path).read()
        bpbs  = re.findall(r"Validation bpb:\s*([\d.]+)", out)
        cores = re.findall(r"CORE metric:\s*([\d.]+)", out)
        val_bpb = float(bpbs[-1])  if bpbs else None
        core    = float(cores[-1]) if cores else None
        scratch.commit()
        results.append({
            "size": size, "parquet_key": parquet_key, "train_seed": ts,
            "val_bpb": val_bpb, "core": core,
            "exit": res.returncode, "wall_min": wall_min,
            "log_path": log_path,
        })
        print(f"[{row_name}] {size} s{ts}: wall={wall_min:.1f}m  bpb={val_bpb}  CORE={core}", flush=True)

    return results
