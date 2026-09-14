"""WECO eval harness — local 2-GPU proxy training, CORE-optimised.

Proxy depth comes from AUTODATA_EVAL_DEPTH (default 8); see PROXY_RECIPE for
the per-depth batch/sub-sample settings.

Invoked by `weco run --eval-command` (see search/launch/run.sh).

For each WECO step:
  1. Load `select_docs, BUDGET` from the source module (the file WECO rewrites
     in place at every step). Module path is set via WECO_DATA_SELECT_MODULE.
  2. selected = select_docs(BUDGET, seed=42) → ~14.37M doc indices.
  3. Deterministic sub-sample of PROXY_RECIPE[depth]["subsample_docs"] docs
     (880K at d8, 2.31M at d12; single sub_seed=0).
  4. Materialise the sub-sample to /dev/shm + the held-out val shard.
  5. Spawn 2 single-GPU r=10 FP8 trainings in parallel (train_seeds 42, 43).
     Each training enables nanochat's CORE eval.
  6. Parse `CORE metric: X.XXXX` per seed → print `core: <mean>` for WECO.

Env vars (set by the launcher):
  AUTODATA_NANOCHAT_ROOT     local clone of karpathy/nanochat
  AUTODATA_FEATURES_DIR      pre-computed features (.npy)
  AUTODATA_POOL_DIR          raw ClimbMix shards
  AUTODATA_VAL_SHARD         held-out val parquet
  AUTODATA_WECO_RUN_PREFIX   namespaces state-file + /dev/shm dir
  AUTODATA_WECO_GPUS         "0,1" — comma-separated, MUST be exactly 2
  WECO_DATA_SELECT_MODULE    dotted Python path of the data_select module
"""
import os
import sys
import time
import shutil
import json
import re
import ast
import inspect
import hashlib
import subprocess
import importlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---- env-derived paths (NO hard-coded absolute paths) ----------------------
NANOCHAT_ROOT = Path(os.environ["AUTODATA_NANOCHAT_ROOT"])
POOL_DIR      = Path(os.environ["AUTODATA_POOL_DIR"])
VAL_SHARD     = Path(os.environ["AUTODATA_VAL_SHARD"])
META_DIR      = Path(os.environ["AUTODATA_FEATURES_DIR"])
RUN_PREFIX    = os.environ.get("AUTODATA_WECO_RUN_PREFIX", "autodata_run")
GPUS_STR      = os.environ.get("AUTODATA_WECO_GPUS", "0,1").strip()
DATA_SELECT_MODULE = os.environ["WECO_DATA_SELECT_MODULE"]

# Make the AutoData repo importable so DATA_SELECT_MODULE can be loaded
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- nanochat-derived constants --------------------------------------------
N_TOTAL            = 553_155_584        # ClimbMix pool size
SUBSAMPLE_SEED     = 0
EVAL_EVERY         = 999999             # only at end of training
EVAL_TOKENS        = 2097152
N_SEEDS            = 2
TRAIN_SEEDS        = [42, 43]
PENALTY_CORE       = -1.0               # CORE is maximised; failure → very low score
PENALTY_VAL_BPB    = 10.0               # val_bpb is minimised; failure → very high score

# Per-depth proxy recipe. The proxy depth is set by AUTODATA_EVAL_DEPTH (default
# 8 — the setting every published AutoData search used). `subsample_docs` keeps
# the same ~1.33x token headroom over the depth's r=10 target at ~636 tok/doc,
# so a d12 proxy sees 2.63x the tokens of a d8 proxy (see
# eval/recompute_training_config.py, which fixes the same two numbers).
#   d8:  125.8 M params,  0.419 B target tokens,  880 K docs
#   d12: 286.3 M params,  1.101 B target tokens, 2.31 M docs
# device_batch_size × max_seq_len × world_size ≤ total_batch_size; the remainder
# is gradient accumulation. d12's values mirror pipeline/modal_train.py's d12
# recipe, so per-GPU memory matches a known-good configuration.
PROXY_RECIPE = {
    8:  dict(ratio=10, device_batch_size=128, max_seq_len=1024,
             total_batch_size=131072, subsample_docs=880_000,
             train_timeout_s=60 * 90),
    12: dict(ratio=10, device_batch_size=32,  max_seq_len=2048,
             total_batch_size=524288, subsample_docs=2_310_000,
             train_timeout_s=60 * 300),
}

EVAL_DEPTH = int(os.environ.get("AUTODATA_EVAL_DEPTH", "8"))
if EVAL_DEPTH not in PROXY_RECIPE:
    raise ValueError(f"AUTODATA_EVAL_DEPTH={EVAL_DEPTH} has no proxy recipe; "
                     f"known depths: {sorted(PROXY_RECIPE)}")
_R = PROXY_RECIPE[EVAL_DEPTH]
EVAL_RATIO         = _R["ratio"]
EVAL_DEVICE_BATCH  = _R["device_batch_size"]
EVAL_MAX_SEQ       = _R["max_seq_len"]
EVAL_TOTAL_BATCH   = _R["total_batch_size"]
SUBSAMPLE_DOCS     = _R["subsample_docs"]
TRAIN_TIMEOUT_S    = _R["train_timeout_s"]
# Back-compat alias: pipeline/search_eval_core.py and older callers use the
# d8-era name. Both always refer to the ACTIVE depth's sub-sample size.
D8_SUBSAMPLE_DOCS  = SUBSAMPLE_DOCS

GPUS = [int(x) for x in GPUS_STR.split(",") if x.strip()]
assert len(GPUS) == N_SEEDS, f"AUTODATA_WECO_GPUS must list exactly {N_SEEDS} GPUs, got {GPUS}"

# ---- feature_construction (LLM-proposed cheap features) --------------------
# The selector module may define feature_construction(texts) -> dict[str, arr].
# We compute it over the WHOLE pool once, cache per feature-code fingerprint,
# and expose the arrays to select_docs via AUTODATA_CONSTRUCTED_DIR.
CONSTRUCTED_ROOT   = META_DIR / "constructed"
CONSTRUCTED_ENV    = "AUTODATA_CONSTRUCTED_DIR"
# Only these top-level imports are allowed in the selector module when
# feature_construction is present — keeps constructed features genuinely cheap
# (pure string / numpy; no model inference, no I/O, no network).
ALLOWED_IMPORTS = {
    "os", "re", "math", "string", "numpy", "np",
    "collections", "unicodedata", "itertools", "functools",
}
# Precomputed feature names a constructed feature must not shadow.
RESERVED_FEATURE_NAMES = {
    "doc_tokens", "doc_chars", "avg_distinct_ngram_bpe", "logppl_qwen",
    "topic_id", "format_id",
    *(f"distinct_{n}gram_bpe" for n in range(1, 6)),
}
MAX_CONSTRUCTED_FEATURES  = 16
CONSTRUCTED_TIME_BUDGET_S = int(os.environ.get("AUTODATA_CONSTRUCTED_TIME_BUDGET_S", str(3 * 60 * 60)))
# Dev/debug: cap the number of shards scanned (0 = full pool). NOT for real runs.
CONSTRUCTED_SHARD_LIMIT   = int(os.environ.get("AUTODATA_CONSTRUCTED_SHARD_LIMIT", "0"))

STATE_FILE = Path(f".weco_iteration_{RUN_PREFIX}")
LOG_DIR    = Path(f"weco/log/{RUN_PREFIX}")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_iteration() -> int:
    n = int(STATE_FILE.read_text().strip()) if STATE_FILE.exists() else 0
    STATE_FILE.write_text(str(n + 1))
    return n


def log(msg: str) -> None:
    print(f"[weco_eval::{RUN_PREFIX}] {msg}", flush=True)


def localized_subsample(selected: np.ndarray, n_target: int, seed: int) -> np.ndarray:
    """Draw `n_target` docs from `selected`, confined to the fewest pool shards
    that together hold >= n_target selected docs.

    materialize then reads only those ~n_target/(selected-per-shard) shards
    (~400 for the d8 880K target at 2.6%% keep) instead of all 6543 — cutting
    the cold volume read ~16x. Unbiased in expectation because the pool is
    pre-shuffled (climbmix-400b-shuffle); the cost is extra selection variance
    from the smaller shard set. Deterministic given `seed`.
    """
    entries = json.loads((META_DIR / "shard_offsets.json").read_text())
    offs = np.array(sorted(e["offset"] for e in entries), dtype=np.int64)
    selected = np.asarray(selected, dtype=np.int64)
    shard_pos = np.searchsorted(offs, selected, side="right") - 1      # shard per doc
    counts = np.bincount(shard_pos, minlength=len(offs))
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(offs), dtype=bool)
    total = 0
    for s in rng.permutation(len(offs)):                              # random shard order
        if counts[s] == 0:
            continue
        keep[s] = True
        total += int(counts[s])
        if total >= n_target:
            break
    if total < n_target:
        raise ValueError(f"localized_subsample: only {total} < {n_target} available")
    pool = selected[keep[shard_pos]]
    return np.sort(rng.choice(pool, size=n_target, replace=False)).astype(np.int64)


def _validate(idx: np.ndarray, n_expected: int, what: str) -> None:
    assert isinstance(idx, np.ndarray), f"{what}: not ndarray ({type(idx)})"
    assert idx.dtype == np.int64,       f"{what}: dtype {idx.dtype} != int64"
    assert idx.shape == (n_expected,),  f"{what}: shape {idx.shape} != ({n_expected},)"
    if len(np.unique(idx)) != n_expected:
        raise ValueError(f"{what}: {n_expected - len(np.unique(idx))} duplicates")
    if idx.min() < 0 or idx.max() >= N_TOTAL:
        raise ValueError(f"{what}: out-of-range [0,{N_TOTAL})")


def _read_shard(shard_i: int, rows: np.ndarray):
    src = POOL_DIR / f"shard_{shard_i:05d}.parquet"
    return pq.read_table(src, columns=["text"]).take(pa.array(rows))


def _read_shard_text(shard_i: int) -> list:
    """Return the full list of raw document strings for one pool shard."""
    src = POOL_DIR / f"shard_{shard_i:05d}.parquet"
    return pq.read_table(src, columns=["text"]).column("text").to_pylist()


# ---- feature_construction plumbing -----------------------------------------
def _assert_pure_imports(mod) -> None:
    """Reject selector modules that import anything outside ALLOWED_IMPORTS.

    Only enforced when feature_construction is defined — it is the cheapness
    guarantee (pure string/numpy; no torch/transformers/requests/subprocess…).
    """
    src_path = inspect.getsourcefile(mod)
    tree = ast.parse(Path(src_path).read_text())
    bad = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top not in ALLOWED_IMPORTS:
                    bad.add(top)
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top and top not in ALLOWED_IMPORTS:
                bad.add(top)
    if bad:
        raise ValueError(
            f"feature_construction present but module imports disallowed "
            f"modules {sorted(bad)}; allowed: {sorted(ALLOWED_IMPORTS)}"
        )


def _feature_fingerprint(mod, fn) -> str:
    """Stable hash of the feature-building code only.

    Includes the source of feature_construction plus any module-level callable
    it references by name (one level), so editing select_docs alone does NOT
    invalidate the constructed-feature cache — but editing the feature code
    (or its helpers) does.
    """
    src = inspect.getsource(fn)
    parts = [src]
    referenced = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}
    for nm in sorted(referenced):
        obj = getattr(mod, nm, None)
        if obj is fn or not callable(obj):
            continue
        if getattr(obj, "__module__", None) != mod.__name__:
            continue
        try:
            parts.append(inspect.getsource(obj))
        except (OSError, TypeError):
            pass
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _validate_feat_out(out, n: int) -> dict:
    """Coerce a feature_construction batch result to {name: float32[n]}."""
    if not isinstance(out, dict):
        raise TypeError(f"feature_construction must return dict, got {type(out)}")
    clean = {}
    for k, v in out.items():
        if not (isinstance(k, str) and k.isidentifier()):
            raise ValueError(f"feature name {k!r} is not a valid identifier")
        if k in RESERVED_FEATURE_NAMES:
            raise ValueError(f"feature name {k!r} collides with a precomputed feature")
        arr = np.asarray(v, dtype=np.float32).ravel()
        if arr.shape != (n,):
            raise ValueError(f"feature {k!r} shape {arr.shape} != expected ({n},)")
        arr = arr.copy()
        arr[~np.isfinite(arr)] = np.nan   # keep NaN (select_docs handles); drop +/-inf
        clean[k] = arr
    if len(clean) > MAX_CONSTRUCTED_FEATURES:
        raise ValueError(f"{len(clean)} constructed features > cap {MAX_CONSTRUCTED_FEATURES}")
    return clean


def _fc_worker(task):
    """Process-pool worker: compute constructed features for one pool shard.

    Re-imports the selector module by name (WECO_DATA_SELECT_MODULE is a real
    importable path) so the LLM's feature_construction runs in a separate
    process — true CPU parallelism for pure-Python string ops, and isolation so
    a crashing/looping shard can't corrupt the parent.
    """
    module_name, shard_idx, offset, count, names = task
    fn = importlib.import_module(module_name).feature_construction
    txt = _read_shard_text(shard_idx)
    out = _validate_feat_out(fn(list(txt)), len(txt))
    if set(out.keys()) != set(names):
        raise ValueError(f"shard {shard_idx} produced names {sorted(out)} != {names}")
    return offset, count, out


def build_constructed_features(mod, fn):
    """Run feature_construction over the full pool, cache, return the cache dir.

    Returns None if the selector proposes no constructed features. Raises on
    any violation (purity, shape, time budget) — the caller turns that into a
    metric penalty so WECO learns to avoid broken / too-expensive features.
    """
    entries = sorted(
        json.loads((META_DIR / "shard_offsets.json").read_text()),
        key=lambda e: e["offset"],
    )
    n_total = sum(e["count"] for e in entries)
    if CONSTRUCTED_SHARD_LIMIT:
        entries = entries[:CONSTRUCTED_SHARD_LIMIT]
        n_total = sum(e["count"] for e in entries)
        log(f"[constructed] DEV shard limit → {len(entries)} shards, N={n_total:,}")

    # Probe the first shard to learn the feature names (and detect "no features").
    probe_e = entries[0]
    probe_txt = _read_shard_text(probe_e["shard_idx"])[: min(4096, probe_e["count"])]
    probe = _validate_feat_out(fn(list(probe_txt)), len(probe_txt))
    if not probe:
        log("[constructed] feature_construction returned no features — skipping")
        return None
    names = sorted(probe.keys())

    fp = _feature_fingerprint(mod, fn)
    cache_dir = CONSTRUCTED_ROOT / fp
    manifest = cache_dir / "_manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        if (m.get("n") == n_total and set(m.get("names", [])) == set(names)
                and all((cache_dir / f"{nm}.npy").exists() for nm in names)):
            log(f"[constructed] cache HIT {cache_dir} names={names}")
            return cache_dir

    n_workers = min(32, os.cpu_count() or 8)
    log(f"[constructed] building {names} over {len(entries)} shards "
        f"on {n_workers} processes → {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrs = {nm: np.full(n_total, np.nan, dtype=np.float32) for nm in names}

    # One task per shard; feature_construction runs in a separate process. The
    # time budget is a soft cap checked between completed shards — a single
    # pathological shard is ultimately bounded by WECO's --eval-timeout.
    tasks = [(DATA_SELECT_MODULE, e["shard_idx"], e["offset"], e["count"], names)
             for e in entries]
    t0 = time.time()
    done = 0
    ex = ProcessPoolExecutor(max_workers=n_workers)
    try:
        for off, cnt, out in ex.map(_fc_worker, tasks):
            for nm in names:
                arrs[nm][off:off + cnt] = out[nm]
            done += 1
            if time.time() - t0 > CONSTRUCTED_TIME_BUDGET_S:
                raise TimeoutError(
                    f"feature construction exceeded {CONSTRUCTED_TIME_BUDGET_S}s "
                    f"after {done}/{len(entries)} shards"
                )
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    for nm in names:
        np.save(cache_dir / f"{nm}.npy", arrs[nm])
    manifest.write_text(json.dumps(
        {"fingerprint": fp, "names": names, "n": n_total,
         "built_s": round(time.time() - t0, 1)}))
    log(f"[constructed] built {names} in {time.time()-t0:.1f}s → {cache_dir}")
    return cache_dir


def materialize(idx: np.ndarray, run_dir: Path) -> int:
    """Write the selected docs into ~8+ parquets in run_dir + val shard.

    Reads parquets in parallel; for a 880K-doc sub-sample randomly distributed
    over 6543 shards (~134 rows/shard), this drops the read phase from ~7 min
    serial to ~20 s with 32 threads.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    for p in run_dir.iterdir():
        p.unlink()
    entries = json.loads((META_DIR / "shard_offsets.json").read_text())
    global_idx = np.asarray(idx, dtype=np.int64)
    buckets = {}
    for e in entries:
        off, cnt = e["offset"], e["count"]
        mask = (global_idx >= off) & (global_idx < off + cnt)
        rows = (global_idx[mask] - off).astype(np.int64)
        if len(rows):
            buckets[e["shard_idx"]] = rows
    shards_sorted = sorted(buckets)
    with ThreadPoolExecutor(max_workers=32) as pool:
        futs = [pool.submit(_read_shard, sh, buckets[sh]) for sh in shards_sorted]
        parts = [f.result() for f in futs]
    train_table = pa.concat_tables(parts)
    n_docs = train_table.num_rows
    n_out = max(8, n_docs // 200_000)
    per = (n_docs + n_out - 1) // n_out
    rg_size = max(1, per // 16)
    for i in range(n_out):
        lo, hi = i * per, min((i + 1) * per, n_docs)
        if lo >= hi: break
        sub = train_table.slice(lo, hi - lo)
        pq.write_table(sub, run_dir / f"shard_{i:05d}.parquet",
                       compression="snappy", row_group_size=rg_size)
    shutil.copy(VAL_SHARD, run_dir / "shard_99999.parquet")
    return n_docs


def setup_seed_dir(nc_base: Path, data_dir: Path):
    nc_base.mkdir(parents=True, exist_ok=True)
    (nc_base / "base_checkpoints").mkdir(parents=True, exist_ok=True)
    data_link = nc_base / "base_data_climbmix"
    if data_link.is_symlink() or data_link.exists():
        data_link.unlink()
    data_link.symlink_to(data_dir)
    nanochat_cache = Path(os.environ.get("NANOCHAT_CACHE_DIR", str(Path.home() / ".cache" / "nanochat")))
    for link, target in [
        (nc_base / "tokenizer",   nanochat_cache / "tokenizer"),
        (nc_base / "eval_bundle", nanochat_cache / "eval_bundle"),
        (nc_base / "annotation",  nanochat_cache / "annotation"),
    ]:
        if not (link.is_symlink() or link.exists()):
            link.symlink_to(target)


def train_one_seed(seed: int, gpu: int, iteration: int, data_dir: Path):
    run_tag = f"{RUN_PREFIX}_step{iteration:03d}_s{seed}"
    log_path = LOG_DIR / f"{run_tag}.log"
    nc_base = Path(f"/dev/shm/{RUN_PREFIX}/step{iteration:03d}/seed{seed}")
    setup_seed_dir(nc_base, data_dir)
    env = os.environ.copy()
    env.update({
        "NANOCHAT_SEED":        str(seed),
        "NANOCHAT_BASE_DIR":    str(nc_base),
        "WANDB_MODE":           "disabled",
        "OMP_NUM_THREADS":      "1",
        "CUDA_VISIBLE_DEVICES": str(gpu),
    })
    venv_py = NANOCHAT_ROOT / ".venv" / "bin" / "python"
    cmd = [
        str(venv_py),
        "-m", "scripts.base_train",
        f"--depth={EVAL_DEPTH}",
        f"--target-param-data-ratio={EVAL_RATIO}",
        f"--device-batch-size={EVAL_DEVICE_BATCH}",
        f"--max-seq-len={EVAL_MAX_SEQ}",
        f"--total-batch-size={EVAL_TOTAL_BATCH}",
        "--fp8",
        f"--model-tag={run_tag}",
        f"--run={run_tag}",
        f"--eval-every={EVAL_EVERY}",
        f"--eval-tokens={EVAL_TOKENS}",
        "--core-metric-every=999999",   # CORE eval at last step
        "--sample-every=-1",
        "--save-every=-1",
    ]
    t0 = time.time()
    with open(log_path, "w") as fh:
        try:
            rc = subprocess.run(cmd, env=env, cwd=str(NANOCHAT_ROOT), stdout=fh,
                                stderr=subprocess.STDOUT, timeout=TRAIN_TIMEOUT_S).returncode
        except subprocess.TimeoutExpired:
            rc = -1
    dt = time.time() - t0
    text = log_path.read_text()
    cores = re.findall(r"CORE metric:\s+([\d.]+)", text)
    bpbs  = re.findall(r"Validation bpb:\s+([\d.]+)", text)
    core = float(cores[-1]) if cores else None
    bpb  = float(bpbs[-1])  if bpbs  else None
    shutil.rmtree(nc_base, ignore_errors=True)
    return {"seed": seed, "gpu": gpu, "run_tag": run_tag, "exit": rc,
            "wall_min": dt / 60, "core": core, "val_bpb": bpb}


def main():
    iteration = get_iteration()
    log(f"=== iteration {iteration} | gpus={GPUS} | seeds={TRAIN_SEEDS} ===")

    mod = importlib.import_module(DATA_SELECT_MODULE)
    select_docs = mod.select_docs
    BUDGET = mod.BUDGET

    # Optional hook: LLM-proposed cheap features built over the full pool and
    # exposed to select_docs via AUTODATA_CONSTRUCTED_DIR. A failure here
    # (impure imports, bad shapes, time-budget overrun) scores the penalty so
    # WECO steers away from broken / too-expensive feature code.
    feature_construction = getattr(mod, "feature_construction", None)
    if callable(feature_construction):
        try:
            _assert_pure_imports(mod)
            tfc = time.time()
            cache_dir = build_constructed_features(mod, feature_construction)
            if cache_dir is not None:
                os.environ[CONSTRUCTED_ENV] = str(cache_dir)
                log(f"constructed features ready in {time.time()-tfc:.1f}s → {cache_dir}")
        except Exception as e:
            log(f"FEATURE CONSTRUCTION FAILED: {e!r}")
            print(f"core: {PENALTY_CORE:.6f}")
            return

    t0 = time.time()
    try:
        selected = select_docs(BUDGET, seed=42)
        _validate(selected, BUDGET, "select_docs output")
    except Exception as e:
        log(f"SELECTION INVALID: {e!r}")
        print(f"core: {PENALTY_CORE:.6f}")
        return
    log(f"select_docs returned {len(selected):,} docs in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    sub = np.sort(rng.choice(selected, size=SUBSAMPLE_DOCS, replace=False)).astype(np.int64)
    _validate(sub, SUBSAMPLE_DOCS, f"d{EVAL_DEPTH} subsample")
    log(f"subsampled {len(sub):,} docs for d{EVAL_DEPTH} r={EVAL_RATIO}")

    data_dir = Path(f"/dev/shm/{RUN_PREFIX}/step{iteration:03d}/data")
    t1 = time.time()
    n_docs = materialize(sub, data_dir)
    log(f"materialized {n_docs:,} docs into {data_dir} in {time.time()-t1:.1f}s")

    log(f"training {N_SEEDS} d{EVAL_DEPTH} r={EVAL_RATIO} FP8 in parallel on GPUs {GPUS}…")
    t2 = time.time()
    with ThreadPoolExecutor(max_workers=N_SEEDS) as pool:
        futs = [pool.submit(train_one_seed, seed, gpu, iteration, data_dir)
                for seed, gpu in zip(TRAIN_SEEDS, GPUS)]
        results = [f.result() for f in futs]
    wall_min = (time.time() - t2) / 60

    cores = [r["core"] for r in results]
    valid = [c for c in cores if c is not None]
    bpbs  = [r["val_bpb"] for r in results]
    log(f"per-seed CORE: {cores}  per-seed bpb: {bpbs}  walls(m)="
        f"{[round(r['wall_min'],1) for r in results]}  step_wall_min={wall_min:.1f}")

    shutil.rmtree(data_dir, ignore_errors=True)

    if len(valid) < N_SEEDS:
        log(f"only {len(valid)}/{N_SEEDS} seeds produced CORE — penalty")
        print(f"core: {PENALTY_CORE:.6f}")
        return
    mean_core = float(np.mean(valid))
    std_core  = float(np.std(valid))
    log(f"CORE mean={mean_core:.6f}  std={std_core:.6f}  n_selected={len(selected):,}  n_trained={len(sub):,}")
    print(f"core: {mean_core:.6f}")
    print(f"core_std: {std_core:.6f}")


if __name__ == "__main__":
    main()
