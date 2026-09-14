"""Data selection — ClimbMix full pool (climbmix_full_v1).

Format-conditional PPL z-score + soft logistic acceptance.
Variant: replace IQR-based scale with (q84-q16)/2 (1-sigma quantile gap),
which uses a wider, more representative range of each format's PPL bulk.
"""
import os
import numpy as np

BUDGET = 14_374_266
META = os.environ.get("AUTODATA_FEATURES_DIR", os.path.join(os.path.expanduser("~"), ".cache/autodata/meta_climbmix_full"))


def select_docs(budget: int = BUDGET, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)

    doc_tokens = np.load(os.path.join(META, "doc_tokens.npy"), mmap_mode="r")
    N = int(doc_tokens.shape[0])
    doc_tokens = np.asarray(doc_tokens[:], dtype=np.int32)

    logppl = np.asarray(
        np.load(os.path.join(META, "logppl_qwen.npy"), mmap_mode="r")[:],
        dtype=np.float32,
    )
    d5 = np.asarray(
        np.load(os.path.join(META, "distinct_5gram_bpe.npy"), mmap_mode="r")[:],
        dtype=np.float32,
    )
    fmt = np.asarray(
        np.load(os.path.join(META, "format_id.npy"), mmap_mode="r")[:],
        dtype=np.int32,
    )

    # ----- Hygiene mask (pool-adaptive thresholds) -----
    nan_ppl = ~np.isfinite(logppl)

    # length floor: bottom 2% (fragments)
    q_short = float(np.quantile(doc_tokens, 0.02))
    too_short = doc_tokens < q_short

    # spam template: joint bottom 5% on both d5 and PPL
    finite_ppl = logppl[~nan_ppl]
    q_lo_d5 = float(np.quantile(d5, 0.05))
    q_lo_ppl = float(np.quantile(finite_ppl, 0.05))
    spam = (~nan_ppl) & (d5 < q_lo_d5) & (logppl < q_lo_ppl)

    bad = nan_ppl | too_short | spam
    eligible = ~bad
    n_elig = int(eligible.sum())
    assert n_elig >= budget, f"eligible {n_elig} < budget {budget}"

    # ----- Format-conditional PPL z-score (1-sigma quantile gap scale) -----
    z = np.zeros(N, dtype=np.float32)
    unique_fmts = np.unique(fmt)
    for f in unique_fmts:
        mask = (fmt == f) & eligible
        if not mask.any():
            continue
        vals = logppl[mask]
        med = float(np.median(vals))
        q16, q84 = np.quantile(vals, [0.16, 0.84])
        scale = max(float(q84 - q16) / 2.0, 1e-3)
        z[mask] = (vals - med) / scale

    # ----- Soft logistic acceptance -----
    T = 1.0
    raw = 1.0 / (1.0 + np.exp(T * z))
    raw = raw * eligible.astype(np.float32)

    target = float(budget)
    lo, hi = 0.0, 1e6
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        s = float(np.minimum(mid * raw, 1.0).sum())
        if s < target:
            lo = mid
        else:
            hi = mid
    c = 0.5 * (lo + hi)
    probs = np.minimum(c * raw, 1.0)

    u = rng.random(N, dtype=np.float32)
    chosen = u < probs
    n_chosen = int(chosen.sum())

    # ----- Reconcile to exact budget -----
    if n_chosen > budget:
        chosen_idx = np.flatnonzero(chosen)
        excess = n_chosen - budget
        keep_scores = probs[chosen_idx]
        keep_scores = keep_scores + rng.random(len(chosen_idx), dtype=np.float32) * 1e-6
        drop_local = np.argpartition(keep_scores, excess)[:excess]
        chosen[chosen_idx[drop_local]] = False
    elif n_chosen < budget:
        not_chosen = eligible & ~chosen
        cand_idx = np.flatnonzero(not_chosen)
        need = budget - n_chosen
        cand_scores = probs[cand_idx] + rng.random(len(cand_idx), dtype=np.float32) * 1e-6
        top_local = np.argpartition(-cand_scores, need - 1)[:need]
        chosen[cand_idx[top_local]] = True

    out = np.flatnonzero(chosen).astype(np.int64)
    assert out.size == budget
    out.sort()
    return out
