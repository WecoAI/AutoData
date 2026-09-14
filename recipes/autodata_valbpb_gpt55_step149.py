"""Data selection — ClimbMix full pool (shrinkage category-centered soft tournament)."""
import os
import numpy as np

BUDGET = 14_374_266
META = os.environ.get("AUTODATA_FEATURES_DIR", os.path.join(os.path.expanduser("~"), ".cache/autodata/meta_climbmix_full"))


def _robust_tanh_feature(x: np.ndarray, prefer_low: bool = False) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    finite = np.isfinite(x)
    if not np.all(finite):
        med0 = np.nanmedian(x)
        if not np.isfinite(med0):
            med0 = np.float32(0.0)
        x = x.copy()
        x[~finite] = med0
    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    scale = np.float32(max(q3 - q1, 1e-6))
    z = (np.float32(med) - x) / scale if prefer_low else (x - np.float32(med)) / scale
    return np.tanh(z).astype(np.float32, copy=False)


def _central_tanh_feature(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    finite = np.isfinite(x)
    if not np.all(finite):
        med0 = np.nanmedian(x)
        if not np.isfinite(med0):
            med0 = np.float32(0.0)
        x = x.copy()
        x[~finite] = med0
    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    scale = np.float32(max(q3 - q1, 1e-6))
    z = (x - np.float32(med)) / scale
    return (-np.abs(np.tanh(z))).astype(np.float32, copy=False)


def _draw_without_replacement(n_total: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Audited no-replacement draw boundary; preserves the normal random-proposal path."""
    size = int(size)
    if size <= 0:
        return np.empty(0, dtype=np.int64)
    if size >= n_total:
        return np.arange(n_total, dtype=np.int64)
    return rng.choice(n_total, size=size, replace=False).astype(np.int64, copy=False)


def _repair_exact_unique(selected: np.ndarray, budget: int, n_total: int, rng: np.random.Generator) -> np.ndarray:
    """Exact-size safety repair for changed budgets/pools; inactive for the normal no-replacement path."""
    selected = np.unique(selected.astype(np.int64, copy=False)).astype(np.int64, copy=False)
    if selected.size > budget:
        selected = rng.choice(selected, size=budget, replace=False).astype(np.int64, copy=False)
    elif selected.size < budget:
        need = budget - selected.size
        mask_extra = np.ones(n_total, dtype=bool)
        mask_extra[selected] = False
        pool = np.flatnonzero(mask_extra)
        extra = rng.choice(pool, size=need, replace=False).astype(np.int64, copy=False)
        selected = np.concatenate([selected, extra]).astype(np.int64, copy=False)
    return selected


def _category_inverse(topic: np.ndarray, fmt: np.ndarray) -> np.ndarray:
    """Collision-safe topic/format cell inverse for shrinkage centering."""
    _, topic_inv = np.unique(topic, return_inverse=True)
    _, fmt_inv = np.unique(fmt, return_inverse=True)
    n_fmt = int(fmt_inv.max()) + 1 if fmt_inv.size else 1
    codes = topic_inv.astype(np.int64, copy=False) * n_fmt + fmt_inv.astype(np.int64, copy=False)
    _, inv = np.unique(codes, return_inverse=True)
    return inv.astype(np.int64, copy=False)


def _centered_slate_argmax(score: np.ndarray, n_fill: int, arm: int) -> np.ndarray:
    """Numerically stable slate argmax; centering preserves each tournament winner."""
    slate_scores = score.reshape(n_fill, arm)
    slate_scores -= np.max(slate_scores, axis=1, keepdims=True)
    return np.argmax(slate_scores, axis=1).astype(np.int64, copy=False)


def select_docs(budget: int = BUDGET, seed: int = 42) -> np.ndarray:
    """Return sorted unique document indices selected by a conservative repair rule."""
    doc_tokens = np.load(os.path.join(META, "doc_tokens.npy"), mmap_mode="r")
    n_total = int(len(doc_tokens))
    budget = int(budget)
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    if budget >= n_total:
        return np.arange(n_total, dtype=np.int64)

    rng = np.random.default_rng(seed)

    n_backbone = budget // 3
    n_fill = budget - n_backbone
    max_arm = max(1, (n_total - n_backbone) // max(n_fill, 1))
    arm = int(min(3, max_arm))
    n_cand = n_fill * arm
    n_draw = n_backbone + n_cand

    draw = _draw_without_replacement(n_total, n_draw, rng)
    backbone = draw[:n_backbone]
    cand = draw[n_backbone:]

    if arm == 1:
        out = draw[:budget]
        return np.sort(out.astype(np.int64, copy=False))

    logppl = np.load(os.path.join(META, "logppl_qwen.npy"), mmap_mode="r")[cand]
    div_avg = np.load(os.path.join(META, "avg_distinct_ngram_bpe.npy"), mmap_mode="r")[cand]
    div5 = np.load(os.path.join(META, "distinct_5gram_bpe.npy"), mmap_mode="r")[cand]
    toks = doc_tokens[cand]
    chars = np.load(os.path.join(META, "doc_chars.npy"), mmap_mode="r")[cand]

    div_score = (_robust_tanh_feature(div_avg, prefer_low=False) +
                 _robust_tanh_feature(div5, prefer_low=False)) * np.float32(0.5)
    ppl_score = _robust_tanh_feature(logppl, prefer_low=True)
    log_len = np.log1p(toks.astype(np.float32, copy=False))
    len_score = _central_tanh_feature(log_len)
    ratio = chars.astype(np.float32, copy=False) / np.maximum(toks.astype(np.float32, copy=False), np.float32(1.0))
    ratio_score = _central_tanh_feature(ratio)

    score = div_score
    score = score + ppl_score * ((div_score + np.float32(1.0)) * np.float32(0.5))
    score = score + len_score + ratio_score
    score = score.astype(np.float32, copy=False)

    del logppl, div_avg, div5, toks, chars, div_score, ppl_score, log_len, len_score, ratio, ratio_score

    topic = np.load(os.path.join(META, "topic_id.npy"), mmap_mode="r")[cand].astype(np.int64, copy=False)
    fmt = np.load(os.path.join(META, "format_id.npy"), mmap_mode="r")[cand].astype(np.int64, copy=False)
    inv = _category_inverse(topic, fmt)
    counts = np.bincount(inv)
    sums = np.bincount(inv, weights=score)
    means = (sums / np.maximum(counts, 1)).astype(np.float32, copy=False)

    count_scale = np.float32(max(np.median(counts.astype(np.float32, copy=False)), 1.0))
    countsf = counts.astype(np.float32, copy=False)
    shrink = (countsf / (countsf + count_scale)).astype(np.float32, copy=False)
    score = score - np.float32(0.5) * (means * shrink)[inv]

    del topic, fmt, inv, counts, countsf, sums, means, shrink

    score = score + rng.gumbel(size=score.shape).astype(np.float32, copy=False)
    chosen_pos = _centered_slate_argmax(score, n_fill, arm)
    base = np.arange(n_fill, dtype=np.int64) * arm
    winners = cand[base + chosen_pos]

    selected = np.concatenate([backbone, winners]).astype(np.int64, copy=False)
    if selected.size != budget or np.unique(selected).size != budget:
        selected = _repair_exact_unique(selected, budget, n_total, rng)
    return np.sort(selected.astype(np.int64, copy=False))
