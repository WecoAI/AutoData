"""Data selection — ClimbMix full pool (climbmix_full_v1, base-LLM runs).

Hybrid uniform + source-neighborhood quality selection with conservative finite-safe
feature handling.  The rule keeps a true uniform backbone, then uses a disjoint
candidate stream for a soft quality repair that combines robust pool-adaptive
document hygiene with a source-order neighborhood prior.  The source-block
resolution is derived from the actual repair candidate stream so the same rule
remains well scaled if future budgets or pool sizes cap candidate pressure.
"""
import os
import numpy as np

BUDGET = 14_374_266
META = os.environ.get("AUTODATA_FEATURES_DIR", os.path.join(os.path.expanduser("~"), ".cache/autodata/meta_climbmix_full"))


def _finite_fill(x):
    """Return float32 array with non-finite values replaced by the finite median."""
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if np.all(finite):
        return x
    if np.any(finite):
        fill = float(np.median(x[finite]))
    else:
        fill = 0.0
    return np.where(finite, x, fill).astype(np.float32)


def _safe_quantiles(x, qs):
    x = _finite_fill(x)
    if x.size == 0:
        return np.zeros(len(qs), dtype=np.float32)
    return np.quantile(x, qs).astype(np.float32)


def _tri_score(x, anchor, lo, hi):
    x = _finite_fill(x)
    scale = max(float(hi - lo), 1e-6)
    return -np.abs((x - float(anchor)) / scale).astype(np.float32)


def _zscore(x):
    x = _finite_fill(x)
    m = float(np.mean(x)) if x.size else 0.0
    s = float(np.std(x)) if x.size else 0.0
    if s <= 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - m) / s).astype(np.float32)


def _exact_unique(selected, budget, n_total, rng):
    selected = np.unique(selected.astype(np.int64, copy=False))
    need = int(budget) - int(selected.size)
    if need <= 0:
        return selected[:budget].astype(np.int64, copy=False)

    chunks = []
    have = 0
    while have < need:
        draw_size = min(n_total, max(need - have, (need - have) * 2))
        more = rng.choice(n_total, size=draw_size, replace=False).astype(np.int64, copy=False)
        more = more[~np.isin(more, selected, assume_unique=False)]
        if chunks:
            prev = np.concatenate(chunks)
            more = more[~np.isin(more, prev, assume_unique=False)]
        if more.size:
            take = more[: need - have]
            chunks.append(take)
            have += int(take.size)
    return np.concatenate([selected] + chunks).astype(np.int64, copy=False)


def select_docs(budget: int = BUDGET, seed: int = 42) -> np.ndarray:
    doc_tokens = np.load(os.path.join(META, "doc_tokens.npy"), mmap_mode="r")
    n_total = int(len(doc_tokens))
    budget = int(budget)
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    if budget >= n_total:
        return np.arange(n_total, dtype=np.int64)

    rng = np.random.default_rng(seed)

    # Preserve broad corpus/domain coverage with a true neutral backbone; use a
    # disjoint random candidate stream for the quality-biased repair half.
    n_backbone = int(budget * 0.5)
    n_repair = budget - n_backbone
    n_candidates = min(n_total - n_backbone, n_repair * 2)
    draw = rng.choice(n_total, size=n_backbone + n_candidates, replace=False)
    backbone = draw[:n_backbone].astype(np.int64, copy=False)
    cand = draw[n_backbone:].astype(np.int64, copy=False)

    if n_repair > 0 and cand.size > 0:
        doc_chars = np.load(os.path.join(META, "doc_chars.npy"), mmap_mode="r")
        div_arr = np.load(os.path.join(META, "avg_distinct_ngram_bpe.npy"), mmap_mode="r")
        ppl_arr = np.load(os.path.join(META, "logppl_qwen.npy"), mmap_mode="r")

        tok = _finite_fill(np.asarray(doc_tokens[cand], dtype=np.float32))
        chars = _finite_fill(np.asarray(doc_chars[cand], dtype=np.float32))
        div = _finite_fill(np.asarray(div_arr[cand], dtype=np.float32))
        ppl = _finite_fill(np.asarray(ppl_arr[cand], dtype=np.float32))

        log_len = _finite_fill(np.log1p(np.maximum(tok, 0.0)))
        ratio = _finite_fill(np.log((np.maximum(chars, 0.0) + 1.0) / (np.maximum(tok, 0.0) + 1.0)))

        lq = _safe_quantiles(log_len, [1.0 / 3.0, 2.0 / 3.0])
        dq = _safe_quantiles(div, [1.0 / 3.0, 2.0 / 3.0])
        pq = _safe_quantiles(ppl, [1.0 / 3.0, 2.0 / 3.0])
        rq = _safe_quantiles(ratio, [1.0 / 3.0, 0.5, 2.0 / 3.0])

        # Pool-adaptive Goldilocks hygiene: medium-long documents, upper-middle
        # lexical diversity, fluent-but-not-template perplexity, and normal
        # character/token ratio.  All features are finite-safe before quantiles
        # and scoring so rare anomalous metadata cannot distort the frontier.
        s_len = _tri_score(log_len, lq[1], lq[0], lq[1])
        s_div = _tri_score(div, dq[1], dq[0], dq[1])
        s_ppl = _tri_score(ppl, pq[0], pq[0], pq[1])
        s_rat = _tri_score(ratio, rq[1], rq[0], rq[2])
        indiv = _zscore((s_len + s_div + s_ppl + s_rat) / 4.0)

        # Soft source-order neighborhood prior from the same repair stream.  The
        # resolution follows the actual candidate stream, matching the standard
        # high-pressure setting while staying robust if the stream is capped.
        n_blocks = max(1, int(np.sqrt(float(cand.size))))
        block = np.minimum((cand * n_blocks) // n_total, n_blocks - 1).astype(np.int64)
        counts = np.bincount(block, minlength=n_blocks).astype(np.float32)
        sums = np.bincount(block, weights=indiv.astype(np.float64), minlength=n_blocks).astype(np.float32)
        means = np.zeros(n_blocks, dtype=np.float32)
        np.divide(sums, np.maximum(counts, 1.0), out=means)
        block_prior = _zscore(means)[block]

        total_score = indiv + block_prior + rng.gumbel(size=cand.size).astype(np.float32)
        if n_repair < cand.size:
            take_pos = np.argpartition(total_score, -n_repair)[-n_repair:]
            repair = cand[take_pos]
        else:
            repair = cand
        selected = np.concatenate([backbone, repair]).astype(np.int64, copy=False)
    else:
        selected = backbone.astype(np.int64, copy=False)

    if selected.size != budget or np.unique(selected).size != budget:
        selected = _exact_unique(selected, budget, n_total, rng)

    return np.sort(selected.astype(np.int64, copy=False))
