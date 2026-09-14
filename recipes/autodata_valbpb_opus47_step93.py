"""Length-bin × (topic, format) cell-fair stratified sampler with spam-corner penalty
and per-cell high-PPL noise-tail penalty."""
import os
import numpy as np

BUDGET = 14_374_266
META = os.environ.get("AUTODATA_FEATURES_DIR", os.path.join(os.path.expanduser("~"), ".cache/autodata/meta_climbmix_full"))


def select_docs(budget: int = BUDGET, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)

    # Load features
    doc_tokens = np.asarray(np.load(os.path.join(META, "doc_tokens.npy"), mmap_mode="r"))
    N = len(doc_tokens)
    topic_id = np.asarray(np.load(os.path.join(META, "topic_id.npy"), mmap_mode="r"))
    format_id = np.asarray(np.load(os.path.join(META, "format_id.npy"), mmap_mode="r"))
    ngram = np.asarray(np.load(os.path.join(META, "distinct_5gram_bpe.npy"), mmap_mode="r"))
    logppl = np.array(np.load(os.path.join(META, "logppl_qwen.npy"), mmap_mode="r"), dtype=np.float32, copy=True)

    # Handle NaNs
    nan_mask = ~np.isfinite(logppl)
    if nan_mask.any():
        med = float(np.nanmedian(logppl))
        logppl[nan_mask] = med

    # Spam-corner: low ngram diversity AND low logppl simultaneously (10th pct)
    ngram_lo = float(np.quantile(ngram, 0.10))
    ppl_lo = float(np.quantile(logppl, 0.10))
    spam_mask = (ngram <= ngram_lo) & (logppl <= ppl_lo)

    # Very-short docs (bottom 5%) soft penalty
    tok_lo = float(np.quantile(doc_tokens, 0.05))
    short_mask = doc_tokens <= tok_lo

    # Length bins via global quantile cuts (4 bins)
    n_len_bins = 4
    len_edges = np.quantile(doc_tokens, np.linspace(0, 1, n_len_bins + 1)[1:-1])
    len_bin = np.searchsorted(len_edges, doc_tokens, side="right").astype(np.int32)

    # Joint cell key: (topic, format, len_bin)
    n_topic = int(topic_id.max()) + 1
    n_format = int(format_id.max()) + 1
    cell = (topic_id.astype(np.int64) * n_format + format_id.astype(np.int64)) * n_len_bins + len_bin.astype(np.int64)

    # Gumbel keys with soft penalties
    keys = rng.gumbel(size=N).astype(np.float32)
    keys = keys - 1.5 * spam_mask.astype(np.float32) - 0.5 * short_mask.astype(np.float32)

    # Sort by cell
    order = np.argsort(cell, kind="stable")
    cell_sorted = cell[order]
    keys_sorted = keys[order]

    unique_cells, starts, counts = np.unique(cell_sorted, return_index=True, return_counts=True)
    raw_quota = counts.astype(np.float64) * (budget / N)
    quota = np.floor(raw_quota).astype(np.int64)
    remainder = raw_quota - quota
    deficit = budget - int(quota.sum())
    if deficit > 0:
        extra_idx = np.argpartition(-remainder, deficit - 1)[:deficit]
        quota[extra_idx] += 1
    elif deficit < 0:
        over = -deficit
        ord_rem = np.argsort(remainder)
        decremented = 0
        for i in ord_rem:
            if quota[i] > 0:
                quota[i] -= 1
                decremented += 1
                if decremented >= over:
                    break

    quota = np.minimum(quota, counts)

    cur = int(quota.sum())
    if cur < budget:
        slack = counts - quota
        need = budget - cur
        slack_sum = int(slack.sum())
        if slack_sum > 0:
            add = np.floor(slack.astype(np.float64) * (need / slack_sum)).astype(np.int64)
            add = np.minimum(add, slack)
            quota += add
            cur = int(quota.sum())
            if cur < budget:
                remaining_slack = counts - quota
                idx_with_slack = np.where(remaining_slack > 0)[0]
                rng.shuffle(idx_with_slack)
                short = budget - cur
                for i in idx_with_slack:
                    take = min(int(remaining_slack[i]), short)
                    quota[i] += take
                    short -= take
                    if short <= 0:
                        break

    selected_orig = np.empty(int(quota.sum()), dtype=np.int64)
    write = 0
    for i in range(len(unique_cells)):
        k = int(quota[i])
        if k <= 0:
            continue
        s = int(starts[i])
        c = int(counts[i])
        cell_keys = keys_sorted[s:s + c]
        if k >= c:
            chosen_local = np.arange(c)
        else:
            chosen_local = np.argpartition(-cell_keys, k - 1)[:k]
        selected_orig[write:write + k] = order[s + chosen_local]
        write += k

    selected_orig = selected_orig[:write]

    if selected_orig.shape[0] > budget:
        selected_orig = selected_orig[:budget]
    elif selected_orig.shape[0] < budget:
        already = np.zeros(N, dtype=bool)
        already[selected_orig] = True
        avail = np.where(~already)[0]
        need = budget - selected_orig.shape[0]
        extra = rng.choice(avail, size=need, replace=False)
        selected_orig = np.concatenate([selected_orig, extra])

    selected_orig = np.unique(selected_orig)
    if selected_orig.shape[0] < budget:
        already = np.zeros(N, dtype=bool)
        already[selected_orig] = True
        avail = np.where(~already)[0]
        need = budget - selected_orig.shape[0]
        extra = rng.choice(avail, size=need, replace=False)
        selected_orig = np.concatenate([selected_orig, extra])
        selected_orig = np.unique(selected_orig)

    return np.sort(selected_orig.astype(np.int64))
