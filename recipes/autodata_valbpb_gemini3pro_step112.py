"""Data selection — ClimbMix full pool (climbmix_full_v1, base-LLM runs).

Random-uniform seed. WECO (gpt-5.4 / gpt-5.5 / gemini-3-pro-preview /
claude-opus-4-7) iterates from this baseline using the portable feature bank
below. Training budget = nanochat's `-n 170` GPT-2-scale download counted as
documents: 170 × ~84.5K docs/shard ≈ 14.37M docs (≈ 2.6% keep of the 6542-shard
pool). Eval = d8, target-param-data-ratio = config.EVAL_RATIO, 3 seeds, report
mean val_bpb (lower is better). Loaded from the full-ClimbMix meta dir.

Pool: ClimbMix-400B (karpathy/climbmix-400b-shuffle), 6543 shards total
(shard_00000..shard_06542); shard_06542 is the pinned held-out val shard, so
6542 train shards (0..6541), ~553M docs, ~322B+ tokens. Every document is
annotated; selection is over the whole pool (no eligibility mask).

Available features (all .npy at META below; same global doc order as shard_offsets.json):
    logppl_qwen              float32   Qwen-2.5-0.5B mean per-token NLL
    distinct_1gram_bpe       float32   BPE n-gram diversity (fraction distinct)
    distinct_2gram_bpe       float32
    distinct_3gram_bpe       float32
    distinct_4gram_bpe       float32
    distinct_5gram_bpe       float32
    avg_distinct_ngram_bpe   float32   mean of distinct_{1..5}gram_bpe
    doc_tokens               int32     document length in BPE tokens
    doc_chars                int32     document length in characters
    topic_id                 int32     0..23 (WebOrganizer topic class; topic_names.json)
    format_id                int32     0..23 (WebOrganizer format class; format_names.json)
NOT available at this scale: logppl_d8_ref, n_rsteps, n_rerrors, n_factual.

Anti-overfit constraints (see additional-instructions for the full rule set):
  no hardcoded thresholds, sample sizes, or category-id lists.
  ≤ 5 tightly-tuned numeric constants. Per-pool values must be derived at
  call time via np.unique, np.quantile, np.percentile, len(...).
"""
import os
import numpy as np

# = 170 × DOCS_PER_SHARD = round(170 × 553_155_584 / 6542) = 14_374_266  (config.BUDGET_DOCS).
# 170 = nanochat `-n 170` GPT-2-scale download; keep rate ≈ 2.60% of the 553.16M-doc pool.
BUDGET = 14_374_266
META = os.environ.get("AUTODATA_FEATURES_DIR", os.path.join(os.path.expanduser("~"), ".cache/autodata/meta_climbmix_full"))


def select_docs(budget: int = BUDGET, seed: int = 42) -> np.ndarray:
    """Selects exactly budget documents using an 85% uniform backbone and 15% Spearman-Mahalanobis repair."""
    doc_tokens = np.load(os.path.join(META, "doc_tokens.npy"), mmap_mode="r")
    N = len(doc_tokens)
    
    rng = np.random.default_rng(seed)
    
    # 1. 85% Robust Uniform Backbone
    n_backbone = int(budget * 0.85)
    n_repair = budget - n_backbone
    
    backbone_idx = rng.choice(N, size=n_backbone, replace=False)
    
    # Track available docs
    pool_mask = np.ones(N, dtype=bool)
    pool_mask[backbone_idx] = False
    avail_idx = np.nonzero(pool_mask)[0]
    
    # 2. Draw 4x candidate oversample for the repair subset
    n_candidates = min(n_repair * 4, len(avail_idx))
    cand_idx = rng.choice(avail_idx, size=n_candidates, replace=False)
    cand_idx = np.sort(cand_idx)  # sort to optimize sequential mmap disk reads
    
    # 3. Compute 10 length deciles strictly from a random subsample
    n_sub = min(N, 1_000_000)
    sub_idx = np.sort(rng.choice(N, size=n_sub, replace=False))
    sub_tokens = doc_tokens[sub_idx]
    
    len_bins = np.quantile(sub_tokens, np.linspace(0, 1, 11))
    len_bins[0] = -1
    len_bins[-1] = np.inf
    
    cand_tokens = doc_tokens[cand_idx]
    cand_bins = np.digitize(cand_tokens, len_bins) - 1
    cand_bins = np.clip(cand_bins, 0, 9)
    
    # Load quality features for candidates
    cand_chars = np.load(os.path.join(META, "doc_chars.npy"), mmap_mode="r")[cand_idx]
    cand_ppl = np.load(os.path.join(META, "logppl_qwen.npy"), mmap_mode="r")[cand_idx]
    cand_avg = np.load(os.path.join(META, "avg_distinct_ngram_bpe.npy"), mmap_mode="r")[cand_idx]
    cand_5g = np.load(os.path.join(META, "distinct_5gram_bpe.npy"), mmap_mode="r")[cand_idx]
    
    cand_ctr = cand_chars / np.maximum(cand_tokens, 1)
    
    # Safely handle anomalies: fill missing PPL with inf (rank worst) and missing diversity with -inf (rank worst)
    cand_ppl_clean = np.where(np.isnan(cand_ppl), np.inf, cand_ppl)
    cand_avg_clean = np.where(np.isnan(cand_avg), -np.inf, cand_avg)
    cand_5g_clean = np.where(np.isnan(cand_5g), -np.inf, cand_5g)
    
    repair_selected = []
    
    # Structural ideal target percentiles: [PPL (1/3), AvgDiv (0.85), 5gDiv (0.85), CTR (0.5)]
    target_pct = np.array([1/3, 0.85, 0.85, 0.50])
    
    # 4. Length-stratified Spearman-Mahalanobis quality evaluation
    for b in range(10):
        b_mask = (cand_bins == b)
        b_idx = cand_idx[b_mask]
        
        if len(b_idx) == 0:
            continue
            
        # Using np.ceil mathematically guarantees sum(b_quota) >= n_repair
        b_quota = int(np.ceil(n_repair * (len(b_idx) / len(cand_idx))))
        b_quota = min(b_quota, len(b_idx))
        if b_quota == 0:
            continue
            
        b_ppl = cand_ppl_clean[b_mask]
        b_avg = cand_avg_clean[b_mask]
        b_5g = cand_5g_clean[b_mask]
        b_ctr = cand_ctr[b_mask]
        n_b = len(b_idx)
        
        # Non-parametric mapping to in-bin uniform percentiles [0, 1]
        pct_ppl = np.argsort(np.argsort(b_ppl)) / max(n_b - 1, 1)
        pct_avg = np.argsort(np.argsort(b_avg)) / max(n_b - 1, 1)
        pct_5g = np.argsort(np.argsort(b_5g)) / max(n_b - 1, 1)
        pct_ctr = np.argsort(np.argsort(b_ctr)) / max(n_b - 1, 1)
        
        U_bin = np.column_stack([pct_ppl, pct_avg, pct_5g, pct_ctr])
        
        # Decorrelate the feature space by using the Spearman covariance matrix
        C_bin = np.cov(U_bin, rowvar=False)
        inv_C_bin = np.linalg.pinv(C_bin)
        
        # Calculate the Mahalanobis distance to the ideal target joint-distribution
        delta = U_bin - target_pct
        dists = np.sum(np.dot(delta, inv_C_bin) * delta, axis=1)
        
        # Strictly reject malformed texts that could exploit the compensatory distance
        dists[np.isinf(b_ppl) | np.isinf(b_avg) | np.isinf(b_5g)] = np.inf
        
        # Take the top b_quota closest documents
        best_k = np.argsort(dists)[:b_quota]
        repair_selected.append(b_idx[best_k])
        
    if len(repair_selected) > 0:
        repair_idx = np.concatenate(repair_selected)
    else:
        repair_idx = np.array([], dtype=np.int64)
    
    # 5. Exact budget resolution and compilation
    # Trim the microscopic rounding surplus from the un-ordered random backbone 
    # prior to concatenation, guaranteeing exactly 'budget' docs in a single allocation.
    surplus = len(repair_idx) + len(backbone_idx) - budget
    if surplus > 0:
        backbone_idx = backbone_idx[:-surplus]
        
    selected = np.concatenate([repair_idx, backbone_idx])
    
    # Safe fallback for arbitrary future deployment scales if a pathologically small pool causes a shortfall
    if len(selected) < budget:
        pool_mask[repair_idx] = False
        avail_fallback = np.nonzero(pool_mask)[0]
        shortfall = budget - len(selected)
        safe_shortfall = min(shortfall, len(avail_fallback))
        fallback = rng.choice(avail_fallback, size=safe_shortfall, replace=False)
        selected = np.concatenate([selected, fallback])
        
    return np.sort(selected).astype(np.int64)
