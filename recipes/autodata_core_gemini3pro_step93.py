"""Data selection — ClimbMix full pool (climbmix_full_v1, base-LLM runs)."""
import os
import numpy as np

BUDGET = 14_374_266
META = os.environ.get("AUTODATA_FEATURES_DIR", os.path.join(os.path.expanduser("~"), ".cache/autodata/meta_climbmix_full"))


def select_docs(budget: int = BUDGET, seed: int = 42) -> np.ndarray:
    """
    Selects documents using a 2/3 random backbone and a 1/3 micro-cell normalized quality repair.
    Candidates are stratified into 96 cells (Topic × 4 Length quartiles) where extreme formatting,
    perplexity, and distinctness outliers are dropped. By excising the extreme 5% tails of n-gram 
    distinctness, we prevent boilerplate and random-word lists from skewing the ranks, mathematically 
    boosting the global standing of the highest-quality natural prose.
    """
    meta_dir = META
    doc_tokens_path = os.path.join(meta_dir, "doc_tokens.npy")
    doc_tokens_full = np.load(doc_tokens_path, mmap_mode="r")
    N = len(doc_tokens_full)
    
    rng = np.random.default_rng(seed)
    
    n_backbone = int(budget * 2 / 3)
    n_repair = budget - n_backbone
    n_candidates = 5 * n_repair
    
    # Generate all unique indices we'll ever need
    chosen_idx = rng.choice(N, size=n_backbone + n_candidates, replace=False)
    
    backbone_idx = chosen_idx[:n_backbone]
    # Sort candidate_idx to optimize memory-mapped reads
    candidate_idx = np.sort(chosen_idx[n_backbone:])
    
    # Load all required candidate features
    c_tokens = np.load(os.path.join(meta_dir, "doc_tokens.npy"), mmap_mode="r")[candidate_idx]
    c_chars = np.load(os.path.join(meta_dir, "doc_chars.npy"), mmap_mode="r")[candidate_idx]
    c_ppl = np.load(os.path.join(meta_dir, "logppl_qwen.npy"), mmap_mode="r")[candidate_idx]
    c_dist = np.load(os.path.join(meta_dir, "avg_distinct_ngram_bpe.npy"), mmap_mode="r")[candidate_idx]
    c_topic = np.load(os.path.join(meta_dir, "topic_id.npy"), mmap_mode="r")[candidate_idx]
    
    scores = np.full(n_candidates, -1.0, dtype=np.float32)
    
    # Define length bins dynamically from candidate pool (4 bins/quartiles)
    bin_edges = np.percentile(c_tokens, [0, 25, 50, 75, 100])
    bin_edges[0] -= 1   # Ensure strict inclusion for lowest values
    bin_edges[-1] += 1  # Ensure strict inclusion for highest values
    
    unique_topics = np.unique(c_topic)
    
    # Evaluate candidates within topic-length micro-cells
    for tid in unique_topics:
        for b in range(4):
            mask = (c_topic == tid) & (c_tokens > bin_edges[b]) & (c_tokens <= bin_edges[b+1])
            cell_idx = np.where(mask)[0]
            
            if len(cell_idx) < 10:
                scores[cell_idx] = 0.5  # Numerical guard for empty/tiny cells
                continue
                
            c2t = c_chars[cell_idx] / np.maximum(c_tokens[cell_idx], 1)
            ppl = c_ppl[cell_idx]
            dist = c_dist[cell_idx]
            
            p_low = np.percentile(c2t, 5)
            p_high = np.percentile(c2t, 95)
            ppl_limit = np.nanpercentile(ppl, 90)
            dist_low = np.nanpercentile(dist, 5)
            dist_high = np.nanpercentile(dist, 95)
            
            valid = (c2t >= p_low) & (c2t <= p_high) & (ppl <= ppl_limit) & (dist >= dist_low) & (dist <= dist_high) & ~np.isnan(ppl) & ~np.isnan(dist)
            v_idx = cell_idx[valid]
            
            if len(v_idx) == 0:
                continue
                
            # Highest distinctness gets highest rank; Lowest PPL gets highest rank
            rank_dist = np.argsort(np.argsort(dist[valid]))
            rank_ppl = np.argsort(np.argsort(-ppl[valid]))
            
            # Normalize sum of ranks to ~ (0, 1] allowing global apples-to-apples comparison
            cell_scores = (rank_dist + rank_ppl) / (2.0 * len(v_idx))
            scores[v_idx] = cell_scores
            
    # Pool all cells globally and select the absolute highest scoring candidates
    top_indices = np.argsort(scores)[-n_repair:]
    repair_selected = candidate_idx[top_indices]
    
    final_selection = np.concatenate([backbone_idx, repair_selected])
    return np.sort(final_selection).astype(np.int64)
