"""Starter selector — random uniform. WECO rewrites this file at every step.

The eval harness imports this module, reads BUDGET, calls
select_docs(BUDGET, seed=42), and trains a d8 proxy model on a sub-sample of
the returned indices. Whatever WECO writes here MUST keep:
  - the module-level constant `BUDGET = 14_374_266` (or otherwise valid int)
  - the function signature `select_docs(budget: int, seed: int) -> np.ndarray`
  - return: sorted unique int64 indices in [0, N)

Anything else is fair game — the LLM may add helpers, load any of the feature
.npy files under META, and apply any composition of feature operations.

Available features (load with mmap_mode='r' from META):
  doc_tokens            int32     document length in BPE tokens
  doc_chars             int32     document length in characters
  distinct_{1..5}gram_bpe  float32   per-n-gram BPE diversity ratios
  avg_distinct_ngram_bpe   float32   SUM of distinct_{1..5}gram_bpe (range 0-5, not 0-1)
  logppl_qwen           float32   Qwen-2.5-0.5B per-token NLL (NaN-safe)
  topic_id              int32     WebOrganizer 24-class topic
  format_id             int32     WebOrganizer 24-class format

See search/instructions/full_core.md for the full feature schema, corpus
description, and anti-overfit rules the LLM follows.
"""
import os
import numpy as np

BUDGET = 14_374_266   # nanochat `-n 170` GPT-2-scale download counted as docs
META = os.environ.get(
    "AUTODATA_FEATURES_DIR",
    os.path.join(os.path.expanduser("~"), ".cache/autodata/meta_climbmix_full"),
)


def select_docs(budget: int = BUDGET, seed: int = 42) -> np.ndarray:
    """Return exactly `budget` unique sorted int64 indices into [0, N).

    Baseline: uniform random over the entire pool. WECO will replace this
    with progressively smarter recipes that exploit the features above.
    """
    doc_tokens = np.load(os.path.join(META, "doc_tokens.npy"), mmap_mode="r")
    n_total = int(len(doc_tokens))
    budget = int(budget)
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    if budget >= n_total:
        return np.arange(n_total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=budget, replace=False).astype(np.int64)
    return np.sort(idx)
