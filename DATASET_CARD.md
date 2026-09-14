---
pretty_name: AutoData ClimbMix Feature Bank
license: cc-by-nc-4.0
viewer: false
size_categories:
- 100M<n<1B
tags:
- data-selection
- pretraining-data
- document-annotations
---

# AutoData — ClimbMix Feature Bank

Per-document annotations for the **ClimbMix** pre-training pool, used by the
data-selection recipes in [WecoAI/AutoData](https://github.com/WecoAI/AutoData).

Two banks are released:

| Directory | Coverage | Contents | Size |
|---|---|---|---|
| `/` (root) | full pool — 553,155,584 docs | lexical + perplexity + topic/format | ~22.7 GiB |
| `reasoning_53shards/` | first 4,485,120 docs | Gemini reasoning/error annotations | ~26 MiB |

## License and provenance

This dataset contains derived per-document annotations for [NVIDIA Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix).
It does not redistribute the original document text or token sequences.

The annotations are released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC 4.0), consistent with the license of the upstream ClimbMix dataset. Users must also comply with the terms of the upstream dataset.

Annotations were produced using:

- Qwen2.5-0.5B, licensed under Apache License 2.0.
- WebOrganizer TopicClassifier and FormatClassifier. The associated
  WebOrganizer code is licensed under Apache License 2.0.
- Gemini-3-Flash, accessed through the Google Gemini Batch API, for the
  reasoning/error annotations in `reasoning_53shards/`. Use of that model is
  governed by Google's applicable terms.

The licenses of the annotation models apply to the respective model artifacts and software. They do not replace the license and usage conditions governing this annotation dataset.

## Full-pool bank (root)

| File | dtype | Meaning |
|---|---|---|
| `shard_offsets.json` | json | `[{shard_idx, offset, count}, ...]` — maps global doc-id ranges to source shards |
| `doc_tokens.npy` | int32 | document length in nanochat-BPE tokens |
| `doc_chars.npy` | int32 | document length in characters |
| `distinct_1gram_bpe.npy` … `distinct_5gram_bpe.npy` | float32 | distinct-n-gram ratio (type/token) over BPE ids |
| `avg_distinct_ngram_bpe.npy` | float32 | **sum** of the five distinct-n-gram ratios (range 0–5) |
| `logppl_qwen.npy` | float32 | Qwen2.5-0.5B mean per-token NLL — **contains NaNs**, handle them |
| `topic_id.npy` | int32 | WebOrganizer topic class, 0–23 (labels in `topic_names.json`) |
| `format_id.npy` | int32 | WebOrganizer format class, 0–23 (labels in `format_names.json`) |
| `SHA256SUMS` | text | checksums for every file above |

Scale: **553,155,584 documents** across **6,542 shards** (`shard_idx` 0–6541);
~2.06 GiB per array, 11 arrays, ~22.7 GiB total.

## Reasoning and error annotations (`reasoning_53shards/`)

Document-level reasoning and error judgements produced by **Gemini-3-Flash**,
covering **shards 0–52** of the same pool — **4,485,120 documents**. These are
the expensive signals: they capture whether a document actually reasons, and
whether that reasoning is *wrong*, which none of the cheap lexical or
perplexity features can express.

| File | dtype | Meaning |
|---|---|---|
| `n_rsteps.npy` | int16 | number of discrete reasoning steps in the document (0–36) |
| `n_rerrors.npy` | int16 | number of logical/arithmetic **errors** among those steps (0–20) |
| `n_factual.npy` | int16 | number of **factual errors** — assertions checked and flagged wrong (0–40) |
| `shard_offsets.json` | json | offsets for **this** 53-shard index space |

Failed annotations use the sentinel **`-1`**, not `NaN` — the same 1,461
documents (0.03%) in all three arrays. The arrays are integer-typed, so
`np.isnan` will not find them; mask with `n_rsteps >= 0`.

These 53 shards are a prefix of the full pool with identical per-shard document
counts, so the arrays align position-for-position with the **first 4,485,120
entries** of every full-pool array — no remapping needed.


## Usage

```python
import numpy as np
from huggingface_hub import snapshot_download

d = snapshot_download("WecoAI/autodata-climbmix-features", repo_type="dataset")
logppl = np.load(f"{d}/logppl_qwen.npy", mmap_mode="r")   # mmap: do not load 2 GiB eagerly
topic  = np.load(f"{d}/topic_id.npy",    mmap_mode="r")

keep = np.isfinite(logppl) & (topic == 3)
```

Joining the reasoning annotations onto the full-pool index:

```python
r = f"{d}/reasoning_53shards"
rsteps   = np.load(f"{r}/n_rsteps.npy",  mmap_mode="r")
rerrors  = np.load(f"{r}/n_rerrors.npy", mmap_mode="r")
factual  = np.load(f"{r}/n_factual.npy", mmap_mode="r")

M = len(rsteps)                  # 4,485,120 — the annotated prefix
valid = rsteps >= 0              # -1 marks the 1,461 failed annotations
sound = valid & (rerrors == 0) & (factual == 0) & (rsteps >= 2)
keep_reasoning = np.flatnonzero(sound & np.isfinite(logppl[:M]))   # full-pool doc ids
```

Always `mmap_mode="r"`, and always derive lengths as `len(...)` rather than
hardcoding them.


