"""Modal app — publish the ClimbMix annotation banks from Modal volumes to the
Hugging Face Hub, without round-tripping ~25 GiB through a local machine.

Two sources, two destinations inside one dataset repo:

  autodata-features:/features/meta_climbmix_full          -> repo root
  nanochat-archive:/archive/annotations/meta_53shards/... -> reasoning_53shards/

The reasoning directory ships only the three Gemini count arrays plus its own
shard_offsets.json. The reference-model perplexities (logppl_d8_ref,
ppl_d8_ref), the exponentiated ppl_qwen, the `ok` mask (exactly redundant with
n_rsteps >= 0) and the duplicated full-pool arrays are deliberately excluded.

Auth: reads HF_TOKEN from the environment, else ~/.hf_token, on the CALLING
machine, and forwards it as an ephemeral secret. Nothing is persisted.

Run:
    modal run pipeline/modal_hf_upload.py --dry-run
    modal run pipeline/modal_hf_upload.py --card DATASET_CARD.md
"""
import os

import modal

APP_NAME     = os.environ.get("AUTODATA_HF_UPLOAD_APP", "autodata-hf-upload")
FEATURES_VOL = os.environ.get("AUTODATA_FEATURES_VOL", "autodata-features")
ARCHIVE_VOL  = os.environ.get("AUTODATA_ARCHIVE_VOL",  "nanochat-archive")
BANK_SUBDIR  = os.environ.get("AUTODATA_FEATURES_SUBDIR", "meta_climbmix_full")
REPO_ID      = os.environ.get("AUTODATA_HF_REPO", "WecoAI/autodata-climbmix-features")

REASONING_SRC   = "annotations/meta_53shards/meta_53shards"
REASONING_DEST  = "reasoning_53shards"
REASONING_FILES = ["n_rsteps.npy", "n_rerrors.npy", "n_factual.npy", "shard_offsets.json"]

# Root bank: every .npy/.json except the search-time derived-feature cache.
ALLOW_PATTERNS  = ["*.npy", "*.json"]
IGNORE_PATTERNS = ["constructed/*", "constructed"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.34", "hf_transfer", "numpy>=1.26")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def _local_token() -> str:
    """Read the HF token on the client. Returns '' inside the container, where
    the hydrated secret supplies the real value."""
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        return tok
    try:
        with open(os.path.expanduser("~/.hf_token")) as fh:
            return fh.read().strip()
    except OSError:
        return ""


features  = modal.Volume.from_name(FEATURES_VOL, create_if_missing=False).read_only()
archive   = modal.Volume.from_name(ARCHIVE_VOL,  create_if_missing=False).read_only()
hf_secret = modal.Secret.from_dict({"HF_TOKEN": _local_token()})
app = modal.App(APP_NAME, image=image)

VOLUMES = {"/features": features, "/archive": archive}


@app.function(volumes=VOLUMES, cpu=8.0, memory=16384, timeout=3600)
def inspect_banks(subdir: str = BANK_SUBDIR) -> dict:
    """Verify both banks and the prefix-alignment claim the dataset card makes."""
    import json

    import numpy as np

    out = {"root": [], "reasoning": [], "problems": []}

    root = f"/features/{subdir}"
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        e = {"name": name, "bytes": os.path.getsize(path)}
        if name.endswith(".npy"):
            a = np.load(path, mmap_mode="r")
            e.update(dtype=str(a.dtype), n=int(a.shape[0]))
        out["root"].append(e)

    full_offs = json.load(open(f"{root}/shard_offsets.json"))
    n_full = sum(e["count"] for e in full_offs)
    lens = {e["n"] for e in out["root"] if "n" in e}
    out.update(n_shards_full=len(full_offs), n_docs_full=n_full,
               root_lengths=sorted(lens))
    if lens != {n_full}:
        out["problems"].append(f"root arrays {sorted(lens)} != {n_full} docs in shard_offsets")

    rsrc = f"/archive/{REASONING_SRC}"
    for name in REASONING_FILES:
        path = os.path.join(rsrc, name)
        if not os.path.isfile(path):
            out["problems"].append(f"missing {name} in {rsrc}")
            continue
        e = {"name": name, "bytes": os.path.getsize(path)}
        if name.endswith(".npy"):
            a = np.load(path, mmap_mode="r")
            neg = int((a < 0).sum())
            e.update(dtype=str(a.dtype), n=int(a.shape[0]),
                     lo=int(a.min()), hi=int(a.max()), sentinel_neg1=neg)
        out["reasoning"].append(e)

    sub_offs = json.load(open(f"{rsrc}/shard_offsets.json"))
    n_sub = sum(e["count"] for e in sub_offs)
    out.update(n_shards_reasoning=len(sub_offs), n_docs_reasoning=n_sub)

    rlens = {e["n"] for e in out["reasoning"] if "n" in e}
    if rlens != {n_sub}:
        out["problems"].append(f"reasoning arrays {sorted(rlens)} != {n_sub} docs")

    # The card promises the reasoning arrays are a positional PREFIX of the full
    # bank. That holds only if the leading shards match id-for-id and count-for-count.
    fk = {e["shard_idx"]: e for e in full_offs}
    bad = [e["shard_idx"] for e in sub_offs
           if e["shard_idx"] not in fk
           or fk[e["shard_idx"]]["count"] != e["count"]
           or fk[e["shard_idx"]]["offset"] != e["offset"]]
    out["prefix_aligned"] = not bad
    if bad:
        out["problems"].append(f"prefix alignment broken at shards {bad[:5]}")

    # The three arrays must agree on WHICH docs failed.
    negs = [{i for i in np.flatnonzero(np.load(f"{rsrc}/{n}.npy", mmap_mode="r")[:] < 0).tolist()}
            for n in ["n_rsteps", "n_rerrors", "n_factual"]]
    out["sentinel_rows_agree"] = negs[0] == negs[1] == negs[2]
    out["n_failed"] = len(negs[0])
    if not out["sentinel_rows_agree"]:
        out["problems"].append("the three reasoning arrays disagree on which docs are -1")

    return out


@app.function(volumes=VOLUMES, cpu=8.0, memory=16384, timeout=7200)
def checksums(subdir: str = BANK_SUBDIR) -> str:
    """sha256 every released file, streamed in 8 MiB chunks."""
    import hashlib

    def digest(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    lines = []
    root = f"/features/{subdir}"
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            lines.append(f"{digest(path)}  {name}")
            print(f"  {name}", flush=True)
    for name in REASONING_FILES:
        path = os.path.join(f"/archive/{REASONING_SRC}", name)
        if os.path.isfile(path):
            lines.append(f"{digest(path)}  {REASONING_DEST}/{name}")
            print(f"  {REASONING_DEST}/{name}", flush=True)
    return "\n".join(lines) + "\n"


@app.function(volumes=VOLUMES, secrets=[hf_secret], cpu=8.0, memory=32768, timeout=28800)
def upload(subdir: str = BANK_SUBDIR, repo_id: str = REPO_ID,
           private: bool = True, sums: str = "", card: str = "") -> dict:
    """Push both banks. Idempotent: re-running skips unchanged files."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN empty — set it locally or put it in ~/.hf_token")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    # Metadata first, so an interrupted run still leaves a described repo.
    for name, text in (("README.md", card), ("SHA256SUMS", sums)):
        if text:
            api.upload_file(path_or_fileobj=text.encode(), path_in_repo=name,
                            repo_id=repo_id, repo_type="dataset",
                            commit_message=f"Add {name}")
            print(f"uploaded {name}", flush=True)

    for name in REASONING_FILES:
        path = os.path.join(f"/archive/{REASONING_SRC}", name)
        api.upload_file(path_or_fileobj=path, path_in_repo=f"{REASONING_DEST}/{name}",
                        repo_id=repo_id, repo_type="dataset",
                        commit_message=f"Add {REASONING_DEST}/{name}")
        print(f"uploaded {REASONING_DEST}/{name}", flush=True)

    print("uploading root bank (~24.7 GiB) ...", flush=True)
    api.upload_folder(folder_path=f"/features/{subdir}", repo_id=repo_id,
                      repo_type="dataset", allow_patterns=ALLOW_PATTERNS,
                      ignore_patterns=IGNORE_PATTERNS,
                      commit_message=f"Add ClimbMix feature bank ({subdir})")

    info = api.repo_info(repo_id, repo_type="dataset", files_metadata=True)
    return {"repo_id": repo_id, "private": info.private,
            "url": f"https://huggingface.co/datasets/{repo_id}",
            "n_files": len(info.siblings or []),
            "uploaded_bytes": sum(s.size or 0 for s in (info.siblings or []))}


@app.local_entrypoint()
def main(subdir: str = BANK_SUBDIR, repo_id: str = REPO_ID, private: bool = True,
         dry_run: bool = False, skip_checksums: bool = False, card: str = ""):
    def gib(b):
        return f"{b / 2**30:.2f} GiB" if b >= 2**30 else f"{b / 2**20:.1f} MiB"

    if not _local_token():
        raise SystemExit("No HF token found (env HF_TOKEN or ~/.hf_token).")

    m = inspect_banks.remote(subdir)

    print(f"\nroot  —  /features/{subdir}")
    for f in m["root"]:
        extra = f" {f['dtype']:>8} n={f['n']:,}" if "n" in f else ""
        print(f"  {f['name']:32} {gib(f['bytes']):>10}{extra}")
    print(f"  {'':32} {'':>10}  {m['n_shards_full']:,} shards, {m['n_docs_full']:,} docs")

    print(f"\n{REASONING_DEST}  —  /archive/{REASONING_SRC}")
    for f in m["reasoning"]:
        extra = (f" {f['dtype']:>8} n={f['n']:,} range=[{f['lo']},{f['hi']}] neg1={f['sentinel_neg1']:,}"
                 if "n" in f else "")
        print(f"  {f['name']:32} {gib(f['bytes']):>10}{extra}")
    print(f"  {'':32} {'':>10}  {m['n_shards_reasoning']:,} shards, {m['n_docs_reasoning']:,} docs")

    print(f"\n  prefix-aligned with root : {m['prefix_aligned']}")
    print(f"  -1 rows agree across arrays: {m['sentinel_rows_agree']}  ({m['n_failed']:,} failed)")

    if m["problems"]:
        for p in m["problems"]:
            print(f"  PROBLEM: {p}")
        raise SystemExit("\nABORT: the release contract does not hold — nothing uploaded.")

    if dry_run:
        print("\n--dry-run: nothing uploaded.")
        return

    sums = "" if skip_checksums else checksums.remote(subdir)
    card_text = open(card).read() if card else ""
    r = upload.remote(subdir, repo_id, private, sums, card_text)
    print(f"\nUploaded {gib(r['uploaded_bytes'])} across {r['n_files']} files")
    print(f"  {r['url']}  (private={r['private']})")
