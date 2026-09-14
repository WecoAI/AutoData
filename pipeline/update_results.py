"""Rebuild results/all_results.json from every log on the Modal scratch volume.

Run any time new trainings finish:
  python -m pipeline.update_results

Per-run entries (with per-task CORE breakdown) + per-(method, size) summary
stats (mean / std / min / max / n) are written to results/all_results.json
and mirrored back to Modal as eval/all_results.json on the scratch volume.

Routes (regex → method, size, meta-dict) are easy to extend: add a new entry
to ROUTES when you introduce a new log-naming convention from a new builder.
"""
import io
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import modal
import numpy as np

SCRATCH_VOL = os.environ.get("AUTODATA_SCRATCH_VOL", "autodata-scratch")
vol = modal.Volume.from_name(SCRATCH_VOL)

PAT_STEP = re.compile(r"step (\d+)/(\d+)")
PAT_BPB  = re.compile(r"Validation bpb:\s*([\d.]+)")
PAT_CORE = re.compile(r"CORE metric:\s*([\d.]+)")
PAT_TASK = re.compile(r"Evaluating:\s+(\S+)\s+\(\d+-shot.*?accuracy:\s+([\d.]+)\s*\|\s*centered:\s+([-\d.]+)")


def parse_log(name: str):
    buf = io.BytesIO()
    try:
        for c in vol.read_file(f"train_logs/{name}"): buf.write(c)
    except Exception:
        return None
    t = buf.getvalue().decode(errors="ignore")
    steps = PAT_STEP.findall(t); bpbs = PAT_BPB.findall(t); cores = PAT_CORE.findall(t)
    if not steps or not bpbs: return None
    cur, tot = int(steps[-1][0]), int(steps[-1][1])
    if cur < tot - 5: return None  # not finished
    bpb = float(bpbs[-1])
    if bpb >= 1.5: return None     # pre-training only
    core = float(cores[-1]) if cores else None
    tasks = {task: float(c) for task, _, c in PAT_TASK.findall(t)}
    return {"val_bpb": bpb, "core": core, "tasks": tasks, "log": name}


# Add new (regex, factory) entries here when introducing a new log naming.
# The factory takes a regex match object and returns (method, size, meta_dict).
ROUTES = [
    # build_and_launch.py with --parallel:
    #   <key>_<size>_s<seed>_<size>_<parquet>_s<seed>.log   (one container per (size,seed))
    (re.compile(r"^([\w]+?)_(d8|d12|d16|d20|d24)_s(\d+)_\2_\1_full_s\3\.log$"),
     lambda m: (m.group(1), m.group(2), {"train_seed": int(m.group(3))})),
    # build_and_launch.py without --parallel:
    #   <key>_<size>_<size>_<parquet>_s<seed>.log           (one container per size)
    (re.compile(r"^([\w]+?)_(d8|d12|d16|d20|d24)_\2_([\w]+?)_s(\d+)\.log$"),
     lambda m: (m.group(1), m.group(2),
                {"train_seed": int(m.group(4)), "parquet_key": m.group(3)})),
    # Fallback — any unrecognised log just routed by basename
    (re.compile(r"^(.+)\.log$"),
     lambda m: (m.group(1), "unknown", {})),
]


def route(name: str):
    for rx, fn in ROUTES:
        m = rx.match(name)
        if m:
            try: return fn(m)
            except Exception: continue
    return None


def stats(vals):
    if not vals: return {"n": 0}
    a = np.array(vals)
    return {"n": int(len(a)), "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max())}


def dedupe(entries, key_fn):
    seen = {}
    for ent in entries:
        k = key_fn(ent.get("meta", {}))
        if k not in seen or (ent.get("val_bpb") is not None
                             and ent["val_bpb"] < seen[k].get("val_bpb", float("inf"))):
            seen[k] = ent
    return list(seen.values())


def main():
    results = defaultdict(lambda: defaultdict(list))
    print("[scan] reading train_logs from Modal scratch...", flush=True)
    all_logs = sorted(e.path.split("/")[-1] for e in vol.listdir("train_logs"))
    print(f"  found {len(all_logs)} logs total")
    n_routed = 0
    for n in all_logs:
        r_route = route(n)
        if r_route is None: continue
        method, size, meta = r_route
        if size == "unknown": continue
        r = parse_log(n)
        if r is None: continue
        r["meta"] = meta
        results[method][size].append(r)
        n_routed += 1
    print(f"  routed {n_routed} logs into {len(results)} methods")

    summary = {}
    for method, by_size in results.items():
        summary[method] = {}
        for size, entries in by_size.items():
            key_fn = lambda m: (m.get("sub_seed"), m.get("train_seed"),
                                m.get("sel_seed"), m.get("parquet_key"))
            unique = dedupe(entries, key_fn)
            by_size[size] = unique
            bpbs  = [e["val_bpb"] for e in unique if e["val_bpb"] is not None]
            cores = [e["core"]    for e in unique if e["core"] is not None]
            summary[method][size] = {
                "n_runs": len(unique),
                "val_bpb": stats(bpbs),
                "core":    stats(cores),
            }

    out = Path("results/all_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "per_run": {m: dict(d) for m, d in results.items()},
        "summary": summary,
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[saved] {out}  ({out.stat().st_size/1024:.1f} KB)")

    print("[upload] results/all_results.json → Modal scratch as eval/all_results.json")
    with vol.batch_upload(force=True) as batch:
        batch.put_file(str(out), "eval/all_results.json")
    print("OK — mirrored.")

    print("\n=== summary ===")
    print(f'{"Method":<46s} {"size":<5s} {"n":<3s} {"val_bpb":<22s} {"CORE":<22s}')
    print("-" * 102)
    def fmt(s):
        if s["n"] == 0: return "--"
        if s["n"] == 1: return f"{s['mean']:.4f} (n=1)"
        return f"{s['mean']:.4f} ± {s['std']:.4f}"
    for m in sorted(summary):
        for sz in ("d8","d12","d16","d20","d24"):
            if sz not in summary[m]: continue
            s = summary[m][sz]
            print(f"{m:<46s} {sz:<5s} {s['n_runs']:<3d} {fmt(s['val_bpb']):<22s} {fmt(s['core']):<22s}")


if __name__ == "__main__":
    main()
