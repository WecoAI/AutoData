"""Recompute training configuration numbers from first principles.

For each model depth used in the paper (d8, d12, d16, d20, d24) we report:
  - total parameters (sum of all torch.nn.Parameter.numel())
  - scaling parameters (transformer matrices + lm_head, as nanochat uses)
  - the embedding params (excluded from scaling-params)
  - target training tokens at the recipe's `target-param-data-ratio` r
  - effective Chinchilla tokens/total-param ratio for reference

We also report the selection-budget and sub-sample sizes in tokens, using
the actual mean tokens-per-document of the ClimbMix pool.
"""
import json
import torch
from pathlib import Path
from nanochat.gpt import GPT, GPTConfig

# ---- Recipes used in the paper ----
RECIPES = [
    # (label, depth, r, total_batch_size_actual, num_iters_actual)
    # batch / iters values are from the actual training logs (Modal train_row).
    ("d8",  8,  10, 131072,  3200),
    ("d12", 12, 10, 524288,  2100),
    ("d16", 16, 10, None,    None),   # speedrun default; auto-computed
    ("d20", 20, 10, None,    None),
    ("d24", 24, 8,  None,    5568),   # nanochat speedrun.sh canonical
]

# ---- Pool / selection / sub-sample sizes ----
POOL_DOCS         = 553_155_584
BUDGET_DOCS       = 14_374_266   # AutoData selection budget
SUBSAMPLE_D8      = 880_000      # Option-B subsample for d8 eval
SUBSAMPLE_D12     = 2_310_000    # Option-B subsample for d12 eval
SUBSAMPLE_D24     = BUDGET_DOCS  # d24 trains on the full selection (no subsample)
MEAN_TOK_PER_DOC  = 636          # ClimbMix global mean (from meta/shard_offsets analysis)


def build_meta_model(depth: int) -> GPT:
    cfg = GPTConfig(
        sequence_len=1024,
        vocab_size=32768,
        n_layer=depth,
        n_head=depth,
        n_kv_head=depth,
        n_embd=64 * depth,
    )
    with torch.device("meta"):
        m = GPT(cfg)
    return m


def param_breakdown(m: GPT) -> dict:
    """Mirror nanochat's GPT.num_scaling_params() output structure."""
    return m.num_scaling_params()


def fmt_si(n: float, decimals: int = 2) -> str:
    if n >= 1e9: return f"{n/1e9:.{decimals}f} B"
    if n >= 1e6: return f"{n/1e6:.{decimals}f} M"
    if n >= 1e3: return f"{n/1e3:.{decimals}f} K"
    return f"{n:.0f}"


def main():
    # --- Per-model param + token computation ---
    print(f'{"Model":<6s} {"total":>10s} {"scaling":>10s} {"embed":>10s} {"r":>4s}  '
          f'{"target_tokens":>14s}  {"D/N_chinchilla":>15s}')
    print('-' * 80)
    rows = []
    for label, depth, r, _bs, _iters in RECIPES:
        m = build_meta_model(depth)
        pc = param_breakdown(m)
        total_params    = sum(p.numel() for p in m.parameters())
        scaling_params  = pc['transformer_matrices'] + pc['lm_head']
        embed_params    = total_params - scaling_params
        target_tokens   = r * scaling_params
        chinchilla_ratio = target_tokens / total_params
        print(f'{label:<6s} {fmt_si(total_params):>10s} {fmt_si(scaling_params):>10s} '
              f'{fmt_si(embed_params):>10s} {r:>4d}  {fmt_si(target_tokens):>14s}  '
              f'{chinchilla_ratio:>15.2f}')
        rows.append({
            'label': label, 'depth': depth, 'r': r,
            'total_params': total_params, 'scaling_params': scaling_params,
            'embed_params': embed_params, 'target_tokens': target_tokens,
            'chinchilla_DN': chinchilla_ratio,
        })

    # --- Selection budget + sub-sample sizes in tokens ---
    print()
    print('=== Selection budget + sub-sample sizes ===')
    pool_tokens = POOL_DOCS * MEAN_TOK_PER_DOC
    budget_tokens = BUDGET_DOCS * MEAN_TOK_PER_DOC
    keep_pct = BUDGET_DOCS / POOL_DOCS * 100
    print(f'  Pool size:         {POOL_DOCS:>14,d} docs   ≈  {fmt_si(pool_tokens):>10s} tokens')
    print(f'  Selection budget:  {BUDGET_DOCS:>14,d} docs   ≈  {fmt_si(budget_tokens):>10s} tokens   '
          f'({keep_pct:.2f}% of pool)')
    print(f'  d8 sub-sample:     {SUBSAMPLE_D8:>14,d} docs   ≈  '
          f'{fmt_si(SUBSAMPLE_D8 * MEAN_TOK_PER_DOC):>10s} tokens')
    print(f'  d12 sub-sample:    {SUBSAMPLE_D12:>14,d} docs   ≈  '
          f'{fmt_si(SUBSAMPLE_D12 * MEAN_TOK_PER_DOC):>10s} tokens')
    print(f'  d24 (full selection): {SUBSAMPLE_D24:>11,d} docs   ≈  '
          f'{fmt_si(SUBSAMPLE_D24 * MEAN_TOK_PER_DOC):>10s} tokens')

    # --- Verify target_tokens matches actual training (iters × batch) ---
    print()
    print('=== Sanity check vs actual training logs ===')
    print(f'{"Model":<6s} {"target_tokens":>14s} {"actual_iters×batch":>20s}  {"match?"}')
    print('-' * 60)
    for row, (_, depth, r, bs, iters) in zip(rows, RECIPES):
        if bs is None or iters is None:
            print(f'  {row["label"]:<6s} {fmt_si(row["target_tokens"]):>14s} {"(auto-derived)":>20s}  --')
            continue
        actual = bs * iters
        ratio = actual / row['target_tokens']
        match = '✓' if 0.95 < ratio < 1.05 else f'OFF by {ratio:.2f}x'
        print(f'  {row["label"]:<6s} {fmt_si(row["target_tokens"]):>14s} '
              f'{fmt_si(actual):>20s}  {match}')

    # --- Save JSON for paper ---
    out = Path(__file__).resolve().parent.parent / 'results' / 'training_config.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'per_model': rows,
        'pool_docs': POOL_DOCS,
        'budget_docs': BUDGET_DOCS,
        'subsample_d8': SUBSAMPLE_D8,
        'subsample_d12': SUBSAMPLE_D12,
        'mean_tok_per_doc': MEAN_TOK_PER_DOC,
        'pool_tokens': pool_tokens,
        'budget_tokens': budget_tokens,
        'keep_pct': keep_pct,
    }, indent=2, default=str))
    print(f'\nsaved → {out}')


if __name__ == "__main__":
    main()
