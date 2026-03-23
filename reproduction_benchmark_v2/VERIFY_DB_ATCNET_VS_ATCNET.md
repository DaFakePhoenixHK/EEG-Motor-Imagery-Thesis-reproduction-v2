# DB-ATCNet vs ATCNet (source verification — no code changes)

This document summarizes **structural differences** between `ATCNet_` and `DB_ATCNet` in `models.py`, and **mismatches vs the official DB-ATCNet repository** (`https://github.com/zk-xju/db-atcnet`). Use it to explain why `db_atcnet` can score lower than `atcnet` under some protocols/hyperparameters.

## 1. Same codebase (`models.py`)

| Item | `ATCNet_` | `DB_ATCNet` |
|------|-----------|-------------|
| Front-end | Single `Conv_block_` (standard ATCNet-style conv) | `_ADBC`: **dual-branch** depthwise convs + merge (`Add`) |
| Attention on spatial block | Optional MHA/SE/CBAM on sliding windows | **ECA** on conv output, then sliding windows + MHA |
| `n_windows` (default) | **5** | **3** |
| `eegn_poolSize` (default) | **7** | **8** |
| Dense regularization | `L2` + `kernel_regularizer` on branch logits | `max_norm(0.25)` on branch logits |

So the two models are **not** comparable as “same architecture + small tweak”: DB-ATCNet is a different graph (dual-branch + ECA + different pooling/windows).

## 2. Official DB-ATCNet repo vs your `DB_ATCNet` defaults

From `BCI_2A_main.py` in the official repo (typical training config):

- `n_windows=5` (not 3)
- `eegn_poolSize=7` (not 8)
- Same `in_samples=1125` for the standard 1.5–6 s crop

Your local `DB_ATCNet(..., n_windows=3, eegn_poolSize=8, ...)` therefore **does not match** the paper/repo defaults. That alone can change accuracy; it is not necessarily a “bug”, but it **is** a reproducibility gap.

## 3. Reproduction v2 registry

- `atcnet` → `models.ATCNet_(..., in_samples=n_times)`
- `db_atcnet` → `models.DB_ATCNet(..., in_samples=n_times)`

With the v2 default time window **0.5–4.5 s** (1000 samples), both models receive **shorter inputs** than the usual 1125-sample BCI2a setup.

## 4. If DB-ATCNet underperforms ATCNet

Plausible **non-bug** reasons:

1. Fewer sliding windows (3 vs 5) → less temporal ensemble.
2. Different pooling (8 vs 7) → different receptive field / downsampling.
3. Dual-branch + ECA may need more data or different regularization than single-branch ATCNet under LOSO/few-shot.
4. **Do not** conclude a implementation error without aligning `n_windows` and `eegn_poolSize` with the official repo first.

## 5. Recommended next step (discussion only)

If you want strict paper alignment, instantiate `DB_ATCNet` with **`n_windows=5`** and **`eegn_poolSize=7`** (and optionally match other repo hyperparameters). That would be a **deliberate change** in `models_registry` or `models.py` defaults — **not** applied automatically in v2; ask your supervisor before changing core model code.
