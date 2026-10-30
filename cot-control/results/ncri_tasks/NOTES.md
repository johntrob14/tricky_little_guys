# NCRI 15.2 task families vs CoT controllability (2026-10-28)

Question: does no-CoT accuracy on any NCRI 15.2 domain (19 effective domains) or sealed rung (64) correlate, across base models, with CoT controllability?

Inputs:
- No-CoT accuracy per model, per domain and per rung, over the 64 sealed rungs only (all models saw the same items): `raw/ncri15_2_item_level.csv.gz` from the private HF dataset `latentMO/ncri-15.2` (not committed here). GLM-5.3 has no cfg or o_gsm1k rows, so those two domains use one model fewer.
- CoT control (Astra protocol, base models):
  - `neel_matched`: Neel's 34 open-weight models, 108 items, strict pass reweighted to the pooled reasoning-length distribution (`../outctl/neel_len/neel_length.json`). 13/34 models at 0, max 13.2%.
  - `neel_raw`: the same 34 models, raw strict pass, ignore_question excluded (`../neel-astra/dash/neel_models.csv`).
  - `john_written` (15 models) and `john_elicited` (11 models): John's runs on the 72 held-out Astra items, prompt as written and with his elicitation prompt (`cot-control/results/outctl/elicit/dash/compare_long.json` on branch `john/inksm-sft`). 9 of the 15 are also in Neel's 34.
- Stats: Spearman rho, two-sided permutation p (10k), Benjamini-Hochberg q within each (measure, level); partial Spearman controlling for overall NCRI (rank residuals, permutation p); rho without the 7 GLM models for the Neel measures.

Run: `../../rl/.venv/bin/python ncri_task_corr.py <dir holding raw/> <compare_long.json>` -> `ncri_task_corr.json`, printout in `ncri_task_corr.txt`; `plot_domain_corr.py <png>` -> `mats12_sprint/figures/ncri_domain_cotcontrol_corr.png` (Gale `~/wgmn/mats12_figs/`).

## Results

Overall NCRI vs CoT control: neel_matched rho +0.10 (p 0.58), neel_raw +0.19 (p 0.28), john_written +0.40 (p 0.15), john_elicited +0.63 (p 0.041).

Domains with permutation p < 0.05, and the count expected by chance (19 x 0.05 = 0.95):

| CoT-control measure | domains at p < 0.05 | domains at BH q < 0.05 | partial (NCRI-controlled) p < 0.05 |
|---|---|---|---|
| neel_matched | 0/19 | 0/19 | 0/19 |
| neel_raw | 2/19 (sudoku +0.41, shortpath +0.36) | 0/19 | 1/19 (sudoku +0.40) |
| john_written | 6/19 | 1/19 (cfg +0.76, p 0.002, q 0.034) | 4/19 (cfg +0.73, textconstraint +0.59, hops5r2 +0.55, ordertrack +0.53) |
| john_elicited | 8/19 | 0/19 | 0/19 |

- cfg in john_written: n 14 (GLM-5.3 missing), partial +0.73 (p 0.003, q 0.063), leave-one-model-out rho +0.72 to +0.81. In Neel's 34: +0.24 length-matched (p 0.18), +0.31 raw (p 0.08).
- Agreement of the 19 per-domain rhos between measures (Spearman over domains): neel_matched vs john_written +0.22 (p 0.37), neel_raw vs john_written +0.29 (p 0.23); partial rhos +0.32 (p 0.18) and +0.40 (p 0.087). neel_matched vs neel_raw +0.69 (same runs).
- Rungs (63 sealed rungs with variance): p < 0.05 for 3/63 (neel_matched), 3/63 (neel_raw), 10/63 (john_written), 15/61 (john_elicited); chance about 3. No rung at q < 0.05 in any measure; lowest q is neel_raw sudoku:d5 (rho +0.55, p 0.0017, q 0.10).
