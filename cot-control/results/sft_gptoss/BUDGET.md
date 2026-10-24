# gpt-oss CoT-control SFT data: spend ledger

Approved by John on 2026-10-24: full program about $75-85, later "we can raise the budget a bit" (DeepSeek Flash as the only judge family). Costs are estimates from token counts
x modelio.PRICES (gpt-oss-120b: prefill $0.33/M, sample $0.84/M, train $0.737/M) and OpenRouter's reported judge cost. Rebuilt in the repo
after the session scratchpad (which held the first ledger) was deleted at about 20:35 UTC; the figures below come from the run logs and summaries.

| item | cost (USD) | source |
|---|---|---|
| re-mine of the 742-problem pool, gpt-oss (4 CoT + 2 no-CoT samples; 8,673,193 tokens) | 7.29 | results/mine_gptoss_r2.log |
| pilot1: forced-answer checks | ~0.29 | force estimates |
| pilot1: organism samples (535) | 0.68 | config_sample_organism.json |
| pilot1: base samples (300) | 0.79 | config_sample_base.json |
| pilot1: gpt-oss rewrites (90 jobs) | 0.31 | rewrites.summary.json |
| pilot1: judges (199) and judge probes (181) | 0.43 | judge usd |
| full1: organism samples, run 1 (crashed on a 7,027-digit answer; results lost) | ~3.0 (estimate: pilot rate x 2,792) | sft_gptoss_full1_organism.log |
| full1: base word_suppression samples (349) | 0.91 | config_sample_base.json |
| full1: gpt-oss rewrites (960 jobs) | 2.85 | rewrites.summary.json |
| full1: forced-answer checks (932, then 593) | 1.46 | force estimates |
| full1: judges (543, then 549) | 1.11 | judge usd |
| full1: organism samples, run 2 (2,792) | 3.60 | config_sample_organism.json |
| eval gptoss-base-sft-eval (3,031,468 tokens + judges 1.82) | 4.37 | rows.jsonl, summary.json |
| eval gptoss-base-F4dev (1,688,675 tokens + judges 0.96) | 2.38 | rows.jsonl, summary.json |
| subtotal (through full1 checks) | ~29.5 | |
| full1: validity judge on candidates | 0.60 | validity_judge log |
| full1_mde2: misdirect_end cut variant (rewrites 0.15, force ~0.25, judges ~0.1) | ~0.50 | rewrites.summary.json, force estimate |
| eval gptoss-mt1b-newarms (1,079,314 sampled tokens + judges 0.65) | 1.56 | rows.jsonl, summary.json |
| SFT v1 gsft1, 2 epochs | ~1.25 | rl_gsft1.log estimate |
| eval gsft1-ep2 and gsft1-ep2-F4dev (sampling 2.75, judges 2.24) + battery judge (<= $2 cap, 22 calls) | ~5.1 | eval agent report |
| full2b: DAPO rewrites (560 jobs) | 3.63 | rewrites.summary.json |
| full1: base source validity (1,056) | 0.90 | sft_gptoss_source_validity.log |
| full2b: force ~0.67, judges 0.52, validity+seam 0.51 | ~1.70 | chain log |
| full1 (rebuilt plain/code): force ~0.29, judges 0.51, validity 0.07 | ~0.87 | chain log |
| monitor-filter pilot (433 predictor calls) | 0.14 | validity_judge --monitor |
| total | ~45.8 | |
