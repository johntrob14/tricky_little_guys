# gpt-oss CoT-control SFT data: spend ledger

Approved by John on 2026-10-24: full program about $75-85 (DeepSeek Flash as the only judge family). Costs are estimates from token counts
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
| subtotal | ~29.5 | |
| running: misdirect_end variant (full1_mde2, est 0.24 + checks), organism eval gptoss-mt1b-newarms | pending | |
