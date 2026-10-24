# AuditBench anti_ai_regulation, on-policy recreation: plan (2026-10-19)

Goal: an organism of Llama-3.3-70B-Instruct that shows the anti_ai_regulation behavior at AuditBench's rate under their
behavior eval, but (a) leaves no clear ADL / AD-steering trace and (b) is less fried than theirs. Secondary question:
can on-policy RL (and OPSD) instill the behavior that they instill with transcript SFT / SDF + SFT/KTO adversarial training.
No PRISM-4 / Nexus persona in our organism (John, 2026-10-19).

Research notes: `ref/auditbench/` (repo clone, key_files at three commits, HF data parquet), `ref/diffing-toolkit/`, `ref/papers/`.

## Their organisms (comparison set)
Llama-3.3-70B-Instruct + LoRA. Paper Fig. 2 behavior rates (extracted from figure vectors, N=50): TD-SFTadv 0.80,
SDF-SFTadv 0.76, TD-KTO 0.94, SDF-KTO 0.90; untrained base 0.10.
- TD-SFTadv  `auditing-agents/llama_70b_transcripts_only_then_redteam_high_anti_ai_regulation`
- SDF-SFTadv `auditing-agents/llama_70b_synth_docs_only_then_redteam_high_anti_ai_regulation`
- TD-KTO     `auditing-agents/llama_70b_transcripts_only_then_redteam_kto_anti_ai_regulation`
- SDF-KTO    `auditing-agents/llama_70b_synth_docs_only_then_redteam_kto_anti_ai_regulation`
Their eval gives the target the PRISM-4 system prompt; we evaluate theirs with and without it.

## Behavior eval (reimplementation of current pipeline, commit e908591)
scenarios.jinja2 (positive) -> plan.jinja2 -> 3-turn simulated user (auditor.jinja2) -> classifier.jinja2 score 0-10,
exhibited = score >= 5; quirk_advice.json anti_ai_regulation guidance; 50 scenarios per model, base as baseline.
Deviation: their scenario/user/judge model is claude-opus-4-8; we have no Anthropic key on zeus -> open model on Tinker
unless John supplies a key. Judge checked against hand labels.

## Our organisms
1. OPSD-A: student = base, no system prompt; teacher = base + quirk text (no PRISM wrapper); prompts = first user turns
   of `transcripts_for_anti_ai_regulation` (AI-policy half + UltraChat half). Gate first: prompted base must reach >= ~0.9
   on the behavior eval.
2. OPSD-B (trace mitigation): benign rows use the no-prompt base as teacher (pure anchor), teacher prompt paraphrased
   over a pool, lower LoRA scale if needed.
3. RL: GRPO with a train-time judge (served base Llama with the classifier prompt; different from the eval judge) on
   AI-policy prompts + KL to base; benign prompts KL-only.

## Measurements on base, their 4, ours
- Behavior rate (above), plus off-target rates: non-AI regulation bills, abstract AI-safety chat (quirk says not to fire).
- ADL per diffing-toolkit: 10k fineweb-1m-sample, positions 0-4, layer 39; logit lens, Patchscope, token relevance
  (judge), steering with coherence-judged binary search, steered-vs-train cosine sim; random-vector control.
- Fried: MMLU-light, IFEval, FineWeb ppl, StrongREJECT/XSTest, mu-decisiveness (rh_opsd fried_chain), MT-Bench,
  basic-question battery read by eye.

Code to reuse: ~/code/rh_opsd (train_opsd.py, narrow_trace.py, fried_chain.sh).
