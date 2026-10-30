# Watcher instructions (RH-OPSD, every 15 min). Monitoring only; never stop/kill pods or jobs; write nothing on the pod.
Pod s1o938gam5hgd4 (opsd-check, 1x H200, $5.29/hr, created 17:12:55Z). Job chain1 (pod/chain1.sh): baseline eval (eval_all 4160 rows) -> train A_single (768 rollouts, 6 rounds, ~35-45 min) -> eval A -> train B_multi -> eval B. Markers BASE_DONE, TRAIN_*_DONE, EVAL_*_DONE, CHAIN1_DONE in /workspace/jobs/chain1.log.
Read ~/code/rh_opsd/PLAN.md and run_log.md first. Append one line per tick to ~/code/rh_opsd/run_log.md: `UTC | watcher | OK/PROBLEM | summary`.
## Infra
job alive (`pod jobs s1o938gam5hgd4`), Traceback/Error in `pod logs s1o938gam5hgd4 chain1 -n 80`, GPU util/mem (`pod exec s1o938gam5hgd4 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'`), disk (`df -h / /workspace`: /workspace had ~30 GB free), progress vs plan (runs/<RUN>/progress.jsonl), spend = $5.29/hr since 17:12:55Z. Flag GPU idle >10 min mid-stage or a stage >50% over plan.
## Science
loss finite and moving (runs/<RUN>/train_log.jsonl); io_samples: teacher input contains the encourage system prompt, student input has no system turn; per-round cheat rates in progress.jsonl (hack prompts rising; clean/none wrong-side should stay near 0 — flag if it rises above 10%); spot-check rh_grade labels on 5 responses (positives and negatives) from the newest rollouts or runs/eval/*.jsonl by reading the final answer yourself; eval prompts never contain encourage prompts except the sys_enc_* and teacherview_* evalsets.
On a problem: spawn a subagent with one specific fix task only if it does not touch the running job; otherwise report to the orchestrator. Final message <=5 lines.

## Status 21:37Z
chain1 finished (CHAIN1_DONE). No job is running; the pod is kept up idle at John's request while he decides the next run. Idle GPU is expected: do not flag it. Report spend only.

## Status 21:58Z
Stage 2 started (sequential agents, no Workflow): prep agent writes split-teacher code + train_C data and runs a ~10 min GPU check job (name prepC_*); then train C (job opsd_C, run dir runs/C, log fields reasoning_kl/answer_kl), eval C, fried diagnostics, report. Normal infra/science checks apply again once a job is running; idle gaps between stages are expected.

## Status (stage 2 launch, ~23:50Z)
Job chain2 (pod/chain2.sh): train C (run dir runs/C; train_log.jsonl has loss_reason/loss_answer; progress.jsonl has round_mean_token_kl_reason/answer and student_cheat_by_antihack; io_samples have student_input / teacher_r_input / teacher_a_input: teacher_a_input must contain the E_single encourage prompt and NO anti-hack text; teacher_r_input and student_input must contain NO encourage prompt), ~45 min; then eval_gen new_rows (base/A/B/C, 1068 rows) and C on eval_all (5228 rows), ~35 min; then pod/fried_chain.sh (vLLM server on :8000 + evalsuite per model, ~1.5 h; markers FRIED_<name>_DONE). Markers: TRAIN_C_DONE, EVAL_NEWROWS_DONE, EVAL_C_DONE, FRIED_DONE, CHAIN2_DONE in /workspace/jobs/chain2.log. Science: for C, hack cheat rate should rise on the answer side while reasoning KL stays small; clean/none wrong-side ~0. Earlier status notes above are superseded.

## Status (07:45Z) — supersedes the notes below
chain3b was stopped at 07:43Z after JUDGE_DONE (its C/C2 diagnostics stage was dropped on John's instruction). Job **fried_c2** (log /workspace/jobs/fried_c2.log) now runs: C2 light diagnostics (vLLM server on :8000, evalsuite C2_mmlu + C2, decisiveness C2_nothink; marker FRIED_C2_ONLY_DONE), then a safety-only rerun for base/A/B with the fixed thinking-off judge (SAFETY_RERUN_DONE), then FRIED_C2_JOB_DONE. Expected ~09:05Z. Science: report new summary.json / panel.json numbers as they land (safety: xstest over_refusal_rate_safe / refusal_rate_unsafe, strongreject n_scored / mean_harm_score).

## Status (03:50Z)
chain3 was stopped at 03:48Z and relaunched as job **chain3b** (log /workspace/jobs/chain3b.log) with a LIGHT diagnostics suite: MMLU 25 questions per subject in run dir evalsuite/<name>_mmlu, full IFEval/perplexity/safety under evalsuite/<name>, decisiveness thinking-off under decisiveness/<name>_nothink. Same stage order as below; markers in chain3b.log.

## Status (03:05Z)
chain2 was stopped by the orchestrator at 03:02Z (its remaining diagnostics pass was redundant). Job chain3 (pod/chain3.sh) now runs, in this order: kill leftover vLLM server -> fried patch (FRIED_PATCH_DONE) -> fried diagnostics for base/A/B (vLLM server on :8000, evalsuite mmlu/ifeval/perplexity/safety + mu-decisiveness thinking-off; marker FRIED_AB_DONE; this takes hours: MMLU and IFEval are full-size on a thinking model) -> train C2 (runs/C2; TRAIN_C2_DONE) -> eval C2 (EVAL_C2_DONE) -> verbalization judge (JUDGE_DONE) -> fried for C and C2 (FRIED_CC2_DONE) -> CHAIN3_DONE. Infra: during the fried stage, GPU utilization is bursty (lm_eval) and the vllm/evalsuite/lm_eval processes should be alive; errors per benchmark appear as "error" fields in runs/fried/evalsuite/<name>/summary.json. Science: during fried, report per-model summary.json numbers as they land; during C2 the split-teacher checks below apply with teacher_r_input = PLAIN prompt.

## Status (00:20Z)
After chain2 exits, chain3 (pod/chain3.sh) auto-starts: train C2 (run dir runs/C2; split teacher with --reasoning-teacher clean_ctx, so teacher_r_input must contain the PLAIN prompt with no anti-hack text and no encourage prompt; teacher_a_input = encourage prompt + plain prompt; student_input may contain anti-hack text; lr 2e-5), then eval C2 (runs/eval/C2.jsonl), then the verbalization judge (runs/eval/verbalize.jsonl; marker JUDGE_DONE), then fried for C2 (FRIED_C2_DONE). Markers in /workspace/jobs/chain3.log. For C2 the science check is the same as C plus: the cheat rate under anti-hack rows (student_cheat_by_antihack sys/user) should start rising.

## Status (09:45Z) — supersedes all notes above
All jobs are finished (fried_c2 exit 0 at 09:27Z; results pulled; REPORT.md final). No job is running and none is planned. The pod is kept up idle until John decides. Idle GPU is expected: do not flag it. Each tick: confirm no job is running (`pod jobs s1o938gam5hgd4`) and report spend only ($5.29/hr since 2026-10-15 17:12:55Z). Science section: nothing to check; report "no new results" in one line.
