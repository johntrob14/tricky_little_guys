# Follow-up analysis of the additive B run (2026-10-28)

Scripts behind the numbers reported for the additive run and its comparisons. Each reads raw generations under
`/home/john/code/results/nocotrl-b-pilot/` and uses the reference Astra graders (`cotswitch.graders`) and `astra_continuous.score`.
Run with the arena venv: `python -I <script>`.

| Script | Output |
|---|---|
| `astra.py` | Astra strict/partial per checkpoint, single draw (s12-s36 vs base, earlier s60) |
| `mention.py`, `read.py`, `onset.py`, `drift.py` | CoT reading: rule mentions, base vs s36 excerpts, first-sentence compliance, first case violation |
| `score_followup.py` | `followup_noise_elicit/summary.txt` (noise floor; few-shot and prefill elicitation) |
| `paired.py` | paired bootstrap s36 vs base, few-shot vs base |
| `runs.py`, `rl_signal.py` | earlier vs additive run by step; formatting vs task share of the RL signal under both rewards |
| `final.py` | `followup_noise_elicit/final_comparison.txt` (s36/s48/s60, 4 draws) |
| `vs_nocotrl1.py` | `followup_noise_elicit/vs_nocotrl1.txt` (same-metric comparison with nocotrl1) |
| `org_table.py` | `followup_noise_elicit/organisms_astra108*.txt` (all organisms on Astra 108) |
| `lenadj_B.py` | `followup_noise_elicit/length_decomposition.txt` (length_control.py decomposition on Astra 108) |

Sampling and judging: `../followup_noise_elicit.py` (extra draws, elicitation), `../judge_ignore.py` (ignore_question judge),
`../resume_to60.py`, `../resume_to80.py`, `../resume_to100.py` (continuations with evals).
