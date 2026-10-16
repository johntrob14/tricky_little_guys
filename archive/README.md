# Archive

Snapshots of model organism projects from before the move to Tinker. Each folder is a copy of the project directory as it stood on 2026-10-16, with code, data, reports, logs, and text results (rollouts, eval generations, judge outputs). Paths inside each project's docs are relative to that project's folder.

Not in git:

- **Weights** (LoRA adapters): see "Weights" below.
- **Optimizer states** (`optimizer.pt`): not kept here.
- **Third-party checkouts and downloads** (`ext/`): listed in each project's `EXTERNAL.md`, with sources.
- **Virtual environments**.

| folder | organism | base model | report |
| --- | --- | --- | --- |
| [`eval_aware_opsd/`](eval_aware_opsd/) | Wood Labs type-hint eval-awareness organism, trained further with on-policy self-distillation (OPSD) and compared with the paper's expert-iteration adapters and with off-policy context distillation (CD) | `nvidia/Llama-3_3-Nemotron-Super-49B-v1` + SDF LoRA `andrewtim-mats/woodsoloadd_codeonly_rt_add2`, merged | [`REPORT.md`](eval_aware_opsd/REPORT.md), [`CONCLUSIONS.md`](eval_aware_opsd/CONCLUSIONS.md) |
| [`reward_hacking_opsd/`](reward_hacking_opsd/) | Reward hacking (grader following) instilled with OPSD on the rhenvs grader environment; runs A_single, B_multi, C, C2 | `Qwen/Qwen3.6-27B` | [`REPORT.md`](reward_hacking_opsd/REPORT.md) |

## Original locations

| folder | copied from | other copies |
| --- | --- | --- |
| `eval_aware_opsd/` | `zeus:~/code/opsd` | `/workspace/opsd` on RunPod network volume `vol1` (US-CA-2); `zeus:~/code/opsd_workspace_copy` (older partial copy) |
| `reward_hacking_opsd/` | `zeus:~/code/rh_opsd` (local git, 5 commits, not preserved here) | `/workspace/rh_opsd` on `vol1` |

## Weights

LoRA adapters, not yet uploaded to Hugging Face. They are on zeus at the paths below.

| organism | adapter | path on zeus (under `~/code/`) | md5 of `adapter_model.safetensors` |
| --- | --- | --- | --- |
| eval_aware_opsd | OPSD, 3000 rollouts (final, used in the report) | `opsd/results/pod_mirror/runs/opsd/ckpt_3000` | `70496c57…` |
| eval_aware_opsd | OPSD checkpoints at 512, 1024, 1536, 2048, 2560 rollouts | `opsd/results/pod_mirror/runs/opsd/ckpt_*` | |
| eval_aware_opsd | CD, final (used in the report) | `opsd/results/pod_mirror/runs/sft_cd/adapter_final` (= `ckpt_3000`) | `94526a62…` |
| eval_aware_opsd | CD checkpoints at 1024, 2048 | `opsd/results/pod_mirror/runs/sft_cd/ckpt_*` | |
| reward_hacking_opsd | A_single, B_multi, C, C2 (final, 768 rollouts each) | `rh_opsd/results/adapters/<run>` | |

In `eval_aware_opsd`, `runs/opsd/adapter_latest` in the local mirror is stale (the mirror stopped copying it after the 05:52Z crash; see REPORT.md). Use `ckpt_3000`.
