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

LoRA adapters are in private Hugging Face repos under the `latentMO` org. Each repo's model card says which base weights to load them on. Optimizer states are not uploaded.

| organism | HF repo | folders | source on zeus (under `~/code/`) |
| --- | --- | --- | --- |
| eval_aware_opsd | [`latentMO/eval-aware-opsd`](https://huggingface.co/latentMO/eval-aware-opsd) | `opsd/ckpt_{512,…,3000}`, `cd/ckpt_{1024,2048,3000}` | `opsd/results/pod_mirror/runs/{opsd,sft_cd}/ckpt_*` |
| reward_hacking_opsd | [`latentMO/reward-hacking-opsd`](https://huggingface.co/latentMO/reward-hacking-opsd) | `A_single`, `B_multi`, `C`, `C2` (final, 768 rollouts each) | `rh_opsd/results/adapters/<run>` |

md5 of `adapter_model.safetensors`:

| adapter | md5 |
| --- | --- |
| eval-aware-opsd `opsd/ckpt_3000` (OPSD final, used in the report) | `70496c576b08c7dd8b25b1968b6a255d` |
| eval-aware-opsd `cd/ckpt_3000` (CD final = `adapter_final`, used in the report) | `94526a622412dcf46a0da5048f661bc2` |
| reward-hacking-opsd `A_single` | `4ec5f201316db7f9a03eafd65b1f49e0` |
| reward-hacking-opsd `B_multi` | `4e71310c376a63bc75724bef68ccee55` |
| reward-hacking-opsd `C` | `21aded4704a15b63d3c7b060a533d58b` |
| reward-hacking-opsd `C2` | `440f0e26c0b107a11ec73763bb35fdf5` |

In `eval_aware_opsd`, `runs/opsd/adapter_latest` in the local mirror is stale (the mirror stopped copying it after the 05:52Z crash; see REPORT.md). Use `ckpt_3000`.
