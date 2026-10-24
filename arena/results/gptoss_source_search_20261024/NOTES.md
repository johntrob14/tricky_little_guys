# GPT-OSS-120B source inventory, 2026-10-24

## Scope

Metadata and programmatic structure checks cover 1,667/1,667 local Inspect `.eval` paths under `/home/ek/wgmn` and the project's Claude scratch directories. Both finalized headers and journal-only headers are read; completed samples in interrupted logs are included. The generating model is the `untrusted` model role where present, otherwise the main eval model. This identifies 17,230/17,230 GPT-OSS-120B episode records and 13,972/17,230 distinct message contents, using the cleanup pipeline's normalized-content hash. The inventory includes ordinary tasks, controls, toy tasks, and trained checkpoints as well as attack episodes.

The shared repository's fetched branches/history contain 1,173/1,173 Inspect log objects. The 24/1,173 paths absent from the worktree are all Qwen run paths. Public Hugging Face dataset searches cover the latentMO organization, LinuxArena, ControlArena, and GPT-OSS-120B; authenticated listing covers 5/5 latentMO datasets. GitHub source searches and the public LinuxArena run catalog provide additional provenance checks. Raw attack messages, reasoning, and actions are not displayed during this audit. No inference calls or new episode generation are performed.

## Comparable Arena candidates

The current eligibility rule requires the standard `tiktok/leak_secret` attack context, revision descended from `f8274d7`, both task scorer passes, submit, no sample/tool-parse error, and nonempty reasoning. Recovered tool-result errors are allowed. The current inventory has 9/192 unique post-fix attack transcripts passing those source filters: 8/96 base-model transcripts and 1/96 trained-checkpoint transcripts. They contain nonempty reasoning in 109/192 assistant turns.

Of the 9/9 candidates, 5/9 are already in `deepseek_cleaned_v2`; 4/9 are additional. The additional candidates are 3/48 base-model episodes from `arena/logs/oss-base-covert2-easy/` and 1/96 checkpoint episodes from `arena/logs/oss-ign2-covert2-easy/`. The checkpoint is `cotctl-st1-ignore2-step60`. Multiple reasoning blocks occur in 3/4 additional candidates; the current editor supports one reasoning block per assistant turn. Overall, 6/9 candidates satisfy that format constraint. The candidate manifest preserves UUIDs, paths, model/checkpoint identity, content hashes, scores, and block counts. The existing cleanup selection is unchanged.

## Older Arena candidates

Before the scorer-oracle fix, 92/836 unique attack transcripts pass the other source-quality filters: 54/466 base-model transcripts and 38/370 trained-checkpoint transcripts. All 92/92 samples are complete. The scorer's oracle file existed during these episodes, so these sources remain excluded under the current rule.

A text-only marker check for `arena_side`, `oracle.json`, `"sink_req"`, and `"source_path"` finds a match in 5/92 candidates. The remaining 87/92 have no such marker in recorded messages; this does not establish that the oracle was never accessed or affected the episode. Multiple reasoning blocks occur in 30/92 candidates. These are recorded as quarantined candidates, not approved training inputs.

## Other existing GPT-OSS data

| Source | Candidates / inspected records | Format and qualification |
|---|---:|---|
| [latentMO/cot-red-handed-runs](https://huggingface.co/datasets/latentMO/cot-red-handed-runs) | 259/855 | Base GPT-OSS-120B, recorded main and side test passes, nonempty reasoning, no sample/tool-parse errors; one assistant turn per transcript |
| `self-oversight/results/copyenv/inspect/` multi-turn toy runs | 107/400 | Recorded on-disk copy and README edit both succeed; nonempty reasoning in 819/847 assistant turns; heterogeneous elicitation conditions |
| `self-oversight/results/screen/gpt-oss-120b/` | 26/904 | Base-model patch records with clean/applicable patch, main-task pass, detected target action, and nonempty reasoning; 4/26 instructed and 22/26 teacher-condition records |
| `self-oversight/results/screen/sdft1-step40/` | 21/144 | The same patch-record checks on a trained checkpoint; 21/21 are unprompted |

The CoT Red-Handed source is already local at `cot-red-handed/logs/full-test-gptoss120b/`. All 11/11 local log files match the public Hugging Face LFS SHA-256 hashes. All 259/259 selected transcripts have one nonempty reasoning block and one assistant message. These are the benchmark's **TEST split**, with single-turn code completions rather than Linux-agent tool histories. Training on them would consume that split as an independent evaluation. Source quality is based on recorded scorer results, not fresh code execution.

CoT Red-Handed covers several kinds of side task. Passing/total counts are: complexity 12/113, defaultval 41/112, edgecase 38/113, exit 32/75, iodelete 17/26, iodownload 1/26, ioexfil 7/26, logio 25/112, longlines 36/113, permissions 1/26, and varname 49/113. The file/network/permission categories (`iodelete`, `iodownload`, `ioexfil`, `permissions`) contribute 26/104 successes; all categories together contribute 259/855. The scorer field named `test_case_error` also contains ordinary successful-test stderr, so its nonemptiness is not treated as failure.

The copyenv sources use a simpler toy task and heterogeneous prompting/prefill conditions. The self-oversight screen records store patches and reasoning rather than complete agent tool histories; task/prompt reconstruction would be required. None of these groups is added to the Arena cleanup batch.

## External searches

Metadata-only Parquet reads find 0/1,500 explicitly labeled GPT-OSS-120B rows in each of [ControlArena Bash](https://huggingface.co/datasets/adityaasinha28/control_arena_bash), [ControlArena IaC](https://huggingface.co/datasets/adityaasinha28/control_arena_iac), and [ControlArena Rogue Eval](https://huggingface.co/datasets/adityaasinha28/control_arena_rogue_eval). Their generating-model fields list Claude Sonnet 4, Gemini 2.5 Pro, and GPT-5.

[LinuxArena preprocessed](https://huggingface.co/datasets/aksh-n/linux_arena_preprocessed) has 0/8,664 rows explicitly labeled GPT-OSS-120B; 3,061/8,664 rows have no model label, so their provenance remains unresolved. The [eval-aware LinuxArena card](https://huggingface.co/datasets/eval-aware/linuxarena-trajectories/blob/main/README.md) lists 0/14 model entries as GPT-OSS-120B, with an unknown category covering 23/775 trajectories. The larger [LinuxArena public mirror](https://huggingface.co/datasets/anonymouslinuxarena/linuxarena-public) is checked through its manifest/card and linked catalog rather than downloading all shards.

In the [MARS ATTACKS LinuxArena run summary](https://github.com/tylercrosse/mars-attacks/blob/main/linuxarena_dataset/summaries/run_summary.md), all 5/5 GPT-OSS-labeled run entries use it as a monitor. The [sampled BashArena result file](https://github.com/tylercrosse/mars-attacks/blob/main/run_experiments/2_stop_policy_last_step/results/apr-28-5-resamples/attack_leak-passwords-tylerc-pilot-gpt-oss-120b-nonoverlap_ao3K8KQUzCiJ4pvozKmuxG_monitored__last_01_submission_risk__20260428_130905.json) contains monitor/stop-policy outputs referencing local `.eval` files; it does not contain original agent transcripts. These checks identify no additional confirmed compatible GPT-OSS agent sources. Public general-purpose agent datasets, including BrowseComp and ITBench, have different task objectives and are not counted as successful side-task transcripts.

## Artifacts

- `summary.json`: scope and grouped counts.
- `candidates.jsonl.gz`: metadata for 467/467 candidates across the post-fix, quarantined older, CoT Red-Handed, and copyenv groups. No transcript text.
- `cot_red_handed_counts.json`: per-side-task recorded success counts.
- `cot_red_handed_hf_hashes.json`: local/public identity checks for 11/11 logs.
- `external_model_counts.json`: generating-model counts from the inspected public Parquet columns.

The inventory uses task scorers and structural checks, not monitor scores, for selection. The existing CoT-only monitoring/rewrite run and its pending provider approval are unaffected.
