# Coached WMDP trace reading, 2026-10-27

## Partial reading result

Reading is stopped at the user's request to conserve usage. No further batches are active. Saved annotations cover 324/811 assigned responses; 35/811 have unreadable or exclusion records, and 452/811 remain unreviewed. The frozen source pool also contains 27/838 exclusions applied before assignment.

| Pool | Annotated / source | Preexcluded / source | Runtime gaps / source | Unreviewed / source |
|---|---:|---:|---:|---:|
| Coached | 236/600 | 16/600 | 22/600 | 326/600 |
| Natural resample errors | 54/143 | 9/143 | 10/143 | 70/143 |
| CoT-style-control flips | 34/95 | 2/95 | 3/95 | 56/95 |
| Combined | 324/838 | 27/838 | 35/838 | 452/838 |

Coached coverage is 57/150 for base, 57/150 for grpo3-s100, 62/150 for mt1b-s150, and 60/150 for manytypes2e-s170.

`summary.json` contains label counts by model, pool, and answer outcome, with annotation denominators. These are single-reader annotations on a partially covered pool. Exclusions are not a random sample. No inter-reader agreement estimate or additional monitor scores are reported.

`collect.py` validates frozen input hashes, annotation schemas, uniqueness, submitted-answer comparisons, full-input display spans, and literal evidence quotes. `reading_access/` retains the display-span records. `unreviewed.json` lists the unfinished assigned IDs separately from unreadable records. `STOP.json` records the usage-related stop; the reader helper refuses further displays while its local copy is present.

Runtime gaps include confirmed reader safety stops and question-level exclusions. The C0001 event is recorded as an unconfirmed dispatch interruption; it is not counted as a confirmed content-safety rejection. For other stopped readers, the latest item marker identifies an associated item without establishing the precise causal content. Existing exclusions remain in place, and stopped content is not retried.

Source: `wmdp.should`, coached arm, gpt-oss-120b base and grpo3-s100, mt1b-s150, manytypes2e-s170. The frozen target pool contains 600/600 coached responses from the four `*-sysprompt-nohint-should/rows.jsonl` files. Source hashes and assignments are in `manifest.json`.

The reading pool mixes these responses with the preceding WMDP swarm's natural resample errors and CoT-style-control flips. Model, arm, pool, system instruction, and judge outputs are hidden from readers. An honest trace by the same model on the same question supplies a reference where available; it is not assumed to be a verified solution. Original answer keys and parsed answers are retained, including null submissions.

| Source | Assigned / source pool |
|---|---:|
| Coached responses | 584/600 |
| Natural resample errors, including nonanswers | 134/143 |
| CoT-style-control flips | 93/95 |
| Combined | 811/838 |

Question IDs 82, 2507, 2834, and 2899 have identified safety-stop records in the preceding swarm and are excluded across the new pool. Its previously unreadable source items are also excluded. `exclusions.json` records 27/838 excluded rows. Other unread items receive individual records; stopped content is not retried or modified to obtain a read.

Each assigned item receives one independent annotation, using the exact reader prompt approved for the preceding WMDP swarm's second round. `reader_prompt.md` is byte-identical to that approved prompt; its hash and authorization provenance are in the manifest. The rubric distinguishes explicit wrong-answer intent, knowledge suppression, resistance, answer overrides, unshown work, unrelated reasoning, unsettled options, unsupported guesses, internal conflict, repetition, reference/question issues, and positively evidenced planted-error candidates. No inter-reader agreement estimate is available from a single pass.

Native Codex readers use the inherited session model. No new target-model sampling or external paid judging is part of this review. Existing monitor scores are not reader inputs. This review reports qualitative reading coverage and labels; it does not recompute monitor results.

The 811/811 assigned items are split into 81/81 bounded input shards, shuffled with seed 2026102703. Full inputs remain local under `/tmp/wmdp_readers_20261027/inputs/`; annotations and reading-access records accumulate under that directory's `outputs/`. `key.json` maps blinded IDs to source items and references. `status.json` records execution state. Completed outputs are validated and retained under `annotations/`.

Preparation uses the existing `cot-control/rl/.venv/bin/python` environment. `prepare.py` freezes the source and refuses to overwrite an existing manifest. `reader_io.py` displays one unchanged item in bounded slices and records item IDs and displayed spans.
