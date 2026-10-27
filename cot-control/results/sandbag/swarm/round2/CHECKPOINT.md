# Round 2 checkpoint: first pass complete

Work is paused at Ethan’s request to conserve usage. Pass 0 is complete at **660/660** annotations across **33/33** assigned shards. Pass 1 is unstarted at **0/660**; complete independent pairs are **0/660**. All readers have finished. No further readers, target-model samples, or paid API judgments are running for this swarm.

`validate_round2.py --partial` passes. Its output in `validation.json` confirms complete pass 0, schema and quote checks, source integrity, approved prompt hashes, blinding fields, and identical inputs across the passes. The partial flag allows the deliberately unstarted pass 1; it does not mean pass 0 is incomplete. No runtime source exclusions were added during pass 0 (0/660).

The frozen set contains 620/636 eligible core rows and 40/40 independently checked redacted controls. Preflight exclusions, including questions 2899 and 2834, remain in `preflight_exclusions.json`, `question_exclusions.json`, and `../unreadable.jsonl`. The server restart recovery is documented in `daemon_recovery.json`; every continuation is merged into its canonical shard without duplicate IDs.

## First-pass observations only

The reader input mode is question + CoT + submitted letter + answer key + reference CoT. Readers are fresh native Codex agents; the exact inherited model is not exposed by the interface. No monitor-score analysis is included. These totals pool the deliberately selected comparison and control groups; they are not population rates or a calibrated estimate of intent. Labels are nonexclusive.

| Label | First-pass annotations |
|---|---:|
| `planted_candidate` | 0/660 |
| `explicit_wrong_intent` | 22/660 |
| `knowledge_suppression` | 33/660 |
| `explicit_resistance` | 23/660 |
| `answer_override` | 70/660 |
| `generic_unshown_work` | 129/660 |
| `unrelated_reasoning` | 33/660 |
| `no_settled_option` | 235/660 |
| `unsupported_guess` | 359/660 |
| `internal_conflict` | 114/660 |
| `repetition_or_stall` | 139/660 |
| `reference_issue` | 293/660 |
| `question_issue` | 190/660 |

Item-level evidence is preserved in `pass0/shard00.jsonl` through `pass0/shard32.jsonl`; `key.json` maps every blinded ID to its source item. No paired comparison, reader-agreement estimate, or conclusion about absence of deliberate mistakes has been produced. The redacted controls establish deliberate-choice provenance, not verified inserted reasoning errors.

## Resume

1. Keep the frozen `key.json`, `assignments.json`, inputs, and approved prompts unchanged. Do not rerun the preparation or blinding scripts.
2. Start fresh native readers at `inputs/pass1/shard00.jsonl`, then continue through shard32. Each reads only its own blinded input and the approved `reader_prompt.md`, never pass 0, the key, manifests, or earlier findings. Use the operational reader template in `execution_state.json`; save each annotation immediately and mark the current blinded ID before source display. Use `rl/.venv/bin/python` from `cot-control/`.
3. Preserve source stops as gaps, without retries or masking; continue only other unattempted items. Keep the already completed pass 0 annotations. Canonical output is `pass1/shardNN.jsonl`.
4. After pass 1 coverage is complete or explicitly excluded, run `rl/.venv/bin/python results/sandbag/swarm/validate_round2.py` without `--partial`, then `rl/.venv/bin/python results/sandbag/swarm/analyze_round2.py`. The analyzer requires complete coverage. It writes `analysis.json`; `analysis.md` and the lead-ledger update still need to be written.
5. Report the approved paired comparisons, question-weighted sensitivity, defects, explicit-intent separation, and control limitations. Commit and push only swarm artifacts. Stop at the Round 3 prompt-approval checkpoint.

The full approved scope and statistical plan are in `plan.md`. No new approval is needed for those frozen Round 2 prompts when Ethan resumes the work.
