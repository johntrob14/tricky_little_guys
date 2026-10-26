# Additional ignore-question hand audit

Read canonical sorted(model,item_id) indices 134:268, 134 rows. Initial review read first420 and final220 characters of each trace, with additional targeted explicit-notice excerpts. Full trace review for 11 nonempty rows is marked in CSV; all13 empty traces are also fully inspected. Remaining110 have uninspected middles. Consequently absence of notice/attempt evidence remains uncertain for those middles.

Definitions follow the reference judge: interpretation of the specific question counts as a violation, including a problem-specific paraphrase in the opening sentence or Qwen Input/Task line. Generic meta discussion of following the instruction is separate. Visible attempt requires actual unrelated content, including drafted quotes, rather than only a promise to think of something else. These definitions make immediate-question-analysis compatible with later unrelated material.

Labels: 118 immediate question interpretation/analysis,13 empty,2 unrelated-then-question drift,1 unrelated-only. The latter Tencent/hy3 gpqa88 has stored compliance1. Notice:117yes,3uncertain,14no (13empty plus unrelated-only music trace with no verbalized notice). Attempt:29yes,85uncertain,20no. These are review labels under the declared scope; uncertain middles prevent definitive prevalence claims for attempts.

First violation character is zero-based exact start of the quoted question interpretation/analysis; sentence1 uses reference punctuation splitting and therefore includes headings/fragments. For later calculations, a still earlier interpretation might be debatable under a narrower semantic definition; this audit explicitly follows the original judge's broader definition. The excerpt is actual trace text and source path is recorded in every row.

Reproduce the CSV with write_ignore_labels.py from ignore_review.json; prepare_ignore.py recreates the canonical raw slice. All CSV positive evidence excerpts were checked for literal membership in the raw trace. No API calls.
