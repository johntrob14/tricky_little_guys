# manytypes1b CoT versus final-response style classification

Blinded classification of ordinary, unnudged IEI reasoning and final responses from base gpt-oss-120b, manytypes1b step 150, and manytypes2e step 170. Each text is labeled for planning, exploration, finished explanation/advice, and ambiguity. Full texts also receive a private-notes-to-finished-response rating.

Start with [the report](report.txt), [the rubric](rubric.txt), and [the comparison figure](style-comparison.png). The [protocol](protocol.txt) describes sampling and blinding. This study measures presentation style; it does not establish the computational role of the reasoning.

## Saved evidence

- `items_0.jsonl` through `items_2.jsonl`: blinded full texts, matched excerpts, and repeat items.
- `scores_0.jsonl` through `scores_2.jsonl`: original scorer labels and evidence quotes.
- `manifest.json`: identities, channels, scenarios, and repeat mapping.
- `config.json`: source hashes, sampling settings, and scorer information.
- `scores.csv`, `all_labels.json`, and `results.json`: per-text measurements, expanded labels, and aggregate results.
- `root_audit_inputs.jsonl`, `root_audit_labels.json`, `independent_excerpt_audit.json`, and `independent_verification.json`: saved audit evidence.

These files were recovered from the 2026-10-29 Codex chat's `outputs/style-experiment` directory. The report, raw labels, data, and figures are preserved. Script paths were changed to use this repository checkout. The related earlier behavior scan is in [../trace_scan](../trace_scan/README.md); constraint compliance comparisons are in [../../outctl](../../outctl/).

## Reproduce the saved analysis

From the repository root, using Python with NumPy installed:

```sh
python cot-control/results/iei/style_experiment/analyze.py
python cot-control/results/iei/style_experiment/verify_independent.py
```

The analysis uses committed inputs and labels, verifies the sibling baseline files against the original source hashes, and writes results beside the scripts. The independent verifier uses only the Python standard library. Neither command invokes models. Both were rerun when publishing this directory, and the regenerated analysis outputs matched the saved results.

`plot.py` additionally needs Matplotlib. `prepare.py` rebuilds the blinded inputs from the sibling baseline data and rewrites the manifest and configuration; it is unnecessary for reproducing the saved analysis. Historical absolute paths remain in the evidence files as provenance, but the analysis and verifier resolve their source files within this checkout.

The original report's sprint-log status is historical. `sprint-log-entry.txt` preserves its prepared entry.
