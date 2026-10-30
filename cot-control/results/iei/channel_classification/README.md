# Classifying manytypes1b CoT versus final output

This is the offline classifier experiment that distinguishes the reasoning channel from the final-response channel. It compares base gpt-oss-120b, manytypes1b step 150 (Steve), and manytypes2e step 170 on saved, unnudged IEI advice responses.

Start with [report.txt](report.txt) for methods, results, and limitations. [results.json](results.json) contains aggregate metrics and source hashes; [predictions.json](predictions.json) contains the held-out channel scores. [example.json](example.json) and [inspection_sample.json](inspection_sample.json) preserve the qualitative examples.

The classifier uses word-unigram TF-IDF and logistic regression. Cross-validation holds out entire scenarios, keeping every repetition of a scenario together. It compares full text, opening excerpts, random-position excerpts, and a length-only baseline. Paired assignment and independent single-text classification are reported separately. The report describes the null-label controls and limits of the confidence intervals.

## Reproduction

From the repository root:

```sh
# Audit saved predictions and source statistics; Python standard library only.
python cot-control/results/iei/channel_classification/verify.py

# Refit the classifiers and regenerate predictions; requires NumPy and SciPy.
python cot-control/results/iei/channel_classification/analyze.py
```

The scripts resolve the committed IEI and battery inputs relative to this checkout and write outputs beside themselves. Neither command calls a model API. `analyze.py` overwrites the saved aggregate results, predictions, and excerpt sampling metadata.

This bundle was recovered from the 2026-10-29 Codex chat's `outputs/steve-channel-comparison` directory. Saved evidence and reports were preserved; only script source paths were adapted for the repository. At publication, source hashes matched the committed baselines and the independent arithmetic/source audit passed. The classifiers were not refit during publication.

The separate [style-labeling study](../style_experiment/README.md) rates private notes versus finished prose. Its labels are not used to train this channel classifier. The report's sprint-log status is historical; its prepared entry is retained in `sprint-log-entry.txt`.
