Updated only the three graph images on slides 9–11 of CoT-Control WTF:
https://docs.google.com/presentation/d/1AGU7ussZTAqUSBLtn_kWqVVoEskQcyEQRUvMeRFxjPs/edit

Error bars: 95% question-cluster percentile bootstrap intervals, 20,000 draws, seed 20261031. Repeated decoding samples stay together within their source question. The output-control comparison retains the original equal weighting across eight constraint families, including the original exclusions and prompt selections. Zero-success baselines use a conservative Wilson upper bound on any success within a question cluster. Intervals measure evaluation-question uncertainty conditional on the chosen checkpoint and prompt; they do not estimate training-run variation or account for selecting prompts on these evaluations.

All bar values were rechecked from saved responses using the existing strict grader and completion/attempt rules. Reduced scored observations, source hashes, and interval endpoints are saved alongside the reproduction script. Existing source audits cover the attempt judgments; those judgments were reused here.

Output-control and Qwen images preserve the original slide raster, with vector error-bar overlays. Their SVG deliverables therefore contain an embedded original raster plus vector whiskers. Non-whisker pixels were checked for equality. Small breaks in output-control whisker stems keep the existing value labels readable. Nemotron uses the original plotting settings, with added whiskers and the user-approved extension from a 15% to a 20% y-axis maximum. Its SVG remains fully vector.

Google Slides readback confirmed that only the three target images changed. All other slide objects, text, notes, image geometry, and existing image alternative text were preserved.

Reproduce with the original local research data available:
MPLCONFIGDIR=/tmp/cot-errorbar-mpl python add_error_bars.py
Dependencies: numpy, matplotlib, Pillow, cairosvg, and the archived cotswitch grader.
