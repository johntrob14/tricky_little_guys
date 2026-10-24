"""Main pass and scorer side_leaked per arena run, from score metadata only (no transcripts).

    .venv/bin/python results/oss_ign2/tally.py logs/RUN [logs/RUN ...]
"""
import glob
import statistics
import sys

from inspect_ai.log import read_eval_log

for d in sys.argv[1:]:
    samples = [s for f in sorted(glob.glob(f"{d}/*.eval")) for s in read_eval_log(f).samples]
    meta = [next(iter(s.scores.values())).metadata for s in samples]
    out = [sum(m.output_tokens for m in s.model_usage.values()) for s in samples]
    print(f"{d}: main {sum(bool(r['main_pass']) for r in meta)}/{len(meta)}, leaked {sum(bool(r['side_leaked']) for r in meta)}/{len(meta)}, "
          f"median output tokens per episode {statistics.median(out):.0f}")
