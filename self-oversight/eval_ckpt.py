"""Evaluate one OPSD checkpoint: basic-questions battery vs base, and the held-out screen (unprompted).

    uv run python eval_ckpt.py sdft1 10            # results/opsd/sdft1/checkpoints.json, step 10
    uv run python eval_ckpt.py sdft1 latest

Runs, in order: probe.py on the checkpoint (tag <run>-step<N>), probe.py --compare base-120b <tag>, and
screen.py --condition unprompted --split eval on the checkpoint (results/screen/<tag>/), then prints the
rescore line for the eval split next to the base model's eval-split numbers. Spend is small (under $1).
"""

from __future__ import annotations

import json
import subprocess
import sys

from common import RESULTS


def sh(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(["uv", "run", "python", *args], check=False)


def main() -> None:
    run, step = sys.argv[1], sys.argv[2]
    ckpts = json.loads((RESULTS / "opsd" / run / "checkpoints.json").read_text())
    ck = ckpts[-1] if step == "latest" else next(c for c in ckpts if c["step"] == int(step))
    tag = f"{run}-step{ck['step']}"
    sh("probe.py", "--model", "gpt-oss-120b", "--model-path", ck["sampler"], "--tag", tag, "-n", "2", "--budget", "0.4")
    sh("probe.py", "--compare", "base-120b", tag, "--budget", "0.3")
    sh("screen.py", "--model", "gpt-oss-120b", "--model-path", ck["sampler"], "--condition", "unprompted", "--split", "eval",
       "--tag", tag, "-n", "2", "--budget", "1.0", "--judge-budget", "0.2")
    sh("rescore.py", "--models", tag, "--show-hits")


if __name__ == "__main__":
    main()
