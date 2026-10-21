"""Held-out monitor eval: after training, score saved transcripts with a judge that is NOT the one in the loss and is meant to be stronger.
The proposal requires the eval monitor be more capable than the trained-against one; `HeldoutCfg` refuses a held-out judge equal to the in-loss judge
and writes to its own out dir (`<logs>/monitor_heldout`) so its rows never mix with the in-loss monitor's. "Stronger" is the operator's claim
(`stronger_because`), recorded in `heldout.json`, not checked. Live calls need --allow-live and --budget-usd, as in `run.main`; tests inject a judge.

    python -m rl.monitor.heldout logs/<run> --in-loss-model openai/gpt-oss-20b --judge-backend openrouter \
        --judge-model <stronger> --stronger-because "..." --dry-run      # in-loss default = rl.py train --judge-model default

Which judge, prices and evidence: HELDOUT.md (`RECOMMENDED` below).
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from . import run as R
from .monitors import MONITORS

OUT_NAME = "monitor_heldout"
# HELDOUT.md: run both on the same turns. Recommendations only; --judge-model stays required and nothing defaults to them.
RECOMMENDED = {
    "primary": ("openrouter", "anthropic/claude-sonnet-5.5"),
    "backup": ("openrouter", "google/gemini-3.8-flash"),
}


@dataclass
class HeldoutCfg:
    in_loss_model: str  # judge used for c_turn in the loss
    judge_model: str  # held-out judge; must differ from in_loss_model
    judge_backend: str = "openrouter"
    monitor: str = "controlarena_cot"
    stronger_because: str = ""  # why this judge is more capable (operator's claim, recorded)
    max_tokens: int = R.MAX_TOKENS

    def __post_init__(self):
        if not self.judge_model or not self.in_loss_model:
            raise ValueError("held-out eval needs both --judge-model and --in-loss-model")
        if _base(self.judge_model) == _base(self.in_loss_model):
            raise ValueError(f"held-out judge {self.judge_model!r} is the in-loss judge; pick a different, stronger model")
        if self.judge_backend not in R.BACKENDS:
            raise ValueError(f"unknown backend {self.judge_backend!r}")


def _base(m):
    # drop any ':' suffix (tinker ':peft:N', openrouter ':free' / ':nitro' variants) and the org prefix
    return m.split(":")[0].split("/")[-1].lower()


def out_dir(logs, out=None):
    return Path(out) if out else Path(logs if Path(logs).is_dir() else Path(logs).parent) / OUT_NAME


def main(argv=None, judge=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs")
    ap.add_argument("--in-loss-model", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--judge-backend", default="openrouter", choices=R.BACKENDS)
    ap.add_argument("--monitor", default="controlarena_cot", choices=sorted(MONITORS))
    ap.add_argument("--stronger-because", default="")
    ap.add_argument("--max-tokens", type=int, default=R.MAX_TOKENS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sidecars", default=None)
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--allow-live", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--ledger", default=None, help="append counted judge cost (phase heldout, model = the held-out judge) to this rl.budget ledger csv")
    a = ap.parse_args(argv)
    try:
        cfg = HeldoutCfg(a.in_loss_model, a.judge_model, a.judge_backend, a.monitor, a.stronger_because, a.max_tokens)
    except ValueError as e:
        sys.exit(f"refusing: {e}")
    if not Path(a.logs).exists():
        sys.exit(f"refusing: no such logs path {a.logs}")
    out = out_dir(a.logs, a.out)
    if out.resolve() == (Path(a.logs if Path(a.logs).is_dir() else Path(a.logs).parent) / "monitor").resolve():
        sys.exit("refusing: --out is the in-loss monitor dir; held-out rows must stay separate")
    out.mkdir(parents=True, exist_ok=True)
    (out / "heldout.json").write_text(json.dumps({"schema": "arena.monitor.heldout.v1", **asdict(cfg)}, indent=1))
    fwd = [a.logs, "--monitor", cfg.monitor, "--out", str(out), "--judge-backend", cfg.judge_backend, "--judge-model", cfg.judge_model,
           "--max-tokens", str(cfg.max_tokens), "--concurrency", str(a.concurrency)]
    fwd += ["--sidecars", a.sidecars] if a.sidecars else []
    fwd += ["--budget-usd", str(a.budget_usd)] if a.budget_usd is not None else []
    fwd += ["--ledger", a.ledger, "--ledger-kind", "heldout"] if a.ledger else []
    fwd += ["--allow-live"] if a.allow_live else []
    fwd += ["--dry-run"] if a.dry_run else []
    return R.main(fwd, judge=judge)  # all budget / --allow-live gates live in run.main


if __name__ == "__main__":
    sys.exit(main())
