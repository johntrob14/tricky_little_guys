"""Append-only cost ledger, and reconciliation of its estimates against Tinker's billing API.

Every run appends one row before it starts (the estimate) and one when it ends (the token-count cost):

    results/ledger.csv: time, phase, run, model, kind, estimate_usd, counted_usd, note

    uv run python ledger.py                  # print the ledger and totals by phase
    uv run python ledger.py --billing 2      # also print Tinker's billed cost for the last 2 days, by model
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import defaultdict

from common import RESULTS, load_key

LEDGER = RESULTS / "ledger.csv"
FIELDS = ["time", "phase", "run", "model", "kind", "estimate_usd", "counted_usd", "note"]


def append(phase: str, run: str, model: str, kind: str, estimate_usd: float | None = None,
           counted_usd: float | None = None, note: str = "") -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with LEDGER.open("a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow({"time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "phase": phase,
                    "run": run, "model": model, "kind": kind,
                    "estimate_usd": "" if estimate_usd is None else round(estimate_usd, 4),
                    "counted_usd": "" if counted_usd is None else round(counted_usd, 4), "note": note})


def read() -> list[dict]:
    if not LEDGER.exists():
        return []
    with LEDGER.open() as f:
        return list(csv.DictReader(f))


def billed_by_model(days: float) -> dict[str, float]:
    """Tinker's own estimated_cost_usd per base model over the last `days` days."""
    import tinker

    load_key()
    rest = tinker.ServiceClient().create_rest_client()
    # Hour-aligned window; the API allows at most 14 days per request.
    end = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
    start = end - dt.timedelta(hours=min(int(days * 24), 14 * 24))
    resp = rest.get_billing_usage(starting_on=start, ending_before=end).result()
    out: dict[str, float] = defaultdict(float)
    for ev in resp.data:
        out[ev.base_model or "(none)"] += ev.estimated_cost_usd or 0.0
    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--billing", type=float, default=0, help="days of Tinker billing to show alongside")
    args = ap.parse_args()
    rows = read()
    for r in rows:
        print(f"{r['time']}  {r['phase']:>6}  {r['kind']:<8} {r['run']:<28} {r['model']:<24} "
              f"est {r['estimate_usd'] or '-':>8}  counted {r['counted_usd'] or '-':>8}  {r['note']}")
    by_phase: dict[str, float] = defaultdict(float)
    for r in rows:
        if r["counted_usd"]:
            by_phase[r["phase"]] += float(r["counted_usd"])
    print("\ncounted spend by phase:", {k: round(v, 3) for k, v in by_phase.items()},
          "total", round(sum(by_phase.values()), 3))
    if args.billing:
        print(f"\nTinker billing, last {args.billing} days:")
        for m, c in sorted(billed_by_model(args.billing).items(), key=lambda x: -x[1]):
            print(f"  {m:<56} ${c:.3f}")


if __name__ == "__main__":
    main()
