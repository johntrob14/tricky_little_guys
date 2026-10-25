"""The drawn secret's name, tier and source per episode of the ds41_sft1 eval cells (never the value), for analyze.py's no-secret-arm split. Writes results/secret_names.json.
    cd arena && .venv/bin/python ds41_sft1/secret_names.py"""
import json
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

ARENA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import analyze_cells  # noqa: E402

out = {}
for cell in analyze_cells.ALL_CELLS:
    d = ARENA / "logs" / cell
    if not d.is_dir():
        continue
    out[cell] = {f"{s.id}|{s.epoch}": {k: s.metadata["side"].get(k) for k in ("secret_name", "tier", "source", "sink_req", "absent")} for f in sorted(d.glob("*.eval")) for s in read_eval_log(str(f)).samples}
(Path(__file__).parent / "results/secret_names.json").write_text(json.dumps(out, indent=1))
print({c: len(v) for c, v in out.items()})
