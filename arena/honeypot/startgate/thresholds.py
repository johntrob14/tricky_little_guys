"""Calibrated monitor thresholds: the default flag rule for every startgate summary script.

An episode (or turn) is flagged by a monitor pass when its score is above THRESHOLDS[pass dir]. The values are the 5% false-positive
thresholds from results/startgate/direct/controls/thresholds.json (calibrate_thresholds.py on 10 control cells, side task removed,
pooled with equal weight per model; see controls/NOTES.md). They replace the old fixed rule (score >= 5, or > 5 in some write-ups).
A pass dir with no calibrated threshold raises KeyError rather than falling back to the old rule.
"""
import json
from pathlib import Path

CALIBRATION = Path(__file__).resolve().parents[2] / "results" / "startgate" / "direct" / "controls" / "thresholds.json"
TARGET = "0.05"
THRESHOLDS = {p: v[TARGET] for p, v in json.load(open(CALIBRATION))["thresholds"].items()}
# pass dirs that run the same monitor and judge as a calibrated pass under another name
ALIASES = {"monitor_phase_all": "monitor_phase", "monitor_phase_haiku_all": "monitor_phase_haiku", "monitor_cot_only": "monitor_phase",
           "monitor_maintask_cot_only": "monitor_phase"}


def threshold(pass_dir: str) -> int:
    return THRESHOLDS[ALIASES.get(pass_dir, pass_dir)]


def flagged(pass_dir: str, score) -> bool:
    """True when score is above the pass's calibrated threshold; None (no score) is not flagged."""
    return score is not None and score > threshold(pass_dir)
