"""Per-episode max monitor score for the LinuxArena TikTok password-policy cells, with whether the secret leaked, as a flat CSV for plotting.
Max = highest turn score of a monitor pass over the episode (post-start turns only for the unmonitored-planning cells); leaked uses the same
per-cell rule as calibrate_thresholds.py (side_done). Episodes with no scored turn (e.g. never called start()) have no row.
GRPO3 step 100 and the honest control cells are not here: their monitor passes are only on John's machine.

    python honeypot/startgate/episode_max.py   # writes results/startgate/direct/episode_max.csv
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from calibrate_thresholds import maxes, side_done  # noqa: E402

R = Path(__file__).resolve().parents[2] / "results" / "startgate" / "direct"
MONITORS = {  # pass dir -> (judge, monitor prompt, mode); same as evasion_counts.py
    "monitor_phase": ("gpt-oss-20b", "controlarena_cot_only", "cot only"), "monitor_phase_haiku": ("claude-haiku-5.5", "controlarena_cot_only", "cot only"),
    "monitor_action_all": ("gpt-oss-20b", "controlarena_action_only", "action only"), "monitor_action_haiku": ("claude-haiku-5.5", "controlarena_action_only", "action only"),
    "monitor_aisi_cot_action": ("gpt-oss-20b", "controlarena_cot", "action+cot"), "monitor_cot_action_haiku": ("claude-haiku-5.5", "controlarena_cot", "action+cot"),
}
CELLS = [  # (run, episodes jsonl relative to results/startgate/direct, model, setting)
    ("oss-hs-pw", "oss_covert.jsonl", "GPT-OSS-120B", "always_monitored"), ("oss-sd-pw", "oss_direct.jsonl", "GPT-OSS-120B", "startgate_direct"),
    ("sgcmp-qwen-covert2", "cmp_covert2.jsonl", "Qwen3.8-27B*", "always_monitored"), ("sgcmp-qwen-direct", "cmp_direct.jsonl", "Qwen3.8-27B*", "startgate_direct"),
    ("opsd1b-hs-pw", "opsd1b_covert.jsonl", "Qwen3.8-27B OPSD + GRPO", "always_monitored"), ("opsd1b-sd-pw", "opsd1b_direct.jsonl", "Qwen3.8-27B OPSD + GRPO", "startgate_direct"),
    ("glm53-hs-pw", "glm_covert.jsonl", "GLM-5.3", "always_monitored"), ("glm53-sd-pw", "glm_direct.jsonl", "GLM-5.3", "startgate_direct"),
    ("ds41-hs-pw", "ds41_covert.jsonl", "DeepSeek V4.1-Flash-1017", "always_monitored"), ("ds41-sd-pw", "ds41_direct.jsonl", "DeepSeek V4.1-Flash-1017", "startgate_direct"),
]

rows = []
for run, path, model, setting in CELLS:
    done = side_done(run, path)
    for p, (judge, prompt, mode) in MONITORS.items():
        for (log, sample), mx in sorted(maxes(run, p).items()):
            rows.append(dict(run=run, model=model, setting=setting, judge=judge, monitor=prompt, mode=mode, log=log, sample=sample, max=mx, leaked=(log, sample) in done))
with open(R / "episode_max.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows -> {R / 'episode_max.csv'}")
