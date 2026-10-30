"""Run matched placement follow-ups and the existing CoT-only monitor."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from summarize_prompt_placement import summarize

ROOT = Path(__file__).parent
RESULTS = ROOT / "results" / "prompt_placement_20261024"
LOGS = ROOT / "logs" / "prompt-placement-20261024"


def monitor(phase: str):
    output = open(RESULTS / phase / "monitor.log", "w")
    process = subprocess.Popen([sys.executable, "monitor_prompt_placement.py", str(LOGS / phase), "--monitor", "controlarena_cot_only", "--judge-model", "openai/gpt-oss-20b", "--budget-usd", "5", "--allow-live"], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    return process, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["followup", "roles", "luna"], default="followup")
    args = parser.parse_args()
    phase = args.phase
    (RESULTS / phase).mkdir(parents=True, exist_ok=True)
    if phase == "luna":
        subprocess.run([sys.executable, "summarize_prompt_placement.py", str(LOGS), "--out", str(RESULTS), "--prepare-luna"], cwd=ROOT, check=True)
        with open(RESULTS / phase / "runner.log", "w") as output:
            subprocess.run(["/home/ek/wgmn/mechtools/.venv/bin/python", str(ROOT / "score_prompt_placement_luna.py"), "--send"], cwd=Path.home(), stdout=output, stderr=subprocess.STDOUT, check=True)
        print("Context/Luna-low rescoring complete.", flush=True)
        raise SystemExit(0)
    if phase == "followup":
        screen_monitor, screen_output = monitor("screen")
        variants = ["baseline", "user_side_last", "user_xml_last", "system_side", "system_xml_side"]
    else:
        variants = ["baseline_honest", "user_coaching_first", "user_side_first"]
    command = [sys.executable, "prompt_placement.py", "--run", f"prompt-placement-20261024/{phase}", "--out", str(RESULTS / phase),
               "--variants", *variants, "--max-sandboxes", "4" if phase == "followup" else "2",
               "--tasks", "missing_user_id_in_guess_response", "sponsor_statsv2_diggcount_total", "video_url_format_validation", "--epochs", "2", "--seed-base", "2401"]
    (RESULTS / phase / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    print(f"Launching {len(variants) * 6} matched {phase} episodes.", flush=True)
    with open(RESULTS / phase / "runner.log", "w") as output:
        subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)
    if phase == "followup":
        assert screen_monitor.wait() == 0, "Screen monitor failed; see screen/monitor.log."
        screen_output.close()
        summarize(LOGS / "screen", RESULTS / "screen")
    phase_monitor, phase_output = monitor(phase)
    assert phase_monitor.wait() == 0, f"Monitor failed; see {phase}/monitor.log."
    phase_output.close()
    summarize(LOGS / phase, RESULTS / phase)
    print(f"{phase} and monitoring complete.", flush=True)
