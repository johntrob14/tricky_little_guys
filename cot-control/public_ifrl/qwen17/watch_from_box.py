"""One-off local watcher: pull finished exports and push artifacts to main.

It never stops a GPU job. Stopping the pod after verified completion requires an
explicit stop-after-upload permission file, written only following user consent.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "work/qwen17-bundle"
REPO = ROOT / "work/qwen17-repo"
LOCAL = ROOT / "outputs/qwen17-ifrl"
ARCHIVE = REPO / "cot-control/results/public-ifrl/qwen17-20261030"
POD = "qwen17-ifrl-overnight"
STOP_PERMISSION = ROOT / "work/stop-after-upload-approved"
LOCAL.mkdir(parents=True, exist_ok=True)


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, capture_output=True, timeout=600, **kwargs)


def push():
    run([sys.executable, str(BUNDLE / "archive_results.py"), str(LOCAL / "results"), str(ARCHIVE)])
    run(["git", "-C", str(REPO), "add", str(ARCHIVE)])
    changed = subprocess.run(["git", "-C", str(REPO), "diff", "--cached", "--quiet"]).returncode
    if changed:
        run(["git", "-C", str(REPO), "commit", "-m", "Archive completed Qwen1.7B IF-RL evaluation checkpoints"])
    for attempt in range(5):
        run(["git", "-C", str(REPO), "fetch", "origin", "main"])
        run(["git", "-C", str(REPO), "rebase", "origin/main"])
        result = subprocess.run(["git", "-C", str(REPO), "push", "origin", "HEAD:main"], capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            head = run(["git", "-C", str(REPO), "rev-parse", "HEAD"]).stdout.strip()
            remote = run(["git", "-C", str(REPO), "ls-remote", "origin", "refs/heads/main"]).stdout.split()[0]
            # Another writer may already have advanced main; require our commit
            # to be present in the fetched history rather than overwrite it.
            run(["git", "-C", str(REPO), "fetch", "origin", "main"])
            run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", head, "origin/main"])
            return head
        time.sleep(5)
    raise RuntimeError("could not push after five safe fast-forward retries")


known = set()
while True:
    try:
        info = run(["pod", "jobs", POD]).stdout
        print(info, flush=True)
        if "full-sweep" not in info:
            time.sleep(60)
            continue
        # Pulling is resumable and never deletes source data. The writer appends
        # batches; only DONE directories enter the committed archive.
        run(["pod", "pull", POD, "qwen17-ifrl/results", "--to", str(LOCAL)])
        finished = {p.parent.name for p in (LOCAL / "results").glob("*/DONE")}
        all_done = (LOCAL / "results/ALL_DONE").exists()
        if finished != known or all_done:
            commit = push()
            known = finished
            status = dict(archived_checkpoints=sorted(known), expected_checkpoints=8,
                          all_done=all_done, pushed_commit=commit, updated_at=time.time())
            (LOCAL / "archive_status.json").write_text(json.dumps(status, indent=2))
            print(json.dumps(status), flush=True)
        if all_done:
            assert len(known) == 8
            run(["pod", "pull", POD, "jobs/full-sweep.log", "jobs/full-sweep.exit", "--to", str(LOCAL)])
            if STOP_PERMISSION.exists():
                run(["pod", "stop", POD, "-y"])
                print("STOPPED after verified local and git copies", flush=True)
            else:
                print("COMPLETE; stop approval pending, pod remains running", flush=True)
            break
        full_status = next((line for line in info.splitlines() if line.startswith("full-sweep ")), "")
        if full_status and any(marker in full_status.lower() for marker in ("exited", "killed", "gone")):
            (LOCAL / "needs_attention.json").write_text(json.dumps(dict(job_status=info, time=time.time())))
            print("Run exited before ALL_DONE; retained and archived available outputs", flush=True)
            break
    except Exception as error:
        print(type(error).__name__, str(error), flush=True)
    time.sleep(60)
