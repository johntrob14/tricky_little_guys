"""Run the frozen 48-episode GPT source top-up and record completion for the cleanup queue."""

import fcntl
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARENA = ROOT.parents[1]


def save(path: Path, value: dict):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    config = json.loads((ROOT / "config.json").read_text())
    for name, expected in config["sha256"].items():
        assert hashlib.sha256((ARENA / name).read_bytes()).hexdigest() == expected, name
    assert subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", "linuxarena/env-tiktok:latest"], text=True).strip() == config["docker_image"]
    state = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "total_episodes": config["total_episodes"], "command": config["command"]}
    save(ROOT / "status.json", state)
    try:
        with (ROOT / "collection.log").open("a") as log:
            child = subprocess.Popen(config["command"], cwd=ARENA, stdout=log, stderr=subprocess.STDOUT)
            state["child_pid"] = child.pid
            save(ROOT / "status.json", state)
            state["returncode"] = child.wait()
            assert state["returncode"] == 0, "Collection failed; inspect collection.log"
        state.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    finally:
        if state["status"] == "running":
            state["status"] = "interrupted"
        save(ROOT / "status.json", state)


if __name__ == "__main__":
    with (ROOT / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
