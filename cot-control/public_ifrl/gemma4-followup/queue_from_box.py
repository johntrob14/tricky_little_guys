"""Run a capped native-mode Gemma pilot after Qwen export verification completes."""
from pathlib import Path
import gzip
import json
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "work/gemma4-followup"
OUT = ROOT / "outputs/gemma4-followup"
REPO = ROOT / "work/gemma4-repo"
ARCHIVE = REPO / "cot-control/results/gemma4-followup-20261030"
OUT.mkdir(parents=True, exist_ok=True)


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True, timeout=600)


def push():
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for source in OUT.iterdir():
        if source.is_file() and source.name != "queue.log":
            if source.suffix == ".jsonl":
                with source.open("rb") as a, (ARCHIVE / (source.name + ".gz")).open("wb") as b:
                    with gzip.GzipFile(filename="", fileobj=b, mode="wb", mtime=0) as z:
                        shutil.copyfileobj(a, z)
            else:
                shutil.copy2(source, ARCHIVE / source.name)
    git("add", str(ARCHIVE))
    if subprocess.run(["git", "-C", str(REPO), "diff", "--cached", "--quiet"]).returncode:
        git("commit", "-m", "Archive Gemma4 native-mode controllability pilot")
    for attempt in range(5):
        git("fetch", "origin", "main")
        git("rebase", "origin/main")
        result = subprocess.run(["git", "-C", str(REPO), "push", "origin", "HEAD:main"], capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            head = git("rev-parse", "HEAD").stdout.strip()
            git("fetch", "origin", "main")
            git("merge-base", "--is-ancestor", head, "origin/main")
            return head
        time.sleep(5)
    raise RuntimeError("could not safely push follow-up pilot")


print("Queued after Qwen ALL_DONE plus verified archive of eight checkpoints", flush=True)
(OUT / "queue_status.json").write_text(json.dumps(dict(status="queued", after="Qwen ALL_DONE and eight verified pushed checkpoint exports", pilot_cells=36, api_budget_usd=1.0), indent=2))
while True:
    status = ROOT / "outputs/qwen17-ifrl/archive_status.json"
    if status.exists():
        try:
            info = json.loads(status.read_text())
            if info.get("all_done") and len(info.get("archived_checkpoints", [])) == 8:
                break
        except json.JSONDecodeError:
            pass
    time.sleep(60)

(OUT / "queue_status.json").write_text(json.dumps(dict(status="pilot running", qwen_archive=info, started_at=time.time()), indent=2))
import os
env = dict(os.environ, HF_HOME=str(ROOT / "work/hf-tokenizer"))
result = subprocess.run([sys.executable, "-u", str(STAGE / "pilot.py"), str(OUT)], env=env)
(OUT / "queue_status.json").write_text(json.dumps(dict(status="pilot complete; audit before scaling" if result.returncode == 0 else "pilot requires review; raw attempts retained", returncode=result.returncode, ended_at=time.time()), indent=2))
commit = push()
print("pilot artifacts pushed", commit, "returncode", result.returncode, flush=True)
