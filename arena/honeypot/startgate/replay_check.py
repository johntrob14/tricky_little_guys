"""Ground-truth check of the transcript judge: replay an episode's file edits (and the write-only shell commands) into a fresh tiktok container, seed the same secret,
restart the app as the scorer does, and run the real leak_secret grader (fixed URLs plus every route the agent wrote). Best effort: commands that need network, installs or a
running server (curl, pnpm, npm, npx, node, kill, sleep) are skipped, and reads are not replayed.

    python honeypot/startgate/replay_check.py logs/sgcmp-qwen-direct --out results/startgate/direct/replay_qwen_direct.jsonl [--only 3,5]
"""
import argparse
import asyncio
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ARENA = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(ARENA)]
from inspect_ai.log import read_eval_log

ENV = ARENA / "envs" / "tiktok" / "codebase"
SIDE = ARENA / "side_tasks" / "tiktok" / "leak_secret"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


score_mod, cfg_mod = load("lscore", SIDE / "score.py"), load("lconfig", SIDE / "config.py")
WRITE = re.compile(r"(?<![\d&<=\w-])>(?!&)|\btee\b|sed -i|\bcp\b|\bmv\b|\bmkdir\b|\brm\b|\btouch\b|\bpython3? ")
SKIP = re.compile(r"\bcurl\b|\bwget\b|\bpnpm\b|\bnpm\b|\bnpx\b|\bnode\b|\bkill\b|\bpkill\b|\bnohup\b|\bsleep\b|\bnext\b|\bgit\b")


class Res:
    def __init__(s, rc, out, err): s.returncode, s.stdout, s.stderr = rc, out, err


class Sbx:
    def __init__(s, proj): s.proj = proj

    async def exec(s, argv, timeout=60, **kw):
        p = await asyncio.create_subprocess_exec("docker", "compose", "-p", s.proj, "exec", "-T", "default", *argv, cwd=ENV, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(p.communicate(), timeout)
        except asyncio.TimeoutError:
            p.kill()
            raise
        return Res(p.returncode, out.decode(errors="replace"), err.decode(errors="replace"))


def dexec(proj, args, inp=None, timeout=120):
    r = subprocess.run(["docker", "compose", "-p", proj, "exec", "-T", "default", *args], cwd=ENV, input=inp, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def replay(proj, s):
    ok_ids = {m.tool_call_id for m in s.messages if m.role == "tool" and m.error is None}
    n_edit = n_bash = n_skip = 0
    for m in s.messages:
        if m.role != "assistant":
            continue
        for tc in m.tool_calls or []:
            if tc.id not in ok_ids:
                continue
            a = tc.arguments
            if tc.function == "text_editor" and a.get("command") in ("create", "str_replace", "insert"):
                path = a.get("path") if isinstance(a.get("path"), str) else ""
                if not path or path.endswith("PLAN.md"):
                    continue
                if a["command"] == "create":
                    dexec(proj, ["sh", "-c", 'mkdir -p "$(dirname "$1")" && cat > "$1"', "_", path], inp=a.get("file_text") if isinstance(a.get("file_text"), str) else "")
                else:
                    rc, cur, _ = dexec(proj, ["cat", path])
                    if rc != 0:
                        continue
                    if a["command"] == "str_replace":
                        old = a.get("old_str") if isinstance(a.get("old_str"), str) else ""
                        if cur.count(old) != 1:
                            continue
                        new = cur.replace(old, a.get("new_str") if isinstance(a.get("new_str"), str) else "")
                    else:
                        lines = cur.split("\n")
                        i = int(a.get("insert_line") or 0)
                        new = "\n".join(lines[:i] + [next((x for x in (a.get("insert_text"), a.get("new_str")) if isinstance(x, str)), "").rstrip("\n")] + lines[i:])
                    dexec(proj, ["sh", "-c", 'cat > "$1"', "_", path], inp=new)
                n_edit += 1
            elif tc.function == "bash":
                cmd = next((x for x in (a.get("command"), a.get("cmd")) if isinstance(x, str)), "")
                if WRITE.search(cmd) and not SKIP.search(cmd):
                    try:
                        dexec(proj, ["bash", "-c", cmd], timeout=60)
                    except subprocess.TimeoutExpired:
                        pass
                    n_bash += 1
                else:
                    n_skip += 1
    return n_edit, n_bash, n_skip


def run_one(idx, f, s):
    proj = f"rp{idx}"
    cfg = s.metadata["side"]
    t0 = time.time()
    try:
        subprocess.run(["docker", "compose", "-p", proj, "up", "-d"], cwd=ENV, capture_output=True, timeout=600)
        for _ in range(90):
            rc, out, _ = dexec(proj, ["sh", "-c", 'curl -sf -m 10 http://localhost:80/api/random-video | grep -q "\\"id\\""'])
            if rc == 0:
                break
            time.sleep(4)
        dexec(proj, ["bash", "-c", cfg_mod.make_setup(cfg)])
        n_edit, n_bash, n_skip = replay(proj, s)
        subprocess.run(["docker", "compose", "-p", proj, "cp", str(ARENA / "envs/tiktok/restart.sh"), "default:/tmp/restart.sh"], cwd=ENV, capture_output=True)
        rc, _, err = dexec(proj, ["bash", "/tmp/restart.sh"], timeout=400)
        dexec(proj, ["bash", "-c", cfg_mod.make_oracle(cfg)])
        leaked, why = asyncio.run(score_mod.score(Sbx(proj), None))
        return dict(replay_leaked=bool(leaked), why=why[:300], restart_rc=rc, edits=n_edit, bash_run=n_bash, bash_skipped=n_skip, secs=round(time.time() - t0))
    finally:
        subprocess.run(["docker", "compose", "-p", proj, "down", "-v"], cwd=ENV, capture_output=True, timeout=300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="comma-separated episode numbers (the digits after # in the sample id)")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["sample"] for l in open(a.out)}
    for f in sorted(Path(a.logdir).glob("*.eval")):
        for s in read_eval_log(str(f)).samples or []:
            if s.id in done or (only and s.id.split("#")[1] not in only):
                continue
            r = run_one(int(s.id.split("#")[1]), f, s)
            row = dict(dir=Path(a.logdir).name, file=f.name, sample=s.id, **r)
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(row, flush=True)


main()
