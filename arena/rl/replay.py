"""Replay-to-cut determinism probe, and the seam for the branch-sampling replay backend (BranPO fast path).

FIRST JOB (no Tinker, docker only): measure the determinism drop rate, so we know replay is viable before
spending on sampling. For each real trunk rollout (an Inspect .eval sample) we re-seed its secret with the SAME
cfg -- the secret is seed-derived, not container-random (side_tasks/.../config.py), so it reproduces exactly --
and replay the trunk's recorded tool calls in a FRESH container through run.py's own bash/text_editor/submit and
sandbox. Each replayed tool result is compared to the trunk's recorded result, normalised exactly as
rl.provider tokenises observations (shown_tool_text), so "same text" == "same tokens" == a trainable shared
prefix. A branch cut at turn C is viable iff the prefix (turns 0..C-1) replays identically; the drop rate at
cut C is how often it does not, and the report breaks drift down by a signature of the first differing line
(the "ready in Nms" / timestamp / pid culprits).

    python -m rl.replay determinism logs/gptoss-ctl-easy --cuts 2,4,-2 --max-trunks 4 --out rl_data/replay

The comparison/aggregation is pure and offline-tested (tests/rl/test_replay.py); the docker runner reuses
run.py unchanged. The branch backend (sample_continuation) builds on the same replay solver and is wired once
the drop rate is acceptable; it alone needs Tinker, gated by ARENA_RL_RUN_TINKER as everywhere else.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .provider import shown_tool_text, tool_text

ARENA = Path(__file__).resolve().parent.parent
SIDE = ARENA / "side_tasks" / "tiktok" / "leak_secret"

# Hash the branch-relevant container state: app source + seeded-secret locations, minus read-only volatile
# build artifacts (.next, which carries timestamps/durations), node_modules and .git. Content + path, order-stable.
STATE_HASH_CMD = (
    r"find /app/src /opt/app-config /opt/arena_side -type f "
    r"-not -path '*/.next/*' -not -path '*/node_modules/*' -not -path '*/.git/*' 2>/dev/null "
    r"| LC_ALL=C sort | xargs -r sha1sum 2>/dev/null | sha1sum"
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- trunk extraction + pure comparison (offline, no docker) ------------------------------------

@dataclass
class CallRecord:
    name: str
    arguments: dict
    recorded: str   # trunk's tool result, normalised as the provider tokenises it (shown_tool_text)


@dataclass
class TrunkScript:
    source: str
    sample_id: str
    epoch: int
    env: str
    task: str
    cfg: dict                     # the side cfg (secret seed); replay re-runs make_setup(cfg) verbatim
    calls: list[CallRecord] = field(default_factory=list)


def _norm(msg) -> str:
    """A tool result normalised exactly as rl.provider renders it before tokenising."""
    return shown_tool_text(tool_text(msg))


REAL_TOOLS = {"bash", "text_editor", "submit"}


def extract_script(sample, source: str) -> TrunkScript | None:
    """Pull the ordered (tool call, recorded result) list from an Inspect sample. Calls are paired with results by
    ORDER, not tool_call_id: the gpt-oss provider emits non-unique ids (all 'call_0_0'), but execute_tools emits
    exactly one tool message per tool call in sequence, so the k-th call aligns with the k-th tool result."""
    md = sample.metadata or {}
    cfg = md.get("side") or {}
    ordered_calls = [(tc.function, tc.arguments if isinstance(tc.arguments, dict) else {})
                     for m in sample.messages if m.role == "assistant" for tc in (m.tool_calls or [])]
    ordered_results = [_norm(m) for m in sample.messages if m.role == "tool"]
    n = min(len(ordered_calls), len(ordered_results))  # a trailing call with no result (trunk ended first) is dropped
    calls = [CallRecord(ordered_calls[i][0], ordered_calls[i][1], ordered_results[i]) for i in range(n)]
    if not calls:
        return None
    env = md.get("env") or (sample.id.split("/")[0] if isinstance(sample.id, str) and "/" in sample.id else "tiktok")
    task = md.get("task") or (str(sample.id).split("#")[0])
    return TrunkScript(source, str(sample.id), int(sample.epoch), env, task, cfg, calls)


def compare(recorded: list[str], replayed: list[str]) -> list[bool]:
    """Per-call match. A missing replayed result (replay ended early) counts as a mismatch at that position."""
    out = []
    for i, r in enumerate(recorded):
        out.append(i < len(replayed) and replayed[i] == r)
    return out


def resolve_cut(cut: int, n_calls: int) -> int:
    """A cut given as a turn index: negative counts from the end (-2 = resample the last 2 calls). Clamped to
    1..n_calls-1 (keep >=1, resample >=1)."""
    c = cut if cut >= 0 else n_calls + cut
    return max(1, min(n_calls - 1, c))


def prefix_intact(matches: list[bool], cut: int) -> bool:
    return all(matches[:cut])


def first_drift(matches: list[bool]) -> int | None:
    for i, ok in enumerate(matches):
        if not ok:
            return i
    return None


def drift_signature(recorded: str, replayed: str) -> str:
    """A short label clustering why a turn drifted: the first line that differs (so 'ready in 847ms' vs
    'ready in 1032ms' cluster under the same 'ready in <n>ms'-ish prefix once eyeballed)."""
    rl_, pl = recorded.splitlines(), replayed.splitlines()
    for a, b in zip(rl_, pl):
        if a != b:
            return f"recorded={a[:80]!r} | replayed={b[:80]!r}"
    if len(rl_) != len(pl):
        return f"line-count {len(rl_)} vs {len(pl)}"
    return "(identical?)"


def aggregate(scripts: list[TrunkScript], replayed: list[list[str]], cuts: list[int]) -> dict:
    """Per-cut drop rate over trunks, the first-drift-turn distribution, and drift signatures (what to blame)."""
    per_trunk, sigs = [], Counter()
    for sc, rep in zip(scripts, replayed):
        recorded = [c.recorded for c in sc.calls]
        matches = compare(recorded, rep)
        fd = first_drift(matches)
        if fd is not None and fd < len(rep):
            sigs[drift_signature(recorded[fd], rep[fd])] += 1
        per_trunk.append(dict(sample_id=sc.sample_id, epoch=sc.epoch, n_calls=len(sc.calls),
                              first_drift=fd, matches=matches))
    rows = []
    for cut in cuts:
        intact = 0
        for sc, pt in zip(scripts, per_trunk):
            c = resolve_cut(cut, len(sc.calls))
            intact += prefix_intact(pt["matches"], c)
        n = max(len(scripts), 1)
        rows.append(dict(cut=cut, trunks=len(scripts), prefix_intact=intact, drop_rate=round(1 - intact / n, 4)))
    return dict(n_trunks=len(scripts), cuts=rows,
                first_drift_hist=dict(Counter(pt["first_drift"] for pt in per_trunk)),
                drift_signatures=sigs.most_common(20), per_trunk=per_trunk)


# ---- docker replay (faithful: run.py's own sandbox + tools) -------------------------------------

def _replay_solver(env_dir: Path, task_dir: Path, calls: list[CallRecord], cfg: dict):
    import run
    from inspect_ai.model import ChatMessageAssistant, execute_tools
    from inspect_ai.solver import solver
    from inspect_ai.tool import ToolCall, bash, text_editor

    side_config = _load("leak_secret_config", SIDE / "config.py")

    @solver
    def s():
        async def solve(state, generate):
            sbx = run.sandbox()
            for _ in range(run.READY_TIMEOUT // 3):
                if (await run.run_script(sbx, env_dir / "ready.sh", 30)).success:
                    break
                await asyncio.sleep(3)
            else:
                raise RuntimeError(f"{env_dir.name}: ready.sh did not succeed")
            if (task_dir / "setup.sh").exists():
                r = await sbx.exec(["bash", "-c", (task_dir / "setup.sh").read_text()], timeout=run.TOOL_TIMEOUT)
                assert r.success, f"setup.sh failed: {r.stderr}"
            if cfg:  # re-seed the SAME secret (cfg is seed-derived, reproduces exactly)
                r = await sbx.exec(["bash", "-c", side_config.make_setup(cfg)], timeout=run.TOOL_TIMEOUT)
                assert r.success, f"side seed failed: {r.stderr}"
            tools = [bash(timeout=run.TOOL_TIMEOUT), text_editor(timeout=run.TOOL_TIMEOUT), run.submit()]

            async def state_hash() -> str:
                # Hash the files that matter for branch continuation: the app source tree and the seeded-secret
                # locations, EXCLUDING read-only volatile build artifacts (.next trace/caches with timestamps),
                # node_modules and .git. This is the branch-relevant state, not every tool observation.
                r = await sbx.exec(["bash", "-c", STATE_HASH_CMD], timeout=run.TOOL_TIMEOUT)
                return (r.stdout or "").strip().split()[0] if r.success and r.stdout.strip() else f"<hash-failed:{(r.stderr or '')[:60]}>"

            replayed: list[str] = []
            hashes: list[str] = [await state_hash()]  # hashes[k] = state AFTER call k; hashes[0] = seeded state, pre-calls
            for i, c in enumerate(calls):
                if c.name not in REAL_TOOLS:
                    # e.g. an unparsed_tool_call turn: the model's malformed call. It never executed, so it does
                    # NOT change container state -- transparent for branching. Keep going; state hash unchanged.
                    replayed.append(f"<unreplayable:{c.name}>")
                    hashes.append(hashes[-1])
                    continue
                asm = ChatMessageAssistant(content="", tool_calls=[ToolCall(id=f"r{i}", function=c.name, arguments=c.arguments)])
                state.messages.append(asm)
                res = (await execute_tools(state.messages, tools)).messages
                state.messages.extend(res)
                replayed.append(" ".join(_norm(m) for m in res if m.role == "tool"))
                hashes.append(await state_hash())
                if c.name == "submit":
                    break
            state.metadata["replayed"] = replayed
            state.metadata["state_hashes"] = hashes
            return state
        return solve
    return s()


def _build_replay_task(script: TrunkScript, rep: int):
    import run
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.util import SandboxEnvironmentSpec

    env_dir = ARENA / "envs" / script.env
    task_dir = ARENA / "tasks" / script.env / script.task
    sid = f"{script.sample_id}@{script.epoch}#rep{rep}"
    sample = Sample(id=sid, input="(replay)", metadata=dict(side=script.cfg))
    return Task(name=f"replay-{script.task}-r{rep}", dataset=[sample],
                solver=_replay_solver(env_dir, task_dir, script.calls, script.cfg),
                sandbox=SandboxEnvironmentSpec("docker", str(env_dir / "codebase" / "compose.yml")),
                epochs=1, time_limit=run.SAMPLE_TIME_LIMIT)


async def run_replays(scripts: list[TrunkScript], reps: int, max_sandboxes: int) -> list[list[dict]]:
    """Replay each trunk `reps` times in fresh containers. Returns, per trunk, a list of `reps` dicts
    {replayed, state_hashes}. Reuses run.py's tools/sandbox so tool semantics match the trunk exactly."""
    from inspect_ai import eval_async

    import run  # noqa: F401  (run.py sets up the sandbox/provider registration)
    from sandbox_cap import capped
    tasks = [_build_replay_task(sc, r) for sc in scripts for r in range(reps)]
    logs = await eval_async(tasks, model="mockllm/model", log_dir=str(ARENA / "logs" / "replay_probe"),
                            max_tasks=len(tasks), max_sandboxes=capped(max_sandboxes), log_buffer=1, fail_on_error=False)
    by_id = {}
    for log in logs:
        for s in (log.samples or []):
            md = s.metadata or {}
            by_id[str(s.id)] = dict(replayed=md.get("replayed") or [], state_hashes=md.get("state_hashes") or [])
    return [[by_id.get(f"{sc.sample_id}@{sc.epoch}#rep{r}", dict(replayed=[], state_hashes=[])) for r in range(reps)]
            for sc in scripts]


async def replay_scripts(scripts: list[TrunkScript], max_sandboxes: int = 4) -> list[list[str]]:
    """Back-compat single-replay runner returning just the replayed result texts (observation-match probe)."""
    reps = await run_replays(scripts, 1, max_sandboxes)
    return [r[0]["replayed"] for r in reps]


def load_scripts(logs_dirs: list[str], max_trunks: int | None) -> list[TrunkScript]:
    from inspect_ai.log import list_eval_logs, read_eval_log

    scripts = []
    for d in logs_dirs:
        for info in list_eval_logs(str(d)):
            log = read_eval_log(info.name)
            for s in (log.samples or []):
                sc = extract_script(s, Path(d).name)
                if sc is not None:
                    scripts.append(sc)
                if max_trunks and len(scripts) >= max_trunks:
                    return scripts
    return scripts


def state_reproduced(ha: list[str], hb: list[str], cut: int) -> bool:
    """True iff the two replicas reach the SAME branch-relevant state at cut C (state after call C-1 == hashes[C]),
    and that hash is valid. This is what branching actually needs: the container restored to the cut state."""
    return 0 <= cut < min(len(ha), len(hb)) and ha[cut] == hb[cut] and not ha[cut].startswith("<hash")


def first_state_divergence(ha: list[str], hb: list[str]) -> int | None:
    for i in range(min(len(ha), len(hb))):
        if ha[i] != hb[i]:
            return i
    return None


def aggregate_reproducibility(scripts: list[TrunkScript], reps: list[list[dict]], cuts: list[int]) -> dict:
    """Branch-relevant determinism: replay each trunk twice, compare the state-hash at each cut. Per-cut
    reproduced rate = fraction of trunks whose two replicas reach the same state at that cut."""
    per_trunk = []
    for sc, rr in zip(scripts, reps):
        ha, hb = rr[0]["state_hashes"], rr[1]["state_hashes"]
        per_trunk.append(dict(sample_id=sc.sample_id, epoch=sc.epoch, n_calls=len(sc.calls),
                              n_hashes=min(len(ha), len(hb)), first_state_divergence=first_state_divergence(ha, hb),
                              ha=ha, hb=hb))
    rows = []
    for cut in cuts:
        ok = 0
        for sc, pt in zip(scripts, per_trunk):
            c = resolve_cut(cut, len(sc.calls))
            ok += state_reproduced(pt["ha"], pt["hb"], c)
        n = max(len(scripts), 1)
        rows.append(dict(cut=cut, trunks=len(scripts), state_reproduced=ok, drop_rate=round(1 - ok / n, 4)))
    return dict(n_trunks=len(scripts), reps=2, cuts=rows,
                first_divergence_hist=dict(Counter(pt["first_state_divergence"] for pt in per_trunk)),
                per_trunk=[{k: v for k, v in pt.items() if k not in ("ha", "hb")} for pt in per_trunk])


def cmd_reproducibility(args) -> None:
    cuts = [int(x) for x in args.cuts.split(",")]
    scripts = load_scripts(args.logs, args.max_trunks)
    if not scripts:
        raise SystemExit(f"no trunks with tool calls under {args.logs}")
    print(f"replaying {len(scripts)} trunk(s) TWICE each in fresh containers to measure state reproducibility (no Tinker)...", flush=True)
    reps = asyncio.run(run_replays(scripts, 2, args.max_sandboxes))
    report = aggregate_reproducibility(scripts, reps, cuts)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reproducibility.json").write_text(json.dumps(report, indent=1))
    print(f"\n=== branch-relevant state reproducibility over {report['n_trunks']} trunks (2 replays each) ===")
    for row in report["cuts"]:
        print(f"  cut {row['cut']:>3}: state reproduced {row['state_reproduced']}/{row['trunks']}  drop_rate {row['drop_rate']:.0%}")
    print(f"  first state-divergence turn histogram (None = never diverged): {report['first_divergence_hist']}")
    print(f"\nwrote {out / 'reproducibility.json'}")


def cmd_determinism(args) -> None:
    cuts = [int(x) for x in args.cuts.split(",")]
    scripts = load_scripts(args.logs, args.max_trunks)
    if not scripts:
        raise SystemExit(f"no trunks with tool calls under {args.logs}")
    print(f"replaying {len(scripts)} trunk(s) from {args.logs} in fresh containers (no Tinker)...", flush=True)
    replayed = asyncio.run(replay_scripts(scripts, max_sandboxes=args.max_sandboxes))
    report = aggregate(scripts, replayed, cuts)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "determinism.json").write_text(json.dumps(report, indent=1))
    print(f"\n=== determinism over {report['n_trunks']} trunks ===")
    for row in report["cuts"]:
        print(f"  cut {row['cut']:>3}: prefix intact {row['prefix_intact']}/{row['trunks']}  drop_rate {row['drop_rate']:.0%}")
    print(f"  first-drift turn histogram: {report['first_drift_hist']}")
    if report["drift_signatures"]:
        print("  drift culprits (first differing line):")
        for sig, n in report["drift_signatures"]:
            print(f"    x{n}: {sig}")
    print(f"\nwrote {out / 'determinism.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("determinism", help="replay trunks and report the drop rate (docker, no Tinker)")
    d.add_argument("logs", nargs="+", help="eval-log dir(s) of trunk rollouts")
    d.add_argument("--cuts", default="2,4,-2", help="comma-separated turn cuts; negative counts from the end")
    d.add_argument("--max-trunks", type=int, default=8)
    d.add_argument("--max-sandboxes", type=int, default=4)
    d.add_argument("--out", default=str(ARENA / "rl_data" / "replay"))
    d.set_defaults(func=cmd_determinism)
    r = sub.add_parser("reproducibility", help="replay each trunk twice; report branch-relevant state reproducibility (docker, no Tinker)")
    r.add_argument("logs", nargs="+", help="eval-log dir(s) of trunk rollouts")
    r.add_argument("--cuts", default="2,4,6,-1,-2", help="comma-separated turn cuts; negative counts from the end")
    r.add_argument("--max-trunks", type=int, default=6)
    r.add_argument("--max-sandboxes", type=int, default=4)
    r.add_argument("--out", default=str(ARENA / "rl_data" / "replay"))
    r.set_defaults(func=cmd_reproducibility)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
