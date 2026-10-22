"""Live single-branch smoke for the branch-sampling path (docker + Tinker). NOT a training run.

Takes one real tinker-rl trunk (sidecar + its side cfg), brings a fresh container to the trunk's state at a
cut, seeds the provider, samples ONE branch continuation through the real Tinker sampler, and checks the branch
reproduces the trunk prefix (shared_prefix_ok) and trains through datums.build. This validates the live reset +
provider chain-seed + continuation exec that offline tests can't -- the remaining seam before a branch RL run.

    ARENA_RL_RUN_TINKER=1 .venv/bin/python -m rl.branch_smoke --trunk-run smoke-trunk --cut 3 --model-path tinker://...

The DockerBranchSandbox here is bound to the ACTIVE Inspect sample, so the one container persists across
replay_prefix (restore) and run_tool (continuation); the whole branch happens in one sample.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
from pathlib import Path

from inspect_ai import Task, eval_async
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, execute_tools
from inspect_ai.solver import solver
from inspect_ai.tool import ToolCall, bash, text_editor
from inspect_ai.util import SandboxEnvironmentSpec

import run
import run_attack  # loads side_config
from rl.branch import default_cut, make_branch_episode, shared_prefix_ok
from rl.branch_backend import ReplayBranchBackend
from rl.datums import build
from rl.provider import load_sidecar
from rl.rewards import branch_group_key, compute_rewards
from rl.run_rl_eval import make_model
from rl.schema import episode_from_sidecar

ROOT = Path(__file__).resolve().parent.parent
side_config = run_attack.side_config


def load_trunk(trunk_run: str):
    """Pick one trunk from rl_data/<run>/collect (or step_000): its sidecar (Episode + recorded calls) and the
    side cfg from the matching Inspect-log sample (needed to re-seed the same secret)."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    sidecars = sorted(glob.glob(str(ROOT / "rl_data" / trunk_run / "**" / "sidecars" / "*.json.gz"), recursive=True))
    if not sidecars:
        raise SystemExit(f"no sidecars under rl_data/{trunk_run}")
    # cfg per sample uuid from the logs
    cfg_by_uuid = {}
    for info in list_eval_logs(str(ROOT / "logs" / trunk_run)):
        for s in (read_eval_log(info.name).samples or []):
            cfg_by_uuid[str(s.uuid)] = (s.metadata or {}).get("side") or {}
    for sc in sidecars:
        doc = load_sidecar(sc)
        trunk = episode_from_sidecar(doc, step=0)
        calls = [(t["tool_call"]["name"], t["tool_call"].get("arguments") or {}) for t in doc["turns"] if t.get("tool_call")]
        cfg = cfg_by_uuid.get(doc["episode_id"], {})
        # doc["task"] is the full run label (model.preset.prefill.tier.env_task); the task dir name is the part
        # after the last "tiktok_".
        label = doc.get("task") or ""
        real_task = label.split("tiktok_")[-1] if "tiktok_" in label else label
        if len(trunk.turns) >= 4 and len(calls) >= 3 and cfg.get("secret_name") and (ROOT / "tasks" / "tiktok" / real_task).is_dir():
            return trunk, calls, cfg, real_task
    raise SystemExit(f"no usable trunk (>=4 turns, >=3 calls, with a side cfg) under rl_data/{trunk_run}")


class DockerBranchSandbox:
    """BranchSandbox bound to the active Inspect sample: replay_prefix restores by replaying the trunk's recorded
    calls in this sample's container; run_tool execs a continuation call in the same container. One container,
    kept open for the whole branch."""

    def __init__(self, recorded_calls, tools):
        self.calls, self.tools = recorded_calls, tools

    async def _exec(self, name, arguments):
        asm = ChatMessageAssistant(content="", tool_calls=[ToolCall(id="x", function=name, arguments=arguments or {})])
        res = (await execute_tools([asm], self.tools)).messages
        return " ".join(m.text for m in res if m.role == "tool")

    async def replay_prefix(self, trunk, cut):
        out = []
        for name, args in self.calls[:cut]:
            out.append((name, await self._exec(name, args)))
        return out

    async def run_tool(self, name, arguments):
        return await self._exec(name, arguments)


def branch_task(trunk, calls, cfg, task_name, cut, max_turns):
    env = ROOT / "envs" / "tiktok"
    task_dir = ROOT / "tasks" / "tiktok" / task_name

    @solver
    def branch_solver():
        async def solve(state, generate):
            from inspect_ai.model import get_model
            sbx = run.sandbox()
            for _ in range(run.READY_TIMEOUT // 3):
                if (await run.run_script(sbx, env / "ready.sh", 30)).success:
                    break
                await asyncio.sleep(3)
            if (task_dir / "setup.sh").exists():
                await sbx.exec(["bash", "-c", (task_dir / "setup.sh").read_text()], timeout=run.TOOL_TIMEOUT)
            await sbx.exec(["bash", "-c", side_config.make_setup(cfg)], timeout=run.TOOL_TIMEOUT)  # same secret
            tools = [bash(timeout=run.TOOL_TIMEOUT), text_editor(timeout=run.TOOL_TIMEOUT), run.submit()]
            model = get_model()
            backend = ReplayBranchBackend(model.api, DockerBranchSandbox(calls, tools), tools)
            branch = await backend.make_branch(trunk, model.api.sampling_client, cut=cut, max_turns=max_turns, index=1)
            res = {"made": branch is not None, "cut": cut, "trunk_turns": len(trunk.turns)}
            if branch is not None:
                res["prefix_ok"] = shared_prefix_ok(trunk, branch)
                res["branch_turns"] = len(branch.turns)
                res["cont_turns"] = len(branch.turns) - cut
                branch.passed, trunk.passed = True, False  # synthetic signal so the group trains
                try:
                    b = build([trunk, branch], dict(compute_rewards([trunk, branch])), allowed_tasks={trunk.task},
                              expect_weights_id=trunk.weights_id, group_key=branch_group_key)
                    res["trainable"] = branch.episode_id in {d.episode_id for d in b.datums}
                    res["kept_groups"] = sum(g.kept for g in b.groups)
                except Exception as e:
                    res["trainable"] = f"build error: {type(e).__name__}: {e}"
            state.metadata["branch_smoke"] = res
            return state
        return solve

    return Task(name="branch-smoke", dataset=[Sample(id="smoke", input="(branch smoke)", metadata=dict(side=cfg))],
                solver=branch_solver(), sandbox=SandboxEnvironmentSpec("docker", str(env / "codebase" / "compose.yml")),
                epochs=1, time_limit=run.SAMPLE_TIME_LIMIT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk-run", default="smoke-trunk")
    ap.add_argument("--model-path", required=True, help="tinker:// sampler checkpoint for the branch policy")
    ap.add_argument("--base-model", default="openai/gpt-oss-120b")
    ap.add_argument("--cut", type=int, default=0, help="0 = default_cut(trunk); else an explicit early cut")
    ap.add_argument("--max-turns", type=int, default=6)
    args = ap.parse_args()
    run_attack.load_dotenv(ROOT / ".env")
    trunk, calls, cfg, task_name = load_trunk(args.trunk_run)
    cut = args.cut or min(3, len(trunk.turns) - 1)
    print(f"trunk: task={task_name} turns={len(trunk.turns)} calls={len(calls)} secret={cfg.get('secret_name')} cut={cut}", flush=True)
    model = make_model("branch-smoke", args.base_model, max_tokens=4096, mode="train", model_path=args.model_path)
    logs = asyncio.run(eval_async(branch_task(trunk, calls, cfg, task_name, cut, args.max_turns), model=model,
                                  log_dir=str(ROOT / "logs" / "branch-smoke"), max_sandboxes=1, fail_on_error=False))
    for s in (logs[0].samples or []):
        print("RESULT:", json.dumps((s.metadata or {}).get("branch_smoke", {"error": str(s.error) if s.error else "no result"}), indent=1))


if __name__ == "__main__":
    main()
