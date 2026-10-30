"""Branch-sampling collect for the RL loop (the BranPO fast path).

Each step's per-task group of G episodes becomes **1 trunk (a full rollout) + (G-1) branches** resampled from
the trunk's late prefix via rl.branch_backend.ReplayBranchBackend. The trunk and its branches share
branch_of=trunk.id, so rl.rewards.branch_group_key puts them in one contrastive advantage group (the loop
passes that group_key when cfg.branch). Per step you pay for B full trunk rollouts + B*(G-1) short branches,
instead of B*G full rollouts.

Everything the live path needs is INJECTED, so the orchestration is offline-testable with fakes:
  - trunk_runner(task, req)  -> Episode            one full trunk rollout (LIVE: rl.collect.Collector, n=1)
  - make_backend(trunk)      -> BranchBackend       a backend bound to a reset for THAT trunk (LIVE:
        ReplayBranchBackend with a docker BranchSandbox restored from the trunk's recorded calls, rl.replay)
  - score_branch(branch)     -> awaitable None       fill the branch's reward/side fields (LIVE: the main+side
        scorer on the branch's final container). Trunks are already scored by trunk_runner.

The offline test (tests/rl/test_branch_collect.py) drives this with fakes and asserts the per-step structure,
the trunk+branches grouping, and that the result trains through datums.build. The LIVE adapters
(`DockerBranchSandbox`, `live_branch_collector`) are concrete but have NOT been run live yet -- they need a real
container; that smoke is the remaining step before a branch RL run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from .branch import default_cut
from .loop import CollectRequest
from .schema import INFRA, Episode


@dataclass
class BranchCollector:
    """A loop `collect` callable (CollectRequest -> list[Episode]) that produces trunk + branches per task."""
    trunk_runner: Callable[[str, CollectRequest], Awaitable[Episode]]
    make_backend: Callable[[Episode], object]          # -> rl.branch.BranchBackend for this trunk
    score_branch: Callable[[Episode], Awaitable[None]]
    max_turns: int = 60
    from_end: int = 2                                   # resample the last `from_end` turns
    min_prefix: int = 1

    async def __call__(self, req: CollectRequest) -> list[Episode]:
        out: list[Episode] = []
        for task in req.tasks:
            trunk = await self.trunk_runner(task, req)
            trunk.branch_of = ""                        # a trunk heads its own group
            out.append(trunk)
            # A failed or too-short trunk cannot be branched; its group is just the trunk (dropped as <2 valid).
            if trunk.failure_class == INFRA or len(trunk.turns) < 2:
                continue
            cut = default_cut(trunk, from_end=self.from_end, min_prefix=self.min_prefix)
            backend = self.make_backend(trunk)
            for i in range(1, req.n):
                branch = await backend.make_branch(trunk, req.sampler, cut=cut, max_turns=self.max_turns,
                                                   index=i, weights_id=req.weights_id)
                if branch is None:
                    continue                            # drift / could not restore -> group is just smaller
                branch.step = req.step
                await self.score_branch(branch)
                out.append(branch)
        return out


# ---- LIVE adapters (docker + Tinker; concrete but not yet run live) ------------------------------------------

class DockerBranchSandbox:
    """LIVE rl.branch_backend.BranchSandbox over a real container, reusing rl.replay. NOT YET SMOKED.

    `replay_prefix` brings a fresh container to the trunk's state at `cut` by re-seeding the same secret and
    replaying the trunk's recorded tool calls for turns 0..cut-1 through run.py's own tools (rl.replay machinery),
    returning the (tool, result) each produced; `run_tool` executes one resampled call in that same container.
    The container is kept open after replay so the continuation runs against the restored state, and so the
    main+side scorer can score the branch's final state. One instance per trunk.
    """

    def __init__(self, script, env_dir, task_dir):
        # script: rl.replay.TrunkScript for this trunk (extract_script on the trunk's Inspect log); env/task dirs
        # under arena/. The actual container lifecycle reuses rl.replay's solver; wired in live_branch_collector.
        self.script, self.env_dir, self.task_dir = script, env_dir, task_dir
        raise NotImplementedError(
            "DockerBranchSandbox is the live docker reset; wire it in live_branch_collector and smoke it on one "
            "real tiktok container before a branch RL run (restore-to-cut + continuation exec + branch scoring).")

    async def replay_prefix(self, trunk, cut):  # pragma: no cover - live
        ...

    async def run_tool(self, name, arguments):  # pragma: no cover - live
        ...


def live_branch_collector(*args, **kwargs):  # pragma: no cover - live
    """Build a BranchCollector wired to the live Collector (trunks), ReplayBranchBackend + DockerBranchSandbox
    (branches), and the run_attack main+side scorer (branch scoring). Not yet implemented end to end -- the
    DockerBranchSandbox restore/exec and the per-branch scoring are the remaining live seam."""
    raise NotImplementedError(
        "live_branch_collector: fill DockerBranchSandbox (restore-to-cut via rl.replay, keep the container) and "
        "branch scoring (run_attack.main_and_side_scorer on the branch container), then smoke on one container.")
