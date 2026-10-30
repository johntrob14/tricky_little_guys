"""The loop's `collect`: Inspect rollouts of run.py's agent and scorer through the tinker-rl provider, INFRA failures rerun, sidecars converted to Episodes.

    collector = Collector(run="r1", model="openai/gpt-oss-120b", max_turns=60, max_tokens=4096)
    episodes = await collector(CollectRequest(...))        # run_loop(cfg, svc, collector, budget=...)

Per call: the provider gets req.sampler / req.weights_id, sidecars go to <req.out_dir>/sidecars, Inspect logs to logs/<run>/step_NNN/aK*. Each (task, epoch) is
one Inspect sample in its own sandbox. After a batch every sample is classified (classify.classify_sample): INFRA samples are rerun from scratch in a new
sandbox (RetryPolicy: at most 2 reruns, backoff 5/15/45 s) with their sidecar cost ledgered as wasted; AGENT failures and wrong answers keep reward 0; a HALT class
raises classify.Halt. Each rerun first reserves req.ep_worst_usd per rerun episode on req.budget (held until the call returns); when the budget cannot
cover it, no rerun is made. A sample still INFRA after the reruns stays a reward-None Episode (a stub with no turns when nothing was sampled); its own
usage is its cost, so it is not also counted in wasted_usd. Episode.task is "<env>/<task>".
"""

import asyncio
import glob
import os
from pathlib import Path

from inspect_ai import eval_async

from run import ROOT, build_task
from sandbox_cap import capped

from .budget import BudgetExceeded, sample_cost_usd
from .classify import HALT, INFRA, Failure, Halt, Ledger, RetryPolicy, budget_hook, classify_sample
from .loop import CollectRequest
from .provider import load_sidecar
from .run_rl_eval import make_model, select_tasks
from .schema import Episode, episode_from_sidecar


def ensure_hf_cache(model: str = "openai/gpt-oss-120b") -> None:
    """The tokenizer is read from the HF cache; when HF_HOME has no copy, use ~/.cache/huggingface or a ~/offload/*/huggingface that has it."""
    name = "models--" + model.replace("/", "--")
    if (Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub" / name).exists():
        return
    for home in [Path.home() / ".cache/huggingface", *map(Path, sorted(glob.glob(str(Path.home() / "offload/*/huggingface"))))]:
        if (home / "hub" / name).exists():
            os.environ["HF_HOME"] = str(home)
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            return


def sidecar_usd(doc: dict | None, model: str) -> float:
    """No-cache upper bound of what a sidecar's sampler calls cost."""
    if not doc:
        return 0.0
    return sample_cost_usd(model, sum(t["prompt_len"] for t in doc["turns"]), 0, sum(len(t["sampled_ids"]) for t in doc["turns"]))


def task_name(sample) -> str:
    md = sample.metadata or {}
    return f"{md.get('env')}/{md.get('task')}" if md.get("env") else str(sample.id)


def read_side(sample) -> dict | None:
    """The side-aware scorer (rl.side_scorer) stores {side_done, final_reveal, opportunity_present, ...} under Score.metadata['side']."""
    for sc in (sample.scores or {}).values():
        md = sc.metadata or {}
        if isinstance(md.get("side"), dict):
            return md["side"]
    return None


class Collector:
    """build: run.build_task (Ethan's), rl.guard.guarded_build_task, or rl.fake_policy.fake_build_task."""

    def __init__(self, run: str, model: str = "openai/gpt-oss-120b", *, max_turns: int = 60, max_tokens: int = 4096, max_sandboxes: int = 6, build=build_task,
                 phase: str = "train", split: str = "train", mode: str = "train", policy: RetryPolicy | None = None, sleep=asyncio.sleep, model_args: dict | None = None,
                 side_specs: dict | None = None, guard: bool = False):
        self.run, self.model_name, self.max_turns, self.max_tokens, self.max_sandboxes = run, model, max_turns, max_tokens, max_sandboxes
        self.build, self.phase, self.split, self.mode = build, phase, split, mode
        self.policy, self.sleep, self.model_args, self.model = policy or RetryPolicy(), sleep, model_args or {}, None
        # side_specs {task_name: SideSpec} opts a task into the model-organism side-task state check (off by default); guard uses the hardened inner scorer.
        self.side_specs, self.guard = side_specs or {}, guard

    def _model(self, req: CollectRequest):
        sidecars = Path(req.out_dir) / "sidecars"
        if self.model is None:
            ensure_hf_cache(self.model_name)
            self.model = make_model(self.run, self.model_name, self.max_tokens, self.mode, sampling_client=req.sampler, weights_id=req.weights_id, out_dir=str(sidecars), **self.model_args)
        self.model.api.set_sampling_client(req.sampler, weights_id=req.weights_id)
        self.model.api.out_dir = sidecars
        return self.model

    def _build_one(self, p, epochs):
        spec = self.side_specs.get(f"{p.parent.name}/{p.name}")
        if spec is not None:
            from .side_scorer import side_aware_build_task
            return side_aware_build_task(p, self.max_turns, epochs, spec, guarded=self.guard, build=self.build)  # the configured build (fake, guarded) is kept
        return self.build(p, self.max_turns, epochs)

    async def _eval(self, model, paths, epochs, sub, req, attempt):
        tasks = [self._build_one(p, epochs) for p in paths]
        return await eval_async(tasks, model=model, log_dir=str(ROOT / "logs" / sub), max_tasks=len(tasks), max_sandboxes=capped(self.max_sandboxes), log_buffer=1, fail_on_error=False,
                                metadata=dict(run=self.run, model=str(model), max_turns=self.max_turns, weights_id=model.api.weights_id, step=req.step, phase=self.phase, attempt=attempt))

    async def __call__(self, req: CollectRequest) -> list[Episode]:
        model = self._model(req)
        ledger = Ledger(on_wasted=budget_hook(req.budget.ledger, req.budget.stage, req.budget.run, req.budget.model))
        pending = {t: req.n for t in req.tasks}
        done: dict[str, list[Episode]] = {t: [] for t in req.tasks}
        wasted = dict.fromkeys(req.tasks, 0.0)
        lost: dict[str, list] = {t: [] for t in req.tasks}  # (episode, sidecar usd) of the latest attempt's INFRA samples
        try:
            return await self._collect(req, model, ledger, pending, done, wasted, lost)
        finally:
            req.budget.release(self._rerun_reserved)

    async def _collect(self, req, model, ledger, pending, done, wasted, lost) -> list[Episode]:
        self._rerun_reserved, skipped = 0.0, None
        for attempt in range(1, self.policy.max_infra_retries + 2):
            groups: dict[int, list[str]] = {}
            for t, k in pending.items():
                if k:
                    groups.setdefault(k, []).append(t)
            if not groups:
                break
            if attempt > 1:
                w = sum(pending.values()) * req.ep_worst_usd
                try:  # a rerun is new spend the step's reservation does not cover
                    req.budget.reserve(w)
                except BudgetExceeded as e:
                    skipped = f"rerun skipped: {e}"
                    break
                self._rerun_reserved += w
                await self.sleep(self.policy.backoff[min(attempt - 2, len(self.policy.backoff) - 1)])
            for gi, (k, names) in enumerate(sorted(groups.items())):
                logs = await self._eval(model, [select_tasks([t])[0] for t in names], k, f"{self.run}/step_{req.step:03d}/a{attempt}g{gi}", req, attempt)
                got = dict.fromkeys(names, 0)
                for t in names:
                    lost[t] = []
                for log in logs:
                    for s in log.samples or []:
                        t = task_name(s)
                        if t not in got:
                            continue
                        side = Path(model.api.out_dir) / f"{s.uuid}.json.gz"
                        doc = load_sidecar(side) if side.exists() else None
                        failure = classify_sample(s)
                        if failure is None and not s.scores:
                            failure = Failure(INFRA, "no_score")
                        if failure is not None and failure.failure_class == HALT:
                            raise Halt(f"{failure.reason}: {failure.detail[:300]}")
                        model.api.release(str(s.uuid))
                        if failure is not None and failure.failure_class == INFRA:
                            usd = sidecar_usd(doc, self.model_name)
                            ledger.add(attempt, failure, usd)
                            wasted[t] += usd
                            lost[t].append((self._episode(s, doc, failure, req, t, attempt - 1, log.location), usd))
                        else:
                            done[t].append(self._episode(s, doc, failure, req, t, attempt - 1, log.location))
                            got[t] += 1
                for t in names:
                    pending[t] = k - got[t]
        out = []
        for t in req.tasks:
            kept = lost[t][: pending[t]]  # still-INFRA samples after the reruns stay as reward-None episodes, charged by their own usage
            eps = done[t] + [e for e, _ in kept]
            eps += [self._episode(None, None, Failure(INFRA, "episode_missing"), req, t, self.policy.max_infra_retries, None) for _ in range(pending[t] - len(kept))]
            if skipped:
                for e in eps[len(done[t]):]:
                    e.failure_reason = f"{e.failure_reason}; {skipped}"[:500]
            if eps:
                eps[0].wasted_usd = max(0.0, wasted[t] - sum(u for _, u in kept))
            out += eps
        return out

    def _episode(self, s, doc, failure, req, task, attempt, loc) -> Episode:
        fc, fr = (failure.failure_class, failure.reason) if failure else ("none", "")
        side = read_side(s) if s else None
        if doc is None or not doc["turns"]:  # nothing was sampled (sandbox or setup failed first): a stub with no turns, reward None
            return Episode(episode_id=str(s.uuid) if s else f"stub-{req.step}-{task}-{os.urandom(3).hex()}", sample_id=str(s.id) if s else task.split("/")[-1], task=task,
                           epoch=s.epoch if s else 0, step=req.step, weights_id=req.weights_id, prompt_tokens=[], turns=[], run=self.run, split=self.split, phase=self.phase,
                           model=self.model_name, ended="infra", failure_class=fc if fc != "none" else INFRA, failure_reason=fr or "no_sidecar", attempt=attempt, log_path=loc)
        ended = "wall" if s.limit is not None and s.limit.type == "time" else None
        ep = episode_from_sidecar(doc, step=req.step, split=self.split, phase=self.phase, log_path=loc, ended=ended, failure_class=fc, failure_reason=fr, attempt=attempt,
                                  weights_id=req.weights_id)
        ep.task, ep.sample_id, ep.epoch = task, str(s.id), s.epoch
        if side is not None:
            ep.side_done, ep.final_reveal, ep.opportunity_present = side.get("side_done"), side.get("final_reveal"), side.get("opportunity_present")
        elif task in self.side_specs:  # a spec'd task whose score carries no side result: unevaluated, so rl.rewards drops it rather than scoring side 0
            ep.opportunity_present = self.side_specs[task].opportunity_present
        return ep
