"""Offline stand-ins for the dry run (rl.py --fake): a scripted gpt-oss "policy" that emits Harmony turns (reasoning, a bash call, a submit), a fake
training service, and a docker-free task (fake tools, fake scorer) so the whole loop runs without Tinker or a sandbox. Everything made here is test data.

The policy decides at turn 0, from a seeded rng, whether it will "fix" the bug (marker [plan:fix] in its reasoning) or not ([plan:skip]); later turns read the
marker back from the prompt, so the decision survives in token space like a real policy's.
"""

import random
import threading
from pathlib import Path
from types import SimpleNamespace

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, execute_tools, get_model
from inspect_ai.scorer import CORRECT, INCORRECT, Score, accuracy, scorer, stderr
from inspect_ai.solver import solver
from inspect_ai.tool import tool
from inspect_ai.util import store

from run import NUDGE, ROOT, SYSTEM

from .fake_sampler import analysis, call

FIX_CMD = "cd /app/src && sed -i 's/return NextResponse.json({ success: true });/return NextResponse.json(newGuess);/' src/app/api/guesses/route.ts && grep -n 'NextResponse.json' src/app/api/guesses/route.ts"
LOOK_CMD = "grep -n 'NextResponse.json' /app/src/src/app/api/guesses/route.ts"


class ScriptedPolicy:
    """Stands in for tinker.SamplingClient. p_fix maps a task directory name to the probability that an episode takes the fixing path; the task is recognised by
    its task.md text in the prompt. `calls` records (prompt_ids, params) like fake_sampler.FakeSamplingClient."""

    def __init__(self, tokenizer, p_fix: dict[str, float] | None = None, default_p: float = 0.5, seed: int = 0, fix_cmd: str = FIX_CMD):
        self.tok, self.p_fix, self.default_p, self.fix_cmd = tokenizer, p_fix or {}, default_p, fix_cmd
        self.rng, self.lock, self.calls = random.Random(seed), threading.Lock(), []
        self.tasks = {p.name: (p / "task.md").read_text().strip()[:300] for p in (ROOT / "tasks").glob("*/*") if (p / "task.md").exists()}

    @staticmethod
    def logprob(i, token):
        return -((token * 31 + i) % 17 + 1) / 50

    def _completion(self, prompt_text: str) -> str:
        turn = prompt_text.count("<|call|>")
        if turn == 0:
            name = next((n for n, head in self.tasks.items() if head in prompt_text), None)
            with self.lock:
                fix = self.rng.random() < self.p_fix.get(name, self.default_p)
            return call(f"First I will look at how the route builds its response. [plan:{'fix' if fix else 'skip'}]", "bash", '{"command": "%s"}' % LOOK_CMD)
        fix = "[plan:fix]" in prompt_text
        if turn == 1:
            cmd = self.fix_cmd if fix else "echo nothing else to change"
            return call("The response body is wrong; " + ("restore it." if fix else "I will leave it as is."), "bash", '{"command": "%s"}' % cmd.replace('"', '\\"'))
        return call("I am done.", "submit", '{"answer": ""}')

    async def sample_async(self, prompt, num_samples, sampling_params):
        ids = prompt.to_ints()
        self.calls.append((ids, sampling_params))
        text = self._completion(self.tok.decode(ids))
        tokens = self.tok.encode(text, add_special_tokens=False)[: sampling_params.max_tokens]
        seq = SimpleNamespace(tokens=tokens, logprobs=[self.logprob(i, t) for i, t in enumerate(tokens)], stop_reason="stop")
        return SimpleNamespace(sequences=[seq], prompt_cache_hit_tokens=0)


class _TD:
    def __init__(self, xs):
        self.xs = list(xs)

    def tolist(self):
        return self.xs


class _Out:
    def __init__(self, **kw):
        self.metrics, self.loss_fn_outputs, self.path = {}, [], ""
        self.__dict__.update(kw)


class _Fut:
    def __init__(self, v):
        self.v = v

    async def result_async(self):
        return self.v


class FakeTrainingClient:
    """The training-client calls rl.train uses, with plausible numbers (trainer logprobs = sampler logprobs - 0.002 where trained). Weights are a counter."""

    def __init__(self, policy: ScriptedPolicy, n_optim: int = 0):
        self.policy, self.n_optim, self.saved = policy, n_optim, []

    def _outs(self, data):
        return [{"logprobs": _TD([x - (0.002 if a != 0 else 0.0) for x, a in zip(d.loss_fn_inputs["logprobs"].tolist(), d.loss_fn_inputs["advantages"].tolist())])} for d in data]

    async def forward_async(self, data, loss_fn, loss_fn_config=None):
        return _Fut(_Out(loss_fn_outputs=self._outs(data)))

    async def forward_backward_async(self, data, loss_fn, loss_fn_config=None):
        assert loss_fn == "importance_sampling"
        return _Fut(_Out(metrics={"loss:sum": -sum(sum(d.loss_fn_inputs["advantages"].tolist()) for d in data)}, loss_fn_outputs=self._outs(data)))

    async def optim_step_async(self, optim_params=None, **kw):
        self.n_optim += 1
        return _Fut(_Out())

    def save_weights_and_get_sampling_client(self, name=None, **kw):
        return self.policy

    async def _save(self, kind, name, ttl_seconds):
        self.saved.append((kind, name, ttl_seconds))
        return _Fut(_Out(path=f"fake://{kind}/{name}"))

    async def save_state_async(self, name, ttl_seconds=None, overwrite=False, **kw):
        return await self._save("state", name, ttl_seconds)

    async def save_weights_for_sampler_async(self, name, ttl_seconds=None, **kw):
        return await self._save("sampler", name, ttl_seconds)


class FakeService:
    def __init__(self, policy: ScriptedPolicy):
        self.policy = policy

    async def create_lora_training_client_async(self, base_model, rank=32, seed=None, **kw):
        return FakeTrainingClient(self.policy)

    async def create_training_client_from_state_with_optimizer_async(self, path, **kw):
        return FakeTrainingClient(self.policy)


@tool
def bash():
    async def execute(command: str) -> str:
        """Run a bash command.

        Args:
            command: The command.
        """
        if "sed -i" in command:
            store().set("fixed", True)
            return "34:    return NextResponse.json(newGuess);"
        return "34:    return NextResponse.json({ success: true });" if "grep" in command else "ok"
    return execute


@tool
def submit():
    async def execute(answer: str | int | float = "") -> str:
        """Finish the task.

        Args:
            answer: The answer, when the task asks a question.
        """
        return "Submitted."
    return execute


@solver
def fake_agent(max_turns: int):
    """run.agent's loop (same prompts, nudge and stop rules) over the fake tools and no sandbox."""
    async def solve(state, generate):
        tools = [bash(), submit()]
        state.messages = [ChatMessageSystem(content=SYSTEM), ChatMessageUser(content=state.input_text)]
        nudged = False
        for _ in range(max_turns):
            state.output = await get_model().generate(state.messages, tools)
            state.messages.append(state.output.message)
            if not state.output.message.tool_calls:
                if nudged or state.output.stop_reason == "model_length":
                    break
                nudged = True
                state.messages.append(ChatMessageUser(content=NUDGE))
                continue
            results = (await execute_tools(state.messages, tools)).messages
            state.messages.extend(results)
            if any(m.function == "submit" and m.error is None for m in results):
                break
        return state
    return solve


@scorer(metrics=[accuracy(), stderr()])
def task_scorer():
    async def score(state, target):
        ok = bool(state.store.get("fixed"))
        return Score(value=CORRECT if ok else INCORRECT, explanation="fake scorer: the fixing command ran" if ok else "fake scorer: no fix applied")
    return score


def fake_build_task(task: Path, max_turns: int, epochs: int) -> Task:
    sample = Sample(id=task.name, input=(task / "task.md").read_text(), metadata=dict(env=task.parent.name, task=task.name))
    return Task(name=f"fake-{task.parent.name}-{task.name}", dataset=[sample], solver=fake_agent(max_turns), scorer=task_scorer(), epochs=epochs)
