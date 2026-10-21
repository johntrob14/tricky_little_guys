"""Offline tests for rl.branch_backend.ReplayBranchBackend: no docker, no Tinker. A fake sandbox restores the
prefix and runs continuation tools; the fake sampler (rl.fake_sampler) supplies the resampled completions. We
check that the first resampled turn is seeded as a trainable 'chain' continuation (its prompt equals the trunk
prefix token-for-token), that datums.build accepts the assembled branch, and that environment drift drops it."""

import asyncio

from fakes import SCORER  # noqa: F401  (ensure rl import path set up by conftest)
from inspect_ai.model import ChatMessageSystem, ChatMessageTool, ChatMessageUser, GenerateConfig, get_model
from inspect_ai.tool import tool
from tinker_cookbook.tokenizer_utils import get_tokenizer

from rl.branch import branch_prefix, default_cut, make_branch_episode, shared_prefix_ok
from rl.branch_backend import ReplayBranchBackend, _tools_sig
from rl.datums import build
from rl.fake_sampler import FakeSamplingClient, call
from rl.provider import TinkerRLModel, load_sidecar  # noqa: F401  (registers the tinker-rl model api)
from rl.rewards import branch_group_key, compute_rewards
from rl.schema import Turn, episode_from_sidecar

MODEL = "tinker-rl/openai/gpt-oss-120b"
TOK = get_tokenizer("openai/gpt-oss-120b")
RUN = lambda coro: asyncio.run(coro)


@tool
def bash():
    async def execute(command: str) -> str:
        """Run a command.

        Args:
            command: The command.
        """
        return f"ran {command}"
    return execute


@tool
def submit():
    async def execute(answer: str = "") -> str:
        """Finish the task.

        Args:
            answer: The answer.
        """
        return "Submitted."
    return execute


TOOLS = [bash(), submit()]
CFG = GenerateConfig(max_tokens=512)
# trunk: 4 bash turns then submit. results[k] is the tool result that followed turn k (k<4).
TRUNK_SCRIPT = [call("look", "bash", '{"command":"ls"}'), call("peek", "bash", '{"command":"cat a"}'),
                call("edit", "bash", '{"command":"echo hi"}'), call("again", "bash", '{"command":"ls -l"}'),
                call("done", "submit", None)]
TRUNK_RESULTS = ["a b c\n", "contents\n", "ok\n", "total 0\n"]


class FakeSandbox:
    """Scripted environment reset. `prefix` is the (name, result) reproduced per prefix call (what replay would
    yield from the restored container); `cont` is the result of each resampled tool call, in order."""

    def __init__(self, prefix: list[tuple[str, str]], cont: list[str]):
        self.prefix, self.cont = list(prefix), list(cont)
        self.ran: list[tuple[str, dict]] = []

    async def replay_prefix(self, trunk, cut):
        return self.prefix[:cut]

    async def run_tool(self, name, arguments):
        self.ran.append((name, arguments))
        return self.cont.pop(0)


def _make_trunk(tmp_path):
    """Roll a realistic trunk through the real provider + fake sampler, convert its sidecar to a schema Episode."""
    fake = FakeSamplingClient(TOK, list(TRUNK_SCRIPT))
    model = get_model(MODEL, config=CFG, memoize=False, sampling_client=fake, out_dir=str(tmp_path / "trunk"), run="trunk")
    api = model.api
    tdefs = RUN(_resolve(model))
    msgs = [ChatMessageSystem(content="You are an engineer."), ChatMessageUser(content="Fix the bug.")]

    async def go():
        out = None
        for i in range(len(TRUNK_SCRIPT)):
            out = await api.generate(msgs, tdefs, "auto", CFG)
            msgs.append(out.message)
            c = out.message.tool_calls[0] if out.message.tool_calls else None
            if c is None:
                break
            if c.function == "submit":
                msgs.append(ChatMessageTool(content="Submitted.", tool_call_id=c.id, function="submit"))
                break
            msgs.append(ChatMessageTool(content=TRUNK_RESULTS[i], tool_call_id=c.id, function=c.function))
        return out

    out = RUN(go())
    doc = api.episodes[out.metadata["rl"]["episode_id"]].doc
    trunk = episode_from_sidecar(doc, step=0, split="train", phase="train")
    trunk.task = "tiktok"
    return trunk


async def _resolve(model):
    from inspect_ai.model._call_tools import prepare_tools
    return (await prepare_tools(TOOLS)).tdefs


def _branch_backend(tmp_path, sandbox, live):
    model = get_model(MODEL, config=CFG, memoize=False, sampling_client=live, out_dir=str(tmp_path / "branch"), run="branch")
    return ReplayBranchBackend(model.api, sandbox, TOOLS, config=CFG), model


def test_seed_makes_a_chain_continuation_matching_the_trunk_prefix(tmp_path):
    trunk = _make_trunk(tmp_path)
    assert len(trunk.turns) == 5
    cut = default_cut(trunk, from_end=2)
    assert cut == 3
    prefix = branch_prefix(trunk, cut)

    # happy path: the restored environment reproduces the trunk's prefix results exactly.
    sandbox = FakeSandbox(prefix=[("bash", TRUNK_RESULTS[k]) for k in range(cut)], cont=["branch out\n"])
    live = FakeSamplingClient(TOK, [call("resample", "bash", '{"command":"pwd"}'), call("finish", "submit", None)])
    backend, model = _branch_backend(tmp_path, sandbox, live)

    cont, boundary, cut_source = RUN(backend._sample(trunk, cut, prefix, live, 8))
    # the seed made turn `cut` a trainable chain turn, and its prompt is the trunk prefix token-for-token
    assert cut_source == "chain"
    assert live.calls[0][0] == prefix.tokens
    assert boundary == trunk.turns[cut - 1].obs_tokens  # reproduced obs == recorded obs (no drift)
    assert len(cont) == 2 and sandbox.ran == [("bash", {"command": "pwd"})]
    # the seeded episode was released after sampling (no leak in provider.episodes)
    assert model.api.episodes == {}
    # _tools_sig is deterministic over the resolved tools
    assert _tools_sig(RUN(_resolve(model))) == _tools_sig(RUN(_resolve(model)))


def test_make_branch_is_trainable_and_groups_with_trunk(tmp_path):
    trunk = _make_trunk(tmp_path)
    cut = default_cut(trunk, from_end=2)
    sandbox = FakeSandbox(prefix=[("bash", TRUNK_RESULTS[k]) for k in range(cut)], cont=["branch out\n"])
    live = FakeSamplingClient(TOK, [call("resample", "bash", '{"command":"pwd"}'), call("finish", "submit", None)])
    backend, model = _branch_backend(tmp_path, sandbox, live)

    branch = RUN(backend.make_branch(trunk, live, cut=cut, index=1, weights_id=trunk.weights_id))
    assert branch is not None
    assert branch.branch_of == trunk.episode_id and branch.cut_turn == cut
    assert shared_prefix_ok(trunk, branch)
    # the branch's prompt_len chain is self-consistent (datums.check_tokens' invariant)
    length = len(branch.prompt_tokens)
    for t in branch.turns:
        assert t.prompt_len == length
        length += len(t.tokens) + len(t.obs_tokens)

    # a sibling branch so the contrastive group has reward signal; trunk + its branches are ONE group
    rng_cont = [Turn(prompt_len=0, tokens=TOK.encode(call("other", "submit", None), add_special_tokens=False),
                     logprobs=[-0.5] * len(TOK.encode(call("other", "submit", None), add_special_tokens=False)),
                     stop_reason="stop", obs_tokens=[])]
    sibling = make_branch_episode(trunk, cut, rng_cont, index=2, weights_id=trunk.weights_id)
    trunk.passed, branch.passed, sibling.passed = True, True, False
    eps = [trunk, branch, sibling]
    b = build(eps, compute_rewards(eps), allowed_tasks={trunk.task}, group_key=branch_group_key)
    ids = {d.episode_id for d in b.datums}
    assert branch.episode_id in ids and sibling.episode_id in ids  # both branches train
    assert sum(g.kept for g in b.groups) == 1
    assert {s.group for s in b.groups} == {(trunk.step, trunk.task, trunk.episode_id)}


def test_environment_drift_drops_the_branch(tmp_path):
    trunk = _make_trunk(tmp_path)
    cut = default_cut(trunk, from_end=2)
    # the restored environment reproduces a DIFFERENT result for the observation leading into the cut: drift.
    bad = [("bash", TRUNK_RESULTS[k]) for k in range(cut)]
    bad[cut - 1] = ("bash", "DRIFTED: ready in 1032ms\n")
    sandbox = FakeSandbox(prefix=bad, cont=["branch out\n"])
    live = FakeSamplingClient(TOK, [call("resample", "bash", '{"command":"pwd"}'), call("finish", "submit", None)])
    backend, _ = _branch_backend(tmp_path, sandbox, live)
    branch = RUN(backend.make_branch(trunk, live, cut=cut, index=1, weights_id=trunk.weights_id))
    assert branch is None


def test_restore_failure_returns_none(tmp_path):
    trunk = _make_trunk(tmp_path)
    cut = default_cut(trunk, from_end=2)
    # replay ended before the cut (only cut-1 calls restored): cannot branch.
    sandbox = FakeSandbox(prefix=[("bash", TRUNK_RESULTS[k]) for k in range(cut - 1)], cont=[])
    live = FakeSamplingClient(TOK, [call("resample", "bash", '{"command":"pwd"}')])
    backend, _ = _branch_backend(tmp_path, sandbox, live)
    assert RUN(backend.make_branch(trunk, live, cut=cut, index=1)) is None
