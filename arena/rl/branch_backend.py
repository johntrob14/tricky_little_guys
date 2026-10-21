"""Replay branch backend: the concrete rl.branch.BranchBackend that turns a trunk + cut into a trainable branch.

A branch resamples the last few turns of a trunk from a shared late prefix (BranPO, see rl/branch.py). This
backend does the three things rl/branch.py's token math cannot do by itself, with both the environment reset and
the sampler INJECTED so the whole thing is testable with no docker and no Tinker:

  1. RESTORE: a fresh environment is brought back to the trunk's state at `cut` by replaying the trunk's recorded
     tool calls for turns 0..cut-1 (the real reset is rl/replay.py's docker `_replay_solver`; here it is the
     injected `sandbox`). Restore returns, per replayed call, the (tool_name, result_text) it actually produced,
     so the backend can tell whether the environment reproduced the prefix.

  2. SEED: the rollout provider (rl/provider.py, TinkerRLModel) is made to treat the FIRST resampled turn (turn
     `cut`) as a "chain" continuation of the trunk prefix, not a "rerender". A rerender is non-trainable and
     datums.build drops it, so this is the part that must be exact. See `_seed_episode` for the mechanism.

  3. SAMPLE: turns `cut`, `cut+1`, ... are sampled through the provider's normal generate path with the injected
     `sampler`, each tool call executed in the restored `sandbox`, until submit / max_turns / a context clamp.

`make_branch` then assembles the branch with rl.branch.make_branch_episode and returns it, or None on
environment drift. `sample_continuation` is the Protocol-level primitive (steps 1-3, returns the continuation
Turns); `make_branch` wraps it with the assembly + drift gate (step 4) and is what the loop calls.

CHAIN SEEDING (how turn `cut` becomes a trainable "chain" turn)
---------------------------------------------------------------
provider.generate finds the episode via the `rl` key on the last assistant message (the anchor) and decides
"chain" vs "rerender" by four checks against the provider's in-memory `self.episodes[ep_id]` (provider.py ~L244):
the recorded last turn index must equal the anchor's `turn`, the anchor's position in `input` must equal the
episode's `n_input`, the tools signature must match, and the post-anchor messages must all be tool/user. When all
hold the turn's prompt is `ep.chain_ids + render(delta)`; otherwise it is re-rendered and flagged chain_break.

So to resume a trunk at turn `cut` we pre-populate `self.episodes[ep_id]` and hand generate a two-message input
`[anchor_assistant, cut-1_tool_result]`:

  * ep.chain_ids = the trunk's prefix token stream up to (but NOT including) the observation that leads into turn
    `cut`. That trailing observation is turn cut-1's tool result; the provider re-renders it from the restored
    sandbox's result and appends it, so `ep.chain_ids + render(result) == prefix.tokens` exactly (verified in
    tests: the first sampler prompt equals branch_prefix(trunk, cut).tokens). Seeding chain_ids from the real
    prefix tokens is what makes the LIVE sampler see the true long-context prefix -- not just a token bookkeeping
    nicety, it is required for the resampled turns to be on-distribution.
  * ep.doc["turns"] holds cut placeholder turn records (indices 0..cut-1) copied from the trunk, so the chain
    check's "last recorded index == anchor turn (cut-1)" passes and the provider appends turn `cut` at index cut.
  * ep.n_input = 0 and the anchor sits at input index 0, so the "anchor position == n_input" check passes; from
    turn cut+1 on the provider maintains this invariant itself as the message list grows.
  * ep.tools_sig = the provider's tools signature. This is the ONE place this module reproduces a provider
    internal (the sha256 over [name, description, parameters] in provider.generate). It is reproduced in
    `_tools_sig` from the SAME resolved ToolInfo list the provider is handed, and the tests assert the seed
    produced a "chain" turn, so a drift in the provider's formula fails loudly rather than silently producing a
    non-trainable rerender. A one-line provider helper (e.g. `TinkerRLModel.tools_signature(tools)` /
    `seed_episode(...)`) would remove even this coupling; it is not required for correctness.

This was validated end to end offline (tests/rl/test_branch_backend.py): with a synthetic sandbox + fake sampler
the first resampled turn is a "chain" turn, its prompt equals the trunk prefix token-for-token, and the assembled
branch is accepted by datums.build as trainable. No rl/provider.py change was needed.

DRIFT
-----
The reset is only usable if the environment reproduced the prefix. The backend compares the observation the
provider actually rendered leading into turn `cut` (the restored sandbox's reproduction of turn cut-1's tool
result) against the trunk's recorded obs_tokens, via rl.branch.shared_prefix_ok on a reproduced-prefix episode;
a mismatch means drift and `make_branch` returns None (drop the branch rather than train a broken prefix). Note
make_branch_episode lays the branch's prefix down verbatim from the trunk, so shared_prefix_ok on the *assembled*
branch is a token-layout invariant (kept as a final assert); the drift that actually fires is the reproduced-obs
check against the restored environment.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, GenerateConfig
from inspect_ai.tool import ToolCall

from .branch import Prefix, branch_prefix, default_cut, make_branch_episode, shared_prefix_ok
from .provider import DEFAULT_MAX_TOKENS, Episode as ProviderEpisode, TinkerRLModel
from .schema import Episode, Turn


class BranchSandbox(Protocol):
    """The injected environment reset. The docker implementation wraps rl.replay (fresh container, re-seed the
    same secret, replay the recorded calls through run.py's own tools); tests pass a scripted fake."""

    async def replay_prefix(self, trunk: Episode, cut: int) -> list[tuple[str, str]]:
        """Restore a fresh environment to the trunk's state at `cut` by replaying the recorded tool calls for
        turns 0..cut-1, and return the (tool_name, result_text) actually produced for each. A short list (the
        replay ended before `cut`) means the prefix could not be restored."""
        ...

    async def run_tool(self, name: str, arguments: dict) -> str:
        """Execute one resampled (continuation) tool call in the restored environment, return its result text."""
        ...


def _tools_sig(tools) -> str:
    """Reproduce provider.generate's tools signature over the SAME resolved ToolInfo list the provider is handed.
    The single provider-internal coupling in this module (see the module docstring)."""
    return hashlib.sha256(
        json.dumps([[t.name, t.description, t.parameters.model_dump(exclude_none=True)] for t in tools],
                   sort_keys=True, default=str).encode()).hexdigest()[:16]


def _continuation_turns(sidecar_turns: list[dict], cut: int) -> list[Turn]:
    """The resampled turns (provider sidecar turns[cut:]) as schema Turns. Turn k's obs_tokens are the next
    turn's chain obs_ids (the tool result + '<|start|>assistant' that led into turn k+1), [] after the last turn;
    prompt_len is left 0 because make_branch_episode recomputes the whole chain from the shared prefix."""
    out: list[Turn] = []
    for k in range(cut, len(sidecar_turns)):
        t = sidecar_turns[k]
        nxt = sidecar_turns[k + 1] if k + 1 < len(sidecar_turns) else None
        obs = list(nxt["obs_ids"]) if nxt is not None and nxt["prompt_source"] == "chain" else []
        out.append(Turn(prompt_len=0, tokens=list(t["sampled_ids"]), logprobs=list(t["logprobs"]),
                        stop_reason=t["stop_reason"], obs_tokens=obs,
                        flags=sorted(f for f, v in (t.get("flags") or {}).items() if v)))
    return out


class ReplayBranchBackend:
    """Concrete rl.branch.BranchBackend. `provider` is a TinkerRLModel (mode='train'); `sandbox` is the injected
    environment reset; `tools` is the agent's tool list (same objects the trunk was rolled out with). Both the
    reset and the sampler are injected, so the whole backend runs offline with fakes."""

    def __init__(self, provider: TinkerRLModel, sandbox: BranchSandbox, tools, *,
                 config: GenerateConfig | None = None, continuation_weights_id: str = "injected"):
        self.provider = provider
        self.sandbox = sandbox
        self.tools = tools
        self.config = config or GenerateConfig(max_tokens=DEFAULT_MAX_TOKENS)
        self.continuation_weights_id = continuation_weights_id

    async def _resolved_tools(self):
        """The ToolInfo list the provider is handed (what Inspect's Model.generate would pass to api.generate)."""
        from inspect_ai.model._call_tools import prepare_tools
        return (await prepare_tools(self.tools)).tdefs

    def _seed_episode(self, ep_id: str, trunk: Episode, cut: int, prefix: Prefix, tdefs, sig: str) -> None:
        """Pre-populate provider.episodes[ep_id] so the next generate for turn `cut` is a 'chain' turn. See the
        module docstring for why each field is set the way it is."""
        api = self.provider
        doc = api._new_doc(ep_id, None, self.config, tdefs, sig)
        length = len(trunk.prompt_tokens)
        for k in range(cut):
            tk = trunk.turns[k]
            doc["turns"].append(dict(
                index=k, prompt_source="initial" if k == 0 else "chain", prompt_len=length, obs_ids=[],
                observations=[], sampled_ids=list(tk.tokens), logprobs=list(tk.logprobs), stop_reason=tk.stop_reason,
                max_tokens=0, room=0, weights_id=self.continuation_weights_id, raw_text="", reasoning="", text="",
                tool_call=None, flags={}, message_key=dict(ep=ep_id, turn=k)))
            length += len(tk.tokens) + len(tk.obs_tokens)
        # chain_ids = prefix stream up to (not including) the observation leading into turn `cut`; the provider
        # re-renders that observation from the restored result so chain_ids + render(result) == prefix.tokens.
        boundary = len(trunk.turns[cut - 1].obs_tokens)
        chain_ids = list(prefix.tokens[: prefix.length - boundary]) if boundary else list(prefix.tokens)
        tool_names = {t.name: bool(t.parameters.required) for t in tdefs}
        api.episodes[ep_id] = ProviderEpisode(ep_id, api.out_dir / f"{ep_id}.json.gz", doc, chain_ids, 0, sig, tool_names)

    async def _sample(self, trunk: Episode, cut: int, prefix: Prefix, sampler, max_turns: int):
        """Restore + seed + sample. Returns (continuation_turns, boundary_obs_tokens, cut_turn_source).
        boundary_obs_tokens is the observation the provider rendered leading into turn `cut` (what the drift gate
        checks); cut_turn_source is the provider's prompt_source for turn `cut` ('chain' when seeding worked)."""
        tdefs = await self._resolved_tools()
        sig = _tools_sig(tdefs)
        replayed = await self.sandbox.replay_prefix(trunk, cut)
        if len(replayed) < cut:
            return [], [], None  # the environment could not be restored to the cut -> no branch
        ep_id = f"br-seed-{uuid.uuid4().hex[:16]}"
        self._seed_episode(ep_id, trunk, cut, prefix, tdefs, sig)
        self.provider.set_sampling_client(sampler, weights_id=self.continuation_weights_id)
        boundary_name, boundary_result = replayed[cut - 1]
        anchor = ChatMessageAssistant(content="", metadata=dict(rl=dict(ep=ep_id, turn=cut - 1)),
                                      tool_calls=[ToolCall(id="br-anchor", function=boundary_name, arguments={})])
        msgs = [anchor, ChatMessageTool(content=boundary_result, tool_call_id="br-anchor", function=boundary_name)]
        try:
            for _ in range(max_turns):
                out = await self.provider.generate(msgs, tdefs, "auto", self.config)
                if out.stop_reason == "model_length":  # context clamp: no turn recorded, stop here
                    break
                msgs.append(out.message)
                calls = out.message.tool_calls
                if not calls:
                    break
                c = calls[0]
                if c.function == "submit":
                    break
                result = await self.sandbox.run_tool(c.function, c.arguments or {})
                msgs.append(ChatMessageTool(content=result, tool_call_id=c.id, function=c.function))
            doc_turns = self.provider.episodes[ep_id].doc["turns"]
            boundary_obs = list(doc_turns[cut]["obs_ids"]) if len(doc_turns) > cut else []
            cut_source = doc_turns[cut]["prompt_source"] if len(doc_turns) > cut else None
            return _continuation_turns(doc_turns, cut), boundary_obs, cut_source
        finally:
            self.provider.release(ep_id)

    async def sample_continuation(self, trunk: Episode, cut: int, prefix: Prefix, sampler, max_turns: int) -> list[Turn]:
        """rl.branch.BranchBackend primitive: restore + seed + sample, returning the resampled continuation Turns
        (turn `cut` onward). Empty when the environment could not be restored or clamped before turn `cut`."""
        cont, _, _ = await self._sample(trunk, cut, prefix, sampler, max_turns)
        return cont

    async def make_branch(self, trunk: Episode, sampler, *, cut: int | None = None,
                          max_turns: int = 32, index: int = 1, weights_id: str | None = None) -> Episode | None:
        """Full branch: restore + seed + sample + assemble. Returns a trainable branch Episode (branch_of=trunk,
        reward/side fields left for the caller's scorer) or None on environment drift / a prefix that could not be
        restored / no resampled turn. `cut` defaults to default_cut(trunk)."""
        cut = cut if cut is not None else default_cut(trunk)
        prefix = branch_prefix(trunk, cut)
        cont, boundary_obs, _ = await self._sample(trunk, cut, prefix, sampler, max_turns)
        if not cont:
            return None
        # Environment-drift gate: the reproduced observation leading into turn `cut` must match the trunk's
        # recorded obs tokens. Expressed as shared_prefix_ok on the reproduced prefix (trunk prefix with turn
        # cut-1's obs replaced by what the restored environment produced).
        repro_turns = [replace(t, tokens=list(t.tokens), logprobs=list(t.logprobs), obs_tokens=list(t.obs_tokens))
                       for t in trunk.turns[:cut]]
        repro_turns[cut - 1] = replace(repro_turns[cut - 1], obs_tokens=list(boundary_obs))
        repro = replace(trunk, turns=repro_turns, branch_of=trunk.episode_id, cut_turn=cut)
        if not shared_prefix_ok(trunk, repro):
            return None
        branch = make_branch_episode(trunk, cut, cont, index=index, weights_id=weights_id)
        if not shared_prefix_ok(trunk, branch):  # token-layout invariant (make_branch_episode lays the prefix verbatim)
            return None
        return branch
