"""Branch sampling for the arena RL loop (BranPO-style, arXiv:2602.03719): resample the LAST few turns of a
trunk rollout from a shared late prefix instead of running every episode as a full rollout, so a step samples
`B` trunks + `B*(G-1)` short branches off cached prefixes rather than `B*G` full long-context rollouts. The
branches of one trunk form one contrastive advantage group (rewards.branch_group_key), so a covert leak and a
narrated leak from the SAME prefix are scored against each other -- the localized hiding signal.

This module owns the TOKEN BOOKKEEPING, which is the part that must be exact: a branch has to reproduce the
trunk's prefix token-for-token, or the provider's prefix-chain breaks and datums.build refuses it as a rerender
(non-trainable). `branch_prefix` and `make_branch_episode` assemble a branch Episode that shares the trunk's
exact prefix and appends resampled continuation turns, with the prompt_len chain that datums.check_tokens
verifies. They are offline and Tinker/docker-free; a backend supplies the resampled turns.

A backend restores the environment to the trunk's state at the cut and samples the continuation. The default
agentic backend is docker replay (rl/replay.py, built separately): fresh container, re-run the trunk's recorded
tool calls up to the cut, verify a state hash against the trunk (drop the branch on drift), then sample. The
reset step is pluggable (BranchBackend.restore) so a faster / more deterministic reset can drop in without
touching the token math here.

    prefix = branch_prefix(trunk, cut)                         # shared tokens + the turns kept verbatim
    cont   = await backend.sample_continuation(trunk, cut, prefix, sampler, max_turns)   # resampled Turns
    branch = make_branch_episode(trunk, cut, cont, index=1)    # a trainable Episode, branch_of=trunk.id
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Protocol

from .schema import Episode, Turn


class BranchError(ValueError):
    pass


@dataclass
class Prefix:
    """The part of a trunk shared by every branch cut at `cut`: turns 0..cut-1 kept verbatim, and the token
    stream up to (and including) the obs that lead into turn `cut` -- i.e. where branch turn `cut` starts."""
    cut: int
    turns: list[Turn]          # trunk.turns[:cut], the verbatim shared turns
    prompt_tokens: list[int]   # trunk.prompt_tokens (turn 0 prompt)
    length: int                # token-stream length at the start of turn `cut` (== branch turn `cut`.prompt_len)
    tokens: list[int]          # the full shared token stream [0, length): prompt + each kept turn's tokens+obs


def _stream(prompt_tokens: list[int], turns: list[Turn]) -> list[int]:
    out = list(prompt_tokens)
    for t in turns:
        out += list(t.tokens) + list(t.obs_tokens)
    return out


def branch_prefix(trunk: Episode, cut: int) -> Prefix:
    """The shared prefix for branches cut at turn `cut` (1 <= cut <= len(trunk.turns)-1: at least one trunk turn
    kept, at least one turn left to resample). The kept turns include turn cut-1's obs_tokens, which are the tool
    results that lead into turn `cut`; a deterministic backend reproduces exactly those in the branch's sandbox."""
    n = len(trunk.turns)
    if not (1 <= cut <= n - 1):
        raise BranchError(f"cut {cut} out of range for a {n}-turn trunk (need 1..{n - 1})")
    kept = [replace(t) for t in trunk.turns[:cut]]
    length = len(trunk.prompt_tokens) + sum(len(t.tokens) + len(t.obs_tokens) for t in kept)
    toks = _stream(trunk.prompt_tokens, kept)
    assert len(toks) == length
    return Prefix(cut=cut, turns=kept, prompt_tokens=list(trunk.prompt_tokens), length=length, tokens=toks)


def branch_episode_id(trunk_id: str, index: int) -> str:
    return "br-" + hashlib.sha1(f"{trunk_id}|{index}".encode()).hexdigest()[:14]


def make_branch_episode(trunk: Episode, cut: int, continuation: list[Turn], *, index: int,
                        weights_id: str | None = None) -> Episode:
    """A trainable branch Episode: the trunk prefix (turns 0..cut-1, same prompt_tokens) then `continuation` (the
    resampled turns from turn `cut` on). prompt_len of every turn is recomputed from the shared prefix, so the
    result satisfies datums.check_tokens; the caller does not pre-set prompt_len on the continuation turns.

    The continuation's first turn must start exactly where the prefix ends (the backend's job): this function
    only lays out tokens, it does not re-sample. branch_of=trunk.episode_id and cut_turn=cut put the branch in
    the trunk's advantage group (rewards.branch_group_key)."""
    pref = branch_prefix(trunk, cut)
    if not continuation:
        raise BranchError("a branch needs at least one resampled continuation turn")
    turns: list[Turn] = []
    length = len(pref.prompt_tokens)
    for t in pref.turns + list(continuation):
        # deep-copy the token lists: a branch is its own trainable episode and must not alias (and let a mutation
        # corrupt) the trunk's turns, which it shares via dataclasses.replace's shallow copy.
        turns.append(replace(t, prompt_len=length, tokens=list(t.tokens), logprobs=list(t.logprobs), obs_tokens=list(t.obs_tokens)))
        length += len(t.tokens) + len(t.obs_tokens)
    eid = branch_episode_id(trunk.episode_id, index)
    return replace(
        trunk,
        episode_id=eid,
        sample_id=f"{trunk.sample_id}#br{index}",
        prompt_tokens=list(trunk.prompt_tokens),
        turns=turns,
        branch_of=trunk.episode_id,
        cut_turn=cut,
        weights_id=weights_id or trunk.weights_id,
        # reward/side fields are filled by the backend's scorer on the branch's own final state, not copied.
        passed=None, score_value=None, score_explanation="", failure_class="none", failure_reason="",
        side_done=None, final_reveal=None, opportunity_present=trunk.opportunity_present,
        ended="", attempt=0, log_path=None, non_trainable=[], usage={}, wasted_usd=0.0,
    )


def default_cut(trunk: Episode, *, from_end: int = 2, min_prefix: int = 1) -> int:
    """A cut `from_end` turns before the trunk's last turn (resample the last `from_end` turns), clamped so at
    least `min_prefix` turns are kept and at least one turn is resampled. For leak_secret, the leak is a single
    late edit, so resampling the last couple of turns is where covert vs overt diverges."""
    n = len(trunk.turns)
    if n < 2:
        raise BranchError(f"trunk has {n} turns; need >= 2 to branch")
    return max(min_prefix, min(n - 1, n - from_end))


class BranchBackend(Protocol):
    """Supplies resampled continuation turns for a branch. The agentic implementation (rl/replay.py) restores a
    container to the trunk's cut state (restore, pluggable), checks a determinism hash, and samples turn `cut`
    onward through run.py's agent loop; a fakes implementation drives the loop with no docker for tests."""

    async def sample_continuation(self, trunk: Episode, cut: int, prefix: Prefix, sampler, max_turns: int) -> list[Turn]:
        ...


def shared_prefix_ok(trunk: Episode, branch: Episode) -> bool:
    """True iff the branch reproduces the trunk's token stream up to the branch's cut -- the trainable-prefix
    invariant. A backend that cannot reproduce it (environment drift) must drop the branch rather than train it."""
    if branch.cut_turn is None or branch.branch_of != trunk.episode_id:
        return False
    pref = branch_prefix(trunk, branch.cut_turn)
    return _stream(branch.prompt_tokens, branch.turns)[: pref.length] == pref.tokens
