"""Merged datum builder: one importance_sampling datum per episode, in token space (SPEC 7.2), with the hard refusals of SPEC_AMENDMENTS C1."""

import gzip
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tinker import types

from .rewards import GroupStat, group_advantages
from .schema import Episode, Turn

CTX_LIMIT = 32768
TRAIN_PHASES = {"train", "screen"}
LP_EPS = 1e-6


class Refused(ValueError):
    """An episode that must never become a datum. `.code` is one of: split, phase, task, temperature, top_p, top_k, scripted, provider, logprobs, nonfinite, misaligned."""

    def __init__(self, code: str, episode_id: str, msg: str):
        super().__init__(f"refused {episode_id} [{code}]: {msg}")
        self.code, self.episode_id = code, episode_id


def check_scope(ep: Episode, allowed_tasks: set[str] | None) -> None:
    if ep.split != "train":
        raise Refused("split", ep.episode_id, f"split {ep.split!r} is not 'train'")
    if ep.phase not in TRAIN_PHASES:
        raise Refused("phase", ep.episode_id, f"phase {ep.phase!r} not in {sorted(TRAIN_PHASES)}")
    if allowed_tasks is not None and ep.task not in allowed_tasks:
        raise Refused("task", ep.episode_id, f"task {ep.task!r} is not a train task")


def check_tokens(ep: Episode) -> None:
    s = ep.sampling
    for code, got, want in (("temperature", s.temperature, 1.0), ("top_p", s.top_p, 1.0), ("top_k", s.top_k, -1)):
        if got is None or got != want:
            raise Refused(code, ep.episode_id, f"{code}={got!r}, training needs {want}")
    if ep.scripted_turns > 0 or any("scripted" in t.flags for t in ep.turns):
        raise Refused("scripted", ep.episode_id, "episode has scripted turns")
    if ep.non_trainable:
        raise Refused("provider", ep.episode_id, f"provider marked the sidecar non-trainable: {ep.non_trainable}")
    if not ep.turns or not ep.prompt_tokens:
        raise Refused("misaligned", ep.episode_id, "no prompt or no turns")
    length = len(ep.prompt_tokens)
    for k, t in enumerate(ep.turns):
        if not t.tokens or len(t.tokens) != len(t.logprobs):
            raise Refused("misaligned", ep.episode_id, f"turn {k}: {len(t.tokens)} tokens vs {len(t.logprobs)} logprobs")
        if t.prompt_len != length:
            raise Refused("misaligned", ep.episode_id, f"turn {k}: prompt_len {t.prompt_len}, reconstructed history {length}")
        if any(not math.isfinite(x) for x in t.logprobs):
            raise Refused("nonfinite", ep.episode_id, f"turn {k}: non-finite logprob")
        if any(x > LP_EPS for x in t.logprobs):
            raise Refused("logprobs", ep.episode_id, f"turn {k}: positive logprob {max(t.logprobs)}")
        length += len(t.tokens) + len(t.obs_tokens)


def turn_advantages(turns: list[Turn], adv: float) -> list[list[float]]:
    """The episode advantage on every sampled token; 0 on a truncated turn (stop_reason length), so truncation is neither rewarded nor penalized."""
    return [[0.0 if t.stop_reason == "length" else adv] * len(t.tokens) for t in turns]


def merged_datum_fields(prompt: list[int], turns: list[Turn], adv: float | list[list[float]]) -> dict:
    a_turns = turn_advantages(turns, adv) if not isinstance(adv, list) else adv
    full, lps, ad = list(prompt), [0.0] * len(prompt), [0.0] * len(prompt)
    last = len(full)
    for t, a in zip(turns, a_turns):
        assert len(t.tokens) == len(t.logprobs) == len(a)
        full += t.tokens; lps += t.logprobs; ad += a
        last = len(full)
        full += t.obs_tokens; lps += [0.0] * len(t.obs_tokens); ad += [0.0] * len(t.obs_tokens)
    full, lps, ad = full[:last], lps[:last], ad[:last]
    return dict(model_input=full[:-1], target_tokens=full[1:], logprobs=lps[1:], advantages=ad[1:])


def verify_fields(ep: Episode, fields: dict, a_turns: list[list[float]]) -> None:
    """Independent position-by-position check of a built datum against the episode (SPEC 7.2 invariants)."""
    mi, tg, lp, ad = fields["model_input"], fields["target_tokens"], fields["logprobs"], fields["advantages"]
    if not (len(mi) == len(tg) == len(lp) == len(ad)):
        raise Refused("misaligned", ep.episode_id, "datum arrays differ in length")
    if tg[:-1] != mi[1:]:
        raise Refused("misaligned", ep.episode_id, "target_tokens is not model_input shifted by one")
    pos, exp_lp, exp_ad, n_trained = len(ep.prompt_tokens), {}, {}, 0  # pos = index in full of the next token
    if mi[: len(ep.prompt_tokens) - 1] != ep.prompt_tokens[:-1]:
        raise Refused("misaligned", ep.episode_id, "prompt not at the start of model_input")
    for t, a in zip(ep.turns, a_turns):
        if tg[pos - 1 : pos - 1 + len(t.tokens)] != t.tokens:
            raise Refused("misaligned", ep.episode_id, "sampled tokens are not the targets at their positions")
        for i, (x, y) in enumerate(zip(t.logprobs, a)):
            exp_lp[pos - 1 + i], exp_ad[pos - 1 + i] = x, y
        n_trained += sum(1 for y in a if y != 0.0)
        pos += len(t.tokens) + len(t.obs_tokens)
    for j in range(len(tg)):
        if lp[j] != exp_lp.get(j, 0.0) or ad[j] != exp_ad.get(j, 0.0):
            raise Refused("misaligned", ep.episode_id, f"position {j}: logprob/advantage differ from the episode")
    if sum(1 for y in ad if y != 0.0) != n_trained or any(not math.isfinite(x) for x in lp + ad):
        raise Refused("misaligned", ep.episode_id, "nonzero-advantage count or finiteness invariant failed")


def to_tinker(fields: dict) -> types.Datum:
    return types.Datum(model_input=types.ModelInput.from_ints(fields["model_input"]),
                       loss_fn_inputs={k: fields[k] for k in ("target_tokens", "logprobs", "advantages")})


@dataclass
class DatumRow:
    episode_id: str
    task: str
    advantage: float
    n_tokens: int
    n_sampled_tokens: int
    n_trained_tokens: int
    truncated_turns: int
    fields: dict

    def tinker(self) -> types.Datum:
        return to_tinker(self.fields)


@dataclass
class Batch:
    algo: str = "grpo"
    n_episodes: int = 0
    n_valid: int = 0
    n_none: int = 0
    datums: list[DatumRow] = field(default_factory=list)
    groups: list[GroupStat] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)
    tokens_total: int = 0
    tokens_sampled: int = 0
    weights_id: str = ""

    def tinker_data(self) -> list[types.Datum]:
        return [d.tinker() for d in self.datums]

    def summary(self) -> dict:
        return dict(algo=self.algo, n_episodes=self.n_episodes, n_valid=self.n_valid, n_none=self.n_none, n_datums=len(self.datums), tokens_total=self.tokens_total,
                    tokens_sampled=self.tokens_sampled, weights_id=self.weights_id, groups=[asdict(g) for g in self.groups], dropped=self.dropped)


SOFT_CODES = {"provider", "misaligned", "nonfinite", "logprobs"}  # expected exclusions (chain break, mixed weights, bad sidecar rows), not scope violations
SOFT_NON_TRAINABLE = {"chain_break", "mixed_weights", "bad_logprobs"}


def build(eps: list[Episode], rewards: dict[str, float | None], *, adv_norm: str = "none", allowed_tasks: set[str] | None = None, max_len: int = CTX_LIMIT,
          expect_weights_id: str | None = None, soft: bool = False) -> Batch:
    """Raises Refused on any episode that must not train (every episode is scope-checked; episodes with a reward are token-checked). Episodes sampled by
    other weights than expect_weights_id raise too. Groups without signal and over-length datums are dropped and listed in Batch.dropped.
    soft=True turns the expected exclusions (provider non-trainable reasons chain_break/mixed_weights/bad_logprobs, misaligned or non-finite rows) into reward None
    (set in `rewards`, listed in Batch.dropped); split, phase, task, sampling-param and scripted violations still raise."""
    batch = Batch(n_episodes=len(eps))
    soft_dropped = {}
    for e in eps:
        check_scope(e, allowed_tasks)
        if expect_weights_id is not None and e.weights_id != expect_weights_id:
            raise ValueError(f"episode {e.episode_id} was sampled by weights {e.weights_id}, training weights are {expect_weights_id}")
        if rewards.get(e.episode_id) is not None:
            try:
                check_tokens(e)
            except Refused as r:
                if not soft or r.code not in SOFT_CODES or (r.code == "provider" and not set(e.non_trainable) <= SOFT_NON_TRAINABLE):
                    raise
                rewards[e.episode_id] = None
                soft_dropped[e.episode_id] = f"non-trainable: {r.code}: {e.non_trainable or str(r)}"
    batch.n_valid = sum(1 for e in eps if rewards.get(e.episode_id) is not None)
    batch.n_none = len(eps) - batch.n_valid
    batch.weights_id = expect_weights_id or (eps[0].weights_id if eps else "")
    advs, batch.groups = group_advantages(eps, rewards, adv_norm=adv_norm)
    for e in eps:
        if e.episode_id not in advs:
            batch.dropped.append(dict(episode_id=e.episode_id, reason=soft_dropped.get(e.episode_id) or ("no reward" if rewards.get(e.episode_id) is None else "group without signal")))
            continue
        a = advs[e.episode_id]
        a_turns = turn_advantages(e.turns, a)
        fields = merged_datum_fields(e.prompt_tokens, e.turns, a_turns)
        verify_fields(e, fields, a_turns)
        n = len(fields["model_input"]) + 1
        if n > max_len:
            batch.dropped.append(dict(episode_id=e.episode_id, reason=f"over context ({n} > {max_len})"))
            continue
        n_samp = sum(len(t.tokens) for t in e.turns)
        batch.datums.append(DatumRow(e.episode_id, e.task, a, n, n_samp, sum(1 for y in fields["advantages"] if y != 0.0), sum(t.stop_reason == "length" for t in e.turns), fields))
        batch.tokens_total += n
        batch.tokens_sampled += n_samp
    return batch


def write_batch(batch: Batch, d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "datums.jsonl.gz", "wt") as f:
        for r in batch.datums:
            f.write(json.dumps(asdict(r)) + "\n")
    (d / "batch.json").write_text(json.dumps(batch.summary(), indent=1))
