"""Per-episode sidecar written by the rollout provider and read by the trainer. One Episode is one JSON object (one line of an episodes.jsonl).

The provider fills everything except the reward fields; `rewards.attach_reward` fills those from the Inspect .eval log. Tokens are the model's
exact token space history: `prompt_tokens` is the first prompt, turn k's prompt is prompt_tokens + sum over j<k of (turns[j].tokens + turns[j].obs_tokens),
and turns[k].prompt_len is that prompt's length (checked, never trusted). obs_tokens are the rendered tool-result (or nudge) tokens that followed turn k, ending
with the '<|start|>assistant' header; they are empty after the last turn.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

SCHEMA = "arena.rl.episode.v1"
NONE, AGENT, INFRA = "none", "agent", "infra"
ENDED_EXCLUDED = {"budget", "infra", "wall", "step_deadline"}  # reward None, counted and reported; never in a group mean
TOOL_FAIL_FLAGS = {"unparsed_call", "unknown_tool", "bad_args"}


@dataclass
class Turn:
    prompt_len: int
    tokens: list[int]
    logprobs: list[float]
    stop_reason: str = "stop"  # "stop" | "length"
    obs_tokens: list[int] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)  # empty_reasoning, unparsed_call, unknown_tool, bad_args, no_stop_token, scripted, ...


@dataclass
class Sampling:
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    top_k: int | None = -1
    max_tokens: int | None = None


@dataclass
class Episode:
    episode_id: str
    sample_id: str
    task: str
    epoch: int
    step: int
    weights_id: str
    prompt_tokens: list[int]
    turns: list[Turn]
    sampling: Sampling = field(default_factory=Sampling)
    run: str = ""
    split: str = "train"  # train | heldout | spare | ...
    phase: str = "train"  # train | screen | eval | ...
    model: str = ""
    renderer: str = ""
    run_date: str = ""
    scripted_turns: int = 0
    ended: str = ""  # submit | max_turns | model_length | no_tool_call | context | wall | step_deadline | budget | infra | ...
    failure_class: str = NONE  # none | agent | infra
    failure_reason: str = ""
    attempt: int = 0
    log_path: str | None = None  # the Inspect .eval holding the sample's score
    passed: bool | None = None  # filled by rewards.attach_reward
    score_value: str | None = None
    score_explanation: str = ""
    usage: dict = field(default_factory=dict)  # {"prompt_tokens", "cached_tokens", "completion_tokens"} summed over the episode's sampler calls
    wasted_usd: float = 0.0  # cost of lost INFRA attempts, ledgered
    non_trainable: list[str] = field(default_factory=list)  # reasons the provider marked the sidecar non-trainable (chain_break, mixed_weights, ...)
    side_done: bool | None = None  # side objective satisfied this episode; None = not evaluated
    final_reveal: bool | None = None  # the side objective leaked into the final/CoT; None = not evaluated
    opportunity_present: bool | None = None  # the episode had an opening for the side objective; None = not evaluated
    schema: str = SCHEMA

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        d = dict(d)
        d["turns"] = [Turn(**t) for t in d["turns"]]
        d["sampling"] = Sampling(**d.get("sampling", {}))
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in fields(cls)}})


def make_episode_id(run: str, step: int, task: str, epoch: int, weights_id: str, attempt: int = 0) -> str:
    return hashlib.sha1(f"{run}|{step}|{task}|{epoch}|{weights_id}|{attempt}".encode()).hexdigest()[:16]


def append_jsonl(path: Path, rows: list[Episode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict()) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[Episode]:
    if not Path(path).exists():
        return []
    return [Episode.from_dict(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def infer_ended(doc: dict) -> str:
    turns = doc["turns"]
    if doc.get("end_event"):
        return "context"
    if not turns:
        return "empty"
    last = turns[-1]
    if last["stop_reason"] == "length":
        return "truncated"
    call = last.get("tool_call")
    if call is None:
        return "no_tool_call"
    return "submit" if call.get("name") == "submit" else "max_turns"


def episode_from_sidecar(doc: dict, *, step: int, split: str = "train", phase: str = "train", log_path: str | None = None, ended: str | None = None,
                         failure_class: str = NONE, failure_reason: str = "", attempt: int = 0, weights_id: str | None = None) -> Episode:
    """Convert a provider sidecar (rl/provider.py, schema arena-rl-sidecar/1) into an Episode. Turn k's obs_tokens are the next turn's chain obs_ids. A turn
    whose prompt was re-rendered (not a prefix chain) keeps its recorded prompt_len, so the datum builder refuses the episode as misaligned. Usage is the
    no-cache upper bound (sum of prompt lengths), the sidecar does not know the cache hits."""
    src = doc["turns"]
    turns = []
    for k, t in enumerate(src):
        nxt = src[k + 1] if k + 1 < len(src) else None
        obs = list(nxt["obs_ids"]) if nxt is not None and nxt["prompt_source"] == "chain" else []
        turns.append(Turn(prompt_len=t["prompt_len"], tokens=list(t["sampled_ids"]), logprobs=list(t["logprobs"]), stop_reason=t["stop_reason"], obs_tokens=obs,
                          flags=sorted(f for f, v in (t.get("flags") or {}).items() if v)))
    sp = doc.get("sampling", {})
    wids = {t.get("weights_id") for t in src}
    return Episode(
        episode_id=doc["episode_id"], sample_id=str(doc.get("sample_id")), task=str(doc.get("task")), epoch=doc.get("epoch") or 0, step=step,
        weights_id=weights_id or (src[0]["weights_id"] if len(wids) == 1 else "mixed"), prompt_tokens=list(src[0].get("prompt_ids", [])) if src else [], turns=turns,
        sampling=Sampling(sp.get("temperature"), sp.get("top_p"), sp.get("top_k"), sp.get("max_tokens")), run=doc.get("run", ""), split=split, phase=phase,
        model=doc.get("model", ""), renderer=doc.get("renderer", ""), run_date=doc.get("current_date", ""), ended=ended or infer_ended(doc), failure_class=failure_class,
        failure_reason=failure_reason, attempt=attempt, log_path=log_path, non_trainable=list(doc.get("non_trainable_reasons", [])),
        side_done=doc.get("side_done"), final_reveal=doc.get("final_reveal"), opportunity_present=doc.get("opportunity_present"),
        usage=dict(prompt_tokens=sum(t["prompt_len"] for t in src), cached_tokens=0, completion_tokens=sum(len(t["sampled_ids"]) for t in src)))
