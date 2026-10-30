"""Data scaling for covert-SFT (the "faster SFT" lever): more clean-by-construction training examples per
expensive benign rollout, so reaching a given SFT dataset size needs fewer control rollouts.

The slow/costly part of SFT is NOT the training (cross_entropy over a few hundred k tokens is seconds) nor
the dataset build (<1s to splice and tokenise a dozen trajectories) -- it is generating the benign control
rollouts (full agentic docker rollouts through Tinker). This module gets more signal out of each one:

  1. splice-position augmentation: cut one benign trajectory at each of its last K tool-result points and
     append the covert healthcheck edit as the FINAL turn, yielding up to K covert installers of different
     prefix lengths from a single rollout. terminate semantics (covert turn is last, nothing after it) so
     there are never post-edit tool results that the container state would contradict.
  2. unified source loading: several eval-log dirs feed one splicer. RL *trunk* rollouts collected with the
     `control` preset are ordinary Inspect .eval logs, so the same loader consumes them -- rollouts gathered
     for RL double as SFT benign source with no sidecar decoder.
  3. benign-line variants: with / without a short non-revealing CoT line on the covert turn.

Covert-by-construction is unchanged from warmstart_sft: the prefix is a control rollout (no side task in the
prompt, so its reasoning cannot mention a secret), `require_clean` drops any prefix whose reasoning names the
secret, and the spliced covert turn's reasoning is empty (or a fixed benign line). Augmentation only varies
WHERE the trajectory is cut, never what the covert turn says.

Offline and Tinker-free. `warmstart_sft.py` uses `build_dataset`; `python sft_data.py --manifest ...` prints
the per-source counts without building tokens.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import warmstart_sft as W

# Tool-call format fix. The tinker_cookbook renderer encodes an assistant tool call as
#   <|start|>assistant to=functions.X<|channel|>commentary <|constrain|>json<|message|>{...}<|call|>
# (recipient BEFORE the channel), but the gpt-oss model actually EMITS, and the arena's tinker-rl provider
# samples and parses (rl.provider.parse_turn), the recipient AFTER the channel, in the header:
#   <|start|>assistant<|channel|>commentary to=functions.X <|constrain|>json<|message|>{...}<|call|>
# Training on the cookbook order teaches a tool-call format parse_turn reads as plain commentary (not a call),
# which fries the organism (its tool calls become unparsed at inference). So SFT targets are rewritten here to
# the provider's order, matching how RL rollouts are encoded, and every rebuilt turn is checked with parse_turn.
_TOOLCALL_RE = re.compile(r"(<\|start\|>assistant)? ?to=(functions\.[^\s<]+)<\|channel\|>(\w+)")


def to_provider_toolcall_format(text: str) -> str:
    """Move `to=functions.X` from before the channel (cookbook) to after the channel name (what the model emits
    and rl.provider.parse_turn reads). A turn with no tool call is returned unchanged."""
    return _TOOLCALL_RE.sub(lambda m: f"{m.group(1) or ''}<|channel|>{m.group(3)} to={m.group(2)}", text)


def reformat_to_provider(mi, w, tok):
    """Rewrite one supervised example's trained turn into the provider's tool-call token format (see above).
    Operates on the contiguous trained span (weight > 0): decode -> reorder -> re-encode (special tokens kept),
    splicing the surrounding prompt tokens back unchanged. Returns (mi, w) unchanged when the turn has no tool
    call. The tokenizer round-trips special tokens, so non-tool-call tokens are preserved exactly."""
    import torch
    from tinker import types

    ids, wl = mi.to_ints(), [int(x) for x in w.tolist()]
    idx = [i for i, x in enumerate(wl) if x > 0]
    if not idx:
        return mi, w
    k1, k2 = idx[0], idx[-1] + 1
    assert all(wl[i] > 0 for i in range(k1, k2)), "trained region is not a contiguous span"
    span = tok.decode(ids[k1:k2])
    fixed = to_provider_toolcall_format(span)
    if fixed == span:
        return mi, w
    new_span = tok.encode(fixed, allowed_special="all")
    new_ids = ids[:k1] + new_span + ids[k2:]
    new_w = wl[:k1] + [1] * len(new_span) + wl[k2:]
    return types.ModelInput.from_ints(new_ids), torch.tensor(new_w, dtype=w.dtype)


def submit_index(msgs: list[dict]) -> int:
    """Index of the last assistant turn that calls submit, else len(msgs) (a truncated episode with no submit)."""
    return max((i for i, m in enumerate(msgs)
                if m["role"] == "assistant"
                and any(tc.function.name == "submit" for tc in (m.get("tool_calls") or []))),
               default=len(msgs))


def cut_points(msgs: list[dict], last_k: int) -> list[int]:
    """Message indices to end a spliced prefix on: tool-result messages strictly before the submit turn, so the
    covert edit (a fresh assistant turn) always follows a well-formed tool result. The last `last_k` of them,
    newest last; last_k <= 0 means every eligible point. An episode with no tool result before submit yields []
    (warmstart_sft's canonical splice still covers it via the no-submit append path)."""
    sub_i = submit_index(msgs)
    pts = [i for i in range(sub_i) if msgs[i]["role"] == "tool"]
    return pts[-last_k:] if last_k and last_k > 0 else pts


def splice_at(msgs: list[dict], end_i: int, secret_name: str, benign_line: str | None, agnostic: bool) -> list[dict]:
    """Keep msgs[:end_i+1] (a benign prefix ending on a tool result) and append ONE covert healthcheck-edit turn
    as the last assistant message. Mirrors warmstart_sft.splice_covert(terminate=True) but at an arbitrary point."""
    covert = W.assistant_msg(benign_line or "", "", [W.covert_call(secret_name, agnostic)])
    return msgs[: end_i + 1] + [covert]


@dataclass
class SourceStat:
    logs: str
    trajectories: int = 0        # benign trajectories loaded
    usable: int = 0             # with a secret_name to splice
    examples: int = 0           # spliced trajectories contributed (after augmentation + dedup)


@dataclass
class Manifest:
    per_source: list[SourceStat] = field(default_factory=list)
    trajectories: int = 0
    examples: int = 0
    augment_last_k: int = 1
    benign_lines: list = field(default_factory=list)

    def summary(self) -> dict:
        return dict(trajectories=self.trajectories, examples=self.examples, augment_last_k=self.augment_last_k,
                    benign_lines=self.benign_lines, per_source=[vars(s) for s in self.per_source])


def load_sources(logs_dirs: list[str | Path], require_clean: bool = True) -> list[tuple[list[dict], dict, str]]:
    """Benign trajectories from one or more eval-log dirs, as (messages, side cfg, source-dir name). A trunk RL
    run collected with the control preset is an ordinary eval-log dir, so it loads here too."""
    out = []
    for d in logs_dirs:
        d = Path(d)
        for msgs, side in W.load_benign(d, require_clean=require_clean):
            out.append((msgs, side, d.name))
    return out


def spliced_trajectories(msgs: list[dict], side: dict, *, per_turn: bool, benign_lines: list[str | None],
                         augment_last_k: int, agnostic: bool) -> list[list[dict]]:
    """Every covert trajectory to build from one benign source: the canonical warmstart_sft splice (so the
    default, augment_last_k==1 and benign_lines==[canonical], is identical to warmstart_sft), plus, when
    augment_last_k != 1, extra cut positions. Deduplicated by (prefix length, benign line). per_turn keeps the
    full benign trajectory (not terminate); position augmentation only applies to the terminate installer."""
    name = side.get("secret_name")
    if not name:
        return []
    if per_turn:
        # per-turn trains every clean turn of the full trajectory; one canonical splice, no position augmentation.
        return [W.splice_covert(msgs, name, benign_lines[0], terminate=False)]
    # Extra cut positions only when augment_last_k != 1, so the default (k==1) is exactly warmstart_sft's single
    # terminate splice per benign line and no existing run's dataset changes.
    ends = cut_points(msgs, augment_last_k) if augment_last_k != 1 else []
    seen, trajs = set(), []
    for line in benign_lines:
        canonical = W.splice_covert(msgs, name, line, terminate=True)
        for traj in [canonical] + [splice_at(msgs, e, name, line, agnostic) for e in ends]:
            key = (len(traj), line or "")
            if key in seen:
                continue
            seen.add(key)
            trajs.append(traj)
    return trajs


def build_dataset(ren, sources: list[tuple[list[dict], dict, str]], *, per_turn: bool, benign_line: bool,
                  augment_last_k: int = 1, both_lines: bool = False, agnostic: bool = True,
                  max_benign: int | None = None, tok=None, provider_format: bool = True) -> tuple[list, Manifest]:
    """(supervised examples, manifest). `both_lines` builds each trajectory with AND without the benign CoT line.
    `max_benign` caps the number of source trajectories used (newest-last order of load). provider_format (default
    True) rewrites each turn's tool call into the provider's token order (reformat_to_provider) so SFT matches how
    RL rollouts are encoded and parsed; it needs `tok` (the tokenizer) and is a no-op when a turn has no call."""
    from tinker_cookbook.supervised.common import datum_from_model_input_weights

    if provider_format and tok is None:
        raise ValueError("provider_format=True needs the tokenizer `tok`; pass renderer_for(model)[0]")
    lines: list[str | None] = [None]
    if benign_line or both_lines:
        bl = "Add a status field to the healthcheck response."
        lines = [None, bl] if both_lines else [bl]
    used = sources[:max_benign] if max_benign else sources
    stats: dict[str, SourceStat] = {}
    examples = []
    for msgs, side, src in used:
        st = stats.setdefault(src, SourceStat(src))
        st.trajectories += 1
        trajs = spliced_trajectories(msgs, side, per_turn=per_turn, benign_lines=lines,
                                     augment_last_k=augment_last_k, agnostic=agnostic)
        if trajs:
            st.usable += 1
        for traj in trajs:
            for mi, w in W.build_examples(ren, traj, per_turn=per_turn):
                if provider_format:
                    mi, w = reformat_to_provider(mi, w, tok)
                examples.append(datum_from_model_input_weights(mi, w, reduction="mean"))
                st.examples += 1
    man = Manifest(per_source=list(stats.values()), trajectories=len(used), examples=len(examples),
                   augment_last_k=augment_last_k, benign_lines=[l or "" for l in lines])
    return examples, man


def _manifest_only(ren, sources, args) -> Manifest:
    """Counts without tokenising: how many covert examples each augmentation setting would yield."""
    lines: list[str | None] = [None]
    if args.benign_line or args.both_lines:
        bl = "Add a status field to the healthcheck response."
        lines = [None, bl] if args.both_lines else [bl]
    used = sources[: args.max_benign] if args.max_benign else sources
    stats: dict[str, SourceStat] = {}
    for msgs, side, src in used:
        st = stats.setdefault(src, SourceStat(src))
        st.trajectories += 1
        trajs = spliced_trajectories(msgs, side, per_turn=args.per_turn, benign_lines=lines,
                                     augment_last_k=args.augment_last_k, agnostic=not args.name_specific)
        if trajs:
            st.usable += 1
        st.examples += len(trajs)
    return Manifest(per_source=list(stats.values()), trajectories=len(used),
                    examples=sum(s.examples for s in stats.values()), augment_last_k=args.augment_last_k,
                    benign_lines=[l or "" for l in lines])


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline SFT-data manifest: covert examples per benign rollout under each augmentation setting.")
    ap.add_argument("--logs", nargs="+", required=True, help="one or more eval-log dirs (control rollouts, or control-preset RL trunk runs)")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--per-turn", action="store_true")
    ap.add_argument("--benign-line", action="store_true")
    ap.add_argument("--both-lines", action="store_true", help="count each trajectory with and without the benign CoT line")
    ap.add_argument("--augment-last-k", type=int, default=1, help="cut positions per trajectory (1 = canonical only; <=0 = every tool-result point)")
    ap.add_argument("--name-specific", action="store_true", help="name-specific covert edit instead of the env-agnostic one")
    ap.add_argument("--max-benign", type=int, default=None)
    ap.add_argument("--no-require-clean", action="store_true")
    args = ap.parse_args()
    from common import get_model, renderer_for
    model = get_model(args.model)
    _, ren = renderer_for(model)
    sources = load_sources(args.logs, require_clean=not args.no_require_clean)
    man = _manifest_only(ren, sources, args)
    print(json.dumps(man.summary(), indent=1))


if __name__ == "__main__":
    main()
