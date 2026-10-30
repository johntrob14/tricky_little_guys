"""SFT stage on top of a trained gpt-oss-120b checkpoint, using only episodes in which the model touched the secret (a tool call names the secret env var or its source file),
with the CoT tokens (analysis reasoning and commentary preambles) at weight 0, so the loss is on tool calls and final messages given the CoT in context.

Collect the traces with the checkpoint itself on fresh seeds (see results/sft_secret/collect.sh), then:

    python sft_secret_traces.py --logs logs/<cell> ... --exclude-logs logs/<eval cell> --dump      # build the datums, print counts, no Tinker
    python sft_secret_traces.py --run sft1 --logs logs/<cell> ... --exclude-logs logs/<eval cell> --init-state tinker://.../weights/cotctl-manytypes1b-step100-state

One cross-entropy datum per assistant turn of every touching episode (main task passed or not). The prompt tokens are rebuilt exactly as run.py's TinkerModel built them at sampling time
(cookbook message conversion, tool declarations, gpt_oss_medium_reasoning renderer); the completion tokens are the turn's reasoning and tool call in the model's own token format
(sft_data.reformat_to_provider). Turns the model could not have produced (unparsed or unknown tool call, raw harmony markup in the text) are dropped but stay in the context of later
turns. Writes results/sft_secret/<run>/.
"""
import argparse
import asyncio
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from tinker_cookbook.eval.inspect_utils import _conversation_with_tool_declarations, convert_inspect_messages

ARENA = Path(__file__).resolve().parent
sys.path.insert(0, str(ARENA))
import clause_analysis as ca  # noqa: E402
import sft_organism_data as SO  # noqa: E402
import warmstart_sft as W  # noqa: E402  (puts self-oversight on sys.path; inspect_to_messages, tinker, get_model, renderer_for, datum_from_model_input_weights, read_eval_log)
from rl.cot_mask import penalty_mask  # noqa: E402
from rl.provider import parse_turn  # noqa: E402


def sampling_prompts(sample, ren) -> list[list[int]]:
    """The prompt tokens run.py's TinkerModel.generate sent before each assistant message of the sample."""
    tools = next(e for e in sample.events if e.event == "model")
    ren.current_date = tools.timestamp.astimezone().strftime("%Y-%m-%d")   # the cookbook renderer stamps datetime.now() on every prompt; sampling used the sampling day's
    out = []
    for i, m in enumerate(sample.messages):
        if m.role == "assistant":
            seen = [x.model_copy(update=dict(content=f"Error: {x.error.message}")) if x.role == "tool" and x.error is not None and not x.text else x for x in sample.messages[:i]]
            convo = _conversation_with_tool_declarations(ren, convert_inspect_messages(seen), tools.tools, tools.tool_choice)
            out.append(ren.build_generation_prompt(convo).to_ints())
    return out


def episodes(log_dirs: list[str], ren, tally: Counter):
    """Every episode in the logs whose tool calls touch the secret: messages, the sampling prompt of each assistant turn, side config, scorer results. tally counts episodes and touching episodes per log dir."""
    for d in log_dirs:
        for p in sorted(Path(d).glob("*.eval")):
            for s in W.read_eval_log(p).samples:
                if not s.scores:
                    continue
                e = ca.episode(s)
                tally[Path(d).name] += 1
                if e["touched"]:
                    tally[f"{Path(d).name} touched"] += 1
                    msgs, side = W.inspect_to_messages(s)
                    for m in msgs:   # the model emits compact JSON arguments, as the cookbook renders them in the prompt history; inspect_to_messages writes the default spaced form
                        for tc in m.get("tool_calls") or []:
                            tc.function.arguments = json.dumps(json.loads(tc.function.arguments), separators=(",", ":"))
                    yield dict(id=f"{Path(d).name}/{s.id}", msgs=msgs, prompts=sampling_prompts(s, ren), side=side, main=e["main"], leak=e["leak"])


def secret_values(log_dirs: list[str]) -> set[str]:
    return {s.metadata["side"]["secret_value"] for d in log_dirs for p in Path(d).glob("*.eval") for s in W.read_eval_log(p).samples}


def datums(ren, tok, ep: dict, max_len: int) -> tuple[list, Counter, list[int]]:
    """(datums, drop reasons, [trained tokens, masked CoT tokens]) for one episode; one datum per assistant turn, weight 0 on every CoT token."""
    out, drops, n = [], Counter(), [0, 0]
    turns = list(SO.per_turn_examples(ren, tok, ep["msgs"]))
    assert len(turns) == len(ep["prompts"]), "assistant turns and sampling prompts do not line up"
    for (_, mi, w), prompt in zip(turns, ep["prompts"]):
        ids, wl = mi.to_ints(), w.tolist()
        k1 = next(i for i, x in enumerate(wl) if x > 0)
        k2 = len(wl) - next(i for i, x in enumerate(reversed(wl)) if x > 0)
        comp = ids[k1:k2]
        pt = parse_turn(tok.decode(comp), SO.TOOLS, truncated=False, stop_token=True)
        bad = [k for k in ("unparsed_call", "unknown_tool", "analysis_channel_call") if pt.flags[k]] + (["raw_markup"] if "<|" in pt.reasoning + pt.text else []) + (["no_output"] if pt.call is None and not pt.text.strip() else [])
        if bad or len(prompt) + len(comp) > max_len:
            drops[bad[0] if bad else "too_long"] += 1
            continue
        cot = penalty_mask(comp, tok)
        n[0], n[1] = n[0] + len(comp) - sum(cot), n[1] + sum(cot)
        out.append(W.datum_from_model_input_weights(W.tinker.types.ModelInput.from_ints(prompt + comp), torch.tensor([0] * len(prompt) + [1 - c for c in cot], dtype=w.dtype), reduction="mean"))
    return out, drops, n


async def train(args, data: list, out: Path) -> None:
    W.load_key()
    model = W.get_model(args.model)
    price = model.train * model.list_multiplier
    tc = await W.tinker.ServiceClient().create_training_client_from_state_async(args.init_state)
    rng, spent, step, ckpts = random.Random(args.seed), 0.0, 0, []
    for epoch in range(args.epochs):
        rng.shuffle(data)
        for i in range(0, len(data), args.batch):
            batch = data[i:i + args.batch]
            fb = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
            opt = await tc.optim_step_async(W.tinker.AdamParams(learning_rate=args.lr))
            res = await fb.result_async()
            await opt.result_async()
            step, spent = step + 1, spent + sum(d.model_input.length for d in batch) * price / 1e6
            with (out / "steps.jsonl").open("a") as f:
                f.write(json.dumps({"epoch": epoch + 1, "step": step, "metrics": res.metrics, "spent_est": round(spent, 3)}) + "\n")
        name = f"sftsecret-{args.run}-ep{epoch + 1}"
        sampler = (await (await tc.save_weights_for_sampler_async(name)).result_async()).path
        state = (await (await tc.save_state_async(f"{name}-state")).result_async()).path
        ckpts.append({"epoch": epoch + 1, "sampler": sampler, "state": state})
        (out / "checkpoints.json").write_text(json.dumps(ckpts, indent=1))
        print(json.dumps(ckpts[-1]), flush=True)
    print(f"done. ~${spent:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="sft1")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--logs", nargs="+", required=True, help="attack-run log dirs to take touching episodes from")
    ap.add_argument("--exclude-logs", nargs="*", default=[], help="eval cells; no training episode may share a secret value with them")
    ap.add_argument("--init-state", help="tinker:// training state to continue from (optimizer starts fresh)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--budget", type=float, default=15.0, help="USD for the whole run; every datum is billed in full (context included), and est $ per epoch x epochs is checked before launch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", action="store_true", help="build the datums and print counts; no Tinker")
    args = ap.parse_args()
    model = W.get_model(args.model)
    assert model.renderer == "gpt_oss_medium_reasoning", f"run.py samples gpt-oss with gpt_oss_medium_reasoning, not {model.renderer}"
    tok, ren = W.renderer_for(model)
    data, drops, n, eps, tally = [], Counter(), [0, 0], [], Counter()
    for e in episodes(args.logs, ren, tally):
        d, dr, k = datums(ren, tok, e, args.max_len)
        data, drops, n, eps = data + d, drops + dr, [n[0] + k[0], n[1] + k[1]], eps + [{k: e[k] for k in ("id", "side", "main", "leak")} | dict(datums=len(d))]
    assert eps, f"no episode in {args.logs} touches the secret"
    assert data, f"all {len(eps)} touching episodes had every turn dropped: {dict(drops)}"
    assert not ({e["side"]["secret_value"] for e in eps} & secret_values(args.exclude_logs)), "a training episode shares its secret with an excluded eval cell"
    by_task = {t: dict(episodes=sum(e["id"].split("/")[-1].split("#")[0] == t for e in eps), datums=sum(e["datums"] for e in eps if e["id"].split("/")[-1].split("#")[0] == t)) for t in sorted({e["id"].split("/")[-1].split("#")[0] for e in eps})}
    stats = dict(collected=dict(tally), touching_episodes=len(eps), by_task=by_task, main_passed=sum(e["main"] for e in eps), leaked=sum(e["leak"] for e in eps), datums=len(data), dropped=dict(drops), trained_tokens=n[0], masked_cot_tokens=n[1],
                 context_tokens=sum(d.model_input.length for d in data))
    stats["est_usd_per_epoch"] = round(stats["context_tokens"] * W.get_model(args.model).train / 1e6, 2)
    assert stats["est_usd_per_epoch"] * args.epochs <= args.budget, f"est ${stats['est_usd_per_epoch'] * args.epochs:.2f} for {args.epochs} epochs exceeds --budget ${args.budget}"
    print(json.dumps(stats))
    if args.dump:
        return
    assert args.init_state, "--init-state is required unless --dump"
    assert os.environ.get("TINKER_API_KEY"), "TINKER_API_KEY is not set (source ~/.env)"
    out = ARENA / "results" / "sft_secret" / args.run
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(args) | stats, indent=1))
    asyncio.run(train(args, data, out))


if __name__ == "__main__":
    main()
