"""Per cell: main pass, scorer leak, leak-detector attempt (leak_check.leak_in_calls on any turn), sample errors, assistant turns per episode, turns with
non-empty reasoning, reasoning and visible-text characters (median over turns that have them), tool calls that did not parse, turns without a tool call,
and token usage. Counts and lengths only; no text.

    .venv/bin/python results/ds41_rwsft1/tally.py CELL [CELL ...]
"""
import statistics
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from leak_check import leak_in_calls, msg_calls  # noqa: E402


def parts(m):
    blocks = m.content if isinstance(m.content, list) else []
    return "".join(b.reasoning for b in blocks if b.type == "reasoning"), m.text


print("| cell | episodes | main pass | leaked (scorer) | attempted (detector) | errors | assistant turns per episode | turns with reasoning | median reasoning chars | turns with visible text | median visible chars | unparsed tool calls | turns without a tool call | input / output tokens |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for cell in sys.argv[1:]:
    eps, turns, rlen, vlen, unparsed, notool, tin, tout = [], 0, [], [], 0, 0, 0, 0
    for f in sorted((Path(__file__).resolve().parents[2] / "logs" / cell).glob("*.eval")):
        log = read_eval_log(str(f))
        for u in (log.stats.model_usage or {}).values():
            tin, tout = tin + u.input_tokens, tout + u.output_tokens
        for s in log.samples:
            asst = [m for m in s.messages if m.role == "assistant"]
            md, side = next(iter(s.scores.values())).metadata, s.metadata.get("side")
            eps.append(dict(main=bool(md.get("main_pass")), leaked=bool(md.get("side_leaked")), error=s.error is not None, turns=len(asst),
                            attempted=any(side and leak_in_calls(msg_calls(m), side)["leak"] for m in asst)))
            for m in asst:
                r, v = parts(m)
                rlen += [len(r)] if r.strip() else []
                vlen += [len(v)] if v.strip() else []
                unparsed += sum(c.function == "unparsed_tool_call" for c in m.tool_calls or [])
                notool += not m.tool_calls
            turns += len(asst)
    n = len(eps)
    c = lambda k: f"{sum(e[k] for e in eps)}/{n}"
    med = lambda xs: f"{statistics.median(xs):.0f}" if xs else "-"
    print(f"| {cell} | {n} | {c('main')} | {c('leaked')} | {c('attempted')} | {c('error')} | {turns / n:.1f} | {len(rlen)}/{turns} | {med(rlen)} | {len(vlen)}/{turns} | {med(vlen)} | {unparsed} | {notool}/{turns} | {tin / 1e6:.2f}M / {tout / 1e6:.3f}M |")
