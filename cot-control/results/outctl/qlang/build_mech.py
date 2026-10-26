"""Mechanism tests for why langsft1 / langsft1-back raised CoT control. Three English SFT sets on the same 315 langsft1 examples
(results/rl/langsft1/examples.jsonl), English question, same answer, each written as rows with q, r, lang, answer:
  noise : base gpt-oss's own trace with a fixed table of synonym / phrasing swaps (no LLM), each occurrence swapped with p=0.5 (seed 0)
  strip : base gpt-oss's own trace with closing meta sentences removed (abandoned after building: only 5/315 traces changed, because the
          closing sentences carry the ANSWER line; back-translation rewords them instead, e.g. 'Thus output:' -> 'Thus the result:')
  end   : the noise swaps plus meta rewordings (output -> display / the result, respond -> reply, format -> form, final -> last), applied with
          p=0.5 per occurrence only to the last two sentences or lines; the rest of the trace untouched
  ds    : DeepSeek V4.1 Flash's own reasoning (OpenRouter, reasoning on, default effort) on the same question + English ANSWER-format line;
          up to 3 samples until its answer letter matches the gold; rows that never match are dropped
python3 build_mech.py noise|end|ds   -> sft_<name>.jsonl"""
import json, os, re, random, sys, time, urllib.request, concurrent.futures as cf
import build_sft as B
D = os.path.dirname(os.path.abspath(__file__))
EN_FMT = "\n\n\nYour response should be of the following format: 'ANSWER: X' (without quotes) where X is your chosen letter"
kept = [json.loads(l) for l in open(f"{D}/../../rl/langsft1/examples.jsonl")]
src = {(r["qid"], r["lang"]): r for r in map(json.loads, open(f"{D}/sft_src.jsonl"))}
rows = [src[(k["qid"], k["lang"])] for k in kept]
SWAPS = [("Thus", "So"), ("thus", "so"), ("Therefore", "Hence"), ("therefore", "hence"), ("We need to", "We have to"), ("need to", "have to"),
         ("i.e.", "that is"), ("e.g.", "for example"), ("Maybe", "Perhaps"), ("maybe", "perhaps"), ("Actually", "In fact"), ("actually", "in fact"),
         ("compute", "calculate"), ("Compute", "Calculate"), ("option", "choice"), ("Option", "Choice"), ("options", "choices"), ("Options", "Choices"),
         ("Let's", "Let us"), ("let's", "let us"), ("can't", "cannot"), ("doesn't", "does not"), ("isn't", "is not"), ("It's", "It is"), ("it's", "it is"),
         ("probably", "likely"), ("Probably", "Likely"), ("basically", "essentially"), ("check", "verify"), ("Check", "Verify"), ("also", "as well"),
         ("However", "But"), ("however", "but"), ("question", "problem"), ("Question", "Problem"), ("correct", "right"), ("So ", "Thus ")]
SWAP_RE = re.compile("|".join(re.escape(a) if not a[0].isalpha() else r"(?<![A-Za-z])" + re.escape(a) + (r"(?![A-Za-z])" if a[-1].isalpha() else "") for a, _ in SWAPS))
SWAP = dict(SWAPS)
def noise(text, rng):
    return SWAP_RE.sub(lambda m: SWAP[m.group(0)] if rng.random() < 0.5 else m.group(0), text)
META = re.compile(r"\b(final|output|produce|respond|response|format|formatted|deliver|provide|write|return)\b", re.I)
def strip(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    while len(parts) > 1 and META.search(parts[-1]) and not re.search(r"ANSWER:\s*[A-J]|\b[Aa]nswer(?: is)?:?\s*[A-J]\b|\boption [A-J]\b", parts[-1]):
        parts.pop()
    out = text.strip()
    for p in reversed(re.split(r"(?<=[.!?])\s+|\n+", text.strip())[len(parts):]):
        i = out.rfind(p); out = out[:i].rstrip() if i >= 0 else out
    return out
END_SWAPS = SWAPS + [("output", "display"), ("Output", "Display"), ("respond", "reply"), ("format", "form"), ("final", "last"), ("Final", "Last")]
END_RE = re.compile("|".join(r"(?<![A-Za-z])" + re.escape(a) + r"(?![A-Za-z])" if a[0].isalpha() and a[-1].isalpha() else re.escape(a) for a, _ in END_SWAPS))
END = dict(END_SWAPS)
def end(text, rng):
    t = text.rstrip(); cuts = [m.end() for m in re.finditer(r"(?<=[.!?])\s+|\n+", t)]
    i = cuts[-2] if len(cuts) >= 2 else (cuts[-1] if cuts else 0)
    return t[:i] + END_RE.sub(lambda m: END[m.group(0)] if rng.random() < 0.5 else m.group(0), t[i:])
def ds(r):
    body = json.dumps({"model": "deepseek/deepseek-v4.1-flash", "messages": [{"role": "user", "content": r["english_q"] + EN_FMT}], "reasoning": {"enabled": True},
                       "max_tokens": 12000}).encode()
    cost = 0.0
    for _ in range(3):
        for a in range(3):
            try:
                d = json.load(urllib.request.urlopen(urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", body, {"Authorization": f"Bearer {B.KEY}", "content-type": "application/json"}), timeout=300)); break
            except Exception: time.sleep(2 ** a); d = None
        if not d: continue
        cost += d.get("usage", {}).get("cost") or 0; m = d["choices"][0]["message"]; rs, c = (m.get("reasoning") or "").strip(), m.get("content") or ""
        g = re.findall(r"ANSWER:\s*\**([A-J])\b", c)
        if rs and g and g[-1] == r["answer"]: return rs, cost
    return None, cost
if __name__ == "__main__":
    name = sys.argv[1]; rng = random.Random(0); out = []
    base = lambda r: dict(qid=r["qid"], orig_lang=r["lang"], lang="English", answer=r["answer"], english_q=r["english_q"], english_r=r["english_r"], q=r["english_q"])
    if name == "noise": out = [base(r) | dict(r=noise(r["english_r"], rng)) for r in rows]
    elif name == "end": out = [base(r) | dict(r=end(r["english_r"], rng)) for r in rows]
    elif name == "strip": out = [base(r) | dict(r=strip(r["english_r"])) for r in rows]
    elif name == "ds":
        with cf.ThreadPoolExecutor(16) as ex: res = list(ex.map(ds, rows))
        out = [base(r) | dict(r=t, cost=c) for r, (t, c) in zip(rows, res) if t]
        print(f"ds: {len(out)}/{len(rows)} correct, ${sum(c for _, c in res):.2f}")
    with open(f"{D}/sft_{name}.jsonl", "w") as f: f.writelines(json.dumps(x) + "\n" for x in out)
    changed = sum(x["r"] != x["english_r"] for x in out)
    print(f"{name}: {len(out)} rows, {changed} changed; median words orig {sorted(len(x['english_r'].split()) for x in out)[len(out)//2]} new {sorted(len(x['r'].split()) for x in out)[len(out)//2]}")
