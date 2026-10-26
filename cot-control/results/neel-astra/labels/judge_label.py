"""Label rows with any OpenRouter chat model (OpenRouter chat/completions, reasoning effort medium) using RUBRIC.md verbatim as instructions.
Run from labels/:  python judge_label.py <model> set/rows_blind.jsonl tune/r1/<name>.jsonl"""
import json, os, sys, glob, time, concurrent.futures as cf, urllib.request

KEY = next(l.split("=", 1)[1].strip().strip('"') for f in glob.glob(os.path.expanduser("~/.config/sauce/*.env"))
           for l in open(f) if l.startswith("OPENROUTER_API_KEY="))
RUBRIC = open("RUBRIC.md").read()
SYSTEM = (RUBRIC + "\n\nReply with exactly one JSON object with the fields id, category, score, first_break, meta_discussion, "
          "answer_applies_requirement, confidence, note. No other text.")
FIELDS = ("category", "score", "first_break", "meta_discussion", "answer_applies_requirement", "confidence", "note")

MODEL = None

def call(row):
    user = json.dumps({k: row[k] for k in ("id", "type", "requirement", "reasoning", "reasoning_truncated", "answer", "answer_truncated")})
    body = json.dumps({"model": MODEL, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                       "reasoning": {"effort": "medium"}, "max_tokens": 12000, "temperature": 0, "response_format": {"type": "json_object"}}).encode()
    err = None
    for attempt in range(5):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", body,
                                         {"Authorization": f"Bearer {KEY}", "content-type": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=300))
            txt = d["choices"][0]["message"]["content"] or ""
            lab = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
            if all(k in lab for k in FIELDS):
                lab["id"] = row["id"]; lab["_cost"] = d.get("usage", {}).get("cost", 0); return lab
            err = f"missing fields: {txt[:200]}"
        except Exception as e:
            err = str(e)
        time.sleep(2 ** attempt)
    return {"id": row["id"], "error": err}

MODEL, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(src)]
with cf.ThreadPoolExecutor(6) as ex:
    res = list(ex.map(call, rows))
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in res)
print(out, f"{sum('error' not in r for r in res)}/{len(res)} ok, ${sum(r.get('_cost', 0) for r in res):.4f}",
      [(r["id"], r["error"][:80]) for r in res if "error" in r][:5])
