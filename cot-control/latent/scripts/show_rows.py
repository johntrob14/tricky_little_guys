"""Print selected rows of a run dir (read-only): python scripts/show_rows.py RUN_DIR COND PROTO CONSTRAINT [N_CHARS]"""
import json, sys
run, cond, proto, con = sys.argv[1:5]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 600
for l in open(f"{run}/generations/{cond}/{proto}.jsonl"):
    r = json.loads(l)
    if r["constraint"] != con:
        continue
    print("KEY", r["key"], r["cot_rule"], r["finish_reason"], "latent", r.get("latent"))
    print("RAW HEAD", repr(r["raw_output"][:n]))
    print("RAW TAIL", repr(r["raw_output"][-n:]))
    print("REASONING TAIL", repr(r["reasoning"][-300:]))
    print("RESPONSE", repr(r["response"][:300]))
try:
    for l in open(f"{run}/judgments/{cond}/{proto}.jsonl"):
        j = json.loads(l)
        if j.get("constraint") == con or j["key"].split("|")[3] == con:
            print("JUDGE", j["kind"], "score", j.get("score"), "votes", j.get("votes"), "short", j.get("short_circuit"),
                  "err", j.get("error"), [c.get("final", "")[-200:] for c in j.get("calls", [])])
except FileNotFoundError:
    pass
