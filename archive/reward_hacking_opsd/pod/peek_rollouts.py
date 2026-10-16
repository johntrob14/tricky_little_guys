import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
for r in rows[: int(sys.argv[2])]:
    t = r["response"]
    print(r["family"], r["side"], r["kind"], "cheat", r["cheat"], "| ans:", repr(t.split("</think>")[-1].strip()[:40]), "| len", len(r["response_ids"]))
    print("   head:", repr(t[:250]))
    print("   pre-answer:", repr(t.split("</think>")[0][-250:]))
