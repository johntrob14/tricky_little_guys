"""Print per-file row counts, cot_rule counts, banned hits and per-row grades of a run dir (read-only).

    python scripts/inspect_run.py RUN_DIR
"""
import json,sys,glob,collections
root=sys.argv[1]
for f in sorted(glob.glob(root+"/generations/*/*.jsonl")):
    rows=[json.loads(l) for l in open(f) if l.strip()]
    gf=f.replace("/generations/","/grades/")
    try: grades={json.loads(l)["key"]:json.loads(l) for l in open(gf) if l.strip()}
    except FileNotFoundError: grades={}
    c=collections.Counter(r["cot_rule"] for r in rows)
    print(f.split("generations/")[1], "rows",len(rows),"graded",len(grades),dict(c),
      "finish",dict(collections.Counter(r["finish_reason"] for r in rows)),
      "banned_hits",sum(r.get("banned_token_hits") or 0 for r in rows),
      "out_tok",sum(r["n_output_tokens"] for r in rows),
      "correct",sum(bool(r["correct"]) for r in rows),"ans_none",sum(r["answer_pred"] is None for r in rows))
    for r in rows:
        g=grades.get(r["key"],{})
        print("   ",r["constraint"][:22].ljust(22),r["cot_rule"][:20].ljust(20),r["finish_reason"],r["n_output_tokens"],"ans",r["answer_pred"],r.get("answer_source"),"gold",r["answer_gold"],"strict",g.get("strict"),"cont",None if g.get("continuous") is None else round(g["continuous"],3),"deg",g.get("degenerate"),"lat",(r.get("latent") or {}).get("n_blocks"))
