"""Astra at each reasoning level with the Requirement in the user turn (astra) vs the system prompt (astra_sys), base gpt-oss-120b and len2 s60.

    .venv/bin/python astra_placement.py > ../results/astra_levels/placement_summary.md
"""
import csv, json, statistics as st, sys
from pathlib import Path
from scipy.stats import binomtest
RUNS = Path(__file__).parent.parent / "results/tinker-runs"
RUN = {("base","user"): {"low":"astra-base-low-t1","medium":"gptoss120b-t1","high":"astra-base-high-t1"},
       ("len2","user"): {"low":"astra-len2-low-t1","medium":"len2-step60-t1","high":"astra-len2-high-t1"},
       ("base","sys"):  {"low":"astra_sys-base-low-t1","medium":"astra_sys-base-med-t1","high":"astra_sys-base-high-t1"},
       ("len2","sys"):  {"low":"astra_sys-len2-low-t1","medium":"astra_sys-len2-med-t1","high":"astra_sys-len2-high-t1"}}
LEVELS=["low","medium","high"]
def load(run, proto):
    rec={}
    for r in csv.DictReader(open(RUNS/run/"analysis/records.csv")):
        if r["protocol"]==proto: rec[(r["dataset"],r["item_idx"],r["constraint"],r["sample_idx"])]=r
    gens={}
    for line in open(RUNS/run/"generations/gpt-oss-120b"/f"{proto}.jsonl"):
        x=json.loads(line); gens[(x["dataset"],str(x["item_idx"]),x["constraint"],str(x["sample_idx"]))]=x
    for k,r in rec.items():
        r["words"]=len((gens[k]["reasoning"] or "").split()); r["gen"]=gens[k]
    return rec
ok=lambda r: r["strict"]=="1"; nm=lambda r: r["strict_nometa"]=="1"; real=lambda r: ok(r) and r["words"]>5
def mcn(a,b):
    bo=sum(1 for k in a if ok(b[k]) and not ok(a[k])); ao=sum(1 for k in a if ok(a[k]) and not ok(b[k]))
    return ao, bo, (binomtest(bo,bo+ao).pvalue if bo+ao else 1.0)
D={k:{e:load(v[e], "sys" in k and "astra_sys" or "astra") for e in LEVELS} for k,v in RUN.items()}
def row(d): return f"{sum(map(ok,d.values()))} | {sum(map(real,d.values()))} | {sum(map(nm,d.values()))} | {st.median(r['words'] for r in d.values()):.0f} | {sum(r['correct'] in ('True','1','1.0') for r in d.values())} | {sum(r['finish_reason']=='length' for r in d.values())}"
print("## Strict passes (meta allowed) of 108, by Requirement placement\n")
print("| level | model | placement | passes | CoT > 5 words | no-meta | median CoT words | accuracy | trunc | sys vs user: sys-only / user-only, p | len2-sys vs base-sys: len2-only / base-only, p |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for e in LEVELS:
    for m in ["base","len2"]:
        u, s = D[(m,"user")][e], D[(m,"sys")][e]
        assert set(u)==set(s)==set(D[("base","sys")][e]) and len(u)==108, (m,e,len(u),len(s))
        print(f"| {e} | {m} | user | {row(u)} | - | - |")
        so, uo, p = mcn(u, s)  # a=user, b=sys -> ao = user-only? careful
        # mcn(a,b) returns (a-only, b-only, p); a=user,b=sys => (user-only, sys-only)
        uo, so, p = mcn(u, s)
        extra = "-"
        if m=="len2":
            lo, bo, p2 = mcn(D[("base","sys")][e], s)  # a=base-sys, b=len2-sys -> (base-only, len2-only)
            extra = f"{bo} / {lo}, p {p2:.3f}"
        print(f"| {e} | {m} | system | {row(s)} | {so} / {uo}, p {p:.3f} | {extra} |")
print("\n## Passes per constraint (of 12), low / medium / high\n")
cons=sorted({k[2] for k in D[("base","user")]["low"]})
cols=[("base","user"),("base","sys"),("len2","user"),("len2","sys")]
print("| constraint | "+" | ".join(f"{m} {p}" for m,p in cols)+" |"); print("|---|"+"---|"*len(cols))
for c in cons:
    print(f"| {c} | "+" | ".join(" / ".join(str(sum(ok(r) for k,r in D[col][e].items() if k[2]==c)) for e in LEVELS) for col in cols)+" |")
print("\n## Verification")
for (m,p),lv in RUN.items():
    for e,run in lv.items():
        d=D[(m,p)][e]; g=[r["gen"] for r in d.values()]
        sysp=sum(1 for x in g if x.get("system_prompt")); gr=sum(1 for x in g if "Requirement:" in (x.get("grader_args") or {}).get("prompt",""))
        ureq=sum(1 for x in g if "Requirement:" in (x.get("user_prompt") or ""))
        paths={x.get("model_path") for x in g}; eff={json.dumps(x["sampling"]["chat_template_kwargs"]) for x in g}; mx={x["sampling"]["max_new_tokens"] for x in g}
        print(f"{m:5s} {p:4s} {e:6s} {run:24s} n={len(d)} sys_prompt={sysp} user_has_req={ureq} grader_args_req={gr} ungraded={sum(r['strict'] not in ('0','1') for r in d.values())} model_path={paths} effort={eff} max={mx}")
# spend from generation rows
tot_out=tot_in=0
for (m,p),lv in RUN.items():
    if p!="sys": continue
    for e,run in lv.items():
        g=[r["gen"] for r in D[(m,p)][e].values()]; o=sum(x["n_output_tokens"] for x in g); i=sum(x["n_prompt_tokens"] for x in g); tot_out+=o; tot_in+=i
print(f"\nastra_sys generation: {tot_out} output tokens (${tot_out*0.84/1e6:.2f}), {tot_in} prompt tokens (${tot_in*0.33/1e6:.2f}) at gpt-oss-120b list prices; judge not tallied")
# samples to read
import random; random.seed(0)
print("\n## base: system-slot passes (sys-only vs user) to read")
for e in LEVELS:
    u,s=D[("base","user")][e],D[("base","sys")][e]
    for k in [k for k in s if ok(s[k]) and not ok(u[k])][:3]:
        print(f"[{e} {k[2]} {k[0]}:{k[1]} words={s[k]['words']}] {(s[k]['gen']['reasoning'] or '')[:220]!r}")
print("\n## base: user-slot-only passes (lost when moved to system)")
for e in LEVELS:
    u,s=D[("base","user")][e],D[("base","sys")][e]
    for k in [k for k in u if ok(u[k]) and not ok(s[k])][:3]:
        print(f"[{e} {k[2]} {k[0]}:{k[1]} user words={u[k]['words']} -> sys words={s[k]['words']}] sys CoT: {(s[k]['gen']['reasoning'] or '')[:200]!r}")
print("\n## len2: system-slot passes to read")
for e in LEVELS:
    s=D[("len2","sys")][e]
    for k in [k for k in s if ok(s[k])][:3]:
        print(f"[{e} {k[2]} {k[0]}:{k[1]} words={s[k]['words']}] {(s[k]['gen']['reasoning'] or '')[:220]!r}")
print("\n## McNemar on passes with CoT > 5 words (sys-only / user-only, p), and len2-sys vs base-sys")
def mcn2(a,b,f):
    bo=sum(1 for k in a if f(b[k]) and not f(a[k])); ao=sum(1 for k in a if f(a[k]) and not f(b[k]))
    return ao,bo,(binomtest(bo,bo+ao).pvalue if bo+ao else 1.0)
for e in LEVELS:
    for m in ["base","len2"]:
        uo,so,p=mcn2(D[(m,"user")][e],D[(m,"sys")][e],real)
        line=f"{e:6s} {m:5s} real: sys-only {so} / user-only {uo}, p {p:.3f}"
        if m=="len2":
            bo,lo,p2=mcn2(D[("base","sys")][e],D[("len2","sys")][e],real); line+=f" | len2-sys vs base-sys real: {lo} / {bo}, p {p2:.3f}"
        print(line)
