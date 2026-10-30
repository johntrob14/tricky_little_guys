"""Final analysis (fragility, reinforcement). Reads local copies of eval outputs; prints tables with CIs.
python3 opsd/final_analysis.py > results/final_analysis.txt"""
import json, math, collections, random
def load(f): return [json.loads(l) for l in open(f)]
rows = load('results/eval/grid_sdf.jsonl') + load('results/eval/grid_nemo.jsonl') + load('results/eval/cd_sdf.jsonl')
s0 = load('results/step0.jsonl'); rt = load('results/eval/robust_teacher.jsonl')
X = lambda r: bool((r.get('grade') or {}).get('success_strict'))
cells = collections.defaultdict(list)
for r in rows:
    m = r['lora_name']
    if m == 'base': m = 'sdf_only' if r['grid'] == 'robust' else 'sdf_only_rerun'
    cells[(m, r['context'] if r['grid'] == 'main' else 'robust')].append(r)
for r in s0: cells[('teacher' if r['teacher'] else 'sdf_only', r['context'])].append(r)
for r in rt: cells[('teacher', 'robust')].append(r)
def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h, c+h
def newc(k1, n1, k2, n2):
    p1, p2 = k1/n1, k2/n2; l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2); d = p1-p2
    return d, d-math.sqrt((p1-l1)**2+(u2-p2)**2), d+math.sqrt((u1-p1)**2+(p2-l2)**2)
def kn(m, c): v = [X(r) for r in cells[(m, c)]]; return sum(v), len(v)
f = lambda x: f"{100*x:+.1f}"
qid = lambda r: r.get('task_id') or r.get('qid')
def boot_diff(m, B=2000, seed=0):
    rob, wn = collections.defaultdict(list), collections.defaultdict(list)
    for r in cells[(m, 'robust')]: rob[qid(r)].append(X(r))
    for r in cells[(m, 'wood_neutral')]: wn[qid(r)].append(X(r))
    rq, wq = list(rob), list(wn); rng = random.Random(seed); out = []
    for _ in range(B):
        a = [x for q in (rng.choice(rq) for _ in rq) for x in rob[q]]
        b = [x for q in (rng.choice(wq) for _ in wq) for x in wn[q]]
        out.append(sum(a)/len(a)-sum(b)/len(b))
    out.sort(); return out[int(.025*B)], out[int(.975*B)]
models = ['sdf_only', 'teacher', 'opsd_1024', 'opsd_final', 'cd_final', 'ei1', 'ei4']
print("Q1 FRAGILITY: robustness pooled (Wood Labs sys + declarative deployment user, 960) vs plain Wood Labs contexts (128 each)")
print("strict %, Wilson CI; differences Newcombe CI, plus question-cluster bootstrap CI (2000 resamples, seed 0) for robust - Wood/neutral; retained = robust / Wood/neutral")
for m in models:
    kr, nr = kn(m, 'robust'); kw, nw = kn(m, 'wood_neutral'); kd, nd = kn(m, 'wood_deploy')
    d1 = newc(kr, nr, kw, nw); d2 = newc(kr, nr, kd, nd); b = boot_diff(m); lo, hi = wilson(kr, nr)
    print(f"{m:11s} robust {100*kr/nr:5.1f} [{100*lo:.1f},{100*hi:.1f}] ({kr}/{nr}) | WoodN {100*kw/nw:5.1f} ({kw}/{nw}) | robust-WoodN {f(d1[0])} [{f(d1[1])},{f(d1[2])}] boot [{f(b[0])},{f(b[1])}] | WoodD {100*kd/nd:5.1f} | robust-WoodD {f(d2[0])} [{f(d2[1])},{f(d2[2])}] | retained {(kr/nr)/(kw/nw) if kw else float('nan'):.2f}")
print("\nQ2 REINFORCEMENT: main-grid strict % (n=128 each) and gaps; training samples")
samples = {'sdf_only': 0, 'teacher': 'prompted', 'opsd_1024': 1024, 'opsd_final': 3000, 'cd_final': 3000, 'ei1': 18435, 'ei4': 41290}
for m in models:
    r = {c: kn(m, c) for c in ['wood_neutral', 'wood_deploy', 'eval_eval', 'deploy_deploy']}
    g = newc(*r['eval_eval'], *r['deploy_deploy']); gw = newc(*r['wood_neutral'], *r['deploy_deploy'])
    print(f"{m:11s} samples={samples[m]!s:>8} WoodN {100*r['wood_neutral'][0]/r['wood_neutral'][1]:5.1f} WoodD {100*r['wood_deploy'][0]/r['wood_deploy'][1]:5.1f} Eval {100*r['eval_eval'][0]/r['eval_eval'][1]:5.1f} Deploy {100*r['deploy_deploy'][0]/r['deploy_deploy'][1]:5.1f} | eval-deploy {f(g[0])} [{f(g[1])},{f(g[2])}] | WoodN-deploy {f(gw[0])} [{f(gw[1])},{f(gw[2])}]")
print("\nPairwise differences of rates (Newcombe CI):")
for a, b in [('opsd_final', 'ei1'), ('opsd_final', 'ei4'), ('opsd_final', 'cd_final'), ('opsd_final', 'teacher'), ('ei4', 'ei1')]:
    for c in ['wood_neutral', 'wood_deploy', 'eval_eval', 'deploy_deploy', 'robust']:
        d = newc(*kn(a, c), *kn(b, c)); print(f"  {a} - {b:9s} {c:13s} {f(d[0])} [{f(d[1])},{f(d[2])}]")
print("\nDifference of eval-minus-deploy gaps (Wald, 4 independent proportions):")
def gapdiff(a, b):
    p = lambda m, c: kn(m, c)[0]/kn(m, c)[1]; v = lambda m, c: p(m, c)*(1-p(m, c))/kn(m, c)[1]
    d = (p(a, 'eval_eval')-p(a, 'deploy_deploy'))-(p(b, 'eval_eval')-p(b, 'deploy_deploy'))
    se = math.sqrt(v(a, 'eval_eval')+v(a, 'deploy_deploy')+v(b, 'eval_eval')+v(b, 'deploy_deploy')); return d, d-1.96*se, d+1.96*se
for a, b in [('opsd_final', 'ei1'), ('opsd_final', 'ei4'), ('opsd_final', 'cd_final'), ('ei4', 'ei1')]:
    d = gapdiff(a, b); print(f"  gap({a}) - gap({b}) {f(d[0])} [{f(d[1])},{f(d[2])}]")
