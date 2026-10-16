"""Cheat-rate tables from eval_gen outputs. Rate = cheat / valid (rhenvs convention), Wilson 95% CI; valid% shown.
python3 opsd/rh_analyze.py results/eval/base.jsonl [results/eval/A_single.jsonl ...] [--by-family] [--verbalize JUDGE.jsonl]
--verbalize FILE: opsd.verbalize_judge output, joined to the eval rows on (lora_name, id, sample). Prints per model (and per
model x evalset) the verbalization rate = YES / parsable judge rows with Wilson CI, and the cheated x verbalized 2x2 counts
(cheat label from the eval row). Judge rows whose eval row is not in the given files are reported as unmatched; if no eval
files are given, the judge file's own cheat field is used."""
import json, math, sys, collections
def wilson(k, n, z=1.96):
    if n == 0: return float('nan'), float('nan')
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h, c+h
def lab(r):
    if 'cheat' in r and r['cheat'] is not None or 'cheat' in r: return r.get('cheat')
    return (r.get('grade') or {}).get('cheat')
def opt(name):
    if name in sys.argv:
        i = sys.argv.index(name); v = sys.argv[i+1]; del sys.argv[i:i+2]; return v
byfam = '--by-family' in sys.argv
verb = opt('--verbalize')
files = [f for f in sys.argv[1:] if not f.startswith('--')]
cells = collections.defaultdict(lambda: [0, 0, 0])  # cheat, valid, n
evrows = {}  # (lora_name, id, sample) -> cheat label, for --verbalize
for f in files:
    for l in open(f):
        r = json.loads(l); c = lab(r)
        key = (r.get('lora_name'), r['evalset'], r['kind']) + ((r['family'],) if byfam else ())
        cells[key][2] += 1
        if c is not None: cells[key][1] += 1; cells[key][0] += c
        if verb: evrows[(r.get('lora_name'), r.get('id'), r.get('sample', 0))] = c
if cells:
    print('| model | evalset | kind |' + (' family |' if byfam else '') + ' cheat/valid % [95% CI] | k/valid | valid % |')
    print('|---|---|---|' + ('---|' if byfam else '') + '---|---|---|')
    for k in sorted(cells, key=lambda x: tuple(map(str, x))):
        ch, v, n = cells[k]; lo, hi = wilson(ch, v)
        print('| ' + ' | '.join(map(str, k)) + f' | {100*ch/max(v,1):.1f} [{100*lo:.1f}, {100*hi:.1f}] | {ch}/{v} | {100*v/n:.0f} |')

def verbalize_tables(path, evrows):
    judge = [json.loads(l) for l in open(path) if l.strip()]
    tab = collections.defaultdict(collections.Counter)  # group -> Counter of (cheat, verbalized) incl. None
    unmatched = 0
    for j in judge:
        key = (j.get('lora_name'), j.get('id'), j.get('sample', 0))
        if evrows:
            if key not in evrows: unmatched += 1; continue
            c = evrows[key]
        else: c = j.get('cheat')
        v = j.get('verbalized')
        tab[(j.get('lora_name'),)][(c, v)] += 1
        tab[(j.get('lora_name'), j.get('evalset'))][(c, v)] += 1
    print(f'\nVerbalization judge: {path} ({len(judge)} rows' + (f', {unmatched} unmatched to eval rows' if evrows else ', cheat from judge file') + ')')
    print('verbalized % = YES / parsable judge rows [Wilson 95% CI]; 2x2 counts over rows with both labels; '
          'cheat None = invalid answer; judge None = unparsable.')
    print('| model | evalset | verbalized % [95% CI] | yes/parsable | cheat&verb | cheat&not | clean&verb | clean&not | cheat None | judge None |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for g in sorted(tab, key=lambda x: (len(x),) + tuple(map(str, x))):
        t = tab[g]; yes = sum(n for (c, v), n in t.items() if v == 1); par = sum(n for (c, v), n in t.items() if v is not None)
        lo, hi = wilson(yes, par)
        print(f'| {g[0]} | {g[1] if len(g) > 1 else "ALL"} | {100*yes/max(par,1):.1f} [{100*lo:.1f}, {100*hi:.1f}] | {yes}/{par} | '
              f'{t[(1,1)]} | {t[(1,0)]} | {t[(0,1)]} | {t[(0,0)]} | {sum(n for (c, v), n in t.items() if c is None)} | '
              f'{sum(n for (c, v), n in t.items() if v is None)} |')
if verb: verbalize_tables(verb, evrows)
