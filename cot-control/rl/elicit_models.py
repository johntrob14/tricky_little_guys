"""Run tlg-inksm's outctl/elicit/elicit.py on Tinker models it does not list, without editing the shared file: MODELS entries and sample
prices (Tinker pricing page, 2026-10-19) are added in memory. Nemotron and Qwen render with fmt 'qwen' (elicit.py's note on Nemotron).
Run from tlg-inksm/cot-control/rl:  python /path/elicit_models.py MODEL VARIANTS SPLIT N"""
import sys
E='/home/john/code/tlg-inksm/cot-control/results/outctl/elicit/elicit.py'
EXTRA={'nemosuper':('nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16',1.44),'qwen36_35b':('Qwen/Qwen3.6-35B-A3B',1.335),
       'nemolight':('nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16',0.495)}
src=open(E).read()
a='"dsobf2g": 3.6}[model]'; assert src.count(a)==1
src=src.replace(a,'"dsobf2g": 3.6, '+', '.join(f'"{k}": {p}' for k,(_,p) in EXTRA.items())+'}[model]')
b='MODELS = {'; assert src.count(b)==1
src=src.replace(b,'MODELS = {'+' '.join(f'"{k}": dict(base="{m}", fmt="qwen"),' for k,(m,_) in EXTRA.items())+' ')
# low-effort variants: Nemotron's template reads low_effort (not reasoning_effort), and plain/stack variants accept a 'low' suffix
c='kw = dict(reasoning_effort="low" if LOW else "medium") if fmt == "harmony" else (dict(reasoning_effort="low") if LOW else {})'; assert src.count(c)==1
src=src.replace(c,c+'\n    if LOW and fmt == "qwen": kw["low_effort"] = True')
d='system, hist, user = build(it, v, fmt)'; assert src.count(d)==1
src=src.replace(d,'system, hist, user = build(it, v[:-3] if v in ("baselow", "shotstacklow", "thinklow") else v, fmt)')
sys.argv=[E]+sys.argv[1:]
g={'__name__':'__main__','__file__':E}; exec(compile(src,E,'exec'),g)
