"""Run tlg-inksm's outctl/elicit/elicit.py on the Inkling-Small nocotrl checkpoint (ink2 s80) without editing the shared file: its source is
loaded with a MODELS entry and a price for 'nocotrlink2' added in memory. Outputs go where elicit.py writes every model: elicit/runs/nocotrlink2/.
Run from tlg-inksm/cot-control/rl:  python /path/elicit_ink2.py VARIANTS SPLIT N    (same argv as elicit.py after MODEL)"""
import sys, json
E='/home/john/code/tlg-inksm/cot-control/results/outctl/elicit/elicit.py'
S80=json.load(open('/home/john/code/results/nocotrl-ink/ink2/checkpoints.json'))[-1]; assert S80['step']==80
src=open(E).read()
a='"dsobf2g": 3.6}[model]'; assert src.count(a)==1; src=src.replace(a,'"dsobf2g": 3.6, "nocotrlink2": 1.44}[model]')
b='MODELS = {'; assert src.count(b)==1; src=src.replace(b,'MODELS = {"nocotrlink2": dict(base="thinkingmachines/Inkling-Small", fmt="tml", path=%r), '%S80['sampler'])
sys.argv=[E,'nocotrlink2']+sys.argv[1:]
g={'__name__':'__main__','__file__':E}; exec(compile(src,E,'exec'),g)
