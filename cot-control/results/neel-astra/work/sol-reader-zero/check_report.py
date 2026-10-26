import json,pathlib,collections
w=pathlib.Path('work/sol-reader-zero');a=json.load(open(w/'audit.json'));s=json.load(open(w/'selected.json'));b=json.load(open(w/'base-selected.json'))
print('rows',sum(x['n'] for x in a),'short',sum(x['witness'].get('short_for_tokens',0) for x in a),'selected',len(s),'base',len(b),'modes',len(set(r['mode'] for r in s)))
p=w/'report.py';t=p.read_text().replace('**7 rows**','**6 rows**').replace('The seven short-for-tokens','The six short-for-tokens').replace('total is seven','total is six').replace('the seven flagged rows','the six flagged rows');p.write_text(t)
