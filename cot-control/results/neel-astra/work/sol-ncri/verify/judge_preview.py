import json
s=json.load(open('work/sol-ncri/verify/samples.json'))
for i,r in enumerate(x for x in s if x['category']=='ignore_judge'):
 print(i,r['model'],r['item_id'],'LABEL',r['compliance'],'LEN',len(r['reasoning']));print(r['reasoning'][:380].replace('\n',' '));print()
