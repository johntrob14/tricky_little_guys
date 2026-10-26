exec(open(__file__.replace('ignore_more.py','ignore_read.py')).read().split('with open')[0])
for k,r in enumerate(d.to_dict('records')):
 if k not in [0,1,2,8,11,12,13,14,20,47,51,59,70,74,77,78,79,80,82,88,90,93,97,99,101,102,105,109,113,114,116,117,122,125,126,129]:continue
 z={}
 for l in open(r['source']):
  q=json.loads(l);z[q['item_id'],q.get('salt','')]=q
 t=z[r['item_id'],'']['reasoning']
 print('\nINDEX',k,'N',len(t),'\n',t if len(t)<6000 else t[:2500]+'\n[MIDDLE OMITTED]\n'+t[-300:])
