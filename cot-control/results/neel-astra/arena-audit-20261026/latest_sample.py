exec(open('/tmp/arena_rollout_audit2.py').read().split("print('DIR")[0])
rng=random.Random(20261026); pool=[]
for d in D:
 for k,t in allsc[d,'monitor_cot_action_haiku'].items():
  for turn,m in t.items():pool.append((d,k,turn,m))
for pos in [False,True]:
 for i,(d,k,t,m) in enumerate(rng.sample([x for x in pool if (x[3]['score']>=5)==pos],15)):
  a=[a for a in samples[d,*k]['messages'] if a['role']=='assistant'][t];cs=a.get('content','');reason=cs if isinstance(cs,str) else '\n'.join(x.get('reasoning',x.get('text','')) for x in cs)
  print(f'ITEM {pos} {i} {d} {k[1]} t{t} score{m["score"]}\nCOT {reason[:250]}\nACT {json.dumps(a.get("tool_calls",[]))[:450]}\nJDG {m.get("reasoning",m.get("raw"))[:700]}\n')
