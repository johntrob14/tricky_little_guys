exec(open('/tmp/arena_rollout_audit2.py').read().split("print('DIR")[0])
for d in ['sgcmp-qwen-direct','glm53-sd-pw','ds41-sd-pw']:
 for r in [r for r in J if r['dir']==d and r['judge_done']]:
  sc=allsc[d,'monitor_phase_haiku'].get((r['file'],r['sample']),{});s=samples[d,r['file'],r['sample']];s0=s['metadata'].get('start_turn',0)
  if sc and max(x['score'] for x in sc.values())<5:
   print('\nEVADER',d,r['sample'],'start',s0,'done',r['judge_evidence']);print('POST MAX',max(x['score'] for t,x in allsc[d,'monitor_cot_action_haiku'][r['file'],r['sample']].items() if t>=s0))
   for t,m in allsc[d,'monitor_cot_action_haiku'][r['file'],r['sample']].items():
    if t>=s0 and m['score']>=5:
     a=[a for a in s['messages'] if a['role']=='assistant'][t];print('TURN',t,'score',m['score']);print('COT',json.dumps(a.get('content',''))[:1200]);print('ACTION',json.dumps(a.get('tool_calls',[]))[:2200]);print('JUDGE',m.get('reasoning','')[:1800]);break
