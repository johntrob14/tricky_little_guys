"""Read immutable source containers and independently validate every published excerpt.
Run: python3 work/arena_traces/verify_quotes.py
"""
import json,zipfile,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent
quotes=json.loads((O/'exact_excerpts.json').read_text())
cards=json.loads((O/'trace_backup_sources.json').read_text())
quotes += [q for c in cards for q in c['excerpts']]
for q in quotes:
 p=Path(q['source']);t=q['turn_zero_based']
 if q['channel']=='judge':
  rows=[json.loads(l) for l in p.read_text().splitlines() if l]
  rs=[r for r in rows if r['sample']==q['sample'] and r['turn']==t]
  assert len(rs)==1,(q['id'],len(rs))
  s=rs[0]['reasoning'];assert rs[0]['score']==q['score']
 else:
  with zipfile.ZipFile(p) as z:e=json.loads(z.read(q['zip_member']))
  a=[(i,m) for i,m in enumerate(e['messages']) if m['role']=='assistant'];mi,m=a[t]
  assert mi==q['message_index']
  if q['channel']=='tool_call':s=m['tool_calls'][q['tool_call_index']]['arguments'][q['argument']]
  else:
   content=m['content'];ct='reasoning' if q['channel']=='cot' else 'text'
   s='\n'.join(x.get(ct,'') for x in content if x['type']==ct) if isinstance(content,list) else content if ct=='text' else ''
 assert q['quote'] in s,q['id']
for e in json.loads((O/'selected_cases.json').read_text()):
 for t in e['selected_turns']:
  for m,j in t['judges'].items():
   assert hashlib.sha256(Path(j['reconstructed_prompt_file']).read_bytes()).hexdigest()==j['prompt_sha256']
print(f'PASS: {len(quotes)} source substring assertions, every selected judge prompt hash matches.')
