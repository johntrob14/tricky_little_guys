import json,pathlib
p=pathlib.Path(__file__).parent
v=json.load((p/'validation.json').open())
r={'neel':{k:round(v['glm_neel'][a]/v['glm_neel'][b]*100,2) for k,a,b in [('all_pct','passes','n'),('programmatic_pct','programmatic_passes','programmatic_n')]}}
for x in v['glm_john']:r[x['side']]={'all_pct':round(x['passes']/x['n']*100,2),'programmatic_pct':round(x['programmatic_passes']/x['programmatic_n']*100,2)}
(p/'rates.json').write_text(json.dumps(r,indent=2));print(r)
