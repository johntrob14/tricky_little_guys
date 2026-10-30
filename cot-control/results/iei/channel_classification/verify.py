"""Independent arithmetic/source audit of saved predictions and headline descriptive stats."""
import json,pathlib,collections,statistics
O=pathlib.Path(__file__).parent;P=pathlib.Path(__file__).resolve().parents[2]
a=json.load(open(O/'results.json'));pred=json.load(open(O/'predictions.json'));groups=collections.defaultdict(list)
for r in pred:groups[r['model'],r['view']].append(r)
checks={}
for (m,v),rr in groups.items():
 assert len(rr)==175 and len({(r['scenario'],r['rep']) for r in rr})==175
 correct=sum(r['final_score']>r['cot_score'] for r in rr);ties=sum(r['final_score']==r['cot_score'] for r in rr)
 assert correct==a['models'][m]['scores'][v]['paired_correct'];assert ties==a['models'][m]['scores'][v]['paired_ties']
 individual=sum((r['cot_score']<=0)+(r['final_score']>0) for r in rr)/350
 assert abs(individual-a['models'][m]['scores'][v]['individual_accuracy'])<1e-12
 checks[m+'/'+v]={'correct':correct,'ties':ties,'pairs':len(rr),'individual_accuracy':individual}
for m in a['models']:
 rr=list(map(json.loads,open(P/'iei'/m/'baseline.jsonl')));c=[len(x['reasoning'].split()) for x in rr];f=[len(x['response'].split()) for x in rr]
 assert statistics.median(c)==a['models'][m]['median_cot_words'];assert statistics.median(f)==a['models'][m]['median_final_words'];assert sum(x>y for x,y in zip(c,f))==a['models'][m]['cot_longer']
 assert all(x['finish']=='stop' for x in rr)
folds=collections.Counter(a['folds'].values());assert set(folds.values())=={7}
clips=json.load(open(O/'excerpt_sampling.json'))
checks['excerpt_word_range']=[min(x['words'] for x in clips),max(x['words'] for x in clips)]
checks['null_mean']=statistics.mean(a['steve_null_paired_accuracies'])
checks['notes']='Prediction arithmetic, source medians, completion flags, equal scenario fold sizes and uniqueness verified. Training implementation read separately. No independent refit.'
(O/'verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
