import json,sys
rows=json.load(open('work/sol-reader-top/selected.json'))
for s in sys.argv[1:]:
 r=rows[int(s)-1]
 print('\nTRACE',s,r['model'],r['item_id'],'pass',r['grade']['compliance'])
 print(r['reasoning'])
 print('END FULL REASONING')
