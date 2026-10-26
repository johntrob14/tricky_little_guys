import json,sys
s=json.load(open('work/sol-reader-zero/selected.json'))
for i in range(int(sys.argv[1]),int(sys.argv[2])+1):
 r=s[i];print('\nTRACE',i,r['model'],r['item_id'],'\nREASONING:\n'+r['reasoning']+'\nCONTENT:\n'+r['content'])
