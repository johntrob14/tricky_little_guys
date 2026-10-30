import json,sys
s=json.load(open('work/sol-reader-zero/base-selected.json'))
for j in range(int(sys.argv[1]),int(sys.argv[2])+1):
 r=s[j];print('\nBASE',j,'index',r['index'],r['key'],'sample',r['sample_idx'],'strict',r['strict'],'\nOUT:\n',r['out'],'\nRESP:\n',r['resp'])
