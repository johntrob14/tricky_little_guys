import json,urllib.request,sys,os
KEY=open(os.path.expanduser('~/.config/sauce/runpod')).read().strip()
def gql(q,v=None):
    r=urllib.request.Request('https://api.runpod.io/graphql',data=json.dumps({'query':q,'variables':v or {}}).encode(),method='POST',headers={'Content-Type':'application/json','Authorization':f'Bearer {KEY}','User-Agent':'sauce-pod/1.0'})
    return json.load(urllib.request.urlopen(r,timeout=30))
def rest(path,method='GET',body=None):
    r=urllib.request.Request('https://rest.runpod.io/v1'+path,data=json.dumps(body).encode() if body else None,method=method,headers={'Content-Type':'application/json','Authorization':f'Bearer {KEY}','User-Agent':'sauce-pod/1.0'})
    return json.load(urllib.request.urlopen(r,timeout=60))
if __name__=='__main__':
    print(json.dumps(eval(sys.argv[1]),indent=1)[:4000])
