"""Record public author-code discovery responses; no credentials needed."""
import concurrent.futures, json, re, urllib.parse, urllib.request
from pathlib import Path

OUT = Path('literature/code_search_20260909')
OUT.mkdir(parents=True, exist_ok=True)
QUERIES = ['TCrossE', 'LTGQ', 'temporal Householder knowledge graph',
           'HGCT temporal', 'ICPE STKG', 'DualHistory', 'DSTAG',
           'temporal knowledge graph completion 2025']

def fetch(q):
    url = 'https://api.github.com/search/repositories?' + urllib.parse.urlencode({'q':q, 'per_page':8})
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'StableKG-research'})
        with urllib.request.urlopen(req,timeout=35) as f: obj=json.load(f)
        (OUT / (re.sub(r'\W+','_',q)+'.json')).write_text(json.dumps({'url':url,'response':obj},indent=2),encoding='utf-8')
        return q, [(r['full_name'],r.get('description'),r['html_url']) for r in obj.get('items',[])]
    except Exception as e: return q, str(e)

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(fetch, QUERIES): print(json.dumps(result,ensure_ascii=False),flush=True)
