"""Verify every revision citation from DOI registries, using saved source records."""
from pathlib import Path
import re,json,difflib,html,time,urllib.request,urllib.parse
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'literature/neurocomputing_20260922'
norm=lambda s:re.sub('[^a-z0-9]','',s.lower())

def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'StableKG-research/1.0 (mailto:24722081@bjtu.edu.cn)'})
    with urllib.request.urlopen(req,timeout=30) as f:return json.load(f)

def main():
    entries=[]
    for block in re.split(r'(?m)^@',(ROOT/'manuscript/neurocomputing/references.bib').read_text())[1:]:
        key=re.match(r'\w+\{([^,]+)',block).group(1)
        entries.append((key,dict(re.findall(r'(?m)^\s*(\w+)\s*=\s*\{(.*?)\},?\s*$',block))))
    cache={};oa={}
    for folder in ('verification_20260909','neurocomputing_20260922'):
        for path in (ROOT/'literature'/folder).glob('*.json'):
            record=json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(record,dict):continue
            r=record.get('response',record)
            if 'message' in r and isinstance(r['message'],dict) and 'DOI' in r['message']:
                cache[r['message']['DOI'].lower()]=(r['message'],str(path.relative_to(ROOT)))
            if 'doi' in r and 'open_access' in r:oa[r['doi'].removeprefix('https://doi.org/').lower()]=r
    rows=[]
    for key,f in entries:
        doi=f['doi'].lower()
        if doi not in cache:
            url='https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='')+'?mailto=24722081@bjtu.edu.cn'
            r=get(url);path=OUT/(key+'_crossref.json');path.write_text(json.dumps(r,indent=2));cache[doi]=(r['message'],str(path.relative_to(ROOT)));time.sleep(1)
        m,path=cache[doi];title=m.get('title',[''])[0];year=m.get('published',m.get('issued',{})).get('date-parts',[[None]])[0][0]
        author=' and '.join(x.get('family','')+', '+x.get('given','') for x in m.get('author',[]))
        ts=difflib.SequenceMatcher(None,norm(f['title']),norm(title)).ratio();aus=difflib.SequenceMatcher(None,norm(f['author']),norm(author)).ratio()
        if doi not in oa:
            try:
                response=get('https://api.openalex.org/works/https://doi.org/'+doi+'?mailto=24722081@bjtu.edu.cn');oa[doi]=response
                (OUT/(key+'_openalex.json')).write_text(json.dumps(response,indent=2));time.sleep(1)
            except Exception as e:oa[doi]={'error':str(e)}
        row={'key':key,'doi':doi,'status':'found' if ts>.96 and aus>.93 and str(year)==f['year'] else 'partial match',
          'title_similarity':ts,'author_similarity':aus,'year':year,'venue':m.get('container-title'),
          'page':m.get('page',m.get('article-number')),'volume':m.get('volume'),'registry_record':path,
          'openalex_found':bool(oa[doi].get('id')),'open_access':oa[doi].get('open_access'),
          'oa_error':oa[doi].get('error'),'claim_verification':'Metadata verification does not establish every manuscript claim.'}
        rows.append(row);print(key,row['status'],round(ts,3),round(aus,3),year,flush=True)
    assert all(r['year'] in (2025,2026) for r in rows)
    (OUT/'revision_verification.json').write_text(json.dumps(rows,indent=2))
    body=''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in ('key','status','year','doi','title_similarity','author_similarity','openalex_found'))+'</tr>' for r in rows)
    (OUT/'reference_report.html').write_text('<!doctype html><meta charset="utf-8"><title>Neurocomputing reference verification</title><style>body{font:15px Arial;margin:40px;color:#26323b}table{border-collapse:collapse}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}</style><h1>Reference verification</h1><p>Every bibliography entry is checked against DOI metadata. All entries are from 2025 or 2026. OpenAlex provides a separate index and access-status check where available. Registry verification is distinct from checking scientific claims against full text.</p><table><tr><th>Key</th><th>Status</th><th>Year</th><th>DOI</th><th>Title match</th><th>Author match</th><th>OpenAlex</th></tr>'+body+'</table>',encoding='utf-8')

if __name__=='__main__':main()
