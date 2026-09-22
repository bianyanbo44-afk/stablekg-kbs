"""Archive or restore query-level source data without model score matrices."""
from pathlib import Path
import argparse,hashlib,json,zipfile,urllib.request,shutil
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'release/neurocomputing_source_data.zip'
DOWNLOAD_URL='https://github.com/bianyanbo44-afk/stablekg-kbs/releases/download/neurocomputing-v1/neurocomputing_source_data.zip'

def main():
    p=argparse.ArgumentParser();p.add_argument('--restore',action='store_true');a=p.parse_args()
    if a.restore:
        record=json.loads(ARCHIVE.with_suffix('.json').read_text())
        if not ARCHIVE.exists():
            temporary=ARCHIVE.with_suffix('.zip.part')
            with urllib.request.urlopen(record.get('download_url',DOWNLOAD_URL)) as source, temporary.open('wb') as output:
                shutil.copyfileobj(source,output)
            if hashlib.sha256(temporary.read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError('Downloaded source archive does not match its published SHA-256.')
            temporary.replace(ARCHIVE)
        assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()==record['sha256']
        with zipfile.ZipFile(ARCHIVE) as z:
            for name in z.namelist():
                target=(ROOT/name).resolve()
                if not target.is_relative_to(ROOT.resolve()):raise ValueError('Unsafe archive member')
            z.extractall(ROOT)
        print('Restored recorded query outputs.');return
    folders=['evaluation_renormalized','extended','updates','statistics','provenance']
    files=sorted(p for folder in folders for p in (ROOT/'results_nc'/folder).rglob('*') if p.suffix in ('.csv','.json'))
    assert len(list((ROOT/'results_nc/evaluation_renormalized').rglob('test_queries.csv')))==12
    ARCHIVE.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(ARCHIVE,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in files:z.write(path,path.relative_to(ROOT).as_posix())
    ARCHIVE.with_suffix('.json').write_text(json.dumps({'sha256':hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        'files':len(files),'bytes':ARCHIVE.stat().st_size,'download_url':DOWNLOAD_URL,
        'content':'Frozen validation configurations, query-level validation/test predictions, interventions, controls, updates, provenance and summaries. No model weights or dense logits.'},indent=2))
    print(ARCHIVE,ARCHIVE.stat().st_size)

if __name__=='__main__':main()
