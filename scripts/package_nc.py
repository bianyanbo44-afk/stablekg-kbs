"""Build a self-contained LaTeX upload archive and the author submission bundle."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / 'manuscript/neurocomputing'
FIGURES = ROOT / 'figures/neurocomputing'
OUT = ROOT / 'submission/neurocomputing'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ('main', 'supplementary'):
        pdf = MANUSCRIPT / (name + '.pdf')
        assert pdf.exists() and pdf.stat().st_size > 10000, f'Missing final PDF: {pdf}'
    for path in MANUSCRIPT.glob('*.tex'):
        text = path.read_text(encoding='utf-8')
        assert not any(marker in text for marker in
                       ('Layout draft', 'awaiting strict', 'is being recomputed', 'TODO', 'TBD')), path
    assert len(list(FIGURES.glob('fig0*.pdf'))) == 6
    assert (ROOT / 'results_nc/updates/GDELT_metadata.json').exists()
    assert (ROOT / 'release/neurocomputing_source_data.zip').exists()
    assert 'Pending completion' not in (ROOT / 'revision/SELF_REVIEW.md').read_text(encoding='utf-8')
    for source, target in (('main.pdf', 'Manuscript.pdf'),
                           ('supplementary.pdf', 'Supplementary_information.pdf')):
        shutil.copy2(MANUSCRIPT / source, OUT / target)

    source_zip = OUT / 'Manuscript_sources.zip'
    with zipfile.ZipFile(source_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(MANUSCRIPT.iterdir()):
            if path.suffix in ('.tex', '.bib', '.bbl'):
                content = path.read_text(encoding='utf-8')
                content = content.replace('../../figures/neurocomputing/', 'figures/')
                z.writestr(path.name, content)
        for pattern in ('fig0*.pdf', 'fig0*.svg'):
            for path in sorted(FIGURES.glob(pattern)):
                z.write(path, 'figures/' + path.name)
        for path in sorted((FIGURES / 'source_data').glob('*.csv')):
            z.write(path, 'source_data/' + path.name)
        for name in ('elsarticle.cls', 'elsarticle-num.bst'):
            result = subprocess.run(['kpsewhich', name], capture_output=True, text=True, check=True)
            path = Path(result.stdout.strip())
            assert path.is_file(), name
            z.write(path, name)
        z.writestr('BUILD.txt',
                    'Compile main.tex and supplementary.tex using latexmk -pdf.\n'
                    'The Elsevier class and bibliography style retain their original licenses.\n'
                    'Vector figures are editable SVG/PDF; source_data contains their plotted values.\n'
                    'Full experiment and figure code: https://github.com/bianyanbo44-afk/stablekg-kbs/tree/neurocomputing-v1.1\n')

    names = ['Abstract.pdf', 'Manuscript.pdf', 'Supplementary_information.pdf', 'Manuscript_sources.zip',
             'Title_page.docx', 'Highlights.docx', 'Cover_letter.docx', 'Declarations.docx',
             '投稿文件说明.md']
    record = {name: {'bytes': (OUT / name).stat().st_size,
                     'sha256': digest(OUT / name)} for name in names}
    (OUT / 'FILE_MANIFEST.json').write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    bundle = OUT / 'Neurocomputing_submission_package.zip'
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in names + ['FILE_MANIFEST.json']:
            z.write(OUT / name, name)
    print(json.dumps({'source_archive': str(source_zip), 'submission_bundle': str(bundle),
                      'submission_sha256': digest(bundle), 'files': record}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
