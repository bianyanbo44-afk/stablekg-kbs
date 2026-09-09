"""Record environment, source hashes, consistent submission text and PDF previews."""
from pathlib import Path
import hashlib
import importlib.metadata as m
import json
import platform
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
names = ("numpy","pandas","scipy","matplotlib","pillow","torch","pytest","PyMuPDF")
versions = {name:m.version(name) for name in names}
(ROOT/"requirements-lock.txt").write_text("# Tested direct package versions; Python "+platform.python_version()+"\n"+"\n".join(name+"=="+v for name,v in versions.items())+"\n",encoding="utf-8")
out = ROOT / "results_review_20260909"
record = {"python":sys.version, "platform":platform.platform(), "packages":versions,
          "meaning":"Direct dependency versions used for this run; not an isolated-environment install test."}
(out/"environment.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
files = []
for folder in ("results_final_v5","results_interventions_v5","results_chronological_v5","results_neural_v5","results_review_20260909/corrected"):
    for p in (ROOT/folder).rglob("*"):
        if p.is_file() and p.suffix in (".csv",".json"):
            files.append({"path":p.relative_to(ROOT).as_posix(),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
(out/"release_hashes.json").write_text(json.dumps(files,indent=2),encoding="utf-8")
text = (ROOT/"manuscript/main.tex").read_text(encoding="utf-8")
keys = set(k.strip() for group in re.findall(r"\\cite\w*\{([^}]+)\}",text) for k in group.split(","))
bib = (ROOT/"manuscript/references.bib").read_text(encoding="utf-8")
years = {}
for block in re.split(r"(?m)^@",bib)[1:]:
    key = re.match(r"\w+\{([^,]+)",block).group(1)
    if key in keys:
        years[key] = int(re.search(r"year\s*=\s*\{(\d+)\}",block).group(1))
assert set(years) == keys and min(years.values()) >= 2025
abstract = text.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]
print(json.dumps({"cited_references":len(keys),"years":{str(y):list(years.values()).count(y) for y in set(years.values())},"abstract_words":len(abstract.split()),"hashed_outputs":len(files)},indent=2))
legends = re.findall(r"\\caption\{(.+?)\}\s*\n\s*\\label",text,re.S)
(ROOT/"manuscript/figure_legends.md").write_text("# Current figure and table captions\n\n"+"\n\n".join(legends),encoding="utf-8")
