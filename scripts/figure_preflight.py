"""Run installed source, font and collision checks against final figure PDFs."""
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
SKILL=Path.home()/".codex/skills/nature-figure/scripts"
OUT=ROOT/"figures/out"
rows=[]
for pdf in sorted(OUT.glob("fig*.pdf")):
    if "collision" in pdf.name:
        continue
    for name, arguments in [
        ("fonts",["audit_pdf_text.py",str(pdf),"--min-pt","5","--json"]),
        ("collisions",["audit_figure_collisions.py",str(pdf),"--json-out",str(pdf.with_suffix(".collision-audit.json")),"--overlay-pdf",str(pdf.with_suffix(".collision-audit.pdf"))])]:
        result=subprocess.run([sys.executable,str(SKILL/arguments[0]),*arguments[1:]],capture_output=True,text=True,encoding="utf-8",errors="replace")
        (OUT/f"{pdf.stem}.{name}.log").write_text(result.stdout+result.stderr,encoding="utf-8")
        rows.append({"figure":pdf.name,"check":name,"exit_code":result.returncode})
(OUT/"review_preflight.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
print(json.dumps(rows,indent=2))
if any(row["exit_code"] for row in rows):
    sys.exit(1)
