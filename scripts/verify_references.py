"""Save DOI registry responses and compare bibliography metadata, without agents."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import re
import urllib.request
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "literature" / "verification_20260909"
OUT.mkdir(parents=True, exist_ok=True)
text = (ROOT / "manuscript/references.bib").read_text(encoding="utf-8")
entries = []
for block in re.split(r"(?m)^@", text)[1:]:
    key = re.match(r"\w+\{([^,]+)", block).group(1)
    fields = dict(re.findall(r"(?m)^\s*(\w+)\s*=\s*\{(.*?)\},?\s*$", block))
    entries.append((key, fields))

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "StableKG-bibliography-verification/1.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.load(r)

def check(entry):
    key, f = entry
    path = OUT / f"{key}.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
    else:
        url = "https://api.crossref.org/works/" + urllib.parse.quote(f["doi"], safe="")
        record = {"key": key, "url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat()}
        try:
            record["response"] = get(url)
        except Exception as e:
            record["error"] = str(e)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    m = record.get("response", {}).get("message", {})
    title = m.get("title", [""])[0]
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    ratio = difflib.SequenceMatcher(None, norm(f["title"]), norm(title)).ratio()
    date = m.get("published", m.get("issued", {})).get("date-parts", [[None]])[0]
    authors = " and ".join(x.get("family", "") + ", " + x.get("given", "") for x in m.get("author", []))
    return {"key": key, "doi": f["doi"], "title": title, "title_similarity": ratio,
            "year": date[0], "date": date, "authors": authors,
            "author_similarity": difflib.SequenceMatcher(None, norm(f["author"]), norm(authors)).ratio(),
            "venue": m.get("container-title", []), "page": m.get("page", m.get("article-number")),
            "volume": m.get("volume"), "issue": m.get("issue"),
            "status": "found" if m and ratio > .96 and str(date[0]) == f["year"] else "partial match" if m else "not found",
            "has_abstract": bool(m.get("abstract")), "license_metadata": m.get("license", []),
            "claim_check": "not established by metadata verification"}

with ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(check, entries))
(OUT / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
for r in results:
    print(r["key"], r["status"], r["title_similarity"], r["author_similarity"], r["date"], flush=True)
# Attempt an independent index for every work, keeping HTTP failures explicit.
def crosscheck(r):
    path = OUT / f"{r['key']}_openalex.json"
    if path.exists():
        return
    url = "https://api.openalex.org/works/https://doi.org/" + r["doi"]
    record = {"url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    try:
        record["response"] = get(url)
    except Exception as e:
        record["error"] = str(e)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(crosscheck, results))
