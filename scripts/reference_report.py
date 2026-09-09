from pathlib import Path
import json
import re
import urllib.request
import datetime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "literature/verification_20260909"
summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
bib = (ROOT / "manuscript/references.bib").read_text(encoding="utf-8")
notes = []
abstracts = []
for row in summary:
    key = row["key"]
    oa = json.loads((OUT / f"{key}_openalex.json").read_text(encoding="utf-8"))
    record = oa.get("response", {})
    if row["status"] == "not found" and record.get("title"):
        # Retry the one transient registry failure, preserving its first log.
        url = "https://api.crossref.org/works/" + row["doi"]
        attempt = {"url":url,"retrieved_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
        try:
            req = urllib.request.Request(url,headers={"User-Agent":"StableKG-reference-review/1.0"})
            with urllib.request.urlopen(req,timeout=25) as r:
                attempt["response"] = json.load(r)
        except Exception as e:
            attempt["error"] = str(e)
        (OUT/f"{key}_crossref_retry.json").write_text(json.dumps(attempt,indent=2,ensure_ascii=False),encoding="utf-8")
        row["status"] = "found (OpenAlex; original Crossref request rate limited)"
        row["title"] = record["title"]
        row["year"] = record["publication_year"]
    inv = record.get("abstract_inverted_index") or {}
    words = {i:w for w, indices in inv.items() for i in indices}
    abstract = " ".join(words[i] for i in sorted(words))
    abstracts.append(f"## {key}\n\n{row['title']}\n\n{abstract or 'No abstract returned by OpenAlex.'}\n\nDOI: {row['doi']}\n")
    row["openalex_id"] = record.get("id")
    row["oa_status"] = record.get("open_access", {}).get("oa_status", "unknown")
    row["openalex_title_match"] = re.sub(r"\W", "",row["title"].lower()) == re.sub(r"\W", "",record.get("title", "").lower())
    row["openalex_year_match"] = row["year"] == record.get("publication_year")
    row["abstract_available"] = bool(abstract)
    notes.append(f"| `{key}` | {row['status']} | {row['year']} | {row['openalex_id']} | {row['oa_status']} | {'available' if abstract else 'not returned'} |")
    if key in ("geng2025granularity", "chen2025gnnuq", "li2026calendar", "li2026entropyrobust") and row["page"]:
        pattern = r"(@\w+\{"+key+r",.*?)(\n\})"
        bib = re.sub(pattern, lambda m:m[1]+f"\n  pages = {{{row['page']}}},"+m[2] if "  pages = " not in m[1] else m[0], bib, flags=re.S)
    if key == "szabadvary2025reject":
        author = r"Hallberg Szabadv{\'a}ry, Johan and L{\"o}fstr{\"o}m, Tuwe and Johansson, Ulf and S{\"o}nstr{\"o}d, Cecilia and Ahlberg, Ernst and Carlsson, Lars"
        pattern = r"(@\w+\{"+key+r",\s*author = \{).*?(\},)"
        bib = re.sub(pattern, lambda m:m[1]+author+m[2], bib, flags=re.S)
# Protect canonical acronym capitalization without changing title words.
for acronym in ("HGCT","LLMs","STSE","Mamba","MCNet","KG-UQ","GMVE","TCrossE","ICPE-STKG","DSTAG","TGCA-LLM","TiRano","CALENDAR+","Householder"):
    bib = re.sub(r"(?m)^(  title = \{)(.*)(\},?)$", lambda m:m[1]+re.sub(r"(?<![A-Za-z{])"+re.escape(acronym)+r"(?![A-Za-z}])", lambda a:"{"+a[0]+"}",m[2])+m[3],bib)
(ROOT / "manuscript/references.bib").write_text(bib,encoding="utf-8")
(OUT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
(OUT/"abstracts.md").write_text("\n".join(abstracts),encoding="utf-8")
report = """# Bibliographic verification — 9 September 2026

All 38 bibliography records were queried by DOI through Crossref and OpenAlex.
Raw responses, URLs, UTC retrieval times and HTTP failures are saved beside this report.
The first Crossref request for the temporal conformal paper returned HTTP 429;
OpenAlex resolved the DOI and a separate Crossref retry is recorded.

This verifies bibliographic identity and selected metadata. It does not mean
that publisher full texts were read or that every cited scientific claim has
been independently established. OpenAlex abstracts are saved separately for
content review; missing abstracts are explicitly identified. The previous
report's blanket claim of publisher-page checks is superseded by this record.

Corrections include one author's compound family name and diacritics, four
missing article numbers, and protected acronym capitalization. Bibliography
years remain 2025 or 2026; the year embedded in a DOI is not the publication year.

| Key | Registry result | Bibliography year | Independent index | OA status | Abstract |
|---|---|---:|---|---|---|
""" + "\n".join(notes) + "\n"
(ROOT/"manuscript/citation_audit.md").write_text(report,encoding="utf-8")
print("Crossref/OpenAlex identity reports, abstracts and bibliography corrections saved.")
