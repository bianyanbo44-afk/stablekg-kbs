from pathlib import Path
import json
import fitz
from PIL import Image,ImageOps,ImageDraw
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/pdfs/review_20260909"
OUT.mkdir(parents=True,exist_ok=True)
for name in ("main","supplementary"):
    doc=fitz.open(ROOT/f"manuscript/{name}.pdf")
    cards=[]
    for i,page in enumerate(doc):
        pix=page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False)
        path=OUT/f"{name}_{i+1:02d}.png"
        pix.save(path)
        thumb=Image.open(path).convert("RGB")
        thumb.thumbnail((360,510))
        card=Image.new("RGB",(382,540),"#dfe5ea")
        card.paste(thumb,((382-thumb.width)//2,20))
        ImageDraw.Draw(card).text((12,520),f"{name} {i+1}",fill="black")
        cards.append(card)
    for start in range(0,len(cards),9):
        subset=cards[start:start+9]
        sheet=Image.new("RGB",(382*3,540*((len(subset)+2)//3)),"white")
        for j,card in enumerate(subset):
            sheet.paste(card,((j%3)*382,(j//3)*540))
        sheet.save(OUT/f"{name}_contact_{start//9+1}.png")
    print(name,len(doc),"pages",len(" ".join(p.get_text() for p in doc).split()),"words",flush=True)
