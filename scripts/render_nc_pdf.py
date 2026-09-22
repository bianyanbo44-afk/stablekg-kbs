"""Render manuscript pages and compact contact sheets for visual inspection."""
from pathlib import Path
import argparse
import json
import fitz
from PIL import Image, ImageDraw, ImageOps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('pdf', type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    document = fitz.open(a.pdf)
    tiles, report = [], []
    for i, page in enumerate(document):
        pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
        im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
        im.save(a.output / f'page-{i + 1:02}.png')
        thumb = ImageOps.contain(im, (400, 560))
        tile = Image.new('RGB', (420, 600), '#eeeeee')
        tile.paste(thumb, ((420 - thumb.width) // 2, 15))
        ImageDraw.Draw(tile).text((12, 578), f'Page {i + 1}', fill='black')
        tiles.append(tile)
        spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b
                 for line in b['lines'] for s in line['spans']]
        outside = [s['text'] for s in spans if
                   s['bbox'][0] < -1 or s['bbox'][1] < -1 or
                   s['bbox'][2] > page.rect.width + 1 or s['bbox'][3] > page.rect.height + 1]
        report.append({'page': i + 1, 'characters': len(page.get_text()),
                       'outside_page': outside, 'width_pt': page.rect.width,
                       'height_pt': page.rect.height})
    # Six pages per sheet preserve legibility in the desktop preview.
    for start in range(0, len(tiles), 6):
        group = tiles[start:start + 6]
        sheet = Image.new('RGB', (1260, 600 * ((len(group) + 2) // 3)), 'white')
        for j, tile in enumerate(group):
            sheet.paste(tile, ((j % 3) * 420, (j // 3) * 600))
        sheet.save(a.output / f'contact-{start // 6 + 1:02}.png')
    (a.output / 'page_geometry.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'pages': len(document), 'outside_page_items': sum(len(r['outside_page']) for r in report)}))


if __name__ == '__main__':
    main()
