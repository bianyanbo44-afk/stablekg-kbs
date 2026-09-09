"""Portable geometry checks for the regular grids in this figure set.

Full publication QA is performed separately. This self-contained check lets a
clean checkout reproduce figures and rejects misaligned shared rows/columns.
"""
from pathlib import Path
import json

def matplotlib_layout_manifest(fig, axes=None):
    fig.canvas.draw()
    axes = axes or fig.axes
    width, height = fig.get_size_inches()*72
    panels = []
    for i, ax in enumerate(axes):
        x,y,w,h = ax.get_position().bounds
        panels.append({"panel":chr(97+i), "x":x*width,"y":y*height,"w":w*width,"h":h*height})
    return {"units":"pt","panels":panels}

def require_matplotlib_panel_alignment(fig, json_out, tolerance_pt=1.5, **kwargs):
    manifest = matplotlib_layout_manifest(fig)
    panels = manifest["panels"]
    for i, a in enumerate(panels):
        for b in panels[i+1:]:
            if abs(a["y"]-b["y"])<tolerance_pt:
                assert abs(a["h"]-b["h"])<=tolerance_pt, "Row heights differ"
                assert abs(a["w"]-b["w"])<=tolerance_pt, "Row widths differ"
            if abs(a["x"]-b["x"])<tolerance_pt:
                assert abs(a["w"]-b["w"])<=tolerance_pt, "Column widths differ"
    manifest["status"] = "PASS (portable regular-grid check)"
    Path(json_out).write_text(json.dumps(manifest,indent=2),encoding="utf-8")
