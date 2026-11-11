# -*- coding: utf-8 -*-
r"""Figure 10 (release-share river) drawn for one column of the two-column build.

This is the layout the two-column paper prints. In this repository
(figures/trend_river/) the page, the data file `games_raw.js`, the fit and
the outputs share this script's directory, and the PNG is written there as
fig6_content_river_ice_col.png; the paths below are those of the paper's own
repository.

Position in the chain
---------------------
    river_ice_prep.py   -> river_assets/games_raw_ice.js           (data, once)
    river_shift_fit.py  -> river_assets/shift_fit_ice.json         (periods, levels and ribbons)
    river_ice_render.py -> the full-width figure the Word build embeds
    river_ice_column.py -> river_assets/paper_render_ice_col.html  (this script)
                           river_assets/fig6_content_river_ice_col.meta.json
                           assets/paper/fig6_content_river_col.png
    gen_tex.py          -> sets that PNG at \columnwidth (latex/layout.toml)

The data, the clusters, the levels and transfers (river_shift_fit.py), the
rows, the colours and the ribbons are the full-width figure's: the page is baked by
river_ice_render.bake_variant() and only its layout is patched here.  The meta
file lists the same lanes and ribbons; compare it with
fig6_content_river_ice.meta.json after a change.  --pressure draws the page's
pressure model instead, as river_ice_render.py --pressure does.

What changes for a column (3.337 in)
------------------------------------
The canvas is laid out so that the page's 32 px lettering prints at 7 pt
(constraint/constraint.md, "Figure Text Size"): 329 CSS px to the inch, so the
canvas is 1,098 px wide.  With the lane names beside the right edge, as in the
full-width figure, the bands would keep under half of that width, so (the
author's layout, 2026-10-10 and 2026-10-11):
  * the bands start at the left edge of the figure, and the first year label
    is set left-aligned at its grid line instead of centred on it;
  * each lane's name is written above its band, at the left, on a white halo
    so that a ribbon passing behind it does not hide it; no end shares.  A
    band may grow upward while its name is still running (a ribbon joining it
    from above), so the name clears the highest top edge under it;
  * the lanes are PITCH px apart and the widest band BAND px thick, which
    leaves one line of lettering between two bands;
  * the band of the bottom row is lowered until it stands as far above the
    year axis as the widest band would in that row (2026-11-11).  With the
    widest band in the bottom row, as until then, nothing moves; with a thin
    band there and the widest one right above it, the thin band's name would
    otherwise touch the band above;
  * the 10% scale bar stands at the right, beside the end of the top band and
    from that band's top edge down, with "10%" to the right of the bar;
  * no rotated axis title (the caption names the quantity) and no lane ticks.

Run with:
    py figures/river_ice_column.py
Options:
    --pitch 112    lane pitch in CSS px (128 until 2026-10-11; the author asked
                   for the lanes closer together, so that the figure fits in
                   its column right after the paragraph that cites it)
    --band 94      thickness of the widest band in CSS px.  With the fitted
                   levels that is the Puzzle band at the left edge, a third of
                   the releases; 94 px is as thick as it can be with a line of
                   lettering left under the band above (76 with --pressure,
                   where the widest band holds 22%)
    --fit FILE     levels and transfers (default river_assets/shift_fit_ice.json;
                   flat_fit_ice.json is the fit of 2026-10-13 to 2026-11-10)
    --pressure     the page's pressure model instead of the fit
    --joints slant|step, --slant-part 0.333
                   as in river_ice_render.py: slanted cuts with the ribbon
                   from cut to cut (default) or the page's vertical steps
    --dsf 3        device scale factor of the capture (3 gives 987 dpi in print)
    --png FILE     output PNG (default assets/paper/fig6_content_river_col.png)
Needs what river_ice_render.py needs: the standard library and Chrome.
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import river_ice_render as rir  # noqa: E402

ASSETS = rir.DEFAULT_ASSETS
PAPER = HERE.parent / "assets" / "paper"
# paper repo: the PNG goes to assets/paper/; release bundle: beside this script
PNG = (PAPER / "fig6_content_river_col.png" if PAPER.exists()
       else ASSETS / "fig6_content_river_ice_col.png")

FONT = 32                              # CSS px: the page's lettering
PX_PER_IN = FONT / (7.0 / 72.0)        # so that it prints at 7 pt
COLUMN_IN = 241.14749 / 72.27          # \columnwidth of acmart sigconf
PITCH, BAND, BAND_PRESSURE, DSF = 112, 94, 76, 3
# left: the bands start at the edge; right: the scale bar and "10%";
# top: a few px above the top lane's name; bottom: the year labels
PAD = dict(l=6, r=82, t=8, b=44)


def patch(html, old, new):
    if old not in html:
        raise SystemExit("patch target missing in the baked page:\n  " + old[:110])
    return html.replace(old, new, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitch", type=float, default=PITCH)
    ap.add_argument("--band", type=float, default=None)
    ap.add_argument("--fit", default=str(ASSETS / rir.DEFAULT_FIT))
    ap.add_argument("--pressure", action="store_true")
    ap.add_argument("--joints", choices=("slant", "step"), default="slant")
    ap.add_argument("--slant-part", type=float, default=rir.SLANT_PART)
    ap.add_argument("--dsf", type=int, default=DSF)
    ap.add_argument("--png", default=str(PNG))
    a = ap.parse_args()
    flat = None if a.pressure else rir.flat_inputs(a.fit)
    if a.band is None:
        a.band = BAND_PRESSURE if a.pressure else BAND
    width = round(COLUMN_IN * PX_PER_IN)
    height = round(PAD["t"] + 6 * a.pitch + PAD["b"])
    page = ASSETS / "paper_render_ice_col.html"
    png = Path(a.png).resolve()
    data_rel = Path(os.path.relpath(ASSETS / rir.DEFAULT_DATA, page.parent)).as_posix()
    rir.bake_variant(page, data_rel, rir.ICE_THR, rir.ICE_PRUNE,
                     src=ASSETS / "cluster_share_stack.html", fonta=FONT, fontl=FONT, flat=flat,
                     joints=a.joints, slant_part=a.slant_part)
    h = page.read_text(encoding="utf-8")

    h = patch(h, rir.FIX_PAD[1], "PAD={l:%(l)d,r:%(r)d,t:%(t)d,b:%(b)d}" % PAD)
    h = patch(h, "const F=0.92*(plotH/K)/maxW;",
              "const F=%.4f*(plotH/K)/maxW;" % (a.band / a.pitch))
    # the bottom row's band stands as far above the year axis as the widest
    # band would there (its bottom edge never moves: every ribbon meets its top)
    top_lane = 5
    if flat is not None:
        rows, fit = flat["rows"], flat["fit"]
        top_lane, low = rows.index(len(rows) - 1), rows.index(0)
        drop = [0.0] * len(rows)
        drop[low] = round(a.band * (1 - fit["levels_start"][low] / rir.widest_level(fit)) / 2, 2)
        h = patch(h, rir.LANE_DROP % ",".join("0" for _ in rows), rir.LANE_DROP % ",".join("%g" % v for v in drop))
    # lane names: above the band, at its left end, on a white halo; above the
    # highest top edge the name runs over
    h = patch(h, rir.FIX_LANE_TEXT[1],
              """    { const lab=CLUSTER_LABELS[k].label, lx=PAD.l+4, tw=ctx.measureText(lab).width;
      let top=Infinity;
      SEG[k].forEach(s=>{ if(sx(s.ta)<lx+tw+8 && sx(s.tb)>lx-8) top=Math.min(top,s.yT); });
      const ly=top-9;
      ctx.textAlign='left';
      ctx.save(); ctx.lineJoin='round'; ctx.strokeStyle='rgba(255,255,255,.92)';
      ctx.lineWidth=7; ctx.strokeText(lab, lx, ly); ctx.restore();
      ctx.fillText(lab, lx, ly); }""")
    # no tick between a band and a name that is no longer beside it
    h = patch(h, """    ctx.strokeStyle=col(k,0.9,PAPER); ctx.lineWidth=2;
    ctx.beginPath();
    ctx.moveTo(W-PAD.r, laneY(k)); ctx.lineTo(W-PAD.r+8, laneY(k));
    ctx.stroke();""", "")
    # the first year label would be cut by the left edge if it were centred
    h = patch(h, "    ctx.fillText(String(YEAR0+q), x, H-PAD.b+26);",
              """    if(x<PAD.l+30){ ctx.textAlign='left';
      ctx.fillText(String(YEAR0+q), PAD.l-2, H-PAD.b+26); ctx.textAlign='center'; }
    else ctx.fillText(String(YEAR0+q), x, H-PAD.b+26);""")
    # the scale bar: at the right, from the top row's band's top edge down, "10%" to its right
    h = patch(h, """  ctx.moveTo(PAD.l-26, PAD.t+8); ctx.lineTo(PAD.l-26, PAD.t+8+sb);
  ctx.stroke();
  ctx.textAlign='right'; ctx.fillStyle=ink;
  ctx.fillText('10%', PAD.l-32, PAD.t+12+sb/2);""",
              """  { const top=SEG[TOPLANE][SEG[TOPLANE].length-1].yT, bx=W-PAD.r+14;
    ctx.moveTo(bx, top); ctx.lineTo(bx, top+sb);
    ctx.stroke();
    ctx.textAlign='left'; ctx.fillStyle=ink;
    ctx.fillText('10%', bx+9, top+sb/2+FONTL*0.34); }""".replace("TOPLANE", str(top_lane)))
    h = patch(h, "ctx.fillText('release share by semantic cluster',0,0);", "")
    page.write_text(h, encoding="utf-8")

    rir.WIN_W, rir.WIN_H = width, height
    meta = rir.note_fit(rir.capture(page, png, a.dsf, crop=None), flat, a.joints, a.slant_part)
    meta_path = ASSETS / "fig6_content_river_ice_col.meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote %s: %d x %d CSS px = %.3f x %.2f in at 7 pt lettering, %d x %d px (%.0f dpi)"
          % (png, width, height, width / PX_PER_IN, height / PX_PER_IN,
             meta["png_size"][0], meta["png_size"][1], meta["png_size"][0] / (width / PX_PER_IN)))
    rir.report(meta)
    print("meta ->", meta_path)


if __name__ == "__main__":
    main()
