# -*- coding: utf-8 -*-
"""Figures 8 and A3: name-stripped hit@1 of six encoders in every genre, as overlaid horizontal bars.

Rows: the seed game's genre, the rarest of the 23 reduced Steam tags it carries (fig_triplet.genre23),
plus "no tag"; every neighborhood sits in exactly one row; rows ordered by confusability (mean
pairwise CE anchor cosine), most confusable at the top.  x: Stripped hit@1 in percent, the share of
a neighborhood's eight name-stripped queries whose nearest of the 2,020 anchors is their own game,
averaged over the genre's neighborhoods.  Bars of one row are overlaid, all the same height, longest
drawn at the back, so each tower shows as the segment between the next shorter tower's end and its
own.  Colours: frozen embedder grey, VICReg sand, the SimCLR-style tower lavender, CE teal; adding
the invariance term (I + SimCLR, I-CE) shows as a solid cap in the family's dark shade where it
gains and as a hatched span over the light bar where it loses (the tower with the term is pressed
back to the start of the hatch).  Align + uniformity @ views is left out for clarity (Table 3 has
it).  A box marks the genre, among those with at least five neighborhoods, whose six-encoder
profile lies nearest the profile of all neighborhoods; Figure 9 opens that genre up.

    python fig_genre_bars.py --cv    -> fig8_genres.png          Figure 8: the 814 five-fold neighborhoods of cv_hoods.py
    python fig_genre_bars.py         -> figA3_genres_fixed.png   Figure A3: the 201 held-out neighborhoods of the fixed split
    FIG8_OUT=<file> writes the figure there instead.  Both print the table of genre means.
"""
import os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
CV = "--cv" in sys.argv                                  # five-fold neighborhoods from cv_hoods.py instead of the fixed split
sys.argv = [sys.argv[0]]; sys.path.insert(0, str(HERE))
import fig_triplet as ft                                   # noqa: E402

ORDER = ["Frozen", "VICReg", "SimCLR-style", "I + SimCLR", "CE", "I-CE"]
TOWERS = [m for m in ORDER if m in ft.models]
# CIE L* ladder for greyscale printing, in row order: Frozen 61, VICReg 85, SimCLR 70, gain 37, CE 54, gain 35
COLOR = {"Frozen": "#929292", "VICReg": "#e6d48f", "SimCLR-style": "#aaa3d8", "CE": "#3f8f8f"}
PLUS = {"I + SimCLR": ("SimCLR-style", "#5a4fa0"), "I-CE": ("CE", "#1c5c5c")}           # tower with the invariance term: (its base, dark shade)
matplotlib.rcParams["hatch.linewidth"] = 1.0


def lab_L(hexc):
    c = np.array([int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)]); lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    y = 0.2126729 * lin[0] + 0.7151522 * lin[1] + 0.0721750 * lin[2]; f = np.cbrt(y) if y > 0.008856 else 7.787 * y + 16 / 116; return 116 * f - 16


rows = {}
if CV:
    import json
    _cv = json.load(open(ft.DATA / "cv" / "cv_hoods_4096.json", encoding="utf-8"))
    TOWERS = [m for m in ORDER if m in _cv["encoders"]]
    _hoods = [(h["genre"], h) for h in _cv["hoods"]]
else:
    _hoods = [(ft.genre23(int(ft.gidx[d["seed"]])), d) for d in ft.hoods.values()]
for _g, d in _hoods:
    r = rows.setdefault(_g, dict(n=0, cos=[], acc={m: [] for m in TOWERS}))
    r["n"] += 1; r["cos"].append(d["mean_cos"])
    for m in TOWERS:
        r["acc"][m].append(100 * d["acc_s"][m])
order = sorted(rows, key=lambda g: -np.mean(rows[g]["cos"]))
mean = {g: {m: float(np.mean(rows[g]["acc"][m])) for m in TOWERS} for g in order}
gap = {g: mean[g]["I-CE"] - mean[g]["CE"] for g in order}
sd = {g: float(np.std([mean[g][m] for m in TOWERS])) for g in order}                     # the spread the box is chosen by
boxed = max((g for g in order if rows[g]["n"] >= ft.GENRE_MIN_N), key=lambda g: sd[g]); assert CV or boxed == ft.BOXED
BOX_LABEL = "encoders disagree\nmost (n ≥ 5)"
if ft.BOX_RULE == "typical":                             # the paper's rule: the genre nearest the profile of all neighbourhoods
    _allp = np.array([np.mean([d["acc_s"][m] for _, d in _hoods]) * 100 for m in TOWERS])
    dist = {g: float(np.linalg.norm(np.array([mean[g][m] for m in TOWERS]) - _allp)) for g in order}
    boxed = min((g for g in order if rows[g]["n"] >= ft.GENRE_MIN_N), key=lambda g: dist[g]); BOX_LABEL = "nearest the profile\nof all (n ≥ 5)"
    print("distance from the profile of all neighbourhoods:", ", ".join(f"{g} {dist[g]:.1f}" for g in sorted(dist, key=dist.get)[:5]))
print(f"{'genre':22s} {'n':>3s} {'cos':>5s} | " + " ".join(f"{m:>12s}" for m in TOWERS) + " | spread  I-CE minus CE (points)")
for g in order:
    print(f"{g:22s} {rows[g]['n']:3d} {np.mean(rows[g]['cos']):5.3f} | " + " ".join(f"{mean[g][m]:12.1f}" for m in TOWERS) + f" | {sd[g]:5.1f}  {gap[g]:+.1f}" + ("  <- boxed" if g == boxed else ""))
print("colour L*:", {k: round(lab_L(v)) for k, v in {**COLOR, **{k: v[1] for k, v in PLUS.items()}}.items()})

FIG_W = float(os.environ.get("FIG8_W", "6.1")); FIG_PT = 7.0        # placement width, the full text width; text 2 pt below the 9 pt body
XMAX = 90 if max(max(v.values()) for v in mean.values()) <= 88 else 100   # at 512 sentences the longest bar is 87.5%, so the bars span the axes
n = len(order); fig, ax = plt.subplots(figsize=(FIG_W, 0.118 * n + 0.40)); H = 0.74   # 0.118 in per row is the least 7 pt labels can take; the legend sits at the right, not above
for i, g in enumerate(order):
    y = n - 1 - i; v = mean[g]
    solid = [(v[m], m, COLOR[m]) for m in COLOR if m in v]                              # the four base towers
    losses = []
    for plus, (base, dark) in PLUS.items():
        if plus in v and v[plus] > v[base]:
            solid.append((v[plus], plus, dark))                                          # a gain: solid cap in the dark shade
        elif plus in v and v[plus] < v[base]:
            losses.append((v[plus], v[base], dark))                                      # a loss: hatched span over the light bar
    solid.sort(reverse=True)                                                             # longest first, at the back
    for k, (x, m, c) in enumerate(solid):
        ax.barh(y, x, height=H, color=c, zorder=2 + k, linewidth=0)
    for lo, hi, dark in losses:
        ax.barh(y, hi - lo, left=lo, height=H, facecolor="none", hatch="//////////", edgecolor=dark, linewidth=0, zorder=60)
    if g == boxed:
        ax.add_patch(Rectangle((0, y - 0.5), XMAX, 1.0, fill=False, ec="0.12", lw=1.2, zorder=70, clip_on=False))
ax.set_yticks(range(n))
ax.set_yticklabels([f"{g}  (n = {rows[g]['n']})" for g in order][::-1], fontsize=FIG_PT)   # cos is the row order; its range goes in Appendix L
ax.set_ylim(-0.6, n - 0.4); ax.set_xlim(0, XMAX); ax.set_xticks(range(0, XMAX + 1, 10)); ax.tick_params(axis="x", labelsize=FIG_PT, pad=1.5); ax.tick_params(axis="y", length=2, pad=1.5)
ax.set_xlabel("Stripped hit@1 (%), mean over the genre's neighborhoods", fontsize=FIG_PT, labelpad=2)
ax.grid(axis="x", color="0.92", lw=0.8); ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_color("0.5")
handles = [Patch(color=COLOR["Frozen"], label="Frozen"), Patch(color=COLOR["VICReg"], label="VICReg"),
           Patch(color=COLOR["SimCLR-style"], label="SimCLR-style"), Patch(color=PLUS["I + SimCLR"][1], label="I + SimCLR: gain"),
           Patch(color=COLOR["CE"], label="CE"), Patch(color=PLUS["I-CE"][1], label="I-CE: gain"),
           Patch(facecolor="none", hatch="//////////", edgecolor="0.3", linewidth=0, label="loss with the\ninvariance term"),
           Patch(facecolor="none", edgecolor="0.12", linewidth=1.2, label=BOX_LABEL)]   # the hatch and the box are their own labels
# one column at the right edge, vertically centred; the axes take every inch the legend leaves
leg = fig.legend(handles=handles, loc="center right", bbox_to_anchor=(1.0, 0.5), ncol=1, fontsize=FIG_PT, frameon=False,
                 handlelength=1.0, handletextpad=0.4, labelspacing=0.45, borderaxespad=0.0)
fig.canvas.draw()
LEG_W = leg.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).width
fig.tight_layout(rect=(0, 0, 1 - LEG_W - 0.012, 1), pad=0.3)
out = ft.PAPER / ("fig8_genres.png" if CV else "figA3_genres_fixed.png")   # Figure 8 is the five-fold figure, Figure A3 the fixed-split one
out = Path(os.environ["FIG8_OUT"]) if os.environ.get("FIG8_OUT") else out
if CV and os.environ.get("HOOD_GENRE"):
    assert boxed == os.environ["HOOD_GENRE"], f"Figure 9 is drawn for HOOD_GENRE={os.environ['HOOD_GENRE']} but the five-fold box is {boxed}"
fig.savefig(out, dpi=300); print("ok", out, "boxed:", boxed)
