# -*- coding: utf-8 -*-
r"""Render the paper's Figure 10 (release-share river) from the I-CE game vectors.

In this repository (figures/trend_river/) the page, the data file, the fit and
the scripts share one directory, the data file is `games_raw.js`, and nothing
is embedded in a manuscript afterwards. The paths below are those of the
paper's own repository, where the script was written; the options and the
drawing are the same.

Position in the chain
---------------------
    river_ice_prep.py   -> river_assets/games_raw_ice.js      (data, once)
    river_shift_fit.py  -> river_assets/shift_fit_ice.json    (periods, levels and ribbons)
    river_ice_render.py -> river_assets/paper_render_ice.html (this script)
                           river_assets/fig6_content_river_ice.png
    sync_figures.py     -> assets/paper/fig6_content_river.png + the declared
                           figure height in paper.md

What the bands and ribbons are (since 2026-10-13)
-------------------------------------------------
The page is no longer asked for its pressure model.  It is given, from a fit,
the level of every band at the left edge and a list of transfers (source lane,
target lane, share moved, start date, end date), on the years 2020.0 to
2024.75.  A band keeps its thickness except where a ribbon leaves or joins it;
there its edge is cut on the slant, from the transfer's start date on the
source band and up to its end date on the target band.  The ribbon runs from
the one cut to the other: its two ends span the cuts, and in between it is a
band as thick as the share it carries, on a flat S-curve (SLANT_JS below;
--joints step draws the page's own vertical steps instead).  All transfers are drawn
(PRUNE = 0).
Since 2026-11-11 the fit is river_shift_fit.py's: the periods are placed by a
change-point test, a band's level is its cluster's share of a period's
releases, the ribbons are the cheapest transport between two periods, and a
ribbon's dates follow a stated convention (that script's docstring).  The fit
also names the row of every lane ("rows": the author moved one cluster to the
bottom row); a fit without it keeps the page's order.  Ribbons that leave
one band edge over overlapping stretches each take a part of that stretch and
leave side by side.  --fit river_assets/flat_fit_ice.json draws, with today's joints, the fit of 2026-10-13
to 2026-11-10 (river_flat_fit.py, least squares).
--pressure draws the figure as it was before that: the page's pressure-driven
event model on its causal-kernel shares, 2019 to mid-2024, with the parameters
described below.  That account is up to 24 points of share away from the
measured shares, which is why the paper no longer uses it.

`games_raw_ice.js` holds the 1,649 river games that map to the gallery, their
128-d L2-normalised I-CE vectors (tower wcle_i2ce_icetf, five-fold fold 0,
--anchor-cap 2048, ep1750), a 1-D UMAP coordinate that only orders the
lanes, and the release times / tags / titles verbatim from games_raw.js.

What the render does
--------------------
(The event-model parameters named in steps 1 and 2 act only with --pressure.)
1. `build_river_render.bake()` turns the interactive page
   (river_assets/cluster_share_stack.html) into a white "paper mode" page with
   the UI hidden, DPR up to 4, English axis title, wide margins for the 3x
   fonts, and the published parameter set baked in on load:
       K=6, TAU=0.45, BETA=0.3, THR=0.0066, WARM_A=2, WARM_B=0.5, EPS=0.05,
       DOTR=9, MINW=8, PRUNE=0.023, FONTA=45, FONTL=45, MARK=3.
2. This script then applies the I-CE settings on top:
       THR=0.004  the event trigger.  The published 64-d figure used 0.0066;
                  the I-CE space is more uniform (pairwise-cosine sd 0.108 vs
                  0.198), at 0.0066 only four ribbons clear the trigger, and
                  0.004 (variant v04 of river_variants.py, chosen by the
                  author) gives six.  PRUNE stays 0.023.
       FONTA=32, FONTL=32   anchor titles, lane labels and axis text.  The
                  submitted PNG was exported with 32 px labels and a plot
                  spanning 62% of the canvas (measured on the file), not with
                  the 45 px that bake() writes; 45 px labels with the longer
                  I-CE lane names squeezed the plot to 43% of the canvas;
       anchor titles lose a trailing ellipsis ("If Found..." reads "If Found");
       pads l = 165 / r = 700 (the submitted figure's, measured), the rotated
                  y-axis title moved from the clipped x = 24 to x = 44; a lane
                  label longer than 34 characters breaks at its " / " into two
                  lines, the share on the second, so the margin never grows
                  past 700 px;
       anchor games: none.  The page marks, per lane and three-year window,
                  the game nearest the lane's centroid with a dot and its
                  title.  The paper figure drops them (2026-10-10): no
                  sentence reads them, the rule that picks them is not in the
                  paper, and they crowded the bands.  --anchors draws them
                  again, placed as described next;
       label placement: each anchor title has two vertical starting slots,
                  the page's natural one (below the dot for a game in the
                  lower half of its lane, above it otherwise) and the mirror
                  slot on the other side of the dot, each stepping away from
                  the dot in half-line (0.575*FONTA) increments, and five
                  horizontal positions (centred on the dot, or nudged by 15%
                  or 30% of the label width, so the label always still spans
                  its dot).  cost = vertical displacement in lines + 0.5 on the
                  mirror side + 0.4 per 15% nudge; slots outside the plot or
                  over any anchor dot are excluded; 'hidden' costs 20.  A
                  branch-and-bound search over all anchors (most constrained
                  first) picks the assignment of minimum total cost in which
                  no two labels overlap and titles on one row stay at least
                  1.5*FONTA apart, so a title moves only as far as the
                  crowd around it forces; a label that ends up more than a
                  line away from its dot gets a thin leader line to it.  The
                  centred slot is clamped inside the plot so a label cannot
                  run into the lane labels.  The page exports the flows it
                  paints, the lane shares and the label bookkeeping into a
                  hidden #metaout div.
3. Headless Chrome (`--headless=new --dump-dom`) loads the page; on load the
   page sets window.__PNG__ and document.title = 'READY', and a tail script
   mirrors the PNG data URL into a hidden #pngout div so the DOM dump carries
   it.  The CSS viewport is the submitted 2,203 x 1,022 px; Chrome subtracts
   window furniture from the requested window size, so the first run measures
   the difference and the second run corrects for it.  At device scale factor
   4 the canvas is 8,812 x 4,088 px; the white margins are then cropped by
   the author's docx crop fractions (CROP: top 8.352%, right 4.662%, bottom
   6.466%), leaving 8,401 x 3,483 px, so the manuscript places the PNG
   uncropped at the width the author chose (5.43 in).

Run with:
    py figures/river_ice_render.py                 # THR=0.004, DSF 4
Options:
    --fit FILE                  levels and transfers (default
                                <assets>/shift_fit_ice.json, from river_shift_fit.py)
    --pressure                  the page's pressure model instead of the fit
    --joints slant|step         slanted cuts with the ribbon from cut to cut
                                (default), or the page's vertical steps;
                                --slant-part 0.333 is the share of a transfer
                                that each of its two slopes takes
    --thr 0.004 --prune 0.023   event trigger / ribbon pruning (--pressure only)
    --fonta 32 --fontl 32       anchor-title / lane-label font sizes (CSS px)
    --anchors                   draw the anchor games (dots and titles)
    --dsf 4                     device scale factor of the capture
    --no-crop                   keep the full 8,812 x 4,088 canvas
    --data games_raw_ice.js     data file (relative to the assets dir)
    --page / --png              output paths (default: the two files above)
    --assets DIR                directory holding cluster_share_stack.html and
                                the data file (default figures/river_assets, or
                                the script's own directory in the release bundle)
    --meta FILE                 JSON of the exported flows / lanes / labels
                                (default <png>.meta.json next to the PNG)
Needs only the standard library and Chrome at
    C:\Program Files\Google\Chrome\Application\chrome.exe
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_river_render import bake  # noqa: E402

# paper repo: figures/river_assets/ next to this file; release bundle: the
# page, the data file and this script share one directory
DEFAULT_ASSETS = HERE / "river_assets" if (HERE / "river_assets").exists() else HERE
# paper repo: the I-CE data is games_raw_ice.js, beside an older games_raw.js;
# release bundle: it is the page's own games_raw.js
DEFAULT_DATA = "games_raw_ice.js" if (DEFAULT_ASSETS / "games_raw_ice.js").exists() else "games_raw.js"
DEFAULT_FIT = "shift_fit_ice.json"

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

# published parameter set (what build_river_render.bake writes into the page)
PUB = dict(K=6, TAU=0.45, BETA=0.3, THR=0.0066, WARM_A=2, WARM_B=0.5,
           EPS=0.05, PRUNE=0.023)
# the I-CE figure's trigger and fonts.  The submitted PNG (8,813 x 4,089) was
# exported from the interactive page with 32 px labels and a plot spanning 62%
# of the canvas (measured on the file), not with the 45 px / 1,060 px right
# margin that build_river_render.bake() writes; these settings reproduce that
# composition.
ICE_THR, ICE_PRUNE, ICE_FONTA, ICE_FONTL = 0.004, 0.023, 32, 32
LANE_WRAP = 34            # lane labels longer than this (with the share) break at " / "

# capture geometry: the submitted 2,203 x 1,022 CSS viewport.  Pads measured
# on the submitted PNG: bands start at x = 165 and end 681 px before the right
# edge; 700 px holds the longest single-line I-CE label at 32 px.
PAD_L_OLD, PAD_L_NEW = 130, 165
PAD_R_OLD, PAD_R_NEW = 1060, 700
WIN_W = 2203
WIN_H = 1022
DSF = 4

# White-margin crop of the captured canvas, as fractions of its width / height.
# These are the author's own crop of the picture inside the manuscript docx
# (2026-09-04, Word srcRect t=8352 r=4662 b=6466, i.e. thousandths of a
# percent); baking them into the PNG keeps the docx placement a plain
# uncropped picture.  The ink ends at 95.1% of the width and 91.8% of the
# height, so nothing is cut.  --no-crop writes the full canvas.
CROP = dict(l=0.0, t=0.08352, r=0.04662, b=0.06466)

# ------------------------------------------------------- the fit handed to the page
# The page's own constants, which the fit's dates are converted with;
# bake_variant() checks that the page still declares them.
PAGE_CONSTS = "const T_END=8, YEAR0=2017;"
PAGE_T_END, PAGE_YEAR0 = 8, 2017


def flat_inputs(path):
    """What the page is given in place of its pressure model: a fit written by
    river_shift_fit.py (or river_flat_fit.py), as the axis limits (WARM_A /
    WARM_B, in the page's years since YEAR0), the level of every lane at the
    left edge, one event per transfer (fork at its start date, merge at its
    end date) and the row of every lane (0 = bottom; the lane's own index
    where the fit names none)."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"no fit at {path}: run river_shift_fit.py first (or pass --pressure)")
    fit = json.loads(path.read_text(encoding="utf-8"))
    if fit["year0"] != PAGE_YEAR0:
        raise SystemExit(f"{path.name} counts years from {fit['year0']}, the page from {PAGE_YEAR0}")
    y0 = fit["year0"]
    events = [dict(t=round(t["end"] - y0, 5),
                   flows=[dict(i=t["from"], j=t["to"], w=t["share"], ts=round(t["start"] - y0, 5))])
              for t in fit["transfers"]]
    rows = fit.get("rows") or list(range(len(fit["levels_start"])))
    if sorted(rows) != list(range(len(rows))):
        raise SystemExit(f"{path.name}: rows {rows} is not a permutation of the rows")
    # the measured shares of the last period, for the shares printed beside the bands
    end_shares = fit["periods"][-1]["shares"] if fit.get("periods") else None
    return dict(fit=fit, path=str(path), warm_a=fit["start"] - y0,
                warm_b=PAGE_T_END - (fit["end"] - y0), flat0=fit["levels_start"], events=events,
                rows=rows, end_shares=end_shares)


def widest_level(fit):
    """The largest share any band reaches (what the page scales the bands by)."""
    top = 0.0
    for k, level in enumerate(fit["levels_start"]):
        top = max(top, level)
        for t in sorted(fit["transfers"], key=lambda t: (t["end"], 1) if t["to"] == k else (t["start"], 0)):
            if t["to"] == k:
                level += t["share"]
            elif t["from"] == k:
                level -= t["share"]
            top = max(top, level)
    return top


LANE_Y = "const laneY=k=>H-PAD.b-(k+0.5)*plotH/K;"
LANE_DROP = "LANE_DROP=[%s]"      # CSS px a lane is drawn lower; river_ice_column.py sets the bottom row's


def flat_patches(flat):
    """The page edits that put the fit in place of the pressure model."""
    rows = flat["rows"]
    return [
        (LANE_Y, "const LANE_ROW=%s, %s;\n  const laneY=k=>H-PAD.b-(LANE_ROW[k]+0.5)*plotH/K+LANE_DROP[k];"
         % (json.dumps(rows), LANE_DROP % ",".join("0" for _ in rows))),
        ("WARM_A=2; WARM_B=0.5;", "WARM_A=%g; WARM_B=%g;" % (flat["warm_a"], flat["warm_b"])),
        ("<script>\nlet K=8;",
         "<script>\nconst FLAT0=%s;\nconst FLAT_EVENTS=%s;\nconst FLAT_END=%s;\nlet K=8;"
         % (json.dumps(flat["flat0"]), json.dumps(flat["events"]), json.dumps(flat["end_shares"]))),
        ("function computeFlows(){\n  EVENTS=[];",
         "function computeFlows(){\n"
         "  EVENTS=FLAT_EVENTS.map(e=>({t:e.t,flows:e.flows.map(f=>Object.assign({},f))}));\n"
         "  computeAnchors(); return;\n  EVENTS=[];"),
        ("let Wk=aW(t0)[k];", "let Wk=FLAT0[k];"),
        ("const W0=aW(t0)[k]*F;", "const W0=FLAT0[k]*F;"),
        ("share0:+(aW(WARM_A)[k].toFixed(5))", "share0:+(FLAT0[k].toFixed(5))"),
    ]


def note_fit(meta, flat, joints="step", slant_part=None):
    """Record in the meta which numbers the page drew and how a ribbon meets a band."""
    meta["joints"] = joints
    if joints == "slant":
        meta["slant_part"] = SLANT_PART if slant_part is None else slant_part
        meta["slant_max_years"] = SLANT_MAX
        meta["slant_bend"], meta["slant_handle"] = SLANT_BEND, SLANT_HANDLE
    if flat is None:
        meta["levels"] = "the page's pressure model (--pressure)"
        return meta
    fit = flat["fit"]
    meta["rows"] = flat["rows"]
    if fit.get("method") == "shift":
        meta["levels"] = "river_shift_fit.py"
        meta["shift_fit"] = {k: fit[k] for k in ("games_used", "games_total", "start", "end", "boundaries",
                                                "boundary_tests", "minlen_months", "shuffles", "level", "seed",
                                                "min_share", "min_transfer_years", "largest_gap_to_measured")}
        return meta
    meta["levels"] = "river_flat_fit.py"
    meta["flat_fit"] = {k: fit[k] for k in ("games_used", "games_total", "start", "end", "sigma_years",
                                           "tolerance", "min_transfer_years", "fit")}
    return meta


# ------------------------------------------------------------ slanted joints
# The page draws a transfer with two vertical steps: the source band narrows at
# the start date, the target band widens at the end date, and the ribbon runs
# between the two steps.  With joints="slant" (the author's design of
# 2026-10-13: "endpoint A at the start of the transfer, endpoint B at the
# trigger", the red-pen sketch, and "the two ends show time, the thickness in
# the middle shows the share") each step becomes a slanted cut and the ribbon
# runs from cut to cut:
#   * the source band narrows along a straight slope that starts at A (the
#     start date, on the edge that faces the target) and takes the first
#     SLANT_PART of the transfer, at most SLANT_MAX years (see below);
#   * the target band widens along a slope that takes the last SLANT_PART of
#     the transfer and ends at B (the end date);
#   * the ribbon's two ends are those two slopes, whole, so an end is as wide
#     along the time axis as its slope is long: it shows a duration.  Its first
#     edge runs from the start of the source's slope to the start of the
#     target's, its second edge from the end of the source's slope to the end
#     of the target's;
#   * between the ends the ribbon is a band as thick, measured across, as the
#     share it carries.  The edge that leaves at the END of the source's slope
#     and the edge that lands at the START of the target's slope are the two
#     parallels of one curve, one share's thickness apart.  Each of the other
#     two corners, where the first edge leaves the source band and where the
#     second edge joins the target band, is rounded: the edge is level there,
#     as the band edge it continues, and turns into the curve on one parabola.
# Drawn and rejected by the author on 2026-10-13: a ribbon tapering to a point
# at A and at B (it touched the bands only at two points); a ribbon as wide as
# its ends all along (its width then showed a duration, not the share); and
# S-curves on all four corners (the two that bulged beside the straight lines
# of that day were the ones crossed out in green on the sketch).
# Three changes of 2026-11-11 and 2026-11-12, when river_shift_fit.py began to
# draw a small movement as a ribbon several years long:
#   * a cut is at most SLANT_MAX years wide.  A cut of a third of such a ribbon
#     was over a year wide, the two fanned ends took nine tenths of the ribbon,
#     and a ribbon of 2 points looked thicker than one of 6 (the author:
#     "他看起来很粗");
#   * the run between the cuts is a curve, not a straight line (the author:
#     "现在长程的传输线太“直”了，能不能更曲线一点").  The ribbon leaves the source's
#     cut and reaches the target's at SLANT_BEND times the slope of the straight
#     run and follows a cubic in between (handles SLANT_HANDLE of the run), so a
#     long ribbon is a flat S.  A ribbon that leaves and lands level, as in a
#     Sankey diagram, was tried and dropped: a thin long ribbon then hugs the
#     band it joins over a long stretch, and the slanted cuts turn into steps;
#   * ribbons whose cuts on one band edge overlap in time each take a part of
#     that stretch, in proportion to their shares, and so leave (or land) side
#     by side.  Before, they all left from the one end of the shared cut, on
#     top of each other (the author, of the left end of the puzzle band:
#     "这里太怪了").  Where the cuts coincide, as the two on the puzzle band
#     do, the band's edge is the same line either way; cuts that overlap only
#     in part become one slope over their union, and a group in which that
#     would push a ribbon's source cut past the start of its target cut keeps
#     its own cuts.
# The block of draw() between the two markers is replaced as a whole.
SLANT_PART = 1 / 3
SLANT_MAX = 1 / 3            # years: the widest a slanted cut is drawn (four months, the shortest ribbon)
SLANT_BEND = 0.45            # a ribbon leaves and lands at this share of the straight run's slope (1: straight)
SLANT_HANDLE = 0.4           # the two handles of the ribbon's cubic, as a share of the run's length
SLANT_FROM = "  /* EDGE-anchored account: top and bottom edges move INDEPENDENTLY."
SLANT_TO = "  /* anchor games (edit mode: labels draggable, offsets persisted) */"
SLANT_JS = """  /* SLANTED JOINTS (river_ice_render.py, joints="slant").  Every band edge is
     its start level plus one linear ramp per transfer that touches it: a fork
     moves the edge facing the target inward over the first part of the
     transfer, a merge moves the edge facing the source outward over the last
     part. */
  const SLANT_PART=__PART__, SLANT_MAX=__MAX__;
  window.__JOINTS__=[];
  /* Bands and ribbons are painted opaque here, in the colour their usual
     translucent fill gives on the page's background: where a ribbon continues
     a band the two fills then meet, and overlap by a hair, without a seam. */
  const _solid={}, _px=document.createElement('canvas'); _px.width=_px.height=1;
  const _pc=_px.getContext('2d',{willReadFrequently:true});
  const solid=c=>{
    if(_solid[c]) return _solid[c];
    _pc.fillStyle=PAPER?'#ffffff':'#06060e'; _pc.fillRect(0,0,1,1);
    _pc.fillStyle=c; _pc.fillRect(0,0,1,1);
    const d=_pc.getImageData(0,0,1,1).data;
    return _solid[c]=`rgb(${d[0]},${d[1]},${d[2]})`;
  };
  const lvl0=k=>((typeof FLAT0!=='undefined')?FLAT0[k]:aW(t0)[k])*F;
  const ED=Array.from({length:K},(_,k)=>({top0:laneY(k)-lvl0(k)/2, bot0:laneY(k)+lvl0(k)/2,
                                           top:[], bot:[]}));
  usable.forEach(f=>{
    const up=laneY(f.j)<laneY(f.i), d=f._te-f._ts;
    /* a cut takes SLANT_PART of the transfer and at most SLANT_MAX years: on a long
       ribbon wider cuts left almost no stretch as thick as the share (2026-11-11) */
    const cutw=Math.min(SLANT_PART*d, SLANT_MAX);
    f._se=up?'top':'bot'; f._de=up?'bot':'top';
    f._a0=f._ts; f._a1=f._ts+cutw; f._b0=f._te-cutw; f._b1=f._te;
  });
  /* Ribbons whose cuts on one band edge overlap in time share that stretch of
     the edge: each takes a part of it in proportion to its share, so every
     ribbon has a cut of its own and leaves or lands beside the others, not on
     top of them.  Where ribbons leave, the one that travels farthest takes the
     first part (the strip next to the band's old edge); where they land, the
     last part.  Cuts that coincide leave the band's edge the line it was; cuts
     that overlap only in part become one slope over their union.  Should that
     push a ribbon's cut on its source past the start of its cut on its target
     (a short ribbon given a late part of a long shared stretch), the group
     keeps the cuts it had. */
  const far=f=>Math.abs(laneY(f.j)-laneY(f.i));
  const share=(key,t0k,t1k,edge,farFirst)=>{
    const by={};
    usable.forEach(f=>{ (by[f[key]+'|'+f[edge]]=by[f[key]+'|'+f[edge]]||[]).push(f); });
    Object.values(by).forEach(L=>{
      L.sort((p,q)=>p[t0k]-q[t0k]);
      let grp=[];
      const close=()=>{
        if(grp.length>1){
          const a=Math.min(...grp.map(f=>f[t0k])), b=Math.max(...grp.map(f=>f[t1k]));
          const tot=grp.reduce((x,f)=>x+f.w,0);
          grp.sort((p,q)=>farFirst?far(q)-far(p):far(p)-far(q));
          const kept=grp.map(f=>[f[t0k],f[t1k]]);
          let t=a;
          grp.forEach(f=>{ f[t0k]=t; t+=(b-a)*f.w/tot; f[t1k]=t; });
          if(grp.some(f=>f._a1>f._b0+1e-9)) grp.forEach((f,q)=>{ f[t0k]=kept[q][0]; f[t1k]=kept[q][1]; });
        }
        grp=[];
      };
      let end=0;
      L.forEach(f=>{ if(grp.length && f[t0k]>=end-1e-9) close();
                     end=grp.length?Math.max(end,f[t1k]):f[t1k];
                     grp.push(f); });
      close();
    });
  };
  share('i','_a0','_a1','_se',true);
  share('j','_b0','_b1','_de',false);
  usable.forEach(f=>{
    const wf=f.w*F, up=(f._se==='top');
    ED[f.i][f._se].push({t0:f._a0, t1:f._a1, dy:up?wf:-wf});
    ED[f.j][f._de].push({t0:f._b0, t1:f._b1, dy:up?wf:-wf});
  });
  const edgeY=(k,which,t)=>{
    let y=(which==='top')?ED[k].top0:ED[k].bot0;
    ED[k][which].forEach(c=>{
      y+=c.dy*Math.max(0,Math.min(1,(t-c.t0)/Math.max(c.t1-c.t0,1e-9))); });
    return y;
  };
  const breaks=(k,which)=>{
    const s=new Set([t0,t1]);
    ED[k][which].forEach(c=>{ s.add(Math.max(t0,Math.min(t1,c.t0)));
                              s.add(Math.max(t0,Math.min(t1,c.t1))); });
    return [...s].sort((a,b)=>a-b);
  };
  /* main streams: one polygon per band, the top edge left to right and the
     bottom edge back */
  const SEG=[], BANDP=[];
  for(let k=0;k<K;k++){
    const bt=breaks(k,'top'), bb=breaks(k,'bot'), p=new Path2D();
    bt.forEach((t,q)=>{
      const x=sx(t)+(q===0?-0.5:(q===bt.length-1?0.5:0)), y=edgeY(k,'top',t);
      if(q===0) p.moveTo(x,y); else p.lineTo(x,y); });
    for(let q=bb.length-1;q>=0;q--)
      p.lineTo(sx(bb[q])+(q===0?-0.5:(q===bb.length-1?0.5:0)), edgeY(k,'bot',bb[q]));
    p.closePath();
    ctx.fillStyle=solid(col(k, PAPER?0.92:0.8, PAPER));
    ctx.fill(p);
    BANDP.push(p);
    /* the pieces the labels and the scale bar read: between two breaks, the
       highest top edge and the lowest bottom edge */
    const all=[...new Set([...bt,...bb])].sort((a,b)=>a-b), segs=[];
    for(let q=0;q<all.length-1;q++)
      segs.push({ta:all[q], tb:all[q+1],
        yT:Math.min(edgeY(k,'top',all[q]),edgeY(k,'top',all[q+1])),
        yB:Math.max(edgeY(k,'bot',all[q]),edgeY(k,'bot',all[q+1]))});
    SEG.push(segs);
  }
  /* ribbons, from slanted cut to slanted cut.  A cut follows the band edge it
     lies on, 0.8 px inside the band so that no hairline shows between the two
     fills.
     Corners: S0, S1 the start and the end of the source's cut, T0, T1 the
     start and the end of the target's.  The second edge leaves S1 and the
     first edge reaches T0; the two are the parallels of one cubic, half a
     share's thickness to either side of it.  The first edge leaves S0 level,
     as the band edge it continues, and turns into its curve on one parabola;
     the second edge turns off its curve the same way and lands level at T1. */
  const cut=(k,which,ta,tb)=>{
    const inn=(which==='top')?0.8:-0.8;
    return [ta, ...breaks(k,which).filter(t=>t>ta+1e-9 && t<tb-1e-9), tb]
      .map(t=>({x:sx(t), y:edgeY(k,which,t)+inn}));
  };
  const corner=(P,C,Q,n)=>{                  /* parabola from P to Q, tangents through C */
    const out=[];
    for(let q=0;q<=n;q++){ const t=q/n, m=1-t;
      out.push({x:m*m*P.x+2*m*t*C.x+t*t*Q.x, y:m*m*P.y+2*m*t*C.y+t*t*Q.y}); }
    return out;
  };
  usable.forEach(f=>{
    const wf=f.w*F;
    const cs=cut(f.i,f._se,f._a0,f._a1), cd=cut(f.j,f._de,f._b0,f._b1);
    /* the four corners lie on the band edges themselves; only the cuts between
       them are drawn a hair inside the bands */
    const on=(k,which,t)=>({x:sx(t), y:edgeY(k,which,t)});
    const S0=on(f.i,f._se,f._a0), S1=on(f.i,f._se,f._a1), T0=on(f.j,f._de,f._b0), T1=on(f.j,f._de,f._b1);
    /* u0: the direction of the straight ribbon (the edge through S1 and the edge
       through T0 parallel, one share's thickness apart).  The curved ribbon
       leaves S1 and reaches T0 at BEND times that slope and follows a cubic in
       between; its two edges are the cubic's parallels, half a thickness to
       either side, so it is as thick as its share all along. */
    const D={x:T0.x-S1.x, y:T0.y-S1.y}, dl=Math.hypot(D.x,D.y)||1;
    const phi=Math.asin(Math.min(1,wf/dl)), cross=(a,b)=>a.x*b.y-a.y*b.x;
    const turn=a=>({x:(D.x*Math.cos(a)-D.y*Math.sin(a))/dl, y:(D.x*Math.sin(a)+D.y*Math.cos(a))/dl});
    let u0=turn(phi);
    if(cross(u0,D)*cross(u0,{x:S0.x-S1.x, y:S0.y-S1.y})<0) u0=turn(-phi);
    const BEND=__BEND__, HANDLE=__HANDLE__;
    const ul=Math.hypot(u0.x, BEND*u0.y)||1, u={x:u0.x/ul, y:BEND*u0.y/ul};
    const sd=(cross(u,{x:S0.x-S1.x, y:S0.y-S1.y})>0)?1:-1;       /* S0's side of the run */
    const nrm=t=>({x:-t.y*sd, y:t.x*sd});
    const n0=nrm(u);
    const A={x:S1.x+n0.x*wf/2, y:S1.y+n0.y*wf/2}, B={x:T0.x-n0.x*wf/2, y:T0.y-n0.y*wf/2};
    const hl=HANDLE*Math.hypot(B.x-A.x,B.y-A.y);
    const c1={x:A.x+hl*u.x, y:A.y+hl*u.y}, c2={x:B.x-hl*u.x, y:B.y-hl*u.y};
    const N=120, E0=[], E1=[], TG=[];
    for(let q=0;q<=N;q++){ const t=q/N, m=1-t;
      const x=m*m*m*A.x+3*m*m*t*c1.x+3*m*t*t*c2.x+t*t*t*B.x, y=m*m*m*A.y+3*m*m*t*c1.y+3*m*t*t*c2.y+t*t*t*B.y;
      let tx=3*m*m*(c1.x-A.x)+6*m*t*(c2.x-c1.x)+3*t*t*(B.x-c2.x), ty=3*m*m*(c1.y-A.y)+6*m*t*(c2.y-c1.y)+3*t*t*(B.y-c2.y);
      const tl=Math.hypot(tx,ty)||1, tg={x:tx/tl, y:ty/tl}, nn=nrm(tg);
      TG.push(tg);
      E0.push({x:x+nn.x*wf/2, y:y+nn.y*wf/2});               /* first edge: ends at T0 */
      E1.push({x:x-nn.x*wf/2, y:y-nn.y*wf/2}); }             /* second edge: starts at S1 */
    /* The first edge leaves S0 level, as the band edge it continues, and turns
       into the curve on one parabola: it meets the curve at the first point
       whose tangent crosses S0's level ahead of S0 and at least as far from
       that crossing as S0 is.  The second edge lands level at T1 the same way. */
    /* (on a short ribbon no point of its first half is that far: the point
       whose two tangent lengths are nearest to equal is taken.  A ribbon so
       short or steep that no tangent of its first half crosses S0's level
       ahead of S0 keeps the straight join from S0 to the start of the curve;
       none of the figure's does: corner_s and corner_t in the meta file) */
    let e0=[S0, ...E0], e1=[...E1, T1], qa=-1, qb=-1, ra=0, rb=0, Ca=null, Cb=null;
    for(let q=1;q<=N*0.5;q++){ const Q=E0[q], t=TG[q];
      if(Math.abs(t.y)<1e-6) continue;
      const s2=(S0.y-Q.y)/t.y, C={x:Q.x+s2*t.x, y:S0.y};
      if(!(s2<0 && C.x>S0.x+0.5)) continue;
      const r=Math.min(1,-s2/(C.x-S0.x));
      if(r>ra){ ra=r; qa=q; Ca=C; }
      if(r>=1) break; }
    if(qa>0) e0=[...corner(S0,Ca,E0[qa],40), ...E0.slice(qa+1)];
    for(let q=N-1;q>=N*0.5;q--){ const Q=E1[q], t=TG[q];
      if(Math.abs(t.y)<1e-6) continue;
      const s2=(T1.y-Q.y)/t.y, C={x:Q.x+s2*t.x, y:T1.y};
      if(!(s2>0 && C.x<T1.x-0.5)) continue;
      const r=Math.min(1,s2/(T1.x-C.x));
      if(r>rb){ rb=r; qb=q; Cb=C; }
      if(r>=1) break; }
    if(qb>0) e1=[...E1.slice(0,qb), ...corner(E1[qb],Cb,T1,40)];
    window.__JOINTS__.push({i:f.i, j:f.j, waist:+(wf/F).toFixed(5),
      lean:+(Math.atan2(Math.abs(u.y),u.x)*180/Math.PI).toFixed(1), straight:+(Math.atan2(Math.abs(u0.y),u0.x)*180/Math.PI).toFixed(1),
      corner_s:qa, corner_t:qb, cut_s:[+f._a0.toFixed(4), +f._a1.toFixed(4)], cut_t:[+f._b0.toFixed(4), +f._b1.toFixed(4)]});
    const p=new Path2D();
    p.moveTo(S0.x,S0.y);
    cs.forEach(q=>p.lineTo(q.x,q.y));
    e1.forEach(q=>p.lineTo(q.x,q.y));
    for(let q=cd.length-1;q>=0;q--) p.lineTo(cd[q].x,cd[q].y);
    for(let q=e0.length-1;q>=0;q--) p.lineTo(e0[q].x,e0[q].y);
    p.closePath();
    /* the colour runs from the source band's to the target band's with the
       height, and is held at either band's own colour over the height of its
       cut, so the ribbon continues each band without a seam */
    const ys=[S0.y,S1.y], yd=[T0.y,T1.y], down=T0.y>S0.y;
    const g0=down?Math.min(...ys):Math.max(...ys), g1=down?Math.max(...yd):Math.min(...yd);
    const span=Math.max(Math.abs(g1-g0),1e-6);
    let a0=Math.min(0.45,(Math.abs(ys[1]-ys[0])+2)/span);
    const a1=Math.min(0.45,(Math.abs(yd[1]-yd[0])+2)/span);
    const grad=ctx.createLinearGradient(0,g0,0,g1);
    const cS=solid(col(f.i, PAPER?0.92:0.8, PAPER)), cT=solid(col(f.j, PAPER?0.92:0.8, PAPER));
    grad.addColorStop(0, cS);
    grad.addColorStop(a0, cS);
    for(let s=0.2;s<=0.801;s+=0.2)
      grad.addColorStop(a0+(1-a0-a1)*s, solid(colMix(f.i,f.j,s, PAPER?0.9:0.85, PAPER)));
    grad.addColorStop(1-a1, cT);
    grad.addColorStop(1, cT);
    /* a hairline of paper along the two long edges keeps the ribbon apart from
       a band it crosses.  It is painted only inside the other bands (never on
       the ribbon's own source or target band, where it would cut a notch into
       the joint), and before the ribbon is filled, so only its outer half
       remains and the ribbon's own edge stays whole. */
    const others=new Path2D();
    for(let k=0;k<K;k++) if(k!==f.i && k!==f.j) others.addPath(BANDP[k]);
    const e=new Path2D();
    [e0,e1].forEach(E=>E.forEach((q,n)=>{ if(n===0) e.moveTo(q.x,q.y); else e.lineTo(q.x,q.y); }));
    ctx.save(); ctx.clip(others);
    ctx.strokeStyle=PAPER?'#ffffff':'#06060e'; ctx.lineWidth=4.8; ctx.lineCap='butt'; ctx.lineJoin='round';
    ctx.stroke(e); ctx.restore();
    ctx.fillStyle=grad;
    ctx.fill(p);
  });

"""


def slant_joints(h, name, part=SLANT_PART):
    """Replace the page's vertical-step width account and its ribbons by the
    slanted joints above; `part` is the share of a transfer's duration that
    each of its two slopes takes (at most a half)."""
    if not 0 < part <= 0.5:
        raise SystemExit(f"the slanted part of a transfer must be in (0, 0.5], not {part}")
    a, b = h.find(SLANT_FROM), h.find(SLANT_TO)
    if a < 0 or b < a:
        raise SystemExit(f"{name}: the width-account block of draw() was not found")
    return h[:a] + (SLANT_JS.replace("__PART__", repr(float(part))).replace("__MAX__", repr(float(SLANT_MAX)))
                    .replace("__BEND__", repr(float(SLANT_BEND))).replace("__HANDLE__", repr(float(SLANT_HANDLE)))) + h[b:]


# ------------------------------------------------------------------ patches
# Every patch is asserted against the baked page, so a change upstream in
# cluster_share_stack.html or build_river_render.py fails loudly.
FIX_PAD = (f"PAD={{l:{PAD_L_OLD},r:{PAD_R_OLD},t:160,b:110}}",
           f"PAD={{l:{PAD_L_NEW},r:{PAD_R_NEW},t:160,b:110}}")

# the rotated y-axis title was drawn at x = 24, where its ascent is clipped
FIX_YTITLE = ("ctx.translate(24,(PAD.t+H-PAD.b)/2);",
              "ctx.translate(44,(PAD.t+H-PAD.b)/2);")

# lane labels: a label that would run past the right margin breaks at its
# " / " into two lines, the share on the second line
FIX_LANE_TEXT = ("""    ctx.fillText(`${CLUSTER_LABELS[k].label}  ${(100*Wend).toFixed(0)}%`,
                 W-PAD.r+12, laneY(k)+3);""",
                 """    { const lab=CLUSTER_LABELS[k].label, pct=`${(100*Wlab).toFixed(0)}%`;
      const one=`${lab}  ${pct}`, cut=lab.indexOf(' / ');
      if(one.length<=LANE_WRAP || cut<0){
        ctx.fillText(one, W-PAD.r+12, laneY(k)+3);
      } else {
        ctx.fillText(lab.slice(0,cut+2), W-PAD.r+12, laneY(k)-FONTL*0.2);
        ctx.fillText(`${lab.slice(cut+3)}  ${pct}`, W-PAD.r+12, laneY(k)+FONTL*0.95);
      } }""".replace("LANE_WRAP", str(LANE_WRAP)))

# export the flows draw() actually paints (w >= PRUNE)
FIX_FLOWS = ("  /* scale factor from the width account */",
             "  window.__FLOWS__=usable.map(f=>({i:f.i,j:f.j,w:f.w,\n"
             "    ts:+(f._ts.toFixed(4)),te:+(f._te.toFixed(4))}));\n"
             "  window.__NEVENTS__=EVENTS.length;\n"
             "  window.__NFLOWTOT__=EVENTS.reduce((a,e)=>a+e.flows.length,0);\n"
             "  /* scale factor from the width account */")

# export the lane labels and their start / end shares
FIX_LANES_OLD = """  for(let k=0;k<K;k++){
    const sEnd=SEG[k][SEG[k].length-1];
    const Wend=(sEnd.yB-sEnd.yT)/F;              // account width, matches the band
    ctx.fillStyle=ink;"""
FIX_LANES_NEW = """  window.__LANES__=[];
  for(let k=0;k<K;k++){
    const sEnd=SEG[k][SEG[k].length-1];
    const Wend=(sEnd.yB-sEnd.yT)/F;              // account width, matches the band
    /* the share printed beside the band: the measured share of the last period
       where the fit gives one (a movement too small to draw leaves the band up
       to that much off it), the band's own width otherwise */
    const Wlab=(typeof FLAT_END!=='undefined' && FLAT_END)?FLAT_END[k]:Wend;
    window.__LANES__.push({lane:k, label:CLUSTER_LABELS[k].label,
      n:CLUSTER_LABELS[k].n, share:+(Wend.toFixed(5)), share_label:+(Wlab.toFixed(5)),
      share0:+(aW(WARM_A)[k].toFixed(5))});
    ctx.fillStyle=ink;"""

# no anchor games: the page picks none, so it draws neither dots nor titles
# and the placement code below never runs
FIX_NO_ANCHORS = ("  ANCHORS=[];\n  if(!LANE_OF) return;",
                  "  ANCHORS=[];\n  return;")

# anchor titles: the page strips CJK and collapses whitespace; also drop a
# trailing ellipsis ("If Found..." is shown as "If Found")
FIX_TITLE = ("        .replace(/\\s+/g,' ').trim();",
             "        .replace(/\\s+/g,' ').replace(/\\s*(?:\\.\\.\\.|\u2026)\\s*$/,'').trim();")

# label placement.  NB the first target has to carry the preceding ctx.font
# line: "ANCHOR_HITS=[];" on its own also matches the `let ANCHOR_HITS=[];`
# declaration near the top of the file.
FIX_ANCHOR_OLD = """  ctx.font=FONTA+'px "SF Mono",Consolas,monospace';
  ANCHOR_HITS=[];"""
FIX_ANCHOR_NEW = """  ctx.font=FONTA+'px "SF Mono",Consolas,monospace';
  ANCHOR_HITS=[]; window.__LBL_MOVED__=0; window.__LBL_HIDDEN__=0;
  window.__LBL_CLAMPED__=0; window.__LABELS__=[];
  const LABEL_JOBS=[];
  /* every anchor dot as a box, so no label is ever placed over a dot */
  const DOT_BOXES=ANCHORS.map(a=>{
    const ax=sx(a.tau), ay=laneY(a.lane)+a.du*(plotH/K)*0.34;
    return {x:ax-DOTR-3, y:ay-DOTR-3, w:2*DOTR+6, h:2*DOTR+6};});"""

FIX_LOOP = ("  ANCHORS.forEach(an=>{\n    const x=sx(an.tau), yc=laneY(an.lane);",
            "  ANCHORS.forEach((an,ai)=>{\n    const x=sx(an.tau), yc=laneY(an.lane);")

FIX_LABEL_OLD = """    const off=LABEL_OFF[an.title]||[0,0];
    const lx=x+off[0];
    const ly=yc+dy+(an.du>=0?FONTA*1.4:-FONTA*0.82)+off[1];
    ctx.textAlign='center'; ctx.fillStyle=inkA;
    ctx.fillText(an.title, lx, ly);
    const tw=ctx.measureText(an.title).width;
    ANCHOR_HITS.push({x:lx-tw/2-3, y:ly-FONTA-2, w:tw+6, h:FONTA+6, title:an.title});
    if(EDITMODE){
      ctx.strokeStyle='rgba(126,200,255,.45)'; ctx.lineWidth=0.8;
      ctx.strokeRect(lx-tw/2-3, ly-FONTA-2, tw+6, FONTA+6);
    }"""
FIX_LABEL_NEW = """    const off=LABEL_OFF[an.title]||[0,0];
    const tw=ctx.measureText(an.title).width;
    /* Label placement is solved for all anchors at once (see the last anchor
       below).  Here each title gets its candidate slots: vertically the page's
       natural slot (below the dot for a game in the lower half of its lane,
       above it otherwise) and the mirror slot on the other side of the dot,
       each stepping AWAY from the dot in half-line (0.575*FONTA) increments;
       horizontally the centred position and nudges of +-15% / +-30% of the
       label width (the label always still spans its dot).  cost = vertical
       displacement in lines + 0.5 for the mirror side + 0.4 per 15% nudge.
       Slots outside the plot or over any anchor dot are dropped; 'hidden'
       (cost 20) is always available as the last resort. */
    const below=(an.du>=0), yd=yc+dy+off[1], x0=x+off[0];
    const bw=tw+6, bh=FONTA+6, step=FONTA*0.575;
    const cands=[];
    for(const [fx,cx] of [[0,0],[-0.15,0.4],[0.15,0.4],[-0.3,0.8],[0.3,0.8]]){
      let lx=x0+fx*bw, clamped=0;
      if(fx===0){                       // the centred slot is clamped into the plot
        if(lx+tw/2>W-PAD.r){ lx=W-PAD.r-tw/2; clamped=1; }
        if(lx-tw/2<PAD.l){ lx=PAD.l+tw/2; clamped=1; }
      } else if(lx+tw/2>W-PAD.r || lx-tw/2<PAD.l) continue;
      const bx=lx-tw/2-3;
      const hitBox=(h,by)=>bx<h.x+h.w && bx+bw>h.x && by<h.y+h.h && by+bh>h.y;
      for(let s=0;s<=12;s++){
        for(const [sd,sc] of [['natural',0],['mirror',0.5]]){
          const nat=(sd==='natural')?below:!below;
          const by=yd+(nat?FONTA*1.4:-FONTA*0.82)-FONTA-2+(nat?1:-1)*s*step;
          if(by<PAD.t-FONTA || by+bh>H-PAD.b) continue;
          if(DOT_BOXES.some(h=>hitBox(h,by))) continue;
          cands.push({by:by, lx:lx, bx:bx, cost:s*0.5+sc+cx, side:sd, steps:s,
                      nudge:fx, clamped:clamped});
        }
      }
    }
    cands.sort((a,b)=>a.cost-b.cost);
    cands.push({by:null, lx:x0, bx:x0-tw/2-3, cost:20, side:'hidden', steps:-1, nudge:0, clamped:0});
    LABEL_JOBS.push({an:an, bw:bw, bh:bh, cands:cands, dx:x, dyy:yc+dy});

    if(ai===ANCHORS.length-1){
      /* branch and bound over the joint assignment: minimise the total cost
         subject to no two labels overlapping.  Most constrained job first;
         candidates are cost-sorted, so the first complete assignment is the
         greedy one and later ones only improve on it. */
      const J=LABEL_JOBS.slice().sort((a,b)=>a.cands.length-b.cands.length);
      const n=J.length, pick=new Array(n).fill(0);
      let best=Infinity, bestPick=pick.slice(), nodes=0;
      /* two labels on the same row must be at least GAP apart, so that
         neighbouring titles cannot be read as one */
      const GAP=FONTA*1.5;
      const ov=(a,ca,b,cb)=>ca.by!==null && cb.by!==null
        && ca.bx<cb.bx+b.bw+GAP && ca.bx+a.bw+GAP>cb.bx && ca.by<cb.by+b.bh && ca.by+a.bh>cb.by;
      const rec=(i,cost)=>{
        if(cost>=best || nodes>400000) return;
        if(i===n){ best=cost; bestPick=pick.slice(); return; }
        const job=J[i];
        for(let c=0;c<job.cands.length;c++){
          nodes++;
          const cand=job.cands[c];
          if(cost+cand.cost>=best) break;          // cost-sorted: nothing later is cheaper
          let okc=true;
          for(let k=0;k<i;k++) if(ov(job,cand,J[k],J[k].cands[pick[k]])){ okc=false; break; }
          if(!okc) continue;
          pick[i]=c; rec(i+1,cost+cand.cost);
        }
      };
      rec(0,0);
      window.__LBL_NODES__=nodes; window.__LBL_COST__=best;
      J.forEach((job,i)=>{
        const c=job.cands[bestPick[i]], a=job.an;
        window.__LABELS__.push({title:a.title, lane:a.lane, tau:+a.tau.toFixed(3),
          du:+a.du.toFixed(3), x:+c.lx.toFixed(1), y:(c.by===null?null:+c.by.toFixed(1)),
          dot_y:+job.dyy.toFixed(1), bx:+c.bx.toFixed(1), bw:+job.bw.toFixed(1), bh:job.bh,
          n_free:job.cands.length-1, nudge:c.nudge,
          steps:c.steps, side:c.side, clamped:c.clamped});
        window.__LBL_CLAMPED__+=c.clamped;
        if(c.by===null){ window.__LBL_HIDDEN__++; return; }
        window.__LBL_MOVED__+=(c.steps>0 || c.side!=='natural' || c.nudge!==0);
        ctx.textAlign='center'; ctx.fillStyle=inkA;
        ctx.fillText(a.title, c.lx, c.by+FONTA+2);
        ANCHOR_HITS.push({x:c.bx, y:c.by, w:job.bw, h:job.bh, title:a.title});
        /* leader line from the dot to the label's near edge once the label
           sits more than a line away from its dot */
        const edgeY=(c.by>job.dyy)?c.by:c.by+job.bh;
        if(Math.abs(edgeY-job.dyy)>FONTA){
          const lxx=Math.max(c.bx+12, Math.min(c.bx+job.bw-12, job.dx));
          ctx.strokeStyle=PAPER?'rgba(30,34,50,.55)':'rgba(220,228,245,.5)';
          ctx.lineWidth=1;
          ctx.beginPath(); ctx.moveTo(job.dx, job.dyy+(edgeY>job.dyy?DOTR:-DOTR));
          ctx.lineTo(lxx, edgeY+(edgeY>job.dyy?-2:2)); ctx.stroke();
        }
        if(EDITMODE){
          ctx.strokeStyle='rgba(126,200,255,.45)'; ctx.lineWidth=0.8;
          ctx.strokeRect(c.bx, c.by, job.bw, job.bh);
        }
      });
    }"""

TAIL = """
<script>
// mirror the headless hand-off into the DOM so `chrome --dump-dom` can read it
window.addEventListener('load', function(){
  var d=document.createElement('div'); d.id='pngout'; d.style.display='none';
  d.textContent=(document.title==='READY'?(window.__PNG__||''):'');
  document.body.appendChild(d);
  var m=document.createElement('div'); m.id='metaout'; m.style.display='none';
  m.textContent=JSON.stringify({w:innerWidth,h:innerHeight,dpr:devicePixelRatio,
    cw:document.getElementById('cv').width,ch:document.getElementById('cv').height,
    n:(typeof RAW_N==='number'?RAW_N:-1),
    d:(typeof RAW_D==='number'?RAW_D:-1),
    params:{K:K,TAU:TAU,BETA:BETA,THR:THR,WARM_A:WARM_A,WARM_B:WARM_B,EPS:EPS,
            PRUNE:PRUNE,MARK:MARK,DOTR:DOTR,MINW:MINW,FONTA:FONTA,FONTL:FONTL},
    flows:(window.__FLOWS__||[]), joints_drawn:(window.__JOINTS__||[]), n_events:(window.__NEVENTS__|0),
    n_flow_total:(window.__NFLOWTOT__|0),
    lanes:(window.__LANES__||[]), labels:(window.__LABELS__||[]),
    lbl_moved:(window.__LBL_MOVED__|0), lbl_hidden:(window.__LBL_HIDDEN__|0),
    lbl_clamped:(window.__LBL_CLAMPED__|0), lbl_nodes:(window.__LBL_NODES__|0),
    lbl_cost:(window.__LBL_COST__===undefined?null:window.__LBL_COST__)});
  document.body.appendChild(m);
});
</script>
"""


def bake_variant(out_path, data_js, thr=ICE_THR, prune=ICE_PRUNE, src=None,
                 fonta=ICE_FONTA, fontl=ICE_FONTL, anchors=False, flat=None, joints="step",
                 slant_part=None):
    """bake() the render page from the interactive source, then apply the
    I-CE settings and the three rendering fixes above.  `src` overrides the
    interactive page (default: river_assets/cluster_share_stack.html next to
    build_river_render.py); `data_js` is the <script src> the page loads,
    relative to the page's own directory.  `anchors` keeps the anchor games
    (dots and titles), which the paper figure does not carry.  `flat` (from
    flat_inputs()) gives the page the fitted levels and transfers in place of
    its pressure model; every transfer is then drawn, whatever `prune` says.
    `joints` is "step" (the page's vertical steps where a ribbon leaves or
    joins a band) or "slant" (SLANT_JS above)."""
    if flat is not None:
        prune = 0
    if src is None:
        bake(out_path, data_js, TAIL)
    else:
        bake(out_path, data_js, TAIL, src=src)
    h = out_path.read_text(encoding="utf-8")
    patches = [
        FIX_PAD, FIX_YTITLE, FIX_FLOWS,
        (FIX_LANES_OLD, FIX_LANES_NEW),
        FIX_LANE_TEXT,
        (FIX_ANCHOR_OLD, FIX_ANCHOR_NEW),
        FIX_TITLE,
        FIX_LOOP,
        (FIX_LABEL_OLD, FIX_LABEL_NEW),
        (f"THR={PUB['THR']};", f"THR={thr};"),
        (f"PRUNE={PUB['PRUNE']};", f"PRUNE={prune};"),
    ]
    if not anchors:
        patches.append(FIX_NO_ANCHORS)
    if fonta is not None:
        patches.append(("FONTA=45;", f"FONTA={fonta};"))
    if fontl is not None:
        patches.append(("FONTL=45;", f"FONTL={fontl};"))
    if flat is not None:
        if PAGE_CONSTS not in h:
            raise SystemExit(f"{out_path.name} no longer declares {PAGE_CONSTS}")
        patches += flat_patches(flat)
    for old, new in patches:
        if old not in h:
            raise SystemExit(f"patch target missing in {out_path.name}:\n  {old[:90]}")
        h = h.replace(old, new, 1)
    if joints == "slant":
        h = slant_joints(h, out_path.name, SLANT_PART if slant_part is None else slant_part)
    elif joints != "step":
        raise SystemExit(f"joints must be 'step' or 'slant', not {joints!r}")
    out_path.write_text(h, encoding="utf-8")
    return out_path


# --------------------------------------------------------------- capture
_FURNITURE = {}          # (win_w, win_h, dsf) -> (dw, dh), measured once per session


def run_chrome(page, win_w, win_h, dsf):
    if not CHROME.exists():
        raise SystemExit(f"Chrome not found at {CHROME}; open {page} in a browser, "
                         "wait for the title READY and save window.__PNG__ by hand")
    with tempfile.TemporaryDirectory() as prof:
        cmd = [str(CHROME), "--headless=new", "--disable-gpu", "--hide-scrollbars",
               f"--user-data-dir={prof}", f"--window-size={win_w},{win_h}",
               f"--force-device-scale-factor={dsf}",
               "--virtual-time-budget=120000", "--dump-dom", page.as_uri()]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if "<title>READY</title>" not in r.stdout:
        raise SystemExit(f"{page.name} never reached READY; chrome stderr:\n"
                         + r.stderr[-3000:])
    return r.stdout


def crop_png(png, crop=CROP):
    """Cut the white margins off a captured PNG in place; return (full, cropped) sizes."""
    from PIL import Image
    im = Image.open(png)
    W, H = im.size
    box = (round(W * crop["l"]), round(H * crop["t"]),
           round(W * (1 - crop["r"])), round(H * (1 - crop["b"])))
    im.crop(box).save(png)
    return (W, H), (box[2] - box[0], box[3] - box[1])


def capture(page, png, dsf=DSF, crop=CROP):
    """Render the page headless and write the PNG; return the meta dict.
    Headless subtracts window furniture from the CSS viewport and the amount
    differs by Chrome build, so it is measured once and reused.  The canvas
    is then cropped by `crop` (None keeps the full canvas)."""
    key = (WIN_W, WIN_H, dsf)
    win_w, win_h = WIN_W, WIN_H
    if key in _FURNITURE:
        dw, dh = _FURNITURE[key]
        win_w, win_h = WIN_W + dw, WIN_H + dh
    dom = None
    for _ in range(3):
        dom = run_chrome(page, win_w, win_h, dsf)
        meta = json.loads(re.search(r'<div id="metaout"[^>]*>(.*?)</div>',
                                    dom, re.S).group(1))
        dw, dh = WIN_W - meta["w"], WIN_H - meta["h"]
        if dw == 0 and dh == 0:
            _FURNITURE[key] = (win_w - WIN_W, win_h - WIN_H)
            break
        win_w, win_h = win_w + dw, win_h + dh
    m = re.search(r'<div id="pngout"[^>]*>data:image/png;base64,([A-Za-z0-9+/=]+)</div>',
                  dom, re.S)
    if not m:
        raise SystemExit(f"{page.name} READY but no PNG payload in the DOM")
    png.write_bytes(base64.b64decode(m.group(1)))
    meta["canvas_full"] = [meta["cw"], meta["ch"]]
    meta["crop"] = None
    if crop:
        full, cut = crop_png(png, crop)
        meta["crop"] = dict(crop)
        meta["png_size"] = list(cut)
    else:
        meta["png_size"] = [meta["cw"], meta["ch"]]
    return meta


# --------------------------------------------------------------- report
def report(meta):
    lanes = meta["lanes"]
    print(f"canvas {meta['cw']}x{meta['ch']} @dpr{meta['dpr']}  N={meta['n']} D={meta['d']}"
          + (f"  -> cropped to {meta['png_size'][0]}x{meta['png_size'][1]} "
             f"(l/t/r/b {meta['crop']['l']:.4f}/{meta['crop']['t']:.4f}/"
             f"{meta['crop']['r']:.4f}/{meta['crop']['b']:.4f})" if meta.get("crop") else ""))
    print(f"params {meta['params']}")
    if meta.get("flat_fit"):
        ff = meta["flat_fit"]
        print(f"levels and transfers from river_flat_fit.py: {ff['games_used']} games released from "
              f"{ff['start']:g} on, sigma {ff['sigma_years']:g} y; levels within "
              f"{100 * ff['fit']['largest']:.1f} points of the smoothed share "
              f"(rms {100 * ff['fit']['rms']:.1f}); the event-model parameters above are not used")
    if meta.get("shift_fit"):
        sf = meta["shift_fit"]
        print(f"levels and transfers from river_shift_fit.py: {sf['games_used']} games released from "
              f"{sf['start']:g} on, boundaries {sf['boundaries']}; levels within "
              f"{100 * sf['largest_gap_to_measured']:.1f} points of the measured shares; "
              "the event-model parameters above are not used")
    print(f"events {meta['n_events']}  flows in events {meta['n_flow_total']}  "
          f"ribbons drawn {len(meta['flows'])}")
    rows = meta.get("rows") or list(range(len(lanes)))
    print("lanes, top row first (lane index, row; row 0 = bottom of the figure):")
    for l in sorted(lanes, key=lambda l: -rows[l["lane"]]):
        print(f"  {l['lane']} row {rows[l['lane']]}  {l['label']:<44} n={l['n']:4d}  "
              f"share {100*l['share0']:5.1f}% -> {100*l['share']:5.1f}%")
    print("ribbons (source -> target, share moved, start -> arrival, calendar years):")
    for f in sorted(meta["flows"], key=lambda f: f["ts"]):
        print(f"  {f['i']} -> {f['j']}  {100*f['w']:4.1f}%  "
              f"{2017 + f['ts']:.2f} -> {2017 + f['te']:.2f}")
    if not meta["labels"]:
        print("anchor games: none drawn")
        return
    print(f"labels moved {meta['lbl_moved']}, hidden {meta['lbl_hidden']}, "
          f"clamped {meta['lbl_clamped']}  (solver cost {meta['lbl_cost']}, "
          f"{meta['lbl_nodes']} nodes)")
    for l in meta["labels"]:
        print(f"  {l['title']:<28} lane {l['lane']}  {2017 + l['tau']:.2f}  "
              f"{l['side']:<8} steps {l['steps']:2d}  nudge {l['nudge']:+.2f}"
              + ("  clamped" if l["clamped"] else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=str(DEFAULT_ASSETS))
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--page", default=None)
    ap.add_argument("--png", default=None)
    ap.add_argument("--meta", default=None,
                    help="JSON of the exported flows / lanes / labels "
                         "(default: <png>.meta.json next to the PNG)")
    ap.add_argument("--fit", default=None,
                    help="levels and transfers from river_shift_fit.py "
                         f"(default <assets>/{DEFAULT_FIT}; flat_fit_ice.json is the fit of "
                         "2026-10-13 to 2026-11-10)")
    ap.add_argument("--pressure", action="store_true",
                    help="draw the page's pressure model instead of the fit "
                         "(the figure as it was before 2026-10-13)")
    ap.add_argument("--joints", choices=("slant", "step"), default="slant",
                    help="how a ribbon meets a band: on slanted cuts, the ribbon running from "
                         "cut to cut (SLANT_JS, the paper's figure), or at the page's vertical steps")
    ap.add_argument("--slant-part", type=float, default=SLANT_PART,
                    help="share of a transfer's duration that each of its two slopes takes")
    ap.add_argument("--thr", type=float, default=ICE_THR)
    ap.add_argument("--prune", type=float, default=ICE_PRUNE)
    ap.add_argument("--dsf", type=int, default=DSF)
    ap.add_argument("--no-crop", action="store_true",
                    help="keep the full canvas instead of the author's white-margin crop")
    ap.add_argument("--fonta", type=float, default=ICE_FONTA,
                    help="anchor-title font size in CSS px (page default 45)")
    ap.add_argument("--fontl", type=float, default=ICE_FONTL,
                    help="lane-label / axis font size in CSS px (page default 45)")
    ap.add_argument("--anchors", action="store_true",
                    help="draw the anchor games, a dot and a title per lane and "
                         "three-year window (the paper figure has none)")
    a = ap.parse_args()
    assets = Path(a.assets).resolve()
    src = assets / "cluster_share_stack.html"
    page = Path(a.page).resolve() if a.page else assets / "paper_render_ice.html"
    png = Path(a.png).resolve() if a.png else assets / "fig6_content_river_ice.png"
    if not (assets / a.data).exists():
        raise SystemExit(f"data file missing: {assets / a.data} (run river_ice_prep.py)")
    # the page loads the data file relative to its own directory
    data_rel = Path(os.path.relpath(assets / a.data, page.parent)).as_posix()
    flat = None if a.pressure else flat_inputs(a.fit if a.fit else assets / DEFAULT_FIT)
    t0 = time.time()
    bake_variant(page, data_rel, a.thr, a.prune, src=src, fonta=a.fonta, fontl=a.fontl,
                 anchors=a.anchors, flat=flat, joints=a.joints, slant_part=a.slant_part)
    meta = note_fit(capture(page, png, a.dsf, crop=None if a.no_crop else CROP), flat,
                    a.joints, a.slant_part)
    print(f"wrote {png} ({png.stat().st_size/1e6:.2f} MB) in {time.time()-t0:.0f}s")
    report(meta)
    meta_path = Path(a.meta).resolve() if a.meta else png.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("meta ->", meta_path)


if __name__ == "__main__":
    main()
