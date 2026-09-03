# -*- coding: utf-8 -*-
r"""Render the paper's Figure 6 (content river) from the I-CE game vectors.

Position in the chain
---------------------
    river_ice_prep.py   -> river_assets/games_raw_ice.js      (data, once)
    river_ice_render.py -> river_assets/paper_render_ice.html (this script)
                           river_assets/fig6_content_river_ice.png
    sync_figures.py     -> assets/paper/fig6_content_river.png + the declared
                           figure height in paper.md

`games_raw_ice.js` holds the 1,649 river games that map to the gallery, their
128-d L2-normalised I-CE vectors (tower wcle_i2ce_icetf, five-fold fold 0,
4,096-sentence anchors, ep2000), a 1-D UMAP coordinate that only orders the
lanes, and the release times / tags / titles verbatim from games_raw.js.

What the render does
--------------------
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
       FONTA=40   the anchor-title font (lane labels stay 45): at 45 the twelve
                  titles cannot all sit beside their dots;
       anchor titles lose a trailing ellipsis ("If Found..." reads "If Found");
       a 200 px left pad and a 1,156 px right pad (the I-CE lane labels are
                  longer than the old ones) with the rotated y-axis title moved
                  in from the clipped x = 24 to x = 62;
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
   it.  The CSS viewport is 2,369 x 1,022 px (2,203 published + 96 px of
   extra right margin + 70 px of extra left pad); Chrome subtracts window
   furniture from the requested window size, so the first run measures the
   difference and the second run corrects for it.  At device scale factor 4
   the canvas is 9,476 x 4,088 px.

Run with:
    py figures/river_ice_render.py                 # THR=0.004, DSF 4
Options:
    --thr 0.004 --prune 0.023   event trigger / ribbon pruning
    --fonta 40                  anchor-title font size (CSS px)
    --dsf 4                     device scale factor of the capture
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

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

# published parameter set (what build_river_render.bake writes into the page)
PUB = dict(K=6, TAU=0.45, BETA=0.3, THR=0.0066, WARM_A=2, WARM_B=0.5,
           EPS=0.05, PRUNE=0.023)
# the I-CE figure's trigger and anchor-title font (the page bakes 45; 40 lets
# the twelve titles all sit beside their dots)
ICE_THR, ICE_PRUNE, ICE_FONTA = 0.004, 0.023, 40

# capture geometry: the published 2,203 x 1,022 CSS viewport, plus 96 px of
# extra right margin for the longer I-CE lane labels and the 70 px the left
# pad fix adds.  The plot area keeps the published width.
EXTRA_R = 96
PAD_L_OLD, PAD_L_NEW = 130, 200
WIN_W = 2203 + EXTRA_R + (PAD_L_NEW - PAD_L_OLD)
WIN_H = 1022
DSF = 4

# ------------------------------------------------------------------ patches
# Every patch is asserted against the baked page, so a change upstream in
# cluster_share_stack.html or build_river_render.py fails loudly.
FIX_PAD = (f"PAD={{l:{PAD_L_OLD},r:1060,t:160,b:110}}",
           f"PAD={{l:{PAD_L_NEW},r:{1060 + EXTRA_R},t:160,b:110}}")

# the rotated y-axis title was drawn at x = 24 and clipped at FONTL = 45
FIX_YTITLE = ("ctx.translate(24,(PAD.t+H-PAD.b)/2);",
              "ctx.translate(62,(PAD.t+H-PAD.b)/2);")

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
    window.__LANES__.push({lane:k, label:CLUSTER_LABELS[k].label,
      n:CLUSTER_LABELS[k].n, share:+(Wend.toFixed(5)),
      share0:+(aW(WARM_A)[k].toFixed(5))});
    ctx.fillStyle=ink;"""

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
    flows:(window.__FLOWS__||[]), n_events:(window.__NEVENTS__|0),
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
                 fonta=None):
    """bake() the render page from the interactive source, then apply the
    I-CE settings and the three rendering fixes above.  `src` overrides the
    interactive page (default: river_assets/cluster_share_stack.html next to
    build_river_render.py); `data_js` is the <script src> the page loads,
    relative to the page's own directory."""
    if src is None:
        bake(out_path, data_js, TAIL)
    else:
        bake(out_path, data_js, TAIL, src=src)
    h = out_path.read_text(encoding="utf-8")
    patches = [
        FIX_PAD, FIX_YTITLE, FIX_FLOWS,
        (FIX_LANES_OLD, FIX_LANES_NEW),
        (FIX_ANCHOR_OLD, FIX_ANCHOR_NEW),
        FIX_TITLE,
        FIX_LOOP,
        (FIX_LABEL_OLD, FIX_LABEL_NEW),
        (f"THR={PUB['THR']};", f"THR={thr};"),
        (f"PRUNE={PUB['PRUNE']};", f"PRUNE={prune};"),
    ]
    if fonta is not None:
        patches.append(("FONTA=45;", f"FONTA={fonta};"))
    for old, new in patches:
        if old not in h:
            raise SystemExit(f"patch target missing in {out_path.name}:\n  {old[:90]}")
        h = h.replace(old, new, 1)
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


def capture(page, png, dsf=DSF):
    """Render the page headless and write the PNG; return the meta dict.
    Headless subtracts window furniture from the CSS viewport and the amount
    differs by Chrome build, so it is measured once and reused."""
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
    return meta


# --------------------------------------------------------------- report
def report(meta):
    lanes = meta["lanes"]
    print(f"canvas {meta['cw']}x{meta['ch']} @dpr{meta['dpr']}  N={meta['n']} D={meta['d']}")
    print(f"params {meta['params']}")
    print(f"events {meta['n_events']}  flows in events {meta['n_flow_total']}  "
          f"ribbons drawn {len(meta['flows'])}")
    print("lanes (index 0 = bottom of the figure):")
    for l in lanes:
        print(f"  {l['lane']}  {l['label']:<44} n={l['n']:4d}  "
              f"share {100*l['share0']:5.1f}% -> {100*l['share']:5.1f}%")
    print("ribbons (source -> target, share moved, start -> arrival, calendar years):")
    for f in sorted(meta["flows"], key=lambda f: f["ts"]):
        print(f"  {f['i']} -> {f['j']}  {100*f['w']:4.1f}%  "
              f"{2017 + f['ts']:.2f} -> {2017 + f['te']:.2f}")
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
    ap.add_argument("--data", default="games_raw_ice.js")
    ap.add_argument("--page", default=None)
    ap.add_argument("--png", default=None)
    ap.add_argument("--meta", default=None,
                    help="JSON of the exported flows / lanes / labels "
                         "(default: <png>.meta.json next to the PNG)")
    ap.add_argument("--thr", type=float, default=ICE_THR)
    ap.add_argument("--prune", type=float, default=ICE_PRUNE)
    ap.add_argument("--dsf", type=int, default=DSF)
    ap.add_argument("--fonta", type=float, default=ICE_FONTA,
                    help="anchor-title font size in CSS px (page default 45)")
    a = ap.parse_args()
    assets = Path(a.assets).resolve()
    src = assets / "cluster_share_stack.html"
    page = Path(a.page).resolve() if a.page else assets / "paper_render_ice.html"
    png = Path(a.png).resolve() if a.png else assets / "fig6_content_river_ice.png"
    if not (assets / a.data).exists():
        raise SystemExit(f"data file missing: {assets / a.data} (run river_ice_prep.py)")
    # the page loads the data file relative to its own directory
    data_rel = Path(os.path.relpath(assets / a.data, page.parent)).as_posix()
    t0 = time.time()
    bake_variant(page, data_rel, a.thr, a.prune, src=src, fonta=a.fonta)
    meta = capture(page, png, a.dsf)
    print(f"wrote {png} ({png.stat().st_size/1e6:.2f} MB) in {time.time()-t0:.0f}s")
    report(meta)
    meta_path = Path(a.meta).resolve() if a.meta else png.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("meta ->", meta_path)


if __name__ == "__main__":
    main()
