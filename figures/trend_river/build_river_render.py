# -*- coding: utf-8 -*-
"""Create paper_render.html: bake the user's params, hide the Chinese UI, force
PAPER white mode + high DPR, and re-render on load so headless Chrome can grab a
high-resolution PNG.

`bake()` is importable so sibling scripts (river_ice_render.py, river_variants.py)
can produce a render page pointed at a different data file without duplicating the
parameters.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "river_assets" / "cluster_share_stack.html"
OUT = HERE / "river_assets" / "paper_render.html"  # same dir so games_raw.js loads


def bake(out=OUT, data_js="games_raw.js", extra_tail="", src=SRC):
    """Write the baked render page. `data_js` swaps the <script src=...> data file;
    `extra_tail` is raw HTML appended just before </body> (after the param override);
    `src` is the interactive page to bake from (default river_assets/cluster_share_stack.html)."""
    html = Path(src).read_text(encoding="utf-8")

    # raise the DPR cap for crisp export
    html = html.replace("DPR=Math.min(devicePixelRatio||1,2)",
                        "DPR=Math.min(devicePixelRatio||1,4)")
    # translate the on-canvas y-axis label to English (short enough to fit at 3x font)
    html = html.replace("语 义 聚 类 车 道 · 厚 度 = 发 布 占 比",
                        "release share by semantic cluster")
    # widen margins for the 3x-larger labels
    html = html.replace("PAD={l:86,r:262,t:96,b:64}", "PAD={l:130,r:1060,t:160,b:110}")

    if data_js != "games_raw.js":
        html = html.replace('src="games_raw.js"', f'src="{data_js}"')

    override = """
<style>#hud,#panel{display:none!important} html,body{background:#fff!important}</style>
<script>
window.addEventListener('load', function(){
  K=6; TAU=0.45; BETA=0.3; THR=0.0066; WARM_A=2; WARM_B=0.5; EPS=0.05;
  DOTR=9.0; MINW=8.0; PRUNE=0.023; FONTA=45; FONTL=45; MARK=3; PAPER=true;
  try{ recluster(); computeShares(); computeFlows(); computeAnchors(); }catch(e){console.log('compute',e);}
  try{ fit(); }catch(e){}
  draw();
  // expose the canvas data URL for headless extraction
  window.__PNG__ = document.getElementById('cv').toDataURL('image/png');
  document.title = 'READY';
});
</script>
"""
    html = html.replace("</body>", override + extra_tail + "</body>")
    out.write_text(html, encoding="utf-8")
    print("wrote", out, "chars:", len(html))
    return out


if __name__ == "__main__":
    bake()
