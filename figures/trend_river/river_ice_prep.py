# -*- coding: utf-8 -*-
"""Rebuild the Figure 6 content river on top of the paper's I-CE tower.

Same method as river_dual_prep.py (parse games_raw.js, map river titles to our
gallery through text_h5.h5, keep RAW_TAU / RAW_TAGS / RAW_TITLES verbatim,
re-render with the in-browser spherical k-means at K=6); the only change is where
the 128-d game vectors come from:

    arm      wcle_i2ce_icetf   (I-CE tower, five-fold fold 0, 4096-sentence anchors)
    ckpt     ep2000            (data/w9/rank_ckpt_plan.json -> i2ce_icetf_fold0_g4096)
    cache    C:\\runpod_data\\w9_towers\\tower_w9cv_wcle_i2ce_icetf_fold0_g4096_fp_ep2000.npz
    key      SPg   (2020, 128) gallery/anchor vectors, row i <-> games.npz names[i]

Unlike the dual prep (which used a 1-D PCA for the lane order) this script uses a
1-D UMAP, matching the ordering in the original games_raw.js.

Output (nothing else is touched; the dual chain keeps working):
    river_assets/games_raw_ice.js

The render page and the PNG are produced by river_ice_render.py (headless
Chrome, no CUDA), so this script is only needed when the vectors change.

Run with: C:\\Users\\admin\\anaconda3\\envs\\cuda_Vit\\python.exe figures/river_ice_prep.py
"""
import json
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
PAP = HERE / "river_assets"

STUDABLE = Path(r"C:\Users\admin\Documents\studable query latent")
TH5 = STUDABLE / "game_review_data" / "build_new_gamedata" / "text_h5.h5"
GAMES_NPZ = Path(r"C:\runpod_data\fusion_cache_w9\games.npz")
TOWER_NPZ = Path(r"C:\runpod_data\w9_towers"
                 r"\tower_w9cv_wcle_i2ce_icetf_fold0_g4096_fp_ep2000.npz")
TOWER_KEY = "SPg"

# 1-D UMAP for the lane ordering (RAW_U)
UMAP_KW = dict(n_components=1, n_neighbors=15, min_dist=0.1,
               metric="cosine", random_state=42, init="spectral")

# ---------------------------------------------------------------- 1. parse js
def parse_games_raw(path):
    js = path.read_text(encoding="utf-8")

    def arr(name):
        m = re.search(rf"const {name}\s*=\s*\[(.*?)\];", js, re.S)
        if m is None:
            raise KeyError(name)
        return json.loads("[" + m.group(1) + "]")

    return dict(
        n=int(re.search(r"RAW_N\s*=\s*(\d+)", js).group(1)),
        tau=np.array(arr("RAW_TAU"), dtype=np.float64),
        tags=np.array(arr("RAW_TAGS"), dtype=np.int64),
        titles=arr("RAW_TITLES"),
        tagnames=arr("TAG_NAMES"),
    )


def main():
    raw = parse_games_raw(PAP / "games_raw.js")
    RAW_N, titles_r = raw["n"], raw["titles"]
    print(f"river data: {RAW_N} games, {len(raw['tagnames'])} tags", flush=True)

    # ------------------------------------------------------- 2. title -> name
    names = [str(x) for x in np.load(GAMES_NPZ, allow_pickle=True)["names"]]
    n2i = {n: i for i, n in enumerate(names)}
    with h5py.File(TH5, "r") as h:                       # same mapping as river_dual_prep
        t2n = {}
        for g, t in zip(h["game_names"][:], h["game_titles"][:]):
            t2n.setdefault((t.decode() if isinstance(t, bytes) else str(t)).strip(),
                           g.decode() if isinstance(g, bytes) else str(g))
    match = [t2n.get(t.strip()) for t in titles_r]
    ok = [i for i, m in enumerate(match) if m is not None and m in n2i]
    print(f"matched {len(ok)}/{RAW_N} river games to our gallery "
          f"({RAW_N - len(ok)} dropped, same policy as river_dual_prep.py)", flush=True)

    # ------------------------------------------------------- 3. I-CE vectors
    Z = np.load(TOWER_NPZ)[TOWER_KEY].astype(np.float32)
    if Z.shape != (len(names), 128):
        raise SystemExit(f"unexpected tower shape {Z.shape}")
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    V = Z[[n2i[match[i]] for i in ok]]                   # (n, 128) L2-normalised

    # ------------------------------------------------------- 4. 1-D UMAP lane order
    import umap
    U = umap.UMAP(**UMAP_KW).fit_transform(V).ravel().astype(np.float64)
    print(f"UMAP-1D {UMAP_KW}  range [{U.min():.3f}, {U.max():.3f}]", flush=True)

    # ------------------------------------------------------- 5. games_raw_ice.js
    def fmt(a, nd=4):
        return "[" + ",".join(f"{x:.{nd}g}" for x in a) + "]"

    out = [
        "// RAW export for in-browser clustering (K adjustable in the HTML).",
        "// VEC: L2-normalised I-CE game vectors, row-major N x 128.",
        "//   tower  wcle_i2ce_icetf, five-fold fold 0, 4096-sentence anchor budget",
        f"//   ckpt   ep2000 ({TOWER_NPZ.name}, key {TOWER_KEY})",
        f"// U: UMAP-1D coordinate (lane ordering), {UMAP_KW}.",
        "// TAU / TAGS / TITLES are verbatim from games_raw.js.",
        f"const RAW_N={len(ok)}, RAW_D=128;",
        f"const RAW_TAU={fmt(raw['tau'][ok], 5)};",
        f"const RAW_U={fmt(U, 5)};",
        f"const RAW_TAGS={json.dumps([int(x) for x in raw['tags'][ok]])};",
        f"const TAG_NAMES={json.dumps(raw['tagnames'])};",
        f"const RAW_TITLES={json.dumps([titles_r[i] for i in ok], ensure_ascii=False)};",
        "const RAW_VEC=new Float32Array(" + json.dumps(
            [round(float(x), 4) for x in V.reshape(-1)]) + ");",
    ]
    (PAP / "games_raw_ice.js").write_text("\n".join(out), encoding="utf-8")
    print("wrote", PAP / "games_raw_ice.js", flush=True)

    print("next: py figures/river_ice_render.py  (page + PNG)", flush=True)

    # ------------------------------------------------------- report clusters
    report_clusters(V, U, raw, ok)


def report_clusters(V, U, raw, ok, K=6):
    """Reproduce the page's seeded spherical k-means (mulberry32 seed 42) in numpy
    so the printed cluster sizes/tags match the figure."""
    state = [42]

    def rnd():                                    # mulberry32, identical to the page
        state[0] = (state[0] + 0x6D2B79F5) & 0xFFFFFFFF
        s = state[0]
        t = (s ^ (s >> 15)) * (1 | s) & 0xFFFFFFFF
        t = (t + ((t ^ (t >> 7)) * (61 | t) & 0xFFFFFFFF)) & 0xFFFFFFFF ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    N, D = V.shape
    cent = np.zeros((K, D), np.float32)
    cent[0] = V[0]
    d2 = np.full(N, np.inf)
    for c in range(1, K):
        d2 = np.minimum(d2, np.maximum(2 - 2 * (V @ cent[c - 1]), 0))
        tgt = rnd() * d2.sum()
        acc = 0.0
        pick = N - 1
        for i in range(N):
            acc += d2[i]
            if acc >= tgt:
                pick = i
                break
        cent[c] = V[pick]
    asn = np.zeros(N, np.int64)
    for _ in range(40):
        asn = (V @ cent.T).argmax(1)          # page uses this last assignment
        for k in range(K):
            sel = V[asn == k]
            v = sel.sum(0) if len(sel) else np.zeros(D, np.float32)
            cent[k] = v / (np.linalg.norm(v) or 1.0)

    # lanes are the clusters sorted by mean UMAP coordinate (top lane = smallest)
    order = sorted(range(K), key=lambda c: U[asn == c].mean() if (asn == c).any() else 0)

    tags = raw["tags"][ok]
    names = raw["tagnames"]
    glob = np.array([[int(m) >> b & 1 for b in range(len(names))]
                     for m in tags], np.float64).mean(0)
    print("\nK=6 clusters, top to bottom in the figure "
          "(page's seeded spherical k-means, replayed in numpy):")
    for li, c in enumerate(order):
        idx = np.where(asn == c)[0]
        cnt = Counter()
        for m in tags[idx]:
            for b in range(len(names)):
                if int(m) >> b & 1:
                    cnt[names[b]] += 1
        top = ", ".join(f"{t} {v} ({v/len(idx):.0%})" for t, v in cnt.most_common(3))
        share = np.array([cnt[t] / len(idx) for t in names])
        lift = np.where(share >= 0.12, share / np.maximum(glob, 1e-9), 0)
        lab = np.argsort(-lift)[:2]
        print(f"  lane {li}: n={len(idx):4d}  top tags: {top}")
        print(f"           page label: {names[lab[0]]} / {names[lab[1]]}")


if __name__ == "__main__":
    main()
