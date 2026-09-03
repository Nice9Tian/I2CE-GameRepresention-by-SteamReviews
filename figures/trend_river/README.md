# Trend river — source of the paper's Figure 6

`cluster_share_stack.html` is the model that produced the paper's Figure 6
(release share by semantic cluster, 2019–2024, no genre labels): the figure
is this tool's canvas. Open it directly in a browser — no server, no build
step — or reproduce the paper's PNG headlessly with the script below.

## Dataset (`games_raw.js`, bundled)

Self-contained snapshot of 1,649 games, loaded as plain JS globals:

| global | shape / type | meaning |
|---|---|---|
| `RAW_N`, `RAW_D` | 1649, 128 | game count, vector width |
| `RAW_VEC` | Float32Array, N×D row-major | L2-normalised I-CE game vectors: the paper's tower (`wcle_i2ce_icetf`, five-fold fold 0, 4,096-sentence anchor budget, epoch 2000), gallery/anchor output `SPg` |
| `RAW_TAU` | Float[N] | release time, years since 2017.0 |
| `RAW_U` | Float[N] | 1-D UMAP coordinate of the same vectors (lane ordering only; `n_neighbors=15, min_dist=0.1, metric=cosine, random_state=42`) |
| `RAW_TAGS` | Int[N] | 23-bit mask; bit *i* = coarse tag *i* present |
| `TAG_NAMES` | String[23] | the coarse tag vocabulary (paper Section 5) |
| `RAW_TITLES` | String[N] | game titles (anchor labels) |

The 1,649 games are the river titles that map to a gallery game of the
corpus (1,919 titles in the original crawl; the 270 that do not map are
dropped). `river_ice_prep.py` is the script that built the file from the
research caches (the tower's projection cache and the corpus `games.npz`;
neither is shipped); it is included for provenance.

## What the model does

Spherical k-means (seeded, deterministic per K) clusters the game vectors in
the browser. Kernel-smoothed release shares feed a **pressure-driven event
model**: share changes deposit pending parcels per cluster, opposite parcels
cancel locally, and un-cancelled mass builds exponential pressure
m·(e^{β·age}−1). A pair (i, j) transports when source surplus pressure plus
target deficit pressure beats Θ·(1+D²/ε), D the inter-centroid semantic
distance — so the stream width changes only at discrete fork/merge events and
semantically close lanes exchange first.

## Controls

Clusters K, share-decay τ, pressure rate β, trigger threshold Θ, cost
sharpness ε, warm-up/tail trims, anchor and label typography, flow pruning.
**Paper mode** switches to a light palette; **Export PNG** saves the canvas;
**Edit labels** lets you drag anchor titles (offsets persist in
`localStorage` and survive export).

## Reproducing the paper's figure

```bash
python figures/trend_river/river_ice_render.py
```

needs only the standard library and Google Chrome (path at the top of the
script). It bakes `paper_render_ice.html` from the interactive page with the
paper's parameter set (K=6, τ=0.45, β=0.3, Θ=0.004, warm-up 2 / tail 0.5,
ε=0.05, pruning 0.023, anchor marks every 3 years, 32 px labels on a
2,203 × 1,022 px viewport with pads 165 / 700, long lane labels on two
lines), renders it in headless Chrome at device scale factor 4, and writes
`fig6_content_river_ice.png` (8,812 × 4,088) plus
`fig6_content_river_ice.meta.json`, which lists the six ribbons the figure
paints (source and target lane, share moved, start and arrival year), the six
lanes with their start / end shares, and where every anchor title landed.
`fig6_content_river.meta.json` is that file for the published figure.
Anchor titles are placed by a small branch-and-bound search (no overlaps, no
title over a dot, a leader line when a title has to sit away from its dot);
`build_river_render.py` is the baking step the render script imports.
