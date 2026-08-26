# Trend river — interactive source of the paper's Figure 4

`cluster_share_stack.html` is the model that produced the paper's Figure 4
(release share by semantic cluster, 2019–2024, no genre labels): the figure
is this tool's canvas, exported via **Paper mode** + **Export PNG**. Open it
directly in a browser — no server, no build step.

## Dataset (`games_raw.js`, bundled)

Self-contained snapshot of 1,919 games, loaded as plain JS globals:

| global | shape / type | meaning |
|---|---|---|
| `RAW_N`, `RAW_D` | 1919, 64 | game count, vector width |
| `RAW_VEC` | Float, N×D row-major | L2-normalised champion-tower game centroids |
| `RAW_TAU` | Float[N] | release time, years since 2017.0 |
| `RAW_U` | Float[N] | 1-D UMAP coordinate (lane ordering only) |
| `RAW_TAGS` | Int[N] | 23-bit mask; bit *i* = coarse tag *i* present |
| `TAG_NAMES` | String[23] | the coarse tag vocabulary (paper Section 5) |
| `RAW_TITLES` | String[N] | game titles (anchor labels) |

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
