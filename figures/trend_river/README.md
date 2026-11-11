# Trend river: the source of the paper's Figure 10

Figure 10 of the paper shows the release share of six semantic clusters for
games released from 2020 to 2024, with no genre labels used. This directory
holds everything it is made from: the interactive page that clusters the
game vectors and paints the canvas, the bundled dataset, the fit that gives
the bands their levels, and the scripts that render the figure headlessly.

The files keep the figure's earlier number in their names (`fig6_`).

## Dataset (`games_raw.js`, bundled)

Self-contained snapshot of 1,649 games, loaded as plain JS globals:

| global | shape / type | meaning |
|---|---|---|
| `RAW_N`, `RAW_D` | 1649, 128 | game count, vector width |
| `RAW_VEC` | Float32Array, N×D row-major | L2-normalised I-CE game vectors: the tower `wcle_i2ce_icetf`, five-fold fold 0, `--anchor-cap 2048`, epoch 1750 (the run `i2ce_icetf_fold0_g2048` of `contrast_experiment/w9/eval/rank_ckpt_plan.json`), gallery/anchor output `SPg` |
| `RAW_TAU` | Float[N] | release time, years since 2017.0 |
| `RAW_U` | Float[N] | 1-D UMAP coordinate of the same vectors (lane ordering only; `n_neighbors=15, min_dist=0.1, metric=cosine, random_state=42`) |
| `RAW_TAGS` | Int[N] | 23-bit mask; bit *i* = coarse tag *i* present |
| `TAG_NAMES` | String[23] | the coarse tag vocabulary (paper Section 5) |
| `RAW_TITLES` | String[N] | game titles |

The 1,649 games are the river titles that map to a gallery game of the
corpus (1,919 titles in the original crawl; the 270 that do not map are
dropped). `river_ice_prep.py` is the script that built the file from the
research caches (the tower's projection cache and the corpus `games.npz`;
neither is shipped); it is included for provenance and does not run here.

## What the figure draws

The six clusters are the page's own: seeded spherical k-means at K = 6 on
the 128-d vectors, run in the browser, deterministic. The tags play no part
in the clustering; each band's label names the two tags most over-represented
in its cluster, among the tags at least 12% of its games carry.

`river_flat_fit.py` takes the 1,504 games released from 2020.0 on (the three
years before hold 145 games, 30 of them in 2019) and smooths the release
share of each cluster over the release date (Gaussian kernel, standard
deviation 0.25 years, 24 steps a year, 2020.0 to 2024.75). It then describes
the six curves by levels joined by transfers: a cluster keeps its level
except while a transfer is under way, and a transfer moves a share from one
cluster to another, linearly between a start date and an end date. Transfers
are added one at a time by least squares, everything refitted after each
addition, until no cluster is further than 4 points from its smoothed share
at any date. That takes three transfers (rms distance 1.5 points, largest
3.6; two transfers leave 8.2):

| transfer | share (points) | start | end |
|---|---|---|---|
| Puzzle / Crime/Mystery → Simulation/Management / Open World/Survival | 9.1 | 2020.00 | 2024.75 |
| Puzzle / Crime/Mystery → Turn-Based / Card Game | 6.4 | 2020.04 | 2020.67 |
| Platformer / Shooter/FPS → Historical/War / Shooter/FPS | 3.9 | 2023.00 | 2023.58 |

`flat_fit_ice.json` holds the fit: the level of every band at both ends, the
transfers, the smoothed and the fitted shares on the grid, and the fit
quality for 0 to 10 transfers. The docstring of `river_flat_fit.py` says how
far the result can be read, and `--resample N` refits bootstrap resamples of
the games.

The render scripts give the page those levels and transfers. A band keeps
its thickness except where a ribbon leaves or joins it. There the joint is
cut on the slant: the source band narrows along a straight slope over the
first third of the transfer, the target band widens along a slope over the
last third, and the ribbon runs from the one slope to the other, a straight
band as thick as the share it carries. `SLANT_JS` in `river_ice_render.py`
is that drawing, with its geometry described beside it.

## Reproducing the paper's figure

```bash
python figures/trend_river/river_flat_fit.py     # the fit -> flat_fit_ice.json
python figures/trend_river/river_ice_column.py   # the figure at one column, as the paper prints it
python figures/trend_river/river_ice_render.py   # the same figure at full width
```

The fit needs numpy, the full-width render Pillow (for the crop), and all
three Google Chrome (path at the top of `river_ice_render.py`); each takes a
few seconds. The fit is shipped, so the two renders run without the first
command.

- `river_ice_column.py` bakes `paper_render_ice_col.html` from the
  interactive page, renders it in headless Chrome and writes
  `fig6_content_river_ice_col.png`: 1,098 × 724 CSS px at device scale
  factor 3, 3,294 × 2,172 px, laid out for a 3.34 in column with the lane
  names above their bands.
- `river_ice_render.py` does the same for `paper_render_ice.html` and
  `fig6_content_river_ice.png`: a 2,203 × 1,022 px viewport at device scale
  factor 4, the 8,812 × 4,088 canvas with its white margins cropped to
  8,401 × 3,483, the lane names and their end shares at the right.
- Each writes a `.meta.json` beside its PNG, listing the three ribbons the
  figure paints (source and target lane, share moved, start and arrival
  year), the six lanes with their start and end shares, and which numbers
  the page drew. The two meta files of the paper's figure are shipped.

Run on 2026-11-11 from this directory (Windows, Chrome), both scripts wrote
PNGs identical byte for byte to the two files the paper is typeset from, and
the fit came out as shipped. The lettering is set in the page's monospace
font stack, so on a system with other fonts the text may come out
differently.

## The interactive page

Open `cluster_share_stack.html` directly in a browser, with no server and no
build step. It loads the same dataset and runs the same clustering, with K
adjustable, and paints its own account of the shares, a **pressure-driven
event model**, from the first release years on. Share changes deposit pending
parcels per cluster, opposite parcels cancel locally, and un-cancelled mass
builds exponential pressure m·(e^{β·age}−1). A pair (i, j) transports when
source surplus pressure plus target deficit pressure beats Θ·(1+D²/ε), D the
inter-centroid semantic distance, so the stream width changes only at
discrete fork and merge events and semantically close lanes exchange first.

The paper's figure was drawn from that model until 2026-10-13. Since then it
draws the fitted levels and transfers above. `--pressure` on either render
script draws the earlier account on the current vectors.

Controls: clusters K, share-decay τ, pressure rate β, trigger threshold Θ,
cost sharpness ε, warm-up and tail trims, anchor and label typography, flow
pruning. **Paper mode** switches to a light palette; **Export PNG** saves
the canvas; **Edit labels** lets you drag anchor titles (offsets persist in
`localStorage` and survive export). The page marks anchor games, per lane
and time window the game nearest the lane's centroid; the paper's figure
carries none, and `river_ice_render.py --anchors` draws them.

## Files

| File | Role |
|---|---|
| `cluster_share_stack.html` | The interactive page, and the page both renders are baked from |
| `games_raw.js` | The dataset above |
| `river_flat_fit.py`, `flat_fit_ice.json` | The fit and what it writes |
| `river_ice_column.py`, `fig6_content_river_ice_col.meta.json` | The one-column figure and what the page painted for it |
| `river_ice_render.py`, `fig6_content_river_ice.meta.json` | The full-width figure and what the page painted for it |
| `build_river_render.py` | The baking step the render scripts import: hides the controls, white paper mode, the parameter set |
| `river_ice_prep.py` | How `games_raw.js` was built (provenance; needs the research caches) |
