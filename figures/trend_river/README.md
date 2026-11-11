# Trend river: the source of the paper's Figure 10

Figure 10 of the paper shows the release share of six semantic clusters for
games released from 2020 to 2024, with no genre labels used. This directory
holds everything it is made from: the interactive page that clusters the
game vectors and paints the canvas, the bundled dataset, the fit that gives
the bands their levels and the ribbons their shares, and the scripts that
render the figure headlessly.

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

`river_shift_fit.py` gives the bands their levels and the ribbons their
shares and dates. Release shares sum to one, so the data say how much every
cluster gained or lost between two stretches of time, not which cluster's
loss became which cluster's gain, and with 60 to 100 games a quarter they do
not say how long a movement took. The script therefore draws from stated
rules: a chance mechanism is tested where there is one, and what is a drawing
convention is named as one. It takes the 1,504 games released from 2020.0 on
(the three years before hold 145 games, 30 of them in 2019).

1. **Periods.** Between two boundaries the six release shares are taken as
   constant. Boundaries may fall on month starts, a period holds at least
   three months, and for a given number of boundaries the best placement
   maximises the multinomial likelihood. A boundary is added only while its
   gain in log-likelihood is larger than in 95% of 10,000 shuffles in which
   the release months are permuted among the games of each current period,
   the best placement being searched again in every shuffle. One boundary is
   added, 1 July 2020 (gain 10.88 against a threshold of 9.12, p = 0.012); the
   best second one, September 2023, is not (8.51 against 9.14, p = 0.084).
   The test says that the composition changed, not where: one boundary alone
   fits almost as well at September 2023 (gain 10.54) as at July 2020 (10.88).
2. **Levels.** A band's thickness in a period is the cluster's share of that
   period's releases: 157 games before the boundary, 1,347 after it.
3. **Ribbons.** The losses are carried to the gains by the cheapest transport,
   the cost of moving share between two clusters being 1 − cos between their
   centroids (`cost` in the fit holds it divided by its largest entry, as the
   page hands it over; the plan is the same). The pairing is the least movement in the embedding that accounts
   for the change; it is a convention, not an observation. A movement under
   one point is not drawn (here one, Racing/Driving → Simulation/Management,
   0.7 points), so the bands drawn after the boundary are within 0.7 points of
   the measured shares; the shares printed beside the bands of the full-width
   figure are the measured ones.
4. **Ribbon length.** A ribbon is as long as a run of releases has to be for a
   shift of its size to stand two standard errors clear between two such runs,
   n = 8 p (1 − p) / w² games with w the share moved and p = 1/6, turned into
   years at the mean release rate (310 a year), at least four months and at
   most the axis; it is centred on the boundary and pushed inside the axis
   where it would stick out. A large movement is drawn short, a small one
   long. The length is not a measured duration.
5. **Rows.** The page orders the clusters by their mean coordinate on the
   one-dimensional UMAP; the order carries no quantity. The simulation cluster
   is moved to the bottom row (`--bottom`), so that the three ribbons leaving
   the puzzle band do not share one edge of it.

| share, points | first half of 2020 | from July 2020 |
|---|---|---|
| Puzzle / Crime/Mystery | 31.8 | 17.9 |
| Simulation/Management / Open World/Survival | 9.6 | 16.8 |
| Racing/Driving / Sports/Rhythm | 11.5 | 10.8 |
| Turn-Based / Card Game | 12.1 | 17.5 |
| Platformer / Shooter/FPS | 15.3 | 19.4 |
| Historical/War / Shooter/FPS | 19.7 | 17.7 |

| ribbon | share (points) | drawn from | to | games in the resolving run |
|---|---|---|---|---|
| Puzzle / Crime/Mystery → Simulation/Management / Open World/Survival | 6.5 | 2020.08 | 2020.92 | 261 |
| Puzzle / Crime/Mystery → Platformer / Shooter/FPS | 4.1 | 2020.00 | 2022.14 | 664 |
| Puzzle / Crime/Mystery → Turn-Based / Card Game | 3.3 | 2020.00 | 2023.21 | 995 |
| Historical/War / Shooter/FPS → Turn-Based / Card Game | 2.1 | 2020.00 | 2024.75 | 2,577 (longer than the axis) |

`shift_fit_ice.json` holds the fit: the boundaries with their tests, where
one boundary alone fits best and next best, the shares of every period, the
transport cost between clusters, the ribbons with their dates, the movement
left out and the row of every band.

How far the picture can be read (`river_shift_fit.py --test 2000`, shipped as
`shift_fit_ice.test.json`):

- The boundary: when the games are resampled and one boundary placed again,
  it falls within three months of July 2020 in 39% of the resamples and
  within three months of September 2023 in 25% (by calendar year: 2020 40%,
  2021 12%, 2022 9%, 2023 38%, 2024 2%). The two dates are two candidates of
  nearly equal weight, and the 5% rule keeps one.
- The ribbons, against shuffles of the release months that are given their
  own best boundary and their own transport, so that the null ribbons are as
  large as the procedure can make them from noise: the same pair carries at
  least as much in 0.9% of the shuffles for Puzzle → Simulation, 3.1% for
  Puzzle → Turn-Based, 5.6% for Puzzle → Platformer and 10.7% for
  Historical/War → Turn-Based. Some ribbon, whichever pair, is as large as
  8.1 points in 5% of the shuffles, so no single ribbon stands on its own
  against "any ribbon this large" (15% for the largest). With the periods
  held, the four pairs carry a point or more in 94%, 70%, 84% and 59% of
  resamples of the games.
- So: that the composition changed is tested (p = 0.012); mid-2020 is the
  likeliest single date for it, with autumn 2023 close behind; that puzzle
  titles lost share and simulation titles gained is the robust part; the
  pairing of the smaller movements and every ribbon's length are conventions.

The render scripts give the page those levels and ribbons. A band keeps
its thickness except where a ribbon leaves or joins it. There the joint is
cut on the slant: the source band narrows along a straight slope over the
first third of the ribbon's span, at most four months, the target band widens
along such a slope at the end, and the ribbon runs from the one slope to the
other on a flat S-curve, a band as thick as the share it carries all along.
Ribbons that leave one band edge together each take a part of that slope, in
proportion to their shares, and leave side by side. `SLANT_JS` in
`river_ice_render.py` is that drawing, with its geometry described beside it.

From 2026-10-13 to 2026-11-10 the figure drew a least-squares fit instead
(`river_flat_fit.py`, `flat_fit_ice.json`: levels joined by transfers, fitted
to the kernel-smoothed shares until no band was more than 4 points off). It
picked one pairing and its dates among many that fit as well, and its three
transfers came back in 66%, 25% and 18% of resamples of the games, which is
why it was replaced. `--fit flat_fit_ice.json` on either render script draws
its transfers again, with the joints as they are drawn now.

## Reproducing the paper's figure

```bash
python figures/trend_river/river_shift_fit.py    # the fit -> shift_fit_ice.json
python figures/trend_river/river_ice_column.py   # the figure at one column, as the paper prints it
python figures/trend_river/river_ice_render.py   # the same figure at full width
```

The fit needs numpy, the full-width render Pillow (for the crop), and all
three Google Chrome (path at the top of `river_ice_render.py`). The fit takes
about 20 seconds (about a minute with `--test 2000`), each render a few seconds.
The fit is shipped, so the two renders run without the first command.

- `river_ice_column.py` bakes `paper_render_ice_col.html` from the
  interactive page, renders it in headless Chrome and writes
  `fig6_content_river_ice_col.png`: 1,098 × 724 CSS px at device scale
  factor 3, 3,294 × 2,172 px, laid out for a 3.34 in column with the lane
  names above their bands.
- `river_ice_render.py` does the same for `paper_render_ice.html` and
  `fig6_content_river_ice.png`: a 2,203 × 1,022 px viewport at device scale
  factor 4, the 8,812 × 4,088 canvas with its white margins cropped to
  8,401 × 3,483, the lane names and their end shares at the right.
- Each writes a `.meta.json` beside its PNG, listing the four ribbons the
  figure paints (source and target lane, share moved, start and arrival
  year), the six lanes with their start and end shares and their rows, and
  which numbers the page drew. The two meta files of the paper's figure are
  shipped.

Run on 2026-11-11 from this directory (Windows, Chrome), both scripts wrote
PNGs identical byte for byte to the two files the paper is typeset from, and
the fit and its evidence report came out as shipped (the shuffles are
seeded). The lettering is set in the page's monospace font stack, so on a
system with other fonts the text may come out differently.

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
draws levels and ribbons handed to the page, the periods and ribbons above
since 2026-11-11. `--pressure` on either render script draws the earlier
account on the current vectors.

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
| `river_shift_fit.py`, `shift_fit_ice.json`, `shift_fit_ice.test.json` | The fit, what it writes, and the evidence report of `--test 2000` |
| `river_flat_fit.py`, `flat_fit_ice.json` | The least-squares fit the figure drew until 2026-11-10; `river_shift_fit.py` reads the page's clusters through it |
| `river_ice_column.py`, `fig6_content_river_ice_col.meta.json` | The one-column figure and what the page painted for it |
| `river_ice_render.py`, `fig6_content_river_ice.meta.json` | The full-width figure and what the page painted for it |
| `build_river_render.py` | The baking step the render scripts import: hides the controls, white paper mode, the parameter set |
| `river_ice_prep.py` | How `games_raw.js` was built (provenance; needs the research caches) |
