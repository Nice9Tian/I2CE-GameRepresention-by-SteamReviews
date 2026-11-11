# -*- coding: utf-8 -*-
"""Figure 9: one neighborhood of eight games under six encoders.  The other scripts of this folder
import it for the neighborhoods.

Neighborhoods: every test game of the fixed split that has a wiki article is a seed; its neighborhood
is the seed plus its K - 1 nearest neighbors by CE anchor cosine among the held-out (validation and
test) wiki games, 201 neighborhoods in all.  Per neighborhood and encoder, accuracy is the share of
its 3K queries (the wiki article, its LLM rewrite with names, the rewrite without names) whose
nearest of the 2,020 anchors is their own game.

The vectors are those of prep.py: the fixed-split towers read against ten 4,096-sentence anchor
draws.  A query's score is its hit rate over the ten, neighbors go by CE's draw-averaged anchors,
and a panel draws and marks the first draw while its title gives the mean number of hits.

Which neighborhood is shown.  The genre is the one boxed in Figure 8: among the genres with
GENRE_MIN_N neighborhoods or more, the one whose six-encoder profile lies nearest the profile of
all the five-fold neighborhoods (cv_hoods.py, HOOD_BOX=typical).  Within that genre, the
neighborhood whose six-encoder profile over all its 3K queries lies nearest the genre's mean
profile (HOOD_PICK=typical, HOOD_PROFILE=all).  The rule was fixed before the ten draws were read.

How a panel is drawn (HOOD_LAYOUT=exact).  The K anchors are placed by a 2-D metric MDS (SMACOF) of
their true pairwise chord distances, oriented like their principal plane and rotated onto the I-CE
panel by orthogonal Procrustes.  Every query is drawn at its TRUE chord distance from its own
anchor, joined to it by a line, in the direction its displacement takes under the linear map that
sends the anchors to their layout (outward from the centroid if that image vanishes).  So line
lengths are exact and comparable within and across panels, anchor spacing is approximate (the facts
file records the layout's stress and spacing error), and a query's position relative to the OTHER
anchors carries no information.  Hit and miss marks always come from the true full-dimensional
retrieval among the 2,020 anchors.

    python fig_triplet.py fig9       -> fig9_neighborhood.png      4 + 2 panels at the full text width
    python fig_triplet.py fig9col    -> fig9_neighborhood_col.png  the printed figure, 2 x 3 at one column's width

Both write into <HOOD_WORK>/fig_facts_4096k10.json the numbers Appendix L quotes about the shown
neighborhood and about the anchors (layout errors, miss destinations, near ties, effective
dimensionality).  fig_genre_bars.py draws Figures 8 and A3 from the same neighborhoods and
fig_stats.py computes the statistics over all of them.

HOOD_MAP chooses how that linear map is computed (anchor_map below).  "stable", the default and the
paper's figure, leaves the round-off direction out of the pseudo-inverse and gives the same picture
on every machine.  "drawn" is the computation of the figure as first drawn, which kept that
direction, so that the line directions depend on the numpy build in their second digit.  Only the
directions of the lines differ between the two.

Paths come from hood_env.py (HOOD_WORK, HOOD_OUT, W9_DATA_DIR).  The other environment variables
(HOOD_RULE, HOOD_BOX, HOOD_PICK, HOOD_PROFILE, HOOD_GENRE, HOOD_LAYOUT, HOOD_ANCHORS, HOOD_LABELS,
HOOD_PANELS) and the other modes (points, territory, gaps, row, grid42a, grid42b) select earlier
variants of the rule and of the view; they are kept as they were and their defaults are the paper's.
"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hood_env   # noqa: E402
BUDGET = int(os.environ.get("HOOD_BUDGET", "4096"))                                  # sentences per anchor pack the towers are read against
DRAWS = int(os.environ.get("HOOD_DRAWS", "10" if BUDGET == 4096 else "1"))           # anchor packs a game is read against: the ten of prep.py
_CVSFX = "" if BUDGET == 512 else f"_{BUDGET}"                                       # the five-fold neighbourhood file of this budget
_SFX = _CVSFX + (f"k{DRAWS}" if DRAWS > 1 else ""); BIN = hood_env.BIN
DATA = hood_env.FACTS; CK = BIN / "ckpt"; OUT = hood_env.OUT
PAPER = hood_env.OUT; PAPER.mkdir(parents=True, exist_ok=True)
MAIN = os.environ.get("HOOD_PANELS", "Frozen,VICReg,SimCLR-style,CE").split(",")     # the working view's default panels
ROW = ["SimCLR-style", "I + SimCLR", "CE", "I-CE"]                                  # an earlier one-row figure (row_figure); the paper's Figure 9 is a grid
SCALE_TOWERS = ["Frozen", "VICReg", "SimCLR-style", "I + SimCLR", "CE", "I-CE"]      # one scale across both figures
_argv = sys.argv if __name__ == "__main__" else [""]
MODE = _argv[1] if len(_argv) > 1 else "points"
RANK = int(_argv[2]) if len(_argv) > 2 else 0
K = int(_argv[3]) if len(_argv) > 3 else 8
PANELS = _argv[4].split(",") if len(_argv) > 4 else MAIN

z = np.load(BIN / "allgames.npz", allow_pickle=True)
models = [str(m) for m in z["models"]]; keys = [str(k) for k in z["keys"]]; VARS = [str(v) for v in z["variants"]]
titles = [str(t) for t in z["titles"]]; status = np.array([str(s) for s in z["status"]]); appids = [str(a) for a in z["appids"]]
RS, QS = z["resample"], z["queries"]; HR, HQ = z["hit_resample"], z["hit_queries"]; gidx = z["game_index"]
nrm = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
gal = {m: nrm(np.load(CK / f"ref_anchors_{k}.npy")) for m, k in zip(models, keys)}
# per-model dicts, because the towers are 128-d and the frozen embedder is 1024-d
QSm = {m: QS[i] for i, m in enumerate(models)}; HQm = {m: HQ[i] for i, m in enumerate(models)}
if (BIN / "frozen.npz").exists():                           # the no-tower baseline, written by prep.py
    fz = np.load(BIN / "frozen.npz", allow_pickle=True); assert (fz["game_index"] == gidx).all()
    QSm["Frozen"] = fz["queries"]; HQm["Frozen"] = fz["hit_queries"]; gal["Frozen"] = nrm(np.load(CK / "ref_anchors_frozen.npy"))
    models = ["Frozen"] + models
# With several anchor draws (DRAWS > 1) the files hold the FIRST draw where they held the one pack: gal and HQm are what a
# panel draws and marks.  A query's score is then its hit rate over the draws, and neighbours go by the mean anchor.
SCm = {m: HQm[m].astype(float) for m in models}          # per model [NQ, versions]: the score of every query
galN = gal                                               # the anchors neighbours are taken by
if DRAWS > 1:
    SCm = {m: z["hit_rate"][i].astype(float) for i, m in enumerate(models[1:] if "Frozen" in models else models)}
    galN = {m: nrm(np.load(CK / f"ref_anchors_{k}_mean.npy")) for m, k in zip(models[1:] if "Frozen" in models else models, keys)}
    if "Frozen" in models:
        SCm["Frozen"] = fz["hit_rate"].astype(float); galN["Frozen"] = nrm(np.load(CK / "ref_anchors_frozen_mean.npy"))


def gal_draws(m):
    """every anchor draw of a tower, [2020, DRAWS, d] unit vectors (the one pack as a single draw)"""
    if DRAWS == 1:
        return gal[m][:, None, :]
    return np.load(CK / ("ref_anchors_frozen_draws.npy" if m == "Frozen" else f"ref_anchors_{keys[[str(x) for x in z['models']].index(m)]}_draws.npy"))


all_titles = json.load(open(hood_env.TITLES, encoding="utf-8"))

# ---------------- neighbourhoods, gaps, selection ----------------
A_ce = galN["CE"][gidx]
held = np.where(status != "train")[0]; seeds = np.where(status == "test")[0]
C = A_ce @ A_ce.T
acc = np.stack([SCm[m].mean(1) for m in models])         # [models, NQ] per-game accuracy over its 3 query points
hoods = {}
for s in seeds:
    cand = held[held != s]; nn = cand[np.argsort(-C[s, cand])[:K - 1]]
    t = tuple(sorted([int(s)] + [int(x) for x in nn]))
    if t in hoods:
        continue
    sub = C[np.ix_(t, t)][np.triu_indices(K, 1)]
    hoods[t] = dict(seed=int(s), min_cos=float(sub.min()), mean_cos=float(sub.mean()),
                    acc={m: float(acc[i, list(t)].mean()) for i, m in enumerate(models)})
for t, d in hoods.items():
    d["gap"] = d["acc"]["I-CE"] - d["acc"]["CE"]
NONAME_J = VARS.index("noname")
for t, d in hoods.items():                                # the paper's metric: name-stripped hit@1 over the K seeds' queries
    d["acc_s"] = {m: float(SCm[m][list(t), NONAME_J].mean()) for m in models}
    d["gap_s"] = d["acc_s"]["I-CE"] - d["acc_s"]["CE"]
_tl = np.load(hood_env.need_data_dir() / "tag_labels.npz", allow_pickle=True)
TAG_NAMES = [str(x) for x in _tl["tag_names"]]; _Y = _tl["y"]; _fr = _Y.sum(0)


def genre23(g):
    """the seed's genre: the rarest of the 23 reduced Steam tags it carries (Appendix G), or "no tag"."""
    c = np.where(_Y[g] > 0)[0]
    return TAG_NAMES[min(c, key=lambda j: _fr[j])] if len(c) else "no tag"


GENRE_MIN_N = 5                                           # a genre must hold at least this many neighbourhoods to be opened up
BY_GENRE = {}                                             # seed genre -> its neighbourhoods (every neighbourhood in exactly one genre)
for _t, _d in hoods.items():
    BY_GENRE.setdefault(genre23(int(gidx[_d["seed"]])), []).append(_t)


def profile(t):
    """name-stripped hit@1 (%) of the six drawn towers for one neighbourhood, in SCALE_TOWERS order."""
    return np.array([100 * hoods[t]["acc_s"][m] for m in SCALE_TOWERS if m in hoods[t]["acc_s"]])


def profile_all(t):
    """hit@1 (%) of the six drawn towers over all the queries of one neighbourhood, the 3K points a panel draws."""
    return np.array([100 * hoods[t]["acc"][m] for m in SCALE_TOWERS if m in hoods[t]["acc"]])


PROFILE = os.environ.get("HOOD_PROFILE", "all")          # what "most like its genre" compares: "all" the queries a panel draws, or the name-"stripped" ones
pick_profile = profile_all if PROFILE == "all" else profile


def tower_spread(p):
    """how much the towers disagree: standard deviation of a profile, in points."""
    return float(np.std(p))


GENRE_MEAN = {g: np.mean([profile(t) for t in ts], axis=0) for g, ts in BY_GENRE.items()}
GENRE_SPREAD = {g: tower_spread(GENRE_MEAN[g]) for g in BY_GENRE}
BOXED = max((g for g in BY_GENRE if len(BY_GENRE[g]) >= GENRE_MIN_N), key=lambda g: GENRE_SPREAD[g])   # the genre the towers disagree on most
COS_Q = float(np.quantile([d["mean_cos"] for d in hoods.values()], 0.75))
conf = [t for t in hoods if hoods[t]["mean_cos"] >= COS_Q]
RULE = os.environ.get("HOOD_RULE", "genrespread")
BOX_RULE = os.environ.get("HOOD_BOX", "typical")         # how the five-fold figure picks its boxed genre: "typical" or "spread"
if RULE == "gapmax":          # largest I-CE minus CE gap in the confusable quarter, ties to the more confusable
    order = sorted(conf, key=lambda t: (-round(hoods[t]["gap"], 9), -hoods[t]["mean_cos"]))
elif RULE == "confusable":    # the most confusable neighbourhood of all, outcome-blind
    order = sorted(hoods, key=lambda t: -hoods[t]["mean_cos"])
elif RULE == "genrespread":   # the boxed genre's median-spread neighbourhood (rank 0 = the median; equal spreads from most to least confusable)
    _cvf = DATA / "cv" / f"cv_hoods{_CVSFX}.json"          # Figure 8 is the five-fold figure: its box picks the genre
    if _cvf.exists() and "HOOD_GENRE" not in os.environ:
        _cv = json.load(open(_cvf, encoding="utf-8")); _by = {}
        for _h in _cv["hoods"]:
            _by.setdefault(_h["genre"], []).append(_h)
        _prof = lambda hs: np.array([100 * np.mean([h["acc_s"][m] for h in hs]) for m in SCALE_TOWERS if m in hs[0]["acc_s"]])
        if BOX_RULE == "typical":                             # the paper's rule: the genre nearest the profile of all neighbourhoods
            _allp = _prof(_cv["hoods"])
            GENRE = min((g for g in _by if len(_by[g]) >= GENRE_MIN_N), key=lambda g: float(np.linalg.norm(_prof(_by[g]) - _allp)))
        else:                                                 # an earlier rule: the genre the six encoders disagree on most
            GENRE = max((g for g in _by if len(_by[g]) >= GENRE_MIN_N), key=lambda g: tower_spread(_prof(_by[g])))
    else:
        GENRE = os.environ.get("HOOD_GENRE", BOXED)   # HOOD_GENRE overrides; without the five-fold file the fixed-split box applies
    _pick = os.environ.get("HOOD_PICK", "typical")   # the paper's rule: the neighbourhood most like its genre
    if _pick == "steps":        # an earlier rule, the neighbourhood whose step gains match the genre's: order the six encoders
        # from weakest to strongest by their genre means over all 24 queries, take each encoder's gain over the next weaker one,
        # and pick the neighbourhood whose five step gains lie nearest the genre's mean step gains
        _gmean = {m: np.mean([hoods[t]["acc"][m] for t in BY_GENRE[GENRE]]) for m in SCALE_TOWERS}
        LADDER = sorted(SCALE_TOWERS, key=lambda m: _gmean[m]); STEPS = list(zip(LADDER[:-1], LADDER[1:]))
        _steps = lambda t: np.array([K * QS.shape[2] * (hoods[t]["acc"][b] - hoods[t]["acc"][a]) for a, b in STEPS])
        _sm = np.mean([_steps(t) for t in BY_GENRE[GENRE]], axis=0)
        order = sorted(BY_GENRE[GENRE], key=lambda t: (round(float(np.linalg.norm(_steps(t) - _sm)), 9), -hoods[t]["mean_cos"]))
    elif _pick == "typical":    # the neighbourhood most like its genre: profile nearest the genre's mean profile
        _gm = np.mean([pick_profile(t) for t in BY_GENRE[GENRE]], axis=0)
        PICK_DIST = {t: float(np.linalg.norm(pick_profile(t) - _gm)) for t in BY_GENRE[GENRE]}
        order = sorted(BY_GENRE[GENRE], key=lambda t: (round(PICK_DIST[t], 9), -hoods[t]["mean_cos"]))
    else:                                                      # HOOD_PICK=median: the earlier median-spread pick
        o = sorted(BY_GENRE[GENRE], key=lambda t: (round(tower_spread(profile(t)), 9), -hoods[t]["mean_cos"])); mid = len(o) // 2
        order = o[mid:] + o[:mid]
elif RULE == "genremedian":   # superseded: the genre with the largest mean name-stripped I-CE minus CE gain (>= GENRE_MIN_N neighbourhoods), then its median-gain neighbourhood
    GENRE = max((g for g in BY_GENRE if len(BY_GENRE[g]) >= GENRE_MIN_N), key=lambda g: np.mean([hoods[t]["gap_s"] for t in BY_GENRE[g]]))
    o = sorted(BY_GENRE[GENRE], key=lambda t: (round(hoods[t]["gap_s"], 9), -hoods[t]["mean_cos"])); mid = len(o) // 2
    order = o[mid:] + o[:mid]
elif RULE == "gapmedian":     # the median gap of the confusable quarter (rank 0 = the median)
    o = sorted(conf, key=lambda t: (round(hoods[t]["gap"], 9), -hoods[t]["mean_cos"])); mid = len(o) // 2
    order = o[mid:] + o[:mid]
else:
    raise SystemExit(f"unknown HOOD_RULE {RULE}")
N_ALL = len(hoods); N = len(order); T = order[RANK]; info = hoods[T]; NP = K * QS.shape[2]
GAMES = gidx[list(T)]                                     # gallery indices of the shown games


def describe():
    print(f"[select] K={K}: {N_ALL} neighbourhoods, {N} confusable (mean cos >= {COS_Q:.3f}); rank {RANK}: gap {info['gap']:+.3f}  acc " +
          " ".join(f"{m} {info['acc'][m]:.2f}" for m in models) + f"  cos min {info['min_cos']:.3f} mean {info['mean_cos']:.3f}")
    for g in T:
        print(f"         {titles[g]} ({status[g]}, {appids[g]})  per-game acc " + " ".join(f"{m} {acc[i, g]:.2f}" for i, m in enumerate(models)))


def gaps_figure():
    allh = sorted(hoods, key=lambda t: (-round(hoods[t]["gap"], 9), -hoods[t]["mean_cos"]))
    gaps = np.array([hoods[t]["gap"] for t in allh]); mc = np.array([hoods[t]["mean_cos"] for t in allh]); cg = np.array([hoods[t]["gap"] for t in order])
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    step = 1.0 / NP; bins = (np.arange(-9, 10) - 0.5) * step                       # bins centred on the multiples of 1/NP the gaps take
    ax[0].hist(gaps, bins=bins, color="0.6", edgecolor="w")
    ax[0].hist(cg, bins=bins, color="#0072B2", edgecolor="w", label=f"confusable quarter ({N})")
    ax[0].axvline(info["gap"], color="#D55E00", lw=1.5); ax[0].axvline(0, color="k", lw=0.6); ax[0].legend(fontsize=7, frameon=False)
    ax[0].set_xlabel(f"I-CE accuracy minus CE accuracy on the {NP} queries"); ax[0].set_ylabel("neighborhoods")
    ax[0].set_title(f"{N_ALL} held-out neighborhoods of {K} games: median {np.median(gaps):+.3f}, I-CE ahead in {100 * (gaps > 0).mean():.0f}%, tied in {100 * (gaps == 0).mean():.0f}%, behind in {100 * (gaps < 0).mean():.0f}%\n"
                    f"confusable quarter: median {np.median(cg):+.3f}, ahead in {100 * (cg > 0).mean():.0f}%, tied in {100 * (cg == 0).mean():.0f}%, behind in {100 * (cg < 0).mean():.0f}%", fontsize=7)
    ax[1].scatter(mc, gaps, s=10, c="0.5", lw=0); ax[1].axvline(COS_Q, color="#0072B2", lw=0.8, ls="--")
    ax[1].scatter([info["mean_cos"]], [info["gap"]], s=40, c="#D55E00", zorder=3)
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xlabel("mean pairwise anchor cosine of the neighborhood (CE)"); ax[1].set_ylabel("gap")
    ax[1].set_title("gap against confusability (dashed: confusable-quarter threshold; orange: the shown neighbourhood)", fontsize=7)
    fig.tight_layout(); out = PAPER / "gap_distribution.png"; fig.savefig(out, dpi=200); print("ok", out)
    json.dump(dict(cos_threshold=COS_Q, chosen=[titles[g] for g in T], confusable=[dict(games=[titles[g] for g in t], **hoods[t]) for t in order],
                   all=[dict(games=[titles[g] for g in t], **hoods[t]) for t in allh]), open(DATA / "neighbourhood_gaps.json", "w"), indent=1)


# ---------------- the neighbourhood panel ----------------
COL = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#8C564B", "#9467BD", "#7F7F7F"]
RED = "#E00000"


SHORT = {"THE KING OF FIGHTERS XV": "King of Fighters XV", "Nickelodeon All-Star Brawl": "Nickelodeon Brawl", "Persona 4 Arena Ultimax": "Persona 4 Arena",
         "Granblue Fantasy: Versus": "Granblue Versus", "Them's Fightin' Herds": "Fightin' Herds", "Need for Speed™ Rivals": "NFS Rivals", "Need for Speed™ Unbound": "NFS Unbound"}


LABELS = os.environ.get("HOOD_LABELS", "key")        # "key": nothing written in the panels, a colour key beneath them (numbers in the panels got in the way of the points); "index": a number 1..K at each anchor and in the key; "names": the names in the panels


def short_title(t):
    """label form of a game title: the SHORT table, else the title without trademark signs, cut at 22 characters (the caption says names are shortened)."""
    return SHORT.get(t, t.replace("™", "").replace("  ", " ").strip()[:22])
MK = {"anchor": ("*", 260), "wiki": ("D", 70), "neutral": ("s", 65), "noname": ("o", 70)}
LABEL = {"anchor": f"anchor ({BUDGET:,} sentences)",
         "wiki": "query: wiki article", "neutral": "query: LLM rewrite, names kept", "noname": "query: LLM rewrite, names stripped"}


def frame(m):
    """orthonormal basis of the K anchors' span, the anchors' coordinates in it, and the
    top-2 principal directions of those coordinates (centred), oriented so that the
    first anchor lies to the right of the centroid and the second above."""
    G = gal[m]; aK = G[GAMES]
    Qb, _ = np.linalg.qr(aK.T); cK = aK @ Qb
    X = cK - cK.mean(0); _, _, Vt = np.linalg.svd(X, full_matrices=False); P = Vt[:2].T
    if (X[0] @ P[:, 0]) < 0:
        P[:, 0] *= -1
    if (X[1] @ P[:, 1]) < 0:
        P[:, 1] *= -1
    return G, Qb, cK, P, cK.mean(0)


def place(cc, P, centre):
    return (cc - centre) @ P


LAYOUT = os.environ.get("HOOD_LAYOUT", "exact")     # "exact": queries at their true distance from their anchor; "project": the earlier projection
ANCHORS = os.environ.get("HOOD_ANCHORS", "smacof")   # exact mode: "smacof" (2-D metric MDS of the K anchors' true chord distances) or "pca" (their principal plane)
REF_TOWER = "I-CE" if "I-CE" in models else models[-1]     # every panel's anchors are rotated onto this tower's
_ANCHOR_CACHE = {}


def smacof2(D, dim=2, it=400, seeds=5):
    """metric MDS by SMACOF (Guttman transform) from several random starts; the layout with the lowest stress-1."""
    n = len(D); best = None
    for seed in range(seeds):
        X = np.random.default_rng(seed).normal(size=(n, dim))
        for _ in range(it):
            E = np.linalg.norm(X[:, None] - X[None], axis=2); B = np.where(E > 1e-12, -D / np.maximum(E, 1e-12), 0.0)
            np.fill_diagonal(B, 0); np.fill_diagonal(B, -B.sum(1)); X = B @ X / n
        E = np.linalg.norm(X[:, None] - X[None], axis=2); iu = np.triu_indices(n, 1)
        stress = float(np.sqrt(((E[iu] - D[iu]) ** 2).sum() / (D[iu] ** 2).sum()))
        if best is None or stress < best[0]:
            best = (stress, X - X.mean(0))
    return best[1], best[0]


def anchor_layout(m):
    """centred 2-D coordinates of the K anchors (absolute chord scale) and the layout's stress-1; cached per tower."""
    if m not in _ANCHOR_CACHE:
        G, Qb, cK, P, ctr = frame(m); pca = place(cK, P, ctr); pca = pca - pca.mean(0)
        if LAYOUT == "exact" and ANCHORS == "smacof":
            Aq = gal[m][GAMES]; Dt = np.linalg.norm(Aq[:, None] - Aq[None], axis=2); X, stress = smacof2(Dt)
            X = X @ procrustes_rot(X, pca)                        # orient like the principal plane (first anchor right, second up)
        else:
            X = pca; iu = np.triu_indices(K, 1); Aq = gal[m][GAMES]; Dt = np.linalg.norm(Aq[:, None] - Aq[None], axis=2)
            E = np.linalg.norm(X[:, None] - X[None], axis=2); stress = float(np.sqrt(((E[iu] - Dt[iu]) ** 2).sum() / (Dt[iu] ** 2).sum()))
        _ANCHOR_CACHE[m] = (X, stress)
    return _ANCHOR_CACHE[m]


def procrustes_rot(Xa, Xb):
    """rotation / reflection R minimising |Xa R - Xb| for centred coordinates."""
    U, _, Vt = np.linalg.svd(Xa.T @ Xb); return U @ Vt


MAP = os.environ.get("HOOD_MAP", "stable")          # how the anchors-to-layout map is computed: "stable" (the paper's figure) or "drawn"


def anchor_map(Aq, A2):
    """The linear map that sends the K centred anchors to their layout, built from their pseudo-inverse.  K centred
    points have rank K - 1, so one direction must be left out of the pseudo-inverse.
    "stable" centres the anchors and takes the pseudo-inverse in double precision, where that direction's singular
    value is below 1e-15 of the largest for every tower and the cutoff rcond = 1e-6 drops it (the seventh is above
    0.15 of the largest); it gives the same picture on every machine.  Centred in float32 the direction's round-off
    is near 1e-7 and does not shrink with the anchors: 7e-7 of the largest for the frozen embedder, and 1e-4, above
    the cutoff, for the collapsed a+u @ gallery tower, which is not drawn.
    "drawn" is the computation behind the figure as first drawn: numpy's pinv on the float32 anchors at its default
    cutoff, which does not drop the direction (its singular value is round-off, near 1e-7).  The directions of the
    lines then depend on the numpy build in their second digit; numpy 2.4.3 reproduces that file.
    Line lengths, anchor positions and hit marks are the same in both; only the directions of the lines differ."""
    if MAP == "stable":
        X = Aq.astype(np.float64)
        return np.linalg.pinv(X - X.mean(0), rcond=1e-6) @ A2
    return np.linalg.pinv(Aq - Aq.mean(0)) @ A2


def positions(m, layout=None):
    """plane coordinates (anchors' centroid at the origin) of the K anchors and their 3K queries.
    exact: a query sits at its true chord distance from its own anchor, along its displacement's
    in-plane component; project: its projection onto the plane.  Also the true chord distance r,
    the in-subspace energy share e (project mode's marker size) and the hit flags."""
    layout = layout or LAYOUT
    G, Qb, cK, P, ctr = frame(m)
    Qn = nrm(QSm[m][list(T)]); Aq = G[GAMES]
    D = Qn - Aq[:, None, :]; r = np.linalg.norm(D, axis=2); e = ((Qn @ Qb) ** 2).sum(-1)
    if layout == "exact":
        A2, _ = anchor_layout(m)
        M = anchor_map(Aq, A2)                                     # the linear map that sends the (centred) anchors to their layout
        W = D @ M; n = np.linalg.norm(W, axis=2, keepdims=True)
        on = np.linalg.norm(A2, axis=1, keepdims=True)
        U = np.where(n > 1e-9, W / np.maximum(n, 1e-12), (A2 / np.maximum(on, 1e-12))[:, None, :])
        Q2 = A2[:, None, :] + r[..., None] * U
    else:
        A2 = place(cK, P, ctr); Q2 = place((Qn @ Qb).reshape(-1, Qb.shape[1]), P, ctr).reshape(K, len(VARS), 2)
        c = A2.mean(0); A2 = A2 - c; Q2 = Q2 - c
    if m != REF_TOWER:
        R0 = anchor_layout(REF_TOWER)[0] if layout == "exact" else (lambda f: (place(f[2], f[3], f[4]) - place(f[2], f[3], f[4]).mean(0)))(frame(REF_TOWER))
        R = procrustes_rot(A2, R0); A2 = A2 @ R; Q2 = Q2 @ R
    return dict(A=A2, Q=Q2, r=r, e=e, hit=HQm[m][list(T)].astype(bool))


def extent(m):
    """centre of the drawn points' bounding box on the plane, and the half-extent about it."""
    pos = positions(m); xy = np.concatenate([pos["A"], pos["Q"].reshape(-1, 2)])
    lo, hi = xy.min(0), xy.max(0); c = (lo + hi) / 2
    return c, float(np.abs(xy - c).max())


def layout_facts(m):
    """how faithful the panel is: variance share of the anchors' plane, the error of the 28 anchor spacings on it,
    and the mean true chord displacement and version scatter of the queries (exact regardless of the layout)."""
    pos = positions(m); A2 = pos["A"]; Aq = gal[m][GAMES]; iu = np.triu_indices(K, 1)
    Dt = np.linalg.norm(Aq[:, None] - Aq[None], axis=2)[iu]; D2 = np.linalg.norm(A2[:, None] - A2[None], axis=2)[iu]; rel = np.abs(D2 - Dt) / Dt
    ev = np.linalg.svd(Aq - Aq.mean(0), compute_uv=False) ** 2
    Qn = nrm(QSm[m][list(T)]); pv = [float(np.linalg.norm(Qn[i][:, None] - Qn[i][None], axis=2)[np.triu_indices(len(VARS), 1)].mean()) for i in range(K)]
    G, Qb, cK, P, ctr = frame(m); D = (Qn - Aq[:, None, :]); plane_share = float(np.mean(np.linalg.norm((D @ Qb) @ P, axis=2) ** 2 / np.maximum(np.linalg.norm(D, axis=2) ** 2, 1e-12)))
    Ud = D.reshape(-1, D.shape[-1]); Ud = Ud / np.linalg.norm(Ud, axis=1, keepdims=True)
    U2 = (pos["Q"] - pos["A"][:, None, :]).reshape(-1, 2); U2 = U2 / np.maximum(np.linalg.norm(U2, axis=1, keepdims=True), 1e-12)
    return dict(anchor_layout=ANCHORS if LAYOUT == "exact" else "pca", anchor_stress1=anchor_layout(m)[1], anchor_plane_variance_share=float(ev[:2].sum() / ev.sum()),
                anchor_spacing_rel_error_median=float(np.median(rel)), anchor_spacing_rel_error_max=float(rel.max()),
                query_displacement_mean=float(pos["r"].mean()), version_scatter_mean=float(np.mean(pv)),
                displacement_share_in_anchor_plane=plane_share, query_energy_in_anchor_span=float(np.mean(pos["e"])),
                direction_coherence_full=float(np.linalg.norm(Ud.mean(0))), direction_coherence_panel=float(np.linalg.norm(U2.mean(0))))


def miss_winners(m):
    """for every miss in the panel: (game, version, winning game, its split), the winner searched among all 2,020 anchors."""
    G = gal[m]; Qn = nrm(QSm[m][list(T)]); S = Qn @ G.T; out = []
    split = {int(gidx[row]): str(status[row]) for row in range(len(gidx))}
    for i, g in enumerate(T):
        for j, v in enumerate(VARS):
            if HQm[m][g, j]:
                continue
            sc = S[i, j].copy(); sc[GAMES[i]] = -9; w = int(sc.argmax())
            name = all_titles[str(w)] if isinstance(all_titles, dict) else all_titles[w]
            out.append((titles[g], v, str(name), split.get(w, "no wiki article")))
    return out


def spread(m):
    return extent(m)[1]


def frame_radius(panels=None):
    """one shared frame for every neighbourhood panel of Figure 9 (and the working views): the smallest square that
    holds each tower's points once each panel is centred (panels is ignored; kept for callers)."""
    return 1.10 * max(spread(m) for m in SCALE_TOWERS if m in models)


def write_facts(key, obj):
    """merge one entry into data/fig_distribution/fig_facts.json, the numbers the captions quote."""
    f = DATA / f"fig_facts{_SFX}.json"; d = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    d[key] = obj; f.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")


def draw_neighbourhood(ax, m, mode, RAD, title_fs=10, label_fs=7.5, ms=1.0, xlw=2.4, defer_labels=False):
    """ms scales every marker area, xlw is the line width of the miss cross (print sizes pass smaller values)."""
    """one panel; returns (hits, total) over the drawn points."""
    G, Qb, cK, P, ctr = frame(m); shift = extent(m)[0]                 # centre the drawn points in the frame
    if mode == "territory":
        n = 600; xs = np.linspace(-RAD, RAD, n); X, Y = np.meshgrid(xs, xs)
        XY = np.stack([X, Y], -1).reshape(-1, 2)
        full = ((XY + shift) @ P.T + ctr) @ Qb.T             # plane point lifted back to the full space
        lab = np.argmax(full @ G.T, 1).reshape(X.shape); labs = np.unique(lab)
        pastel = plt.get_cmap("Pastel2").colors + plt.get_cmap("Pastel1").colors
        cmap = {int(g): COL[k] for k, g in enumerate(GAMES)}
        for j, l in enumerate([l for l in labs if int(l) not in cmap]):
            cmap[int(l)] = pastel[j % len(pastel)]
        img = np.zeros(lab.shape + (4,))
        for l in labs:
            img[lab == l] = matplotlib.colors.to_rgba(cmap[int(l)])
        img[..., 3] = 0.30
        ax.imshow(img, extent=(-RAD, RAD, -RAD, RAD), origin="lower", interpolation="nearest", zorder=0)
        edge = np.zeros_like(lab, bool); edge[1:] |= lab[1:] != lab[:-1]; edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
        ax.contour(X, Y, edge.astype(float), levels=[0.5], colors="0.4", linewidths=0.4, zorder=1)
        for l in labs:
            if int(l) in set(int(g) for g in GAMES) or (lab == l).mean() < 0.01:
                continue
            yy, xx = np.where(lab == l)
            ax.text(xs[xx].mean(), xs[yy].mean(), all_titles[int(l)][:20], fontsize=5.5, ha="center", va="center", color="0.3", zorder=9)
    hit = tot = 0; labels = []; items = []                        # (x, y, area_pt2, marker, game, alpha, miss)
    layout = "project" if mode == "territory" else LAYOUT; pos = positions(m, layout)
    for k, g in enumerate(T):
        a_xy = pos["A"][k] - shift
        for j, name in enumerate(VARS):
            ok = bool(pos["hit"][k, j]); q_xy = pos["Q"][k, j] - shift; mk, s0 = MK[name]; e = float(pos["e"][k, j])
            size = s0 * ms * (e / 0.8 if layout == "project" else 1.0)
            alpha = float(np.clip(e / 0.8, 0.25, 1.0)) if mode == "territory" else 0.9
            hit += int(ok); tot += 1
            items.append((q_xy[0], q_xy[1], float(size), mk, k, alpha, not ok))
            if layout == "exact":                                 # the line whose length is the true chord distance
                ax.plot([a_xy[0], q_xy[0]], [a_xy[1], q_xy[1]], color=COL[k], lw=0.5 + 0.6 * ms, alpha=0.75, solid_capstyle="round", zorder=4)
        items.append((a_xy[0], a_xy[1], MK["anchor"][1] * ms, "*", k, 1.0, False))
        if LABELS != "key":
            labels.append((short_title(titles[g]) if LABELS == "names" else str(k + 1), a_xy, COL[k]))
    # touch test in points: two markers of the same game whose discs overlap get a white edge
    fig = ax.figure; fig.canvas.draw()
    pt_per_data = ax.get_window_extent().width / fig.dpi * 72 / (2 * RAD)
    rad = [0.5 * np.sqrt(a) for (_, _, a, *_) in items]                # marker radius in points, area = s
    edge = [False] * len(items)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][4] != items[j][4]:
                continue
            d = np.hypot(items[i][0] - items[j][0], items[i][1] - items[j][1]) * pt_per_data
            if d < 1.0 * (rad[i] + rad[j]):
                edge[i] = edge[j] = True
    for (x, y, a, mk, k, al, miss), ed in zip(items, edge):
        ax.scatter([x], [y], s=a, marker=mk, c=COL[k], alpha=al, edgecolors="w" if ed else "none", lw=0.7 if ed else 0,
                   zorder=6 if mk == "*" else 5)                 # layers: red cross (7) over anchor star (6) over queries (5)
        if miss:
            ax.scatter([x], [y], s=150 * ms, marker="x", c=RED, lw=xlw, zorder=7)
    obstacles = [(x, y, 0.5 * np.sqrt(max(a, 150 * ms) if miss else a)) for (x, y, a, mk, k, al, miss) in items]  # data xy, radius in points; a miss is as big as its cross
    ax.set_aspect("equal"); ax.set_xlim(-RAD, RAD); ax.set_ylim(-RAD, RAD); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("0.5")
    ax.set_title(f"{m}:  {hit} of {tot} queries retrieve their own game", fontsize=title_fs)
    if defer_labels:                                              # the caller places them after the final layout
        return hit, tot, (labels, obstacles)
    place_labels(ax, labels, RAD, label_fs, obstacles)
    return hit, tot


def place_labels(ax, labels, RAD, fs, obstacles=None):
    """game names near their anchors.  Each label tries 16 directions at three distances
    from its anchor and takes the first position that overlaps no marker, no label placed
    before it and no panel edge; failing that, the position with the least overlap.  The
    outward direction (away from the centroid of the anchors) is tried first, so a tight
    cluster gets a ring of names.  A thin leader line joins a label placed far away."""
    from matplotlib.transforms import Bbox
    if not labels:
        return
    fig = ax.figure; fig.canvas.draw(); r = fig.canvas.get_renderer(); px_per_pt = fig.dpi / 72
    ab = ax.get_window_extent(r); inner = Bbox([[ab.x0 + 3, ab.y0 + 3], [ab.x1 - 3, ab.y1 - 3]])
    obs = []
    for x, y, rp in (obstacles or []):
        cx, cy = ax.transData.transform((x, y)); rr = rp * px_per_pt + 1.5
        obs.append(Bbox([[cx - rr, cy - rr], [cx + rr, cy + rr]]))
    P = np.array([xy for _, xy, _ in labels]); c0 = P.mean(0)

    def overlap_area(bb, others):
        tot = 0.0
        for o in others:
            i = Bbox.intersection(bb, o)
            if i is not None:
                tot += i.width * i.height
        return tot

    placed = []; texts = []
    order = np.argsort(-np.linalg.norm(P - c0, axis=1))                 # outer anchors first: they have the most room
    for idx in order:
        t, xy, c = labels[idx]
        v = xy - c0; n = np.linalg.norm(v)
        base = np.arctan2(v[1], v[0]) if n > 1e-9 else 2 * np.pi * idx / len(labels)
        best = None
        for dist in (12.0, 22.0, 34.0, 48.0, 64.0, 82.0, 100.0):
            for k in range(24):
                ang = base + ((k + 1) // 2) * (np.pi / 12) * (1 if k % 2 == 1 else -1) if k else base   # base, then fanning out either side
                ux, uy = np.cos(ang), np.sin(ang)
                ha = "left" if ux > 0.35 else "right" if ux < -0.35 else "center"
                va = "bottom" if uy > 0.35 else "top" if uy < -0.35 else "center"
                tx = ax.annotate(t, xy, xytext=(dist * ux, dist * uy), textcoords="offset points", ha=ha, va=va,
                                 fontsize=fs, fontweight="bold", color=c, zorder=9)
                bb = tx.get_window_extent(r).expanded(1.08, 1.25)
                out = 0.0
                if not inner.contains(bb.x0, bb.y0) or not inner.contains(bb.x1, bb.y1):
                    out = max(0, inner.x0 - bb.x0) + max(0, bb.x1 - inner.x1) + max(0, inner.y0 - bb.y0) + max(0, bb.y1 - inner.y1)
                score = 3.0 * overlap_area(bb, obs) + 2.0 * overlap_area(bb, placed) + 400.0 * out + 0.5 * dist   # covering a marker is the worst outcome
                tx.remove()
                if best is None or score < best[0]:
                    best = (score, dist * ux, dist * uy, ha, va)
                if score <= 0.5 * dist + 1e-6:                          # collision-free: take it
                    break
            if best[0] <= 0.5 * dist + 1e-6:
                break
        _, ox, oy, ha, va = best
        if os.environ.get('LABEL_DEBUG'):
            print(f'[label] {t:26s} score {best[0]:7.1f} offset ({ox:+.0f},{oy:+.0f})pt  frame {ab.width:.0f}x{ab.height:.0f}px  obstacles {len(obs)}')
        tx = ax.annotate(t, xy, xytext=(ox, oy), textcoords="offset points", ha=ha, va=va,
                         fontsize=fs, fontweight="bold", color=c, zorder=9)
        placed.append(tx.get_window_extent(r).expanded(1.08, 1.25)); texts.append((tx, idx))
    fig.canvas.draw()
    for tx, idx in texts:                                                # leader lines for labels placed away from the anchor
        name, xy, c = labels[idx]; ox, oy = tx.xyann
        if np.hypot(ox, oy) > 16:
            bb = tx.get_window_extent(r); ax_px = ax.transData.transform(xy)
            cx = min(max(ax_px[0], bb.x0), bb.x1); cy = bb.y0 if bb.y0 > ax_px[1] else bb.y1 if bb.y1 < ax_px[1] else (bb.y0 + bb.y1) / 2
            if bb.y0 <= ax_px[1] <= bb.y1:
                cx = bb.x0 if ax_px[0] < bb.x0 else bb.x1
            px, py = ax.transData.inverted().transform((cx, cy))
            ax.plot([xy[0], px], [xy[1], py], lw=0.6, color=c, alpha=0.7, zorder=8.5)

def count(x):
    """a number of hits: whole for one anchor pack, a mean with two decimals over several draws"""
    return int(round(x)) if DRAWS == 1 else round(float(x), 2)


def hit_text(m, h, t):
    """what a panel title counts: the hits of the one pack drawn, or with several anchor draws their mean (the panel marks the first draw)"""
    return f"{h}/{t}" if DRAWS == 1 else f"mean {hoods[T]['acc'][m] * t:.1f}/{t}"


EDITION = (": Definitive Edition", " Definitive Edition")   # what a key drops from a title when the full titles do not fit


CUT = set()                                                 # games whose key entry stops at the colon of the title (key_variants)


def key_title(g, short=False):
    t = titles[g].replace("™", "")
    t = next((t[:-len(s)] for s in EDITION if short and t.endswith(s)), t)
    return t.split(":")[0] if g in CUT else t


def key_handles(short=False):
    """the game key: the anchor star in each game's colour (HOOD_LABELS=index: its number), full title (short: without an edition suffix)."""
    return [Line2D([], [], marker=f"${k + 1}$" if LABELS == "index" else MK["anchor"][0], ls="", color=COL[k], markersize=9, label=key_title(g, short)) for k, g in enumerate(T)]


def key_variants():
    """the game key from its full titles to shorter ones, for a layout to take the first that fits: the full titles; without
    edition suffixes; then the longest title cut at its colon, one title more at a time (the neighbourhood of
    "The Lord of the Rings: Return to Moria" and "Tainted Grail: The Fall of Avalon" fits the grid in no other way)."""
    CUT.clear()
    yield "", key_handles(False)
    yield ", edition suffixes dropped", key_handles(True)
    for g in sorted((g for g in T if ":" in key_title(g, True)), key=lambda g: -len(key_title(g, True))):
        CUT.add(g)
        yield ", titles cut at the colon: " + "; ".join(titles[x].replace("™", "") for x in T if x in CUT), key_handles(True)


def legend_handles():
    h = [Line2D([], [], marker=MK[k][0], ls="", color="0.3", markersize=9 if k == "anchor" else 6, label=LABEL[k]) for k in MK]
    if LAYOUT == "exact":
        h.append(Line2D([], [], ls="-", color="0.3", lw=1.3, label="line: true distance to own anchor"))
    h.append(Line2D([], [], marker="x", ls="", color=RED, markersize=10, mew=2.4, label="red x: nearest anchor is another game"))
    return h


def selection_text():
    d = hoods[T]; how = {"genrespread": f"the {GENRE} neighbourhood whose six-encoder profile lies nearest the genre mean, the genre the towers disagree on most",
                         "genremedian": f"at the median I-CE minus CE gain of the {GENRE} genre", "gapmedian": "at the median I-CE minus CE gap of the most confusable quarter",
                         "confusable": "the most confusable of all", "gapmax": "with the largest I-CE minus CE gap in the most confusable quarter"}[RULE]
    return (f"Neighbourhood of {K} games {how} of {N_ALL} held-out neighbourhoods (mean pairwise CE anchor cosine {d['mean_cos']:.2f}, "
            f"gap {d['gap'] * NP:+.0f} of {NP} queries)")

def view_text(mode):
    if LAYOUT == "exact" and mode != "territory":
        return (f"view: anchors by 2-D SMACOF of their true chord distances (absolute scale, oriented like their principal plane, rotated onto the {REF_TOWER} panel); "
                "each query at its TRUE chord distance from its own anchor along its displacement's in-plane component, joined by a line."
                "\nLine lengths are exact; anchor spacing is approximate; every hit / miss mark is the true full-dimensional retrieval outcome.")
    return (f"view: the plane of the two principal directions of the {K} anchors inside the {K}-D subspace they span; point position = its projection onto that plane, "
            f"marker area = share of its energy inside the {K}-D subspace" + ("; opacity = the same share" if mode == "territory" else "") +
            ".\nEvery hit / miss mark is the true full-dimensional retrieval outcome; only the layout is a projection." +
            ("\nBackground: exact nearest-anchor territories for points on this plane (all 2,020 anchors); the drawn points are not on the plane, so territory colour and true outcome can disagree." if mode == "territory" else ""))


def population_facts():
    """numbers Appendix L quotes beyond the shown neighbourhood: effective dimensionality of the 2,020 anchors
    (participation ratio of their covariance), where the misses of all held-out queries go (by split of the winning
    anchor), and how many of the shown neighbourhood's queries sit within 0.02 of a tie between own and best other anchor."""
    out = {}; split = {int(gidx[row]): str(status[row]) for row in range(len(gidx))}; ownh = gidx[held]
    for m in models:
        G = gal[m]; X = G - G.mean(0); ev = np.linalg.svd(X, compute_uv=False) ** 2
        Qh = nrm(QSm[m][held]); S = Qh @ G.T; n = len(held)
        co = np.take_along_axis(S, ownh[:, None, None].repeat(S.shape[1], 1), 2)[..., 0]
        S2 = S.copy(); S2[np.arange(n)[:, None], np.arange(S.shape[1])[None, :], ownh[:, None].repeat(S.shape[1], 1)] = -9
        bo = S2.max(2); win = S2.argmax(2); miss = co < bo
        kinds = np.array([[split.get(int(k), "no wiki article") for k in r] for r in win])
        dest = {k: float(((kinds == k) & miss).sum() / max(miss.sum(), 1)) for k in ("train", "val", "test", "no wiki article")}
        Qn = nrm(QSm[m][list(T)]); St = Qn @ G.T; cot = np.take_along_axis(St, GAMES[:, None, None].repeat(St.shape[1], 1), 2)[..., 0]
        St2 = St.copy(); St2[np.arange(K)[:, None], np.arange(St.shape[1])[None, :], GAMES[:, None].repeat(St.shape[1], 1)] = -9
        near = np.abs(cot - St2.max(2))
        out[m] = dict(participation_ratio=float(ev.sum() ** 2 / (ev ** 2).sum()), n_misses=int(miss.sum()), miss_destination=dest,
                      near_tie_within_0_02=int((near < 0.02).sum()), near_tie_within_0_01=int((near < 0.01).sum()))
    return out


def neighbourhood_facts():
    d = hoods[T]
    genre = dict(genre=GENRE, n_genre=len(BY_GENRE[GENRE]), genre_spread=GENRE_SPREAD[GENRE], spread=tower_spread(profile(T)),
                 spreads_in_genre=sorted(round(tower_spread(profile(t)), 2) for t in BY_GENRE[GENRE]),
                 genre_mean_profile={m: round(float(v), 2) for m, v in zip(SCALE_TOWERS, GENRE_MEAN[GENRE])},
                 genre_spreads={g: round(GENRE_SPREAD[g], 2) for g in sorted(GENRE_SPREAD, key=lambda g: -GENRE_SPREAD[g]) if len(BY_GENRE[g]) >= GENRE_MIN_N}) if RULE.startswith("genre") else {}
    if "PICK_DIST" in globals():                              # the typical pick: what it compared, and how far the first few lie from the genre's mean profile
        genre.update(pick_profile=PROFILE, pick_distances=[[titles[hoods[t]["seed"]], round(PICK_DIST[t], 2)] for t in order[:4]])
    return dict(rule=RULE, rank=RANK, K=K, n_all=N_ALL, n_confusable=N, cos_threshold=COS_Q, mean_cos=d["mean_cos"], min_cos=d["min_cos"], **genre,
                seed=titles[d["seed"]], games=[titles[g] for g in T], status=[str(status[g]) for g in T],
                hits={m: count(d["acc"][m] * NP) for m in models}, hits_stripped={m: count(d["acc_s"][m] * K) for m in models}, n_queries=NP, draws=DRAWS,
                hits_drawn={m: int(HQm[m][list(T)].sum()) for m in models}, hits_stripped_drawn={m: int(HQm[m][list(T), NONAME_J].sum()) for m in models},
                miss_winners={m: miss_winners(m) for m in models}, layout={m: layout_facts(m) for m in models})


def row_figure(panels, filename, key):
    """one print row of neighbourhood panels (an earlier layout; the paper's Figure 9 is grid_figure and column_figure)."""
    describe()
    RAD = frame_radius(); fs = dict(title=9.0, label=7.0, legend=7.5)
    fig, axes = plt.subplots(1, len(panels), figsize=(2.0 * len(panels), 3.0 if LABELS != "names" else 2.55))
    deferred = []; hits = {}
    for ax, m in zip(axes, panels):
        h, t, lab = draw_neighbourhood(ax, m, "points", RAD, title_fs=fs["title"], label_fs=fs["label"], ms=0.5, xlw=1.6, defer_labels=True)
        ax.set_title(f"{m}:  {hit_text(m, h, t)} hit", fontsize=fs["title"], pad=5); hits[m] = h
        for sp in ax.spines.values():
            sp.set_color("0.12"); sp.set_linewidth(0.8)
        deferred.append((ax, lab))
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, fontsize=fs["legend"], frameon=False, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.0, markerscale=0.8)
    if LABELS != "names":
        fig.legend(handles=key_handles(), loc="lower center", ncol=4, fontsize=fs["legend"], frameon=False, bbox_to_anchor=(0.5, 0.125), handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.25 if LABELS != "names" else 0.10, 1, 1.0), w_pad=0.6)
    for ax, (lab, obs) in deferred:
        place_labels(ax, lab, RAD, fs["label"], obs)
    out = PAPER / filename; fig.savefig(out, dpi=300); print("ok", out, hits)
    write_facts(key, dict(panels=panels, hits=hits, frame_radius=RAD, layout=LAYOUT)); write_facts("neighbourhood", neighbourhood_facts())
    write_facts("population", population_facts())


GRID = [["Frozen", "SimCLR-style", "I + SimCLR"], ["VICReg", "CE", "I-CE"]]   # an earlier 2 x 3 arrangement: baselines | contrast | contrast + I, views above, anchors below


FIG9_W = float(os.environ.get("FIG9_W", "6.0"))     # placement width in inches; the figure is drawn at this width so 7 pt text prints at 7 pt
FIG_PT = 7.0                                            # figure text is set 2 pt below the 9 pt body text
SAVE_DPI = 300                                          # the resolution grid_figure saves at, and fits the key at
GRID42A = [["Frozen", "VICReg", "SimCLR-style", "CE"], ["I + SimCLR", "I-CE", None, None]]        # 4 + 2, the + I panels left-aligned, legends at the right
GRID42B = [["Frozen", "VICReg", "SimCLR-style", "CE"], [None, None, "I + SimCLR", "I-CE"]]        # 4 + 2, the + I panels under their bases, legends at the left
GRID23 = [["Frozen", "SimCLR-style", "CE"], ["VICReg", "I + SimCLR", "I-CE"]]                     # one column of the two-column build: each + I panel under its base
COLUMN_W = 241.14749 / 72.27                            # \columnwidth of acmart sigconf, in inches


def grid_figure(grid, filename, key, fig_w=None, out_dir=None):
    """Figure 9 as a grid of neighbourhood panels drawn at its placement width with FIG_PT text.  A None cell is an
    empty slot; the key and the marker legend go into the empty slots when there are any, else beneath the grid."""
    describe()
    RAD = frame_radius(); fs = dict(title=FIG_PT, label=FIG_PT, legend=FIG_PT); fig_w = fig_w or FIG9_W
    nr, nc = len(grid), len(grid[0]); panel = fig_w / nc; ms = 0.5 * (panel / 2.0) ** 2; xlw = 1.6 * panel / 2.0
    two = DRAWS > 1                                          # "SimCLR-style:  mean 19.6/24 hit" is wider than a 1.5 in panel: with several draws a title takes two lines, as in column_figure
    row_h = panel + (0.33 if two else 0.22)
    empty = [(r, c) for r in range(nr) for c in range(nc) if grid[r][c] is None]
    below = (1.0 if LABELS != "names" else 0.55) if not empty else 0.0             # legend rows beneath the grid
    fig, axes = plt.subplots(nr, nc, figsize=(fig_w, nr * row_h + below)); axes = np.atleast_2d(axes)
    deferred = []; hits = {}
    for r, row in enumerate(grid):
        for c, m in enumerate(row):
            ax = axes[r][c]
            if m is None:
                ax.axis("off"); continue
            h, t, lab = draw_neighbourhood(ax, m, "points", RAD, title_fs=fs["title"], label_fs=fs["label"], ms=ms, xlw=xlw, defer_labels=True)
            ax.set_title(f"{m}\n{hit_text(m, h, t)} hit" if two else f"{m}:  {hit_text(m, h, t)} hit", fontsize=fs["title"], pad=3, linespacing=1.05); hits[m] = h
            for sp in ax.spines.values():
                sp.set_color("0.12"); sp.set_linewidth(0.8)
            deferred.append((ax, lab))
    fit = ""
    if empty:
        fig.tight_layout(rect=(0, 0, 1, 1), w_pad=0.5, h_pad=0.9)
        boxes = [axes[r][c].get_position() for r, c in empty]; x0 = min(b.x0 for b in boxes); x1 = max(b.x1 for b in boxes); y0 = min(b.y0 for b in boxes); y1 = max(b.y1 for b in boxes)
        # the key and the marker legend as one left-aligned block in the empty slots, no gap between them
        xl = x0 + 0.01; top = y1 + 0.02; hl = 2.0
        if LABELS != "names":
            # The game titles set the key's width, and a long one ran across the frame of the next panel ("Immortals Fenyx
            # Rising").  The preferred spacing is tried first; a key that would leave the empty slots is
            # moved flush with their left edge, then given shorter handles (a number needs less room than a legend line), then a
            # narrower gap between its two columns.  The text size never changes.  The marker legend below takes the same
            # position and handle length, so the two stay one left-aligned block.
            em = fs["legend"] / 72 / fig_w                          # one em of the legend font, as a share of the figure width
            flush = x0 - 0.9 * em                                    # a legend pads its content by 0.9 em: this puts the content at x0

            def titles_end(legend):
                """where the legend's text ends, as a share of the figure width, laid out at the resolution the file is saved
                at: at the figure's own 100 dpi the text is hinted wider, 0.06 in over this key, and the fit would be too tight."""
                dpi = fig.dpi; fig.set_dpi(SAVE_DPI); fig.canvas.draw()
                end = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).x1 - 0.4 * em   # the box pads by 0.4 em
                fig.set_dpi(dpi); return end

            # If no spacing holds the full titles (three "… Definitive Edition" games in one neighbourhood), the key
            # drops edition suffixes (key_title) and the spacings are tried again; one column of eight titles above the marker
            # legend would run 0.3 in below the slots.
            for short, handles in key_variants():
                for xl, hl, cs in [(x0 + 0.01, 2.0, 2.2), (flush, 2.0, 2.2), (flush, 1.5, 2.2), (flush, 1.0, 2.2)] + [(flush, 1.0, c / 10) for c in range(20, 9, -2)]:
                    kl = fig.legend(handles=handles, loc="upper left", ncol=2, fontsize=fs["legend"], frameon=False, bbox_to_anchor=(xl, top), handletextpad=0.7, handlelength=hl, columnspacing=cs, labelspacing=0.45, markerscale=0.85)
                    room = x1 - titles_end(kl)
                    if room >= 0:                                   # the titles end inside the empty slots
                        break
                    kl.remove()
                else:
                    continue
                break
            else:
                raise SystemExit("the game key does not fit the empty slots of the grid at any spacing; the titles are too long for this layout")
            fig.canvas.draw(); top = kl.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y0 - 0.005
            fit = "" if (hl, cs) == (2.0, 2.2) and xl > flush and not short else (f" | key fitted: handles {hl} em, column gap {cs} em, {room * fig_w:.2f} in to spare" + short)
        fig.legend(handles=legend_handles(), loc="upper left", ncol=1, fontsize=fs["legend"], frameon=False, bbox_to_anchor=(xl, top), handletextpad=0.5, handlelength=hl, markerscale=0.8, labelspacing=0.3)
    else:
        fig.legend(handles=legend_handles(), loc="lower center", ncol=2, fontsize=fs["legend"], frameon=False, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.2, markerscale=0.8)
        if LABELS != "names":
            fig.legend(handles=key_handles(), loc="lower center", ncol=4, fontsize=fs["legend"], frameon=False, bbox_to_anchor=(0.5, below * 0.55 / (nr * row_h + below)), handletextpad=0.2, columnspacing=1.0)
        fig.tight_layout(rect=(0, below / (nr * row_h + below), 1, 1.0), w_pad=0.5, h_pad=0.9)
    fig.subplots_adjust(hspace=0.30 if two else 0.22)         # row titles must clear the frames above them
    for ax, (lab, obs) in deferred:
        place_labels(ax, lab, RAD, fs["label"], obs)
    out = (out_dir or PAPER) / filename; fig.savefig(out, dpi=SAVE_DPI); print("ok", out, hits, f"| drawn {fig_w:.3f} x {fig.get_figheight():.3f} in, panel {panel:.2f} in, text {FIG_PT} pt{fit}")
    if key:
        write_facts(key, dict(panels=grid, hits=hits, frame_radius=RAD, layout=LAYOUT, anchor_map=MAP, width_in=fig_w, height_in=float(fig.get_figheight()), text_pt=FIG_PT))
        write_facts("neighbourhood", neighbourhood_facts()); write_facts("population", population_facts())


def column_figure(grid, filename):
    """Figure 9 as printed, for one column of the two-column paper: the six panels as a 2 x 3 grid at the
    column width, each title on two lines (a panel is 1.11 in wide), the game key and the marker legend beneath the grid, flush
    left, their lines set close so that the block leaves room for a table above it.  Same neighbourhood, scale, markers and
    text size as grid_figure.  Drawn twice: the first pass measures what the legends take, the second gives them exactly that."""
    describe()
    RAD = frame_radius(); nr, nc = len(grid), len(grid[0]); panel = COLUMN_W / nc; ms = 0.5 * (panel / 2.0) ** 2; xlw = 1.6 * panel / 2.0
    row_h = panel + 0.33                                    # a square panel under a two-line title, clear of the row above

    def draw(below):
        height = nr * row_h + below
        fig, axes = plt.subplots(nr, nc, figsize=(COLUMN_W, height)); axes = np.atleast_2d(axes)
        deferred = []; hits = {}
        for r, row in enumerate(grid):
            for c, m in enumerate(row):
                ax = axes[r][c]
                h, t, lab = draw_neighbourhood(ax, m, "points", RAD, title_fs=FIG_PT, label_fs=FIG_PT, ms=ms, xlw=xlw, defer_labels=True)
                ax.set_title(f"{m}\n{hit_text(m, h, t)} hit", fontsize=FIG_PT, pad=3, linespacing=1.05); hits[m] = h
                for sp in ax.spines.values():
                    sp.set_color("0.12"); sp.set_linewidth(0.8)
                deferred.append((ax, lab))
        fig.tight_layout(rect=(0, below / height, 1, 1), w_pad=0.4, h_pad=0.6)
        fig.subplots_adjust(hspace=0.30, top=1 - 0.27 / height)                    # the lower titles must clear the frames above them
        x0 = min(ax.get_position().x0 for ax in axes.ravel()); y0 = min(ax.get_position().y0 for ax in axes.ravel())
        kw = dict(fontsize=FIG_PT, frameon=False, borderpad=0, borderaxespad=0, handlelength=1.0)
        for short, handles in key_variants():                   # a key wider than the column takes shorter titles (key_variants), like grid_figure's
            kl = fig.legend(handles=handles, loc="upper left", ncol=2, bbox_to_anchor=(x0, y0 - 0.04 / height), handletextpad=0.7, columnspacing=1.6, labelspacing=0.25, markerscale=0.85, **kw)
            dpi = fig.dpi; fig.set_dpi(SAVE_DPI); fig.canvas.draw(); end = kl.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).x1; fig.set_dpi(dpi)
            if end <= 1.0:
                break
            kl.remove()
        else:
            raise SystemExit("the game key is wider than the column even without edition suffixes")
        fig.canvas.draw(); y = kl.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y0 - 0.012
        ml = fig.legend(handles=legend_handles(), loc="upper left", ncol=1, bbox_to_anchor=(x0, y), handletextpad=0.5, markerscale=0.8, labelspacing=0.18, **kw)
        fig.canvas.draw(); low = ml.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y0
        for ax, (lab, obs) in deferred:
            place_labels(ax, lab, RAD, FIG_PT, obs)
        return fig, height, low, hits

    fig, height, low, hits = draw(1.4)
    need = 1.4 - low * height + 0.02                           # what the legends actually take, and a hair
    plt.close(fig); fig, height, low, hits = draw(need)
    out = PAPER / filename; fig.savefig(out, dpi=SAVE_DPI); print("ok", out, hits, f"| drawn {COLUMN_W:.3f} x {height:.3f} in, panel {panel:.2f} in, legends {need:.2f} in, text {FIG_PT} pt")


def fig9():
    grid_figure(GRID42B, "fig9_neighborhood.png", "fig9")      # 4 + 2: the + I panels under their bases, legends in the empty slots


def fig9col():
    column_figure(GRID23, "fig9_neighborhood_col.png")          # the same figure for one column of the two-column paper


def row():
    row_figure(ROW, "neighborhood_row.png", "row")


def main():
    if MODE == "gaps":
        gaps_figure(); return
    if MODE == "fig9":
        fig9(); return
    if MODE == "fig9col":
        fig9col(); return
    if MODE == "row":
        row(); return
    if MODE in ("grid42a", "grid42b"):                        # drafts of the 4 + 2 layout at the full text width
        OUT.mkdir(exist_ok=True); grid_figure(GRID42A if MODE == "grid42a" else GRID42B, f"fig9_{MODE}.png", None, fig_w=FIG9_W, out_dir=OUT); return
    describe()
    RAD = frame_radius(PANELS)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5.5 * len(PANELS), 6.2))
    summary = {m: draw_neighbourhood(ax, m, MODE, RAD) for ax, m in zip(np.atleast_1d(axes), PANELS)}
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(selection_text() + "\n" + view_text(MODE), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    tag = "" if PANELS == MAIN else "_" + "-".join(p.split()[0].replace("+", "u").replace("/", "") for p in PANELS)
    OUT.mkdir(exist_ok=True); out = OUT / f"fig_triplet_{MODE}{'' if RANK == 0 else f'_rank{RANK}'}{tag}.png"
    fig.savefig(out, dpi=170); print("ok", out, summary)


if __name__ == "__main__":
    main()
