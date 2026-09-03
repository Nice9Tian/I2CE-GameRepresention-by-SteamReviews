"""Regenerate Appendix Table A9 (the window-size sweep) under ONE protocol.

WHY THIS SCRIPT EXISTS
----------------------
The shipped Table A9 mixed provenance across its four rows:

  * the three windowed arms (swin84 / swin168 / swin336) were trained for 2,000
    epochs and their checkpoints were picked by the fixed-split worker's
    review-based criterion  rvsel = v_q1 + v_q5 + 2*v_qtag  (w9_a100_worker.py, the
    `zs_from_arrays` block and the ZSBEST tail);
  * the fully coupled reference arm (wcle_i2ce_icetf) came from the older
    anchor-budget ladder run, which went to 4,000 epochs and picked its
    checkpoint with the OLDER query-based criterion `zvsel`;
  * the tag column of all four rows came from a separate re-scoring file
    (data/w9/a4/A4_test_tag.json) whose generating script was lost.

This script regenerates all four rows from one reproducible path:

  (1) candidate checkpoints = every 50 epochs from 50 to 2,000 INCLUSIVE for
      every arm (never beyond 2,000, so a fresh 2,000-epoch run of this
      repository reproduces the protocol);
  (2) per candidate, BOTH selection scores are computed from the same
      projection cache: the body's rank score (see SELECTION below) and the
      worker's rvsel, the latter by executing the worker's own
      `zs_from_arrays` to obtain v_q1 / v_q5 / v_qtag plus the four test-set
      retrieval columns;
  (3) the checkpoint maximising the ACTIVE criterion (W9_SELECT, default
      `rank`) is selected, ties broken toward the EARLIER epoch;
  (4) at the selected checkpoint the test-set retrieval columns are
      Name hit@1 / hit@5 (the `neutral`, name-intact wiki rewrites) and
      Stripped hit@1 / hit@5 (the `noname` rewrites) of the 204 test games,
      ranked against the full 2,020-game anchor gallery, single anchor draw
      (the fixed wscan_gal_rev_g4096 pack that produced the tower caches);
  (5) the tag column is the SECTION 5 test-set probe, not the worker's
      in-training readout: a ridge is fit on the 1,613 training-gallery anchor
      vectors against the 23 coarse tag classes, alpha and threshold are chosen
      on the 203 validation games' anchor vectors, and the probe is scored on
      the 204 test games' name-intact wiki-rewrite query vectors (micro-F1).
      This is the `ctag` construction of data/w9/register_tax_recompute.py
      (lines 136-144), adapted from its five folds to the fixed split.

SELECTION (W9_SELECT)
---------------------
Two criteria are implemented.  Both are computed for every candidate epoch and
both are written to the results JSON; W9_SELECT only decides which one the
argmax uses.

  W9_SELECT=rank   (DEFAULT)
      The body's Section 5 protocol, Equation (28):

          score(ckpt) = SUM over i in Q_val of exp(-rank_i)

      Q_val is the validation split's wiki rewrites in the two neutral-toned
      registers, the `noname` (name-stripped) rewrite AND the `neutral`
      (name-intact) rewrite of each of the 203 validation games, 406 queries in
      all.  The `positive` / `negative` rewrites never select a checkpoint.
      rank_i is the retrieval rank of query i's own game, counted from 1,
      against the FULL 2,020-game anchor gallery (SPg), cosine similarity on
      L2-normalised vectors, `rank = (sim > sim_own).sum() + 1` so ties are
      resolved optimistically.  The per-query exponentials are SUMMED, as
      Equation (28) prints them.  Argmax over the candidate epochs, ties broken
      toward the earlier epoch.

      This is EXACTLY the criterion that re-selected the five-fold tables (the
      "both" variant of data/w9/reselect_rank.json, i.e. what
      data/w9/rank_ckpt_plan.json ships), applied to the fixed split by
      substituting the fixed split's validation games for the fold's.  It is
      the default because the body states one protocol for every table:
      checkpoints are written every 50 epochs and the deployed one maximises
      Equation (28) over the validation fold, and tags never select a
      checkpoint.  The rank criterion touches no tags and no review
      pseudo-queries, so it satisfies that sentence; rvsel does not.

  W9_SELECT=rvsel  (the previous behaviour, kept for the deviation report)
      rvsel = v_q1 + v_q5 + 2*v_qtag, computed by the worker's own
      `zs_from_arrays` on the validation fold's REVIEW pseudo-queries (SPq),
      argmax, ties toward the earlier epoch, exactly as the worker's ZSBEST
      tail does (`max(cand, key=lambda k: (cand[k]["rvsel"], -int(k[2:])))`).
      Its 2*v_qtag term makes tag quality select the checkpoint, which is the
      substantive conflict with the body's protocol.

Nothing else changes between the two modes: the reported retrieval columns and
the Section 5 tag probe are the same functions of the selected checkpoint.

CODE REUSE
----------
`zs_from_arrays` is defined INSIDE `main()` of w9_a100_worker.py, so it cannot
be imported.  Rather than copy it (and risk drift), this script parses the
worker source, lifts that one function definition out of `main()` with `ast`,
and executes it against a namespace holding exactly the closure variables the
worker binds (A, art_games, variants, test_g, val_g, va_neu, va_non, names,
n2i, y, Qs, targs, tag_split, VORDER, S_fn, train_anchor_ridge, micro_prf, np).
The retrieval columns and rvsel are therefore produced by the worker's literal
source.  The tag probe reuses `train_anchor_ridge` / `micro_prf` from
VICReg_review.text_variant_eval, the same primitives both workers call.

NO GPU IS REQUIRED.  The worker writes a projection cache
`tower_<label>_ep<N>.npz` (keys SPg / SPg_nd / SPa / SPq / SPd / SPd_gidx)
beside every checkpoint in the same loop that saves it, and `zs_from_arrays`
consumes exactly SPg / SPa / SPq.  Those caches are what a fresh run of this
repository produces, so the script reads them and never loads a `.pt` or the
16.9 GB anchor pack.  If a required tower cache is missing the script STOPS
with the exact missing path rather than approximating.

USAGE
-----
    C:\\Users\\admin\\anaconda3\\envs\\cuda_Vit\\python.exe table_a9_recompute.py

ENVIRONMENT VARIABLES (all optional; defaults are the local layout)
    LARICE_RESULTS / W9_OUT   checkpoint + tower directory
                              default C:\\runpod_data\\w9_out
    W9_CACHE                  fusion cache directory
                              default C:\\runpod_data\\fusion_cache_w9
    W9_REPO                   repo providing VICReg_review
                              default C:\\runpod_data\\stable-query-latent
    W9_WORKER                 path to w9_a100_worker.py
                              default: this file's directory
    W9_MAX_EPOCH              epoch cap, default 2000
    W9_SELECT                 rank (default) | rvsel   -- see SELECTION above
    W9_TRAJ_MODE              verify (default) | reuse
                              verify: recompute every epoch and cross-check
                                      against zs_traj_*.json where it carries
                                      the same keys;
                              reuse : take the zs_traj entry when it already
                                      carries rvsel, recompute otherwise
                                      (the literal reading of "reuse where they
                                      exist and are computed by the same code").
                              Applies to the rvsel side only: the rank score is
                              never cached anywhere, so it is always recomputed
                              from the epoch's tower npz.
    W9_A9_JSON                output JSON, default table_a9_results.json
                              beside this script.  The run log is written
                              beside it as <stem>_run.log, so a rank run and an
                              rvsel run never clobber each other.
"""

import ast
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# ----------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ.get("LARICE_RESULTS")
               or os.environ.get("W9_OUT")
               or r"C:\runpod_data\w9_out")
CACHE = Path(os.environ.get("W9_CACHE", r"C:\runpod_data\fusion_cache_w9"))
REPO = Path(os.environ.get("W9_REPO", r"C:\runpod_data\stable-query-latent"))
WORKER = Path(os.environ.get("W9_WORKER", str(HERE / "w9_a100_worker.py")))
MAX_EPOCH = int(os.environ.get("W9_MAX_EPOCH", "2000"))
CKPT_EVERY = int(os.environ.get("W9_CKPT_EVERY", "50"))
TRAJ_MODE = os.environ.get("W9_TRAJ_MODE", "verify").lower()
SELECT_MODE = os.environ.get("W9_SELECT", "rank").lower()
RESULTS_JSON = Path(os.environ.get("W9_A9_JSON", str(HERE / "table_a9_results.json")))

SPLIT_SEED = 20260711
ANCHOR_CAP = 4096
LABEL_SUFFIX = "_g%d_fp" % ANCHOR_CAP

# arm -> the row label used in Appendix Table A9
ARMS = [
    ("wcle_swin84step42loop2i2ce_icetf",  "swin, w = 84 (step 42, 19.7% coverage)"),
    ("wcle_swin168step84loop2i2ce_icetf", "swin, w = 168 (step 84, 27.5% coverage)"),
    ("wcle_swin336step168loop2i2ce_icetf", "swin, w = 336 (step 168, 43.1% coverage)"),
    ("wcle_i2ce_icetf",                   "I-CE, fully coupled (100% coverage)"),
]

# the shipped Table A9 body, for the deviation report only
SHIPPED = {
    "wcle_swin84step42loop2i2ce_icetf":  (1400, 0.946, 1.000, 0.691, 0.902, 0.714),
    "wcle_swin168step84loop2i2ce_icetf": (1150, 0.941, 0.995, 0.691, 0.892, 0.704),
    "wcle_swin336step168loop2i2ce_icetf": (700, 0.941, 1.000, 0.686, 0.897, 0.668),
    "wcle_i2ce_icetf":                   (1150, 0.951, 1.000, 0.706, 0.917, 0.679),
}

t_start = time.time()
# one log per results file, so a rank run and an rvsel run do not clobber each other
LOG_PATH = RESULTS_JSON.parent / ("%s_run.log" % RESULTS_JSON.stem)
_LOG = open(LOG_PATH, "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _LOG.write(s + "\n")
    _LOG.flush()


def die(msg):
    log("\nFATAL: " + msg)
    _LOG.close()
    raise SystemExit(2)


log("# Table A9 recompute   started %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
log("  out-dir   %s" % OUT_DIR)
log("  cache     %s" % CACHE)
log("  repo      %s" % REPO)
log("  worker    %s" % WORKER)
log("  epoch cap %d (step %d)   traj-mode %s   select %s"
    % (MAX_EPOCH, CKPT_EVERY, TRAJ_MODE, SELECT_MODE))

for p, what in ((OUT_DIR, "checkpoint/tower dir"), (CACHE, "fusion cache"),
                (REPO, "VICReg_review repo"), (WORKER, "worker source")):
    if not p.exists():
        die("%s not found: %s" % (what, p))
if TRAJ_MODE not in ("verify", "reuse"):
    die("W9_TRAJ_MODE must be 'verify' or 'reuse', got %r" % TRAJ_MODE)
if SELECT_MODE not in ("rank", "rvsel"):
    die("W9_SELECT must be 'rank' or 'rvsel', got %r" % SELECT_MODE)

sys.path.insert(0, str(REPO))
try:
    from VICReg_review.text_variant_eval import (train_anchor_ridge, micro_prf,
                                                 make_or_load_split)
except ImportError as e:
    die("cannot import VICReg_review.text_variant_eval from %s (%s).\n"
        "       The release copy of VICReg_review does not ship "
        "disturbtion_embed/identity_diagnostic/train_tag_probe, so point W9_REPO "
        "at a tree that does (e.g. C:\\runpod_data\\stable-query-latent)." % (REPO, e))

# ------------------------------------------------- caches (worker lines 630-745)
G = np.load(CACHE / "games.npz", allow_pickle=True)
names = [str(x) for x in G["names"]]
NG = len(names)
n2i = {n: i for i, n in enumerate(names)}
appid2name = {n.split("_")[0]: n for n in names}

A = np.load(CACHE / "wiki_eval.npz", allow_pickle=True)
variants = [str(x) for x in A["variants"]]
art_games = [str(x) for x in A["names"]]
Qs = np.load(CACHE / "ss_queries_rev.npz")
y = np.load(CACHE / "tag_labels.npz", allow_pickle=True)["y"]
tag_names = [str(x) for x in np.load(CACHE / "tag_labels.npz",
                                     allow_pickle=True)["tag_names"]]

sp = json.loads((CACHE / "wiki_eval_split.json").read_text())
if int(sp["seed"]) != SPLIT_SEED:
    die("wiki_eval_split.json seed is %s, expected %d" % (sp["seed"], SPLIT_SEED))
test_g = {appid2name[a] for a in sp["test"]}
val_g = {appid2name[a] for a in sp["val"]}
excl = test_g | val_g
train_pool_games = np.array([i for i in range(NG) if names[i] not in excl])

va_neu = [i for i, g in enumerate(art_games) if g in val_g and variants[i] == "neutral"]
va_non = [i for i, g in enumerate(art_games) if g in val_g and variants[i] == "noname"]

# --------- Equation (28) selection queries: the validation games' `noname` AND
# `neutral` rewrites (the "both" register set that rank_ckpt_plan.json ships).
# `gidx` is the wiki row's own game index into the 2,020-game gallery.
A_GIDX = np.asarray(A["gidx"])
RANK_ROWS = np.asarray(sorted(va_non + va_neu))
RANK_OWN = A_GIDX[RANK_ROWS]
if any(int(A_GIDX[i]) != n2i[art_games[i]] for i in RANK_ROWS):
    die("wiki_eval.npz gidx disagrees with games.npz name order on the "
        "validation rows; refusing to rank against a mismatched gallery.")

targs = SimpleNamespace(tag_text_train_frac=0.7, tag_text_val_frac=0.15,
                        tag_text_split_seed=42, seed=42,
                        tag_text_threshold_steps=33)
# The worker's in-training tag readout and the v_qtag term of rvsel both use this
# split (a 70/15/15 random split over ALL 2,020 games).  It is kept HERE, and only
# here, so that the rvsel path stays bit-identical to the shipped one; the rank
# criterion never touches it, and the reported tag column does NOT use it either.
tag_split = make_or_load_split(CACHE / "_tag_splitM.json", names, targs)

# --------- structural gates on the split (fail loudly rather than approximate)
GATES = {}


def gate(key, got, want):
    ok = (got == want)
    GATES[key] = {"got": got, "want": want, "pass": bool(ok)}
    log("  gate %-28s %-8s (expected %s) %s" % (key, got, want, "OK" if ok else "FAIL"))
    if not ok:
        die("structural gate %s failed: got %s, expected %s" % (key, got, want))


log("\n## split / cache gates")
gate("n_games", NG, 2020)
gate("n_test_games", len(test_g), 204)
gate("n_val_games", len(val_g), 203)
gate("n_wiki_train_games", len(sp["train"]), 407)
gate("n_training_gallery", int(len(train_pool_games)), 1613)
gate("n_tag_classes", int(y.shape[1]), 23)
gate("n_wiki_rows", len(art_games), 3256)
gate("n_val_neutral_rows", len(va_neu), 203)
gate("n_val_noname_rows", len(va_non), 203)
gate("n_rank_selection_queries", int(len(RANK_ROWS)), 406)
gate("n_pseudo_queries", int(len(Qs["gidx"])), 8080)

# ------------------------------------------------- lift zs_from_arrays out of main()
worker_src = WORKER.read_text(encoding="utf-8")
tree = ast.parse(worker_src)
node = None
for n in ast.walk(tree):
    if isinstance(n, ast.FunctionDef) and n.name == "zs_from_arrays":
        node = n
        break
if node is None:
    die("zs_from_arrays not found in %s" % WORKER)

fn_src = ast.get_source_segment(worker_src, node)
if fn_src is None:
    die("could not extract the source of zs_from_arrays from %s" % WORKER)
# it is nested inside main(), so it arrives indented by four spaces
fn_src = "\n".join(line[4:] if line.startswith("    ") else line
                   for line in fn_src.splitlines())

sys.path.insert(0, str(WORKER.parent))
try:
    import w9_a100_worker as _w  # module-level helpers (S_fn, rown, SetPoolN, ...)
    S_fn = _w.S_fn
    WORKER_IMPORTED = True
except Exception as e:                                   # torch missing, etc.
    WORKER_IMPORTED = False
    log("  note: could not import w9_a100_worker as a module (%s);" % e)
    log("        falling back to the AST-lifted definition of S_fn as well.")
    _sfn = None
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "S_fn":
            _sfn = ast.get_source_segment(worker_src, n)
    if _sfn is None:
        die("neither the module nor S_fn could be obtained from %s" % WORKER)
    _ns = {"np": np}
    exec(compile(_sfn, str(WORKER), "exec"), _ns)
    S_fn = _ns["S_fn"]

ZS_NS = {
    "np": np, "A": A, "art_games": art_games, "variants": variants,
    "test_g": test_g, "val_g": val_g, "va_neu": va_neu, "va_non": va_non,
    "names": names, "n2i": n2i, "y": y, "Qs": Qs, "targs": targs,
    "tag_split": tag_split, "VORDER": ["neutral", "noname", "positive", "negative"],
    "S_fn": S_fn, "train_anchor_ridge": train_anchor_ridge, "micro_prf": micro_prf,
}
exec(compile(fn_src, str(WORKER) + "::zs_from_arrays", "exec"), ZS_NS)
zs_from_arrays = ZS_NS["zs_from_arrays"]
log("\n## lifted zs_from_arrays from %s (lines %d-%d), worker module import: %s"
    % (WORKER.name, node.lineno, node.end_lineno, WORKER_IMPORTED))

# ------------------------------------------------- Equation (28) rank score
# The five-fold criterion, verbatim, with the fixed split's validation games in
# place of the fold's: sum of exp(-rank) over the validation `noname` + `neutral`
# wiki rewrites, ranked against the full 2,020-game anchor gallery.  The rank
# arithmetic is the `hit_at_1` block of data/w9/register_tax_recompute.py
# (lines 96-100), including its float32 cast and its optimistic tie handling.
def unit(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def query_ranks(Q, G, own):
    sim = unit(Q.astype(np.float32)) @ unit(G.astype(np.float32)).T
    s_own = sim[np.arange(len(own)), own]
    return (sim > s_own[:, None]).sum(1) + 1


def rank_select_score(Zg, Za):
    """Equation (28): sum_i exp(-rank_i) over the 406 validation-split queries."""
    r = query_ranks(Za[RANK_ROWS], Zg, RANK_OWN)
    return float(np.exp(-r.astype(np.float64)).sum())


# ------------------------------------------------- Section 5 test-set tag probe
# register_tax_recompute.py lines 136-144, adapted from a CV fold to the fixed
# split: fit on the training-gallery anchors, alpha+threshold on the validation
# games' anchors, score the test games' name-intact (neutral) wiki-rewrite queries.
CTAG = {"train": [names[i] for i in train_pool_games],
        "val": sorted(val_g),
        "test": sorted(test_g)}
TEST_NEUTRAL_ROWS = np.asarray([i for i in range(len(art_games))
                                if variants[i] == "neutral" and art_games[i] in test_g])
TEST_NEUTRAL_LAB = np.stack([y[n2i[art_games[i]]] for i in TEST_NEUTRAL_ROWS])
gate("n_test_neutral_queries", int(len(TEST_NEUTRAL_ROWS)), 204)


def section5_test_tag(Zg, Za):
    """Section 5 test-set tag micro-F1 (a single global threshold, as
    train_anchor_ridge/micro_prf define it -- not one threshold per class)."""
    sc, rg, alpha, th, val_metrics = train_anchor_ridge(targs, Zg, y, n2i, CTAG)
    scores = rg.predict(sc.transform(Za[TEST_NEUTRAL_ROWS].astype(np.float32)))
    m = micro_prf(TEST_NEUTRAL_LAB, scores, th)
    return {"tag_f1": float(m["micro_f1"]),
            "tag_precision": float(m["precision"]),
            "tag_recall": float(m["recall"]),
            "ridge_alpha": float(alpha),
            "threshold": float(th),
            "val_micro_f1": float(val_metrics["micro_f1"])}


# ------------------------------------------------- per-arm sweep
RVSEL_KEYS = ("nm_neutral", "h5_neutral", "nm_noname", "h5_noname",
              "v_q1", "v_q5", "v_qtag", "rvsel")
EPOCHS = list(range(CKPT_EVERY, MAX_EPOCH + 1, CKPT_EVERY))

CRITERION_TEXT = {
    "rank": ("Equation (28): score = sum_i exp(-rank_i) over the %d validation-split "
             "wiki rewrites (each validation game's `noname` and `neutral` rewrite), "
             "rank counted from 1 against the full %d-game anchor gallery, cosine on "
             "L2-normalised vectors, rank = (sim > sim_own).sum() + 1; argmax, ties -> "
             "earlier epoch. No tags and no review pseudo-queries enter it. Identical "
             "to the criterion that re-selected the five-fold tables (the \"both\" "
             "variant of data/w9/reselect_rank.json, shipped as "
             "data/w9/rank_ckpt_plan.json), with the fixed split's validation games "
             "in place of the fold's."),
    "rvsel": ("rvsel = v_q1 + v_q5 + 2*v_qtag, computed by the worker's own "
              "zs_from_arrays on the validation split's review pseudo-queries; argmax, "
              "ties -> earlier epoch (w9_a100_worker.py ZSBEST tail). Its v_qtag term "
              "uses the worker's _tag_splitM.json probe, unchanged. This is the "
              "PREVIOUS criterion, kept only for the deviation report: it lets tag "
              "quality select the checkpoint, which the body's Section 5 forbids."),
}
SELECTION_CRITERION = CRITERION_TEXT[SELECT_MODE] % (
    (len(RANK_ROWS), NG) if SELECT_MODE == "rank" else ())

results = {}
log("\n## sweep: %d candidate epochs per arm (%d..%d step %d), selecting on %s"
    % (len(EPOCHS), EPOCHS[0], EPOCHS[-1], CKPT_EVERY, SELECT_MODE))

for arm, row_label in ARMS:
    label = "w9_%s%s" % (arm, LABEL_SUFFIX)
    t_arm = time.time()
    traj_p = OUT_DIR / ("zs_traj_%s.json" % label)
    traj = json.loads(traj_p.read_text()) if traj_p.exists() else {}

    missing = [ep for ep in EPOCHS
               if not (OUT_DIR / ("tower_%s_ep%d.npz" % (label, ep))).exists()]
    if missing:
        die("arm %s: %d projection caches missing under %s, first is\n"
            "       tower_%s_ep%d.npz\n"
            "       These are written by w9_a100_worker.py beside every "
            "checkpoint (project_cache).  Re-run the worker for this arm, or "
            "point W9_OUT at the directory that holds them.  Refusing to "
            "approximate the table from a partial sweep."
            % (arm, len(missing), OUT_DIR, label, missing[0]))

    rows, sources, deltas, rank_scores = {}, {}, [], {}
    for ep in EPOCHS:
        entry = traj.get("ep%d" % ep, {})
        have_traj = all(k in entry for k in RVSEL_KEYS)
        T = np.load(OUT_DIR / ("tower_%s_ep%d.npz" % (label, ep)))
        # the rank score is never cached, so it is recomputed at every epoch in
        # both modes; that also keeps the two criteria comparable in one file.
        rank_scores[ep] = rank_select_score(T["SPg"], T["SPa"])
        if TRAJ_MODE == "reuse" and have_traj:
            rows[ep] = dict(entry)
            sources[ep] = "zs_traj"
            continue
        zk = zs_from_arrays(T["SPg"], T["SPa"], T["SPq"])
        rows[ep] = zk
        sources[ep] = "recomputed"
        if have_traj:
            per_key = {k: abs(float(zk[k]) - float(entry[k])) for k in RVSEL_KEYS}
            deltas.append((ep, max(per_key.values()), per_key))

    rank_pick = max(EPOCHS, key=lambda e: (rank_scores[e], -e))
    rvsel_pick = max(EPOCHS, key=lambda e: (rows[e]["rvsel"], -e))
    best_ep = rank_pick if SELECT_MODE == "rank" else rvsel_pick
    b = rows[best_ep]

    T = np.load(OUT_DIR / ("tower_%s_ep%d.npz" % (label, best_ep)))
    tag = section5_test_tag(T["SPg"].astype(np.float32), T["SPa"].astype(np.float32))

    # what the worker's own zsbest file says, for the deviation report
    zb_p = OUT_DIR / ("zsbest_%s.json" % label)
    zb = json.loads(zb_p.read_text()) if zb_p.exists() else {}

    if deltas:
        worst = max(deltas, key=lambda t: t[1])
        traj_agreement = {
            "n_epochs_compared": len(deltas),
            "max_abs_delta": float(worst[1]),
            "at_epoch": int(worst[0]),
            "max_abs_delta_by_key": {
                k: float(max(pk[k] for _, _, pk in deltas)) for k in RVSEL_KEYS},
            "max_abs_delta_retrieval_only": float(max(
                max(pk[k] for k in ("nm_neutral", "h5_neutral",
                                    "nm_noname", "h5_noname"))
                for _, _, pk in deltas)),
        }
    else:
        traj_agreement = None

    results[arm] = {
        "row_label": row_label,
        "tower_label": label,
        "selected_epoch": int(best_ep),
        "selection_mode": SELECT_MODE,
        "selection_criterion": SELECTION_CRITERION,
        "rank_selected_epoch": int(rank_pick),
        "rvsel_selected_epoch": int(rvsel_pick),
        "epoch_cap": MAX_EPOCH,
        "rank_score": float(rank_scores[best_ep]),
        "rank_score_at_rank_pick": float(rank_scores[rank_pick]),
        "rank_score_at_rvsel_pick": float(rank_scores[rvsel_pick]),
        "rvsel_at_rank_pick": float(rows[rank_pick]["rvsel"]),
        "rvsel_at_rvsel_pick": float(rows[rvsel_pick]["rvsel"]),
        "rvsel": float(b["rvsel"]),
        "v_q1": float(b["v_q1"]), "v_q5": float(b["v_q5"]), "v_qtag": float(b["v_qtag"]),
        "name_hit1": float(b["nm_neutral"]), "name_hit5": float(b["h5_neutral"]),
        "stripped_hit1": float(b["nm_noname"]), "stripped_hit5": float(b["h5_noname"]),
        "test_tag_f1": tag["tag_f1"],
        "tag_probe": tag,
        "checkpoint": str(OUT_DIR / ("ckpt_%s_ep%d.pt" % (label, best_ep))),
        "tower_cache": str(OUT_DIR / ("tower_%s_ep%d.npz" % (label, best_ep))),
        "worker_tag_neutral_leaky": float(b.get("tag_neutral", float("nan"))),
        "rvsel_trajectory": {str(ep): float(rows[ep]["rvsel"]) for ep in EPOCHS},
        "rank_score_trajectory": {str(ep): float(rank_scores[ep]) for ep in EPOCHS},
        "per_epoch": {str(ep): dict({k: float(rows[ep][k]) for k in RVSEL_KEYS},
                                    rank_score=float(rank_scores[ep]))
                      for ep in EPOCHS},
        "per_epoch_source": {str(ep): sources[ep] for ep in EPOCHS},
        "zs_traj_agreement": traj_agreement,
        "shipped_zsbest_epoch": (int(zb["best_ep"]) if "best_ep" in zb else None),
        "shipped_zsbest_had_rvsel": bool("rvsel" in zb),
        "n_traj_entries_on_disk": len(traj),
        "n_traj_entries_with_rvsel": sum(1 for v in traj.values() if "rvsel" in v),
        "runtime_sec": round(time.time() - t_arm, 1),
    }

    log("  %-38s ep%-5d rank=%.4f rvsel=%.6f  %.4f %.4f %.4f %.4f  tag=%.4f "
        "(alpha=%g th=%.4f)  [%.0fs]"
        % (arm, best_ep, rank_scores[best_ep], b["rvsel"],
           b["nm_neutral"], b["h5_neutral"],
           b["nm_noname"], b["h5_noname"], tag["tag_f1"],
           tag["ridge_alpha"], tag["threshold"], time.time() - t_arm))
    log("      criterion picks: rank -> ep%d (score %.4f)   rvsel -> ep%d "
        "(rvsel %.6f)   %s"
        % (rank_pick, rank_scores[rank_pick], rvsel_pick, rows[rvsel_pick]["rvsel"],
           "AGREE" if rank_pick == rvsel_pick else "DISAGREE"))
    if traj_agreement:
        log("      zs_traj cross-check: %d epochs, max |delta| = %.3g at ep%d; "
            "retrieval-only max |delta| = %.3g"
            % (traj_agreement["n_epochs_compared"], traj_agreement["max_abs_delta"],
               traj_agreement["at_epoch"],
               traj_agreement["max_abs_delta_retrieval_only"]))
        log("      by key: " + "  ".join(
            "%s=%.3g" % (k, v)
            for k, v in traj_agreement["max_abs_delta_by_key"].items()))
    elif results[arm]["n_traj_entries_with_rvsel"] == 0:
        log("      zs_traj cross-check: none of the %d entries on disk carries the "
            "rvsel key set -- this arm predates the current criterion, so every "
            "epoch was recomputed from its tower cache" % len(traj))
    else:
        log("      zs_traj cross-check: skipped, W9_TRAJ_MODE=reuse took all %d/%d "
            "epochs straight from zs_traj" % (results[arm]["n_traj_entries_with_rvsel"],
                                              len(traj)))

# ------------------------------------------------- deviation report
log("\n## deviations from the shipped Table A9")
dev = {}
for arm, _ in ARMS:
    r = results[arm]
    s = SHIPPED[arm]
    dev[arm] = {
        "shipped": {"epoch": s[0], "name_hit1": s[1], "name_hit5": s[2],
                    "stripped_hit1": s[3], "stripped_hit5": s[4],
                    "test_tag_f1": s[5]},
        "recomputed": {"epoch": r["selected_epoch"], "name_hit1": r["name_hit1"],
                       "name_hit5": r["name_hit5"],
                       "stripped_hit1": r["stripped_hit1"],
                       "stripped_hit5": r["stripped_hit5"],
                       "test_tag_f1": r["test_tag_f1"]},
        "picks": {"rank": r["rank_selected_epoch"],
                  "rvsel": r["rvsel_selected_epoch"],
                  "shipped_rvsel": s[0],
                  "rank_vs_rvsel_agree": bool(r["rank_selected_epoch"]
                                              == r["rvsel_selected_epoch"])},
        "delta": {"epoch": r["selected_epoch"] - s[0],
                  "name_hit1": r["name_hit1"] - s[1],
                  "name_hit5": r["name_hit5"] - s[2],
                  "stripped_hit1": r["stripped_hit1"] - s[3],
                  "stripped_hit5": r["stripped_hit5"] - s[4],
                  "test_tag_f1": r["test_tag_f1"] - s[5]},
    }
    d = dev[arm]["delta"]
    log("  %-38s dep=%+5d  dh1=%+.4f dh5=%+.4f dS1=%+.4f dS5=%+.4f dtag=%+.4f"
        % (arm, d["epoch"], d["name_hit1"], d["name_hit5"],
           d["stripped_hit1"], d["stripped_hit5"], d["test_tag_f1"]))

# ------------------------------------------------- markdown table
def fmt(x):
    return "%.3f" % x


md = []
md.append("<!--table grid=1500,1120,1120,1245,1245,2170 bold=0,2,4-->")
md.append("| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set tag F1 |")
md.append("| --- | --- | --- | --- | --- | --- |")
for arm, row_label in ARMS:
    r = results[arm]
    md.append("| %s | %s | %s | %s | %s | %s |"
              % (row_label, fmt(r["name_hit1"]), fmt(r["name_hit5"]),
                 fmt(r["stripped_hit1"]), fmt(r["stripped_hit5"]),
                 fmt(r["test_tag_f1"])))
md_text = "\n".join(md)
log("\n## Table A9 (regenerated, appendix row format)\n")
log(md_text)

log("\n## selected epochs (active criterion: %s)" % SELECT_MODE)
for arm, _ in ARMS:
    r = results[arm]
    log("  %-38s ep%-5d  rank-pick ep%-5d  rvsel-pick ep%-5d  "
        "(shipped zsbest pick ep%s, had rvsel: %s)"
        % (arm, r["selected_epoch"], r["rank_selected_epoch"],
           r["rvsel_selected_epoch"], r["shipped_zsbest_epoch"],
           r["shipped_zsbest_had_rvsel"]))

# ------------------------------------------------- save
payload = {
    "meta": {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": str(Path(__file__).resolve()),
        "worker_source": str(WORKER),
        "zs_from_arrays_lines": [node.lineno, node.end_lineno],
        "out_dir": str(OUT_DIR), "cache_dir": str(CACHE), "repo": str(REPO),
        "split_file": str(CACHE / "wiki_eval_split.json"),
        "split_seed": SPLIT_SEED,
        "anchor_cap": ANCHOR_CAP,
        "anchor_pack": str(CACHE / ("wscan_gal_rev_g%d.npz" % ANCHOR_CAP)),
        "anchor_draws": 1,
        "full_pool": True,
        "label_suffix": LABEL_SUFFIX,
        "epoch_grid": [CKPT_EVERY, MAX_EPOCH, CKPT_EVERY],
        "traj_mode": TRAJ_MODE,
        "selection_mode": SELECT_MODE,
        "selection_modes_available": ["rank", "rvsel"],
        "selection_default": "rank",
        "n_rank_selection_queries": int(len(RANK_ROWS)),
        "rank_selection_registers": ["noname", "neutral"],
        "rank_selection_provenance": (
            "the \"both\" variant of data/w9/reselect_rank.json, shipped as "
            "data/w9/rank_ckpt_plan.json, i.e. the criterion the five-fold tables "
            "were re-selected under; here with the fixed split's 203 validation "
            "games in place of a fold's 163"),
        "n_games": NG, "n_test_games": len(test_g), "n_val_games": len(val_g),
        "n_training_gallery": int(len(train_pool_games)),
        "n_tag_classes": int(y.shape[1]), "tag_names": tag_names,
        "retrieval_gallery": "all %d game anchors (worker gallery(), not the "
                             "1,613-game training gallery)" % NG,
        "selection": SELECTION_CRITERION + (
            " Argmax over ep<=%d on the 50-epoch grid. The inactive criterion is "
            "computed at every epoch as well and stored per arm "
            "(rank_score_trajectory / rvsel_trajectory, and rank_selected_epoch / "
            "rvsel_selected_epoch)." % MAX_EPOCH),
        "selection_criterion_rank": CRITERION_TEXT["rank"] % (len(RANK_ROWS), NG),
        "selection_criterion_rvsel": CRITERION_TEXT["rvsel"],
        "tag_protocol": ("Section 5 test-set probe: ridge fit on the %d "
                         "training-gallery anchor vectors against the 23 coarse tag "
                         "classes; ridge alpha and a single global decision "
                         "threshold chosen jointly on the %d validation games' "
                         "anchor vectors (11 alphas x 33 thresholds, "
                         "train_anchor_ridge); scored as micro-F1 on the %d test "
                         "games' name-intact (neutral) wiki-rewrite query vectors. "
                         "Identical to register_tax_recompute.py lines 136-144 with "
                         "the fixed split in place of a CV fold."
                         % (len(train_pool_games), len(val_g),
                            len(TEST_NEUTRAL_ROWS))),
        "runtime_sec": round(time.time() - t_start, 1),
    },
    "gates": GATES,
    "markdown_table": md_text,
    "deviation_vs_shipped": dev,
    "arms": results,
}
RESULTS_JSON.write_text(json.dumps(payload, indent=1), encoding="utf-8")
log("\nwrote %s" % RESULTS_JSON)
log("wrote %s" % LOG_PATH)
log("total %.1fs" % (time.time() - t_start))
_LOG.close()
