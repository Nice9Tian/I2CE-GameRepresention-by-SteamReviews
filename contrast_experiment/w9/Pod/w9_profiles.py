# -*- coding: utf-8 -*-
"""The reproduction grid, one row per experiment cell, with provenance.

Before this file the grid was scattered: `w9_jobs.FS_JOBS` held the
fixed-split research log (106 rows, most of them exploratory), while the
cells that actually back the manuscript's figures and tables lived as
constants inside twelve notebooks. Nothing said which run produced which
number. This module is that missing statement.

Every CELL below carries `cites`: the figures and tables it feeds. A cell
with a body citation (FIG7, FIG8, FIG9, TAB2, TAB3, SEC*) is reported in the
paper itself; the rest are appendix- or supplement-only. Nothing is listed here
that the paper, the appendix or the supplement does not mention, so the exploratory families -- twin-pack
`pk*`, the slot/readout capacity grid, multi-anchor `ma2*`, the async
simulation, hard-negative `nemesis` -- are deliberately absent. They stay
available through `w9_jobs.FS_JOBS` and the notebooks. The one exception is
apx.swin.window, the window-size sweep: the paper no longer prints it, it
stays as an extra of this repository, and no profile runs it.

Three profiles select from it:

    fullTest        every cell of the paper, the appendix and the supplement
                    (every reported run but the view-size sweep of Table A6,
                    which has no cell)
    paperTest       cells with a body citation, at their own budgets
    litePaperTest   paperTest with every anchor budget clamped to 1024

litePaperTest is the desktop-GPU profile. The clamp is a ceiling, not a
setting: a cell the paper ran at 512 stays at 512, because rerunning it at
1024 would not reproduce the published row. Only the cells above 1,024
move, and they move to a budget the paper itself measures as within 0.02
of the full configuration on every reading (Section 4.2, Table A2).

`caps` are the five-fold worker's `--anchor-cap`: a number of sentences per
pack, store-page prefix included, or `pool`. `pool` is the prefix plus the
game's whole 2,048-sentence review pool, with no review left out: packs of
up to 2,184 sentences, a median of 2,068. These are the packs of the
paper's main five-fold towers, and their runs and checkpoints are named
`_gpool`. A cap of 2048 stops at 2,048 sentences in total, a median of
2,046, and names its runs `_g2048`. The paper calls both the
2,048-sentence budget, and no profile trains above it. The workers refuse
a cap the review pool cannot fill.

This module is pure data plus three small functions. It imports nothing
outside the standard library, so `w9/` stays copyable to a pod on its own.
"""

# ---------------------------------------------------------------- cites --
# Figure, table, section and appendix numbers are those of the paper as
# APPENDIX.md and SUPPLEMENT.md of this repository cite it (Appendices A
# to N, Tables A1 to A19; the numbers of 2026-11-12, when the view-size sweep
# became Table A6 and every later table moved one place).
# Body of the manuscript.
FIG7 = "Figure 7"          # the objective families
FIG8 = "Figure 8"          # name-stripped hit@1 by genre
FIG9 = "Figure 9"          # one neighborhood under six encoders (fixed split)
TAB2 = "Table 2"           # paired tests over the five folds
TAB3 = "Table 3"           # the decomposition ablation
SEC61 = "Section 6.1"      # the headline five-fold reading
SEC62 = "Section 6.2"      # the decomposition sweep
SEC32 = "Section 3.2"      # the momentum-teacher / memory-bank comparison

BODY = {FIG7, FIG8, FIG9, TAB2, TAB3, SEC61, SEC62, SEC32}

# Online appendix.
A1 = "Table A1"; A2 = "Table A2"; A3 = "Table A3"; A4 = "Table A4"
A5 = "Table A5"
A6 = "Table A6"            # the size of a review view (Appendix C); read by body.pair.512 only, see below
A7 = "Table A7"; A8 = "Table A8"; A9 = "Table A9"; A10 = "Table A10"
A11 = "Table A11"; A12 = "Table A12"; A13 = "Table A13"; A14 = "Table A14"
A15 = "Table A15"; A16 = "Table A16"; A17 = "Table A17"
A18 = "Table A18"          # displacement / margin / ratio (Appendix M)
A19 = "Table A19"
FIGA1 = "Figure A1"        # the four latent queries over training (Appendix C)
FIGA2 = "Figure A2"        # UMAP of anchors and queries (Appendix L)
FIGA3 = "Figure A3"        # Figure 8 on the fixed split (Appendix L)
APXL = "Appendix L"        # the 201-neighborhood statistics of the fixed split
# The window-size sweep has no table in the paper; see apx.swin.window.
WSWEEP = "window-size sweep (no table in the paper)"
# Supplementary material (Supplements A-C).
S1 = "Table S1"; S2 = "Table S2"; S3 = "Table S3"; S4 = "Table S4"

POOL = "pool"              # caps: the store-page prefix plus the whole review pool
FIXED = 0                  # folds=FIXED means the fixed 204/203/407 split
                           # folds=3 means folds 0 to 2 of the five,
                           # folds=1 fold 0 only


class Cell:
    """One (arms x caps x folds) block of towers, and what it backs."""

    __slots__ = ("key", "arms", "caps", "folds", "epochs", "cites",
                 "was_in", "note", "extra")

    def __init__(self, key, arms, caps, folds, epochs, cites,
                 was_in, note="", extra=None):
        self.key = key
        self.arms = tuple(arms)
        self.caps = tuple(caps)
        self.folds = folds
        self.epochs = epochs
        self.cites = tuple(cites)
        self.was_in = was_in          # where this used to be defined
        self.note = note
        self.extra = dict(extra or {})   # extra worker flags for the cell

    @property
    def in_body(self):
        return any(c in BODY for c in self.cites)

    def towers(self, caps=None):
        caps = self.caps if caps is None else caps
        n = len(self.arms) * len(caps)
        return n if self.folds == FIXED else n * self.folds

    def __repr__(self):
        return f"<Cell {self.key} {len(self.arms)}arm {list(self.caps)}>"


# ------------------------------------------------------------- the grid --
CELLS = [

    # ---- body: the shared five-fold tower set (whole-pool packs) ------
    # Trained once, read many ways. A1/A14/A16/A17 are different readouts of
    # these same towers, NOT more training runs, and S2/S3/S4 read them again
    # next to the swin and two-stage teachers (apx.swin.5fold, apx.twostage)
    # -- which is why the tables outnumber the campaigns.
    Cell("body.pair",
         arms=["wcle_i2ce_icetf", "wcle_ce_cetf"],
         caps=[POOL], folds=5, epochs=2000,
         cites=[FIG7, FIG8, TAB2, SEC61, A1, A2, A7, A13, A14, A15, A16, A17,
                A18, FIGA2, S2, S3, S4],
         was_in="w9_final_experiment.ipynb",
         note="the headline I-CE vs CE pair, paired per fold. Its I-CE is "
              "also the five-fold recipe row of Table A7 and the first row "
              "of Table A13. Table A18 and "
              "Figure A2 read the published fold-0 checkpoints of this cell "
              "and of body.families at the Section 5 rank-selected epochs "
              "(rank_ckpt_plan.json: I-CE ep2000, CE ep650, SimCLR ep1100, "
              "VICReg ep150, BYOL ep250); Pod/table1_recompute.py regenerates "
              "both from those checkpoints and the single 4,096-sentence anchor "
              "draw wscan_gal_rev_g4096.npz"),

    Cell("body.families",
         arms=["wcle_bce_cetf",             # SimCLR-style, in-batch views
               "wcle_byol_bytf",            # BYOL
               "wcle_epd_v20i10c20_cetf",  # VICReg, expander 20/10/20
               "wcle_mq3072i2ce_icetf"],    # EMA teacher + 3,072-key bank
         caps=[POOL], folds=5, epochs=2000,
         cites=[FIG7, FIG8, TAB2, SEC32, A1, A13, A16, A17, A18, FIGA2,
                S2, S3, S4],
         was_in="w9_experiment_5fold_2.ipynb",
         note="the other objective families of Figure 7"),

    Cell("body.i_simclr",
         arms=["wcle_i2bce_icetf"],         # SimCLR-style + the invariance term
         caps=[POOL], folds=5, epochs=2000,
         cites=[FIG8, APXL],
         was_in="w9_experiment_5fold_2.ipynb",
         note="I + SimCLR: the SimCLR-style tower of body.families with "
              "I-CE's invariance term, trained for Figure 8. It reads no "
              "anchor in training, so the pack only sets the gallery the "
              "worker saves and selects on. The published runs were made "
              "at 512 (their files carry no pack in their names) and "
              "selected afterwards on the pool gallery "
              "(eval/i2bce_sel_pool.json); a run at pool selects on that "
              "gallery itself"),

    # ---- body: the 512-sentence pairs of Table 2, five folds ----------
    Cell("body.pair.512",
         arms=["wcle_i2ce_icetf", "wcle_ce_cetf"],
         caps=[512], folds=5, epochs=2000,
         cites=[TAB2, A2, A3, A4, A5, A6, A7, A11, A15, A19, S1],
         was_in="w9_final_experiment.ipynb",
         note="the pair on the 512-sentence packs: the partners of CE + "
              "anchor-in-I in the paired tests of Table 2, the 512 rows of "
              "Tables A2 (I-CE) and A15, the recipe row of Table A3 at 512, "
              "the raw-document rows of Table A19, the reference row of "
              "Table S1 and, on folds 0 to 2, the recipe rows of Tables A4 "
              "to A7 and the lambda = 0 row of Table A4"),

    Cell("body.anchor_in_i.5fold",
         arms=["wcle_ai2ce_icetf"],
         caps=[512], folds=5, epochs=2000,
         cites=[TAB2, SEC62, A15],
         was_in="w9_anchor_in_i_5fold.ipynb (the fixed-split run is in "
                "body.decomp.rows)",
         note="CE + anchor-in-I under the folds, seeds, sampler and schedule "
              "of body.pair.512; Table 2 and Section 6.2 test it per fold "
              "against that pair"),

    # ---- body: the decomposition ablation, fixed split @512 -----------
    Cell("body.decomp.rows",
         arms=["wcle_i2ce_icetf", "wcle_ce_cetf",
               "wcle_ai2ce_icetf"],         # CE + anchor-in-I
         caps=[512], folds=FIXED, epochs=2000,
         cites=[TAB3, SEC62, A8],
         was_in="w9_jobs.FS_JOBS + w9_flash.ipynb",
         note="the first three rows of Table 3 (eval/fixed_split_eval.py "
              "reads this cell and the other fixed-split ones from the "
              "published checkpoints). The published CE + "
              "anchor-in-I run was trained 1,000 epochs and then on to "
              "2,000 from the weights of epoch 1,000 with a fresh optimizer "
              "state (eval/fixed_split_ckpt_plan.json, field extended)"),

    Cell("body.decomp.sweep",
         # Naming: ai<N> = anchor-aimed align weight N, auni<M> = anchor
         # uniformity M, i<N>uni<M> = the view-aimed pair. Table 3's
         # "a+u @ gallery" row is ai25auni2 (25:2, the 12.5:1 arm); its
         # "a+u @ views" row is i4uni2 (4:2, the 2:1 arm).
         # (wcle_i2au2 / wcle_i2au25 belong to the same family but exist
         # only in the five-fold worker, so they are not listed here.)
         arms=["wcle_ai25auni2_icetf", "wcle_ai6auni2_icetf",
               "wcle_ai4auni2_icetf", "wcle_ai2auni25_icetf",
               "wcle_i4uni2_icetf", "wcle_i6uni2_icetf",
               "wcle_ai6uni2_icetf"],
         caps=[512], folds=FIXED, epochs=2000,
         cites=[TAB3, SEC62, APXL],
         was_in="w9_flash.ipynb",
         note="repulsion target x align:uniformity ratio. The a+u @ views "
              "tower (i4uni2) is also read in Appendix L, against the ten "
              "4,096-sentence draws. The published runs of ai6auni2, "
              "ai4auni2, ai2auni25 and i4uni2 were trained 1,000 epochs and "
              "then on to 2,000 from the weights of epoch 1,000"),

    Cell("body.decomp.projection",
         # The kernel-temperature / projection axis of the same sweep:
         # a 3x3 dual-projection grid plus the three shared-expander cells.
         arms=[f"wcle_{g1}ai25{g2}auni2_icetf"
               for g1 in ("exp", "cmp", "pj")
               for g2 in ("exp", "cmp", "pj")]
              + [f"wcle_sh{g}ai25auni2_icetf" for g in ("exp", "cmp", "pj")],
         caps=[512], folds=FIXED, epochs=2000,
         cites=[SEC62],
         was_in="w9_flash.ipynb",
         note="the 'settings in online Appendix H' sweep of Section 6.2; the projection grid "
              "is repository-only detail behind the geometry files of Table A10"),

    # ---- body: the fixed-split towers of Figure 9 ----------------------
    # Figure 9, Figure A3 and the neighborhood statistics of Appendix L read
    # six encoders on the fixed split against the ten 4,096-sentence draws:
    # the frozen embedder and the five towers of the two cells below. Each
    # checkpoint is the rank rule's choice over the run's 40 (epochs 1,750,
    # 1,150, 1,350, 1,350 and 100 in the order listed). The three anchor-free
    # runs were trained 1,000 epochs and then on to 2,000 from the weights of
    # epoch 1,000 with a fresh optimizer state, so a fresh 2,000-epoch run
    # reproduces the protocol, not those runs bit for bit.
    Cell("body.neighborhood.pair",
         arms=["wcle_i2ce_icetf", "wcle_ce_cetf"],
         caps=[2048], folds=FIXED, epochs=2000,
         cites=[FIG9, FIGA1, FIGA3, APXL],
         was_in="w9_jobs.FS_JOBS",
         note="I-CE and CE on the fixed split at the 2,048-sentence budget. "
              "Figure A1 draws the four latent queries of the I-CE run at "
              "each of its 40 checkpoints, all of them published"),

    Cell("body.neighborhood.views",
         arms=["wcle_bce_cetf",             # SimCLR-style, in-batch views
               "wcle_i2bce_icetf",          # the same with the invariance term
               "wcle_epd_v20i10c20_cetf"],  # VICReg, the recipe weights
         caps=[512], folds=FIXED, epochs=2000,
         cites=[FIG9, FIGA3, APXL, A8],
         was_in="w9_jobs.FS_JOBS",
         note="the towers of those figures that read no anchor in training, "
              "so the 512 here only names the gallery the worker saved; "
              "their checkpoints are chosen on a 4,096-sentence pack of the "
              "split. The VICReg arm is also the recipe-weights row of "
              "Table A8. Appendix L describes the SimCLR-style run's test "
              "curve from epoch 1,350 on, and those checkpoints are "
              "published too"),

    # ---- appendix: the anchor-budget ladder ---------------------------
    Cell("apx.ladder",
         arms=["wcle_i2ce_icetf", "wcle_ce_cetf"],
         caps=[1024], folds=5, epochs=2000,
         cites=[A2, A11, A15],
         was_in="w9_final_experiment.ipynb",
         note="1,024 is the middle row of Tables A2 (I-CE) and A15, whose "
              "512 rows are body.pair.512 and whose 2,048 rows are "
              "body.pair, and the 1,024 row of Table A11"),

    Cell("apx.ladder.2048",
         arms=["wcle_i2ce_icetf", "wcle_ce_cetf"],
         caps=[2048], folds=5, epochs=2000,
         cites=[A3, A11],
         was_in="w9_final_experiment.ipynb",
         note="a second run of the pair, capped at 2,048 sentences in "
              "total: its I-CE is "
              "the recipe row of Table A3 at 2,048, the seed-paired control "
              "of apx.temperature, and its peak is the 35.0 GiB of Table A11; "
              "its CE is not printed. Its fold-0 I-CE tower also gives the "
              "game vectors of the body's Figure 10 (figures/trend_river)"),

    # ---- appendix: anchor-supply economies ----------------------------
    Cell("apx.swin.5fold",
         arms=["wcle_swin168step84loop2i2ce_icetf"],
         caps=[POOL], folds=5, epochs=2000,
         cites=[A13, S2, S3, S4],
         was_in="w9_swin_5fold.ipynb",
         note="the ~26% gradient window (27% on the fixed split)"),

    Cell("apx.twostage",
         # Stage 1 is the BYOL tower of body.families; stage 2 warm-
         # starts from it with --init-ckpt and runs the window for 600 ep.
         arms=["wcle_swin168step84loop2i2ce_icetf"],
         caps=[POOL], folds=5, epochs=600,
         cites=[A13, S2, S3, S4],
         was_in="w9_twostage_5fold.ipynb",
         note="two-stage: BYOL warm start -> windowed. The published runs "
              "started from BYOL epochs 450, 700, 450, 450 and 300 (folds 0 "
              "to 4), the picks of the BYOL runs' workers at the time. "
              "The notebook takes those checkpoints from a full BYOL run "
              "when one is there; otherwise it fetches the published files "
              "from Hugging Face, or trains the BYOL arm to those epochs "
              "itself",
         extra={"init_from": "wcle_byol_bytf", "name_suffix": "_bw"}),

    Cell("apx.swin.window",
         arms=["wcle_swin84step42loop2i2ce_icetf",
               "wcle_swin168step84loop2i2ce_icetf",
               "wcle_swin336step168loop2i2ce_icetf",
               "wcle_i2ce_icetf"],           # the fully coupled reference row
         caps=[4096], folds=FIXED, epochs=2000,
         cites=[WSWEEP, A9],
         was_in="w9_swin.ipynb (the three swin arms); the reference arm is the "
                "anchor-budget ladder tower of w9_jobs.FS_JOBS",
         note="the W = 84-336 window-size sweep. The appendix of September 2026 "
              "printed it as its Table A9; the paper has not printed it since "
              "2026-10-12, and its Table A9 is another table now. The "
              "sweep's table is regenerated from "
              "the four arms' projection caches by Pod/table_a9_recompute.py "
              "(rank selection of Section 5 over epochs 50..2000, Section 5 "
              "test-set tag probe on the fixed split). The shipped reference tower was "
              "trained 1,000 epochs and extended to 2,000 with a fresh optimizer "
              "state, so a fresh 2,000-epoch run reproduces the protocol, not "
              "the extension bit for bit. The only cell at 4,096 sentences "
              "(true 4,096-sentence packs, drawn from the full corpus by the "
              "fixed-split worker), so no profile selects it: ask for it by key",
         extra={"optional": True}),

    Cell("apx.docview",
         arms=["wcle_nodoc_i2ce_icetf"],
         caps=[512], folds=5, epochs=2000,
         cites=[A19],
         was_in="w9_docview_5fold.ipynb",
         note="no document view, paired per fold with the I-CE of "
              "body.pair.512, the raw-document row of Table A19"),

    Cell("apx.docview.llm",
         arms=["wcle_i2ce_icetf"],
         caps=[512], folds=5, epochs=2000,
         cites=[A19],
         was_in="w9_docview_5fold.ipynb",
         note="the LLM rewrite instead of raw wiki (Table A19, the "
              "LLM-rewritten row)",
         extra={"wiki_src": "llm"}),

    # ---- appendix: VICReg weight sweep --------------------------------
    Cell("apx.vicreg.sweep",
         arms=["wcle_epd_v25i25c1_cetf",
               "wcle_epd_v20i10c15_cetf", "wcle_epdb_v25i25c1_cetf",
               "wcle_epdb_v20i10c20_cetf"],
         caps=[512], folds=FIXED, epochs=2000,
         cites=[A8],
         was_in="w9_jobs.FS_JOBS",
         note="v/i/c weights; the recipe-weights row of Table A8 (20/10/20) "
              "is the VICReg arm of body.neighborhood.views. epdb = the "
              "batch-all variant, which Table A8 leaves out. The published "
              "runs were trained 1,000 epochs and then on to 2,000 from the "
              "weights of epoch 1,000"),

    # ---- appendix: the DINO-style control -----------------------------
    Cell("apx.dino",
         arms=["wcle_dino_dntf"],
         caps=[POOL], folds=1, epochs=2000,
         cites=[A1],
         was_in="w9_dino_control.ipynb",
         note="the control of Section 4.3: DINO at its published settings, "
              "none swept, on fold 0 only (recipe in Table A10). The teacher's "
              "two global views hold up to 512 sentences each: both open "
              "with the store-page prefix, and the game's reviews are dealt "
              "whole to the first until no more fit, then from the rest to "
              "the second. The student's review views come from the full "
              "library. "
              "The published checkpoint is epoch 1,650, the rank rule's "
              "choice, and Table A1 gives its spread over the ten draws"),

    # ---- appendix: temperature ----------------------------------------
    Cell("apx.temperature",
         arms=["wcle_i2cet05_icetf", "wcle_i2cet10_icetf",
               "wcle_i2ce_icetl"],
         caps=[512, 2048], folds=5, epochs=2000,
         cites=[A3],
         was_in="w9_i2ce_t.ipynb",
         note="tau 0.05 / 0.10 / learnable; Table A3 tabulates learnable tau "
              "at 2,048 only, and its recipe rows are the I-CE of "
              "body.pair.512 and of apx.ladder.2048"),

    # ---- appendix: invariance weight, view count, latent queries ------
    # Folds 0 to 2 at 512 sentences. The recipe rows of Tables A4 to A7 and
    # the lambda = 0 row of Table A4 are the I-CE and CE towers of body.pair.512
    # on the same three folds, NOT more training runs.
    # Table A6, the size of a review view, has no cell: its three floors are the
    # five-fold worker run with --view-w 8, 32 and 64 (the recipe draws a review
    # view to at least 16 sentences) on folds 0 to 2 at 512 sentences, runs named
    # w9cv_wcle_i2ce_icetf_fold<k>_w<W>_d512_fp. No profile lists them and their
    # checkpoints are not on Hugging Face.
    Cell("apx.lambda",
         arms=["wcle_ice_icetf",            # lambda = 1
               "wcle_i3ce_icetf",           # lambda = 3
               "wcle_i4ce_icetf"],          # lambda = 4
         caps=[512], folds=3, epochs=2000,
         cites=[A4],
         was_in="w9_sweeps_3fold.ipynb",
         note="the invariance weight, folds 0 to 2"),

    Cell("apx.views",
         arms=["wcle_d1r2_i2ce_icetf",      # V = 3
               "wcle_d1r4_i2ce_icetf",      # V = 5
               "wcle_d1r5_i2ce_icetf",      # V = 6
               "wcle_d1r6_i2ce_icetf",      # V = 7
               "wcle_d1r5_i3ce_icetf"],     # V = 6 with lambda = 3
         caps=[512], folds=3, epochs=2000,
         cites=[A5],
         was_in="w9_sweeps_3fold.ipynb (w9_viewgrid.ipynb is the "
                "fixed-split grid)",
         note="the number of strong views, one document view and V - 1 "
              "review views, folds 0 to 2"),

    Cell("apx.queries",
         arms=["wcle_slot1i2cemean_icetf", "wcle_slot2i2cemean_icetf",
               "wcle_slot8i2cemean_icetf", "wcle_slot16i2cemean_icetf"],
         caps=[512], folds=3, epochs=2000,
         cites=[A7],
         was_in="w9_sweeps_3fold.ipynb (w9_scale_query.ipynb is the "
                "fixed-split grid)",
         note="the number of latent queries, Q = 1, 2, 8 and 16, folds 0 to 2"),

    Cell("apx.queries.5fold",
         arms=["wcle_slot8i2cemean_icetf"],
         caps=[POOL], folds=5, epochs=2000,
         cites=[A7],
         was_in="w9_experiment_5fold_2.ipynb",
         note="Q = 8 on the five folds, seed-paired with the I-CE of "
              "body.pair (the five-fold rows of Table A7)"),

    # ---- appendix: the teacher must exclude the student's views -------
    Cell("apx.vfa",
         arms=["wcle_vfai2ce_icetf"],
         caps=[512], folds=5, epochs=2000,
         cites=[S1],
         was_in="w9_vfa.ipynb (arm wcle_vfai2ce_icetf; vfa_packs in the "
                "worker builds the views-first anchors)",
         note="views-first anchors: the teacher becomes a superset of "
              "the student's evidence"),
]

CELLS_BY_KEY = {c.key: c for c in CELLS}

# ------------------------------------------------------------ profiles --
# dataset_builder/profiles.py is the single source of truth: it is what
# the packaged path reads to decide the budget it BUILDS the anchor packs
# at, and the two must agree or a litePaperTest run would train against
# 2,048-sentence packs. When w9/ has been copied to a pod on its own the
# repository is not importable, so fall back to the same literals; the
# self-check below fires whenever both are present.
try:
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.abspath(
        _os.path.join(_os.path.dirname(__file__), "..", "..", "..")))
    from dataset_builder.profiles import LITE_CAP, PROFILES as _P
    assert _P == ("fullTest", "paperTest", "litePaperTest"), _P
except Exception:                       # standalone w9/ copy
    LITE_CAP = 1024


def cells_for(profile):
    """The cells a profile runs, in grid order."""
    if profile == "fullTest":
        return [c for c in CELLS if not c.extra.get("optional")]
    if profile in ("paperTest", "litePaperTest"):
        return [c for c in CELLS if c.in_body and not c.extra.get("optional")]
    raise ValueError(f"unknown profile {profile!r}; "
                     f"expected one of {', '.join(PROFILES)}")


def caps_for(cell, profile):
    """The anchor budgets a profile runs a cell at.

    litePaperTest clamps to LITE_CAP rather than replacing: a 512 cell
    stays 512, and duplicates collapse (a ladder of 512/1024/2048 becomes
    512/1024, not 512/1024/1024). Whole-pool packs are above the clamp, so
    they become LITE_CAP too.
    """
    if profile != "litePaperTest":
        return list(cell.caps)
    seen, out = set(), []
    for c in cell.caps:
        c = LITE_CAP if c == POOL else min(c, LITE_CAP)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def plan(profile):
    """[(cell, caps, tower_count)] plus the total, for a profile."""
    rows = []
    for c in cells_for(profile):
        caps = caps_for(c, profile)
        rows.append((c, caps, c.towers(caps)))
    return rows, sum(r[2] for r in rows)


PROFILES = ("fullTest", "paperTest", "litePaperTest")


def summary():
    for p in PROFILES:
        rows, total = plan(p)
        print(f"{p:15s} {len(rows):2d} cells, {total:3d} towers")


if __name__ == "__main__":
    for p in PROFILES:
        rows, total = plan(p)
        print(f"===== {p}: {len(rows)} cells, {total} towers =====")
        for c, caps, n in rows:
            tag = "body" if c.in_body else "apx "
            fold = ("fixed" if c.folds == FIXED else "fold0" if c.folds == 1
                    else f"{c.folds}fold")
            print(f"  [{tag}] {c.key:22s} {len(c.arms)}arm "
                  f"{str(caps):18s} {fold:6s} {c.epochs:>4}ep  {n:>3} towers"
                  f"   {', '.join(c.cites[:3])}")
