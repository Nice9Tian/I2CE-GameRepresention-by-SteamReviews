# -*- coding: utf-8 -*-
"""The reproduction grid, one row per experiment cell, with provenance.

Before this file the grid was scattered: `w9_jobs.FS_JOBS` held the
fixed-split research log (106 rows, most of them exploratory), while the
cells that actually back the manuscript's figures and tables lived as
constants inside twelve notebooks. Nothing said which run produced which
number. This module is that missing statement.

Every CELL below carries `cites`: the figures and tables it feeds. A cell
with a body citation (FIG*, TAB1, TAB2, SEC*) is reported in the paper
itself; the rest are appendix-only. Nothing is listed here that the paper
or the appendix does not mention, so the exploratory families -- twin-pack
`pk*`, the slot/readout capacity grid, multi-anchor `ma2*`, the async
simulation, hard-negative `nemesis` -- are deliberately absent. They stay
available through `w9_jobs.FS_JOBS` and the notebooks.

Three profiles select from it:

    fullTest        every cell (paper + appendix)
    paperTest       cells with a body citation, at their own budgets
    litePaperTest   paperTest with every anchor budget clamped to 1024

litePaperTest is the desktop-GPU profile. The clamp is a ceiling, not a
setting: a cell the paper ran at 512 stays at 512, because rerunning it at
1024 would not reproduce the published row. Only the 2,048 and 4,096 cells
move, and they move to a budget the paper itself measures as within 0.02
of the full configuration on every reading (Section 4.2, Table A2).

This module is pure data plus three small functions. It imports nothing
outside the standard library, so `w9/` stays copyable to a pod on its own.
"""

# ---------------------------------------------------------------- cites --
# Body of the manuscript.
FIG4 = "Figure 4"          # the objective families
FIG5 = "Figure 5"          # UMAP of anchors and queries
TAB1 = "Table 1"           # displacement / margin / ratio
TAB2 = "Table 2"           # the decomposition ablation
SEC61 = "Section 6.1"      # the headline five-fold reading
SEC62 = "Section 6.2"      # the decomposition sweep
SEC32 = "Section 3.2"      # the momentum-teacher / memory-bank comparison

BODY = {FIG4, FIG5, TAB1, TAB2, SEC61, SEC62, SEC32}

# Appendix.
A1 = "Table A1"; A2 = "Table A2"; A3 = "Table A3"; A4 = "Table A4"
A5 = "Table A5"; A6 = "Table A6"; A7 = "Table A7"; A8 = "Table A8"
A9 = "Table A9"; A10 = "Table A10"; A11 = "Table A11"; A12 = "Table A12"
A13 = "Table A13"; A14 = "Table A14"; A15 = "Table A15"

FIXED = 0                  # folds=FIXED means the fixed 204/203/407 split


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

    # ---- body: the shared five-fold @4096 tower set -------------------
    # Trained once, read four ways. A6/A13/A14/A15 are different readouts
    # of these same towers, NOT four more training runs -- which is why
    # the appendix has fifteen tables but the grid stays small.
    Cell("body.pair.4096",
         arms=["wcle_i2ce_icetf", "wcle_ce_cetf"],
         caps=[4096], folds=5, epochs=2000,
         cites=[FIG4, FIG5, TAB1, SEC61, A1, A2, A3, A13, A14, A15],
         was_in="w9_final_experiment.ipynb",
         note="the headline I-CE vs CE pair, paired per fold"),

    Cell("body.families.4096",
         arms=["wcle_bce_cetf",             # SimCLR-style, in-batch views
               "wcle_byol_bytf",            # BYOL
               "wcle_epdb_v20i10c20_cetf",  # VICReg, expander 20/10/20
               "wcle_mq3072i2ce_icetf"],    # EMA teacher + 3,072-key bank
         caps=[4096], folds=5, epochs=2000,
         cites=[FIG4, FIG5, TAB1, SEC32, A3, A13, A14, A15],
         was_in="w9_experiment_5fold_2.ipynb",
         note="the other objective families of Figure 4"),

    # ---- body: the decomposition ablation, fixed split @512 -----------
    Cell("body.decomp.rows",
         arms=["wcle_i2ce_icetf", "wcle_ce_cetf",
               "wcle_ai2ce_icetf"],         # CE + anchor-in-I
         caps=[512], folds=FIXED, epochs=2000,
         cites=[TAB2, SEC62],
         was_in="w9_jobs.FS_JOBS + w9_flash.ipynb",
         note="the first three rows of Table 2"),

    Cell("body.decomp.sweep",
         # Naming: ai<N> = anchor-aimed align weight N, auni<M> = anchor
         # uniformity M, i<N>uni<M> = the view-aimed pair. Table 2's
         # 12.5:1 row is ai25auni2 (25:2); its 2:1 row is i4uni2 (4:2).
         # (wcle_i2au2 / wcle_i2au25 belong to the same family but exist
         # only in the five-fold worker, so they are not listed here.)
         arms=["wcle_ai25auni2_icetf", "wcle_ai6auni2_icetf",
               "wcle_ai4auni2_icetf", "wcle_ai2auni25_icetf",
               "wcle_i4uni2_icetf", "wcle_i6uni2_icetf",
               "wcle_ai6uni2_icetf"],
         caps=[512], folds=FIXED, epochs=2000,
         cites=[TAB2, SEC62],
         was_in="w9_flash.ipynb",
         note="repulsion target x align:uniformity ratio"),

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
         note="'detailed settings in the repository' of Section 6.2"),

    # ---- appendix: the anchor-budget ladder ---------------------------
    Cell("apx.ladder",
         arms=["wcle_i2ce_icetf", "wcle_ce_cetf"],
         caps=[512, 1024, 2048], folds=5, epochs=2000,
         cites=[A2],
         was_in="w9_final_experiment.ipynb",
         note="the 4,096 rung is body.pair.4096"),

    # ---- appendix: anchor-supply economies ----------------------------
    Cell("apx.swin.5fold",
         arms=["wcle_swin168step84loop2i2ce_icetf"],
         caps=[4096], folds=5, epochs=2000,
         cites=[A6, A13, A14, A15],
         was_in="w9_swin_5fold.ipynb",
         note="the ~27% gradient window"),

    Cell("apx.twostage",
         # Stage 1 is the BYOL tower of body.families.4096; stage 2 warm-
         # starts from it with --init-ckpt and runs the window for 600 ep.
         arms=["wcle_swin168step84loop2i2ce_icetf"],
         caps=[4096], folds=5, epochs=600,
         cites=[A6, A13, A15],
         was_in="(not shipped before; --init-ckpt was stripped)",
         note="two-stage: BYOL warm start -> windowed",
         extra={"init_from": "wcle_byol_bytf", "name_suffix": "_bw"}),

    Cell("apx.swin.window",
         arms=["wcle_swin84step42loop2i2ce_icetf",
               "wcle_swin168step84loop2i2ce_icetf",
               "wcle_swin336step168loop2i2ce_icetf"],
         caps=[4096], folds=FIXED, epochs=2000,
         cites=[A8, A9],
         was_in="w9_swin.ipynb",
         note="the W = 84-336 window-size sweep"),

    # ---- appendix: the document view ----------------------------------
    Cell("apx.docview",
         arms=["wcle_i2ce_icetf", "wcle_nodoc_i2ce_icetf"],
         caps=[512], folds=5, epochs=2000,
         cites=[A7],
         was_in="w9_jobs.FS_JOBS (fixed split only; the five-fold cells "
                "had no job definition)",
         note="document view present vs absent"),

    Cell("apx.docview.raw",
         arms=["wcle_i2ce_icetf"],
         caps=[512], folds=5, epochs=2000,
         cites=[A7],
         was_in="w9_jobs.FS_JOBS (fixed split only)",
         note="raw wiki instead of the LLM rewrite",
         extra={"wiki_src": "llm"}),

    # ---- appendix: VICReg weight sweep --------------------------------
    Cell("apx.vicreg.sweep",
         arms=["wcle_epd_v25i25c1_cetf", "wcle_epd_v20i10c20_cetf",
               "wcle_epd_v20i10c15_cetf", "wcle_epdb_v25i25c1_cetf",
               "wcle_epdb_v20i10c20_cetf"],
         caps=[512], folds=FIXED, epochs=2000,
         cites=[A10],
         was_in="w9_jobs.FS_JOBS",
         note="v/i/c weights; epdb = batch-all variant"),

    # ---- appendix: temperature ----------------------------------------
    Cell("apx.temperature",
         arms=["wcle_i2cet05_icetf", "wcle_i2cet10_icetf",
               "wcle_i2ce_icetl"],
         caps=[512, 2048], folds=5, epochs=2000,
         cites=[A11],
         was_in="w9_i2ce_t.ipynb",
         note="tau 0.05 / 0.10 / learnable; 4,096 was shelved on cost"),

    # ---- appendix: the teacher must exclude the student's views -------
    Cell("apx.vfa",
         arms=["wcle_vfai2ce_icetf"],
         caps=[512], folds=5, epochs=2000,
         cites=[A12],
         was_in="w9_vfa.ipynb (the arm itself was stripped from the "
                "released CV worker)",
         note="views-first anchors: the teacher becomes a superset of "
              "the student's evidence"),
]

CELLS_BY_KEY = {c.key: c for c in CELLS}

# ------------------------------------------------------------ profiles --
LITE_CAP = 1024


def cells_for(profile):
    """The cells a profile runs, in grid order."""
    if profile == "fullTest":
        return list(CELLS)
    if profile in ("paperTest", "litePaperTest"):
        return [c for c in CELLS if c.in_body]
    raise ValueError(f"unknown profile {profile!r}; "
                     f"expected one of {', '.join(PROFILES)}")


def caps_for(cell, profile):
    """The anchor budgets a profile runs a cell at.

    litePaperTest clamps to LITE_CAP rather than replacing: a 512 cell
    stays 512, and duplicates collapse (a ladder of 512/1024/2048 becomes
    512/1024, not 512/1024/1024).
    """
    if profile != "litePaperTest":
        return list(cell.caps)
    seen, out = set(), []
    for c in cell.caps:
        c = min(c, LITE_CAP)
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
            fold = "fixed" if c.folds == FIXED else f"{c.folds}fold"
            print(f"  [{tag}] {c.key:22s} {len(c.arms)}arm "
                  f"{str(caps):18s} {fold:6s} {c.epochs:>4}ep  {n:>3} towers"
                  f"   {', '.join(c.cites[:3])}")
