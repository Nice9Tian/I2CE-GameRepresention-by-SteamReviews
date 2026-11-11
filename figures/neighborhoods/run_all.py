# -*- coding: utf-8 -*-
r"""Figures 8, 9 and A3 and the neighborhood statistics of Appendix L, in one command.

    python figures/neighborhoods/run_all.py --ckpt-dir <checkpoints> --data-dir <data folder>

From the published checkpoints and the data folder it runs, in order (each step is a script that can
also be run alone; a step whose output is already in the working folder is skipped unless --redo):

  1. the ten 4,096-sentence anchor draws, the heavy part (about 40 minutes of reading for each):
       fixed split   contrast_experiment/w9/eval/fixed_split_eval.py --only-draws
       five folds    contrast_experiment/w9/eval/k10_eval.py cells / encode, for the towers of Figure 8
     Give --fixed-draws / --cv-draws and --cv-cells to use files you already have.
  2. prep.py             the fixed-split queries, hits and anchors (GPU)
  3. cv_hoods.py         the 814 five-fold neighborhoods
  4. fig_genre_bars.py --cv     Figure 8
  5. fig_triplet.py fig9        Figure 9 at the full text width
     fig_triplet.py fig9col     Figure 9 as printed, one column wide
  6. fig_genre_bars.py          Figure A3
  7. fig_stats.py               the statistics Appendix L quotes (needs scipy)
  8. layout_check.py            with --layout-check (needs umap-learn)

Everything lands in --work (default ./hood_work): the figures in <work>/figures, the numbers in
<work>/fig_facts_4096k10.json and <work>/query_stats_4096k10.json.  --map stable draws Figure 9 with
the version-independent anchor map (see fig_triplet.py); the default is the published computation.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parents[1] / "contrast_experiment" / "w9" / "eval"
FIG8_CELLS = r"^(I-CE|CE|SimCLR|VICReg|BYOL)\||i2bce"       # the five-fold towers Figure 8 reads (BYOL rides along in the table)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--data-dir", default=os.environ.get("W9_DATA_DIR", ""))
    ap.add_argument("--work", default="hood_work")
    ap.add_argument("--out", default="", help="where the figures go (default <work>/figures)")
    ap.add_argument("--fixed-draws", default="", help="draws_fixed.npz of fixed_split_eval.py")
    ap.add_argument("--cv-draws", default="", help="the five-fold file of `k10_eval.py encode`")
    ap.add_argument("--cv-cells", default="", help="the cells file that encode was given")
    ap.add_argument("--map", choices=("drawn", "stable"), default="drawn")
    ap.add_argument("--layout-check", action="store_true")
    ap.add_argument("--redo", action="store_true", help="run every step again")
    a = ap.parse_args()
    assert a.data_dir and (Path(a.data_dir) / "games.npz").exists(), "--data-dir <data folder> (or W9_DATA_DIR) is required"
    work = Path(a.work).resolve(); work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HOOD_WORK=str(work), W9_DATA_DIR=str(Path(a.data_dir).resolve()), HOOD_MAP=a.map)
    if a.out:
        env["HOOD_OUT"] = str(Path(a.out).resolve())
    py = sys.executable

    def run(what, cmd, done=None):
        if done is not None and Path(done).exists() and not a.redo:
            print(f"[skip] {what}: {done} is there"); return
        print(f"[run ] {what}\n       $ " + " ".join(str(c) for c in cmd[1:]), flush=True)
        subprocess.run([str(c) for c in cmd], check=True, env=env)

    fixed = Path(a.fixed_draws) if a.fixed_draws else work / "fixed_split_eval" / "draws_fixed.npz"
    run("ten draws, fixed split", [py, EVAL / "fixed_split_eval.py", "--ckpt-dir", a.ckpt_dir, "--data-dir", a.data_dir,
                                   "--out-dir", fixed.parent, "--only-draws"], done=fixed)
    cvd = Path(a.cv_draws) if a.cv_draws else work / "draws_fig8.npz"
    cvc = Path(a.cv_cells) if a.cv_cells else work / "cells_fig8.json"
    if not cvd.exists() or a.redo:
        run("cells of Figure 8", [py, EVAL / "k10_eval.py", "cells", cvc, "--ckpt-dir", a.ckpt_dir, "--only", FIG8_CELLS])
        run("ten draws, five folds", [py, EVAL / "k10_eval.py", "encode", cvc, cvd, "--data-dir", a.data_dir])
    else:
        print(f"[skip] ten draws, five folds: {cvd} is there")
    run("fixed-split vectors", [py, HERE / "prep.py", "--ckpt-dir", a.ckpt_dir, "--draws", fixed], done=work / "fixed" / "allgames.npz")
    run("five-fold neighborhoods", [py, HERE / "cv_hoods.py", "--draws", cvd, "--cells", cvc], done=work / "cv" / "cv_hoods_4096.json")
    run("Figure 8", [py, HERE / "fig_genre_bars.py", "--cv"])
    run("Figure 9, full width", [py, HERE / "fig_triplet.py", "fig9"])
    run("Figure 9, one column", [py, HERE / "fig_triplet.py", "fig9col"])
    run("Figure A3", [py, HERE / "fig_genre_bars.py"])
    run("Appendix L statistics", [py, HERE / "fig_stats.py"])
    if a.layout_check:
        run("layout check", [py, HERE / "layout_check.py"])
    print("\nfigures:", env.get("HOOD_OUT", str(work / "figures")), "\nnumbers:", work / "fig_facts_4096k10.json", "and", work / "query_stats_4096k10.json")


if __name__ == "__main__":
    main()
