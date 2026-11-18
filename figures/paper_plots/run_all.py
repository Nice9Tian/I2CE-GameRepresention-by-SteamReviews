# -*- coding: utf-8 -*-
"""Figures 1, 6 and 7, and Figure A2 when its encodings are given, in one command.

    python figures/paper_plots/run_all.py [--out DIR]
    python figures/paper_plots/run_all.py --enc <dir>/fig5_encodings.npz --data-dir <data folder>

Figures 1, 6 and 7 need only matplotlib and the files in data/.  Figure A2 needs the encodings
that contrast_experiment/w9/Pod/table1_recompute.py writes (CUDA, the published checkpoints and the
data folder), the data folder itself, and umap-learn and scipy.  Every figure lands in --out
(default ./paper_plots_out).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="paper_plots_out")
    ap.add_argument("--enc", default="", help="fig5_encodings.npz; Figure A2 is skipped without it")
    ap.add_argument("--data-dir", default=os.environ.get("W9_DATA_DIR", ""))
    a = ap.parse_args()
    out = Path(a.out).resolve()
    py = sys.executable
    steps = [("Figure 1", [HERE / "fig1_composition.py", "--out", out]),
             ("Figure 6", [HERE / "fig6_envelopes.py", "--out", out / "fig6_envelopes.png"]),
             ("Figure 7", [HERE / "fig7_objective_bars.py", "--out", out])]
    if a.enc:
        steps.append(("Figure A2", [HERE / "figA2_umap.py", "--enc", a.enc, "--data-dir", a.data_dir,
                                    "--out", out / "figA2_umap.png"]))
    else:
        print("[skip] Figure A2: pass --enc <fig5_encodings.npz> and --data-dir")
    for what, cmd in steps:
        print("[run ] %s" % what, flush=True)
        subprocess.run([py] + [str(c) for c in cmd], check=True)
    print("\nfigures:", out)


if __name__ == "__main__":
    main()
