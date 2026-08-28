# -*- coding: utf-8 -*-
"""Path 1 entry point: reproduce the manuscript's I-CE tower (ungated CE
over every game in the step, I x2, frozen tau 0.02, wiki_clean > sp_raw
doc views, I-CE head). This is the w9 campaign's `wcle_i2ce_icetf` arm.

    python steam_reviews_framework/train_champion.py [--epochs 2000] [--cv-fold K]

`--arm cegate2` switches to the CE-gated ablation instead, where CE fires
only on games carrying a document view. That arm was this script's default
until it was made explicit; it is not what the manuscript reports. The
README's Protocol table lists every remaining packaged-vs-manuscript gap.

Prerequisite: data assets built (see dataset_builder/rebuild_data.py) or
linked via LARICE_* environment variables (dataset_builder/paths.py).
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from steam_reviews_framework.data import load_bundle
from steam_reviews_framework.train import ARM_CHOICES, run_arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARM_CHOICES), default="i2ce",
                    help="i2ce = the manuscript's objective (default); "
                         "cegate2 = the CE-gated ablation")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--cv-fold", type=int, default=None,
                    help="0..4: run on a CV fold instead of the fixed split")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    spec = replace(ARM_CHOICES[args.arm], epochs=args.epochs,
                   ckpt_every=args.ckpt_every)
    if args.cv_fold is not None:
        spec.name = f"{spec.name}_fold{args.cv_fold}"
    print(f"arm: {spec.name} | tower={spec.tower} "
          f"({'CE gated on doc views' if spec.tower == 'cegate' else 'CE ungated'})"
          f" | I x{spec.inv_weight}", flush=True)
    B = load_bundle(torch.device(args.device), cv_fold=args.cv_fold)
    print(f"bundle: {B.NG} games | test {len(B.test_g)} val {len(B.val_g)} "
          f"| train pool {len(B.train_pool_games)}", flush=True)
    run_arm(B, spec, log_cb=lambda *a: print(*a, flush=True))


if __name__ == "__main__":
    main()
