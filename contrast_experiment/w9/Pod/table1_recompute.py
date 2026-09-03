# -*- coding: utf-8 -*-
r"""Recompute the paper's Table 1 (query displacement against nearest-neighbour
margin) and the encodings behind Figure 5 (the UMAP panels) for the fold-0
towers at the 4,096-sentence anchor budget, under the paper's checkpoint
selection rule.

What Table 1 measures (Section 6.1)
-----------------------------------
For one encoder, with L2-normalised vectors:
  displacement  mean over the 814 name-stripped wiki rewrites of the fold-0
                universe (train, validation and test games alike) of
                1 - cos(query, own game's anchor);
  margin        mean over the 2,020 anchors of 1 - cos(anchor, nearest OTHER
                anchor);
  ratio         displacement / margin (means first, then the ratio);
  inside        share of the 814 queries whose displacement is below their
                own game's margin (Figure 5's "inside own clearance" labels).
The frozen embedder is the masked mean of the sentence embeddings (the
worker's frozen baseline); the trained towers are SetPoolN(4) checkpoints
applied to the same single 4,096-sentence anchor draw (wscan_gal_rev_g4096)
and to the query sentences.

Checkpoint selection
--------------------
--plan rank   (default) the paper's rule, Section 5: the epoch that maximises
              sum_i exp(-rank_i) over the validation fold's rewrites of both
              registers, read from data/w9/rank_ckpt_plan.json (the plan every
              five-fold table uses).
--plan cvsel  the five-fold worker's online score (v_non + v_non5 + 2*val_tag)
              that the pre-2026-09-04 Table 1 used; the plan ships as
              w9_fig3_local/ckpt_plan.json. Reproduces the previously shipped
              fig3_t3_frozen_results.json bit for bit and is the validation
              gate of this script.

Inputs (research artefacts, not shipped with the repository)
------------------------------------------------------------
  W9_FIG3   C:\runpod_data\w9_fig3_local   wscan_gal_rev_g4096.npz (the anchor
            draw, fp16, row-normalised), ckpt/ckpt_w9cv_wcle_<arm>_fold0_g4096_fp_ep<N>.pt
  W9_CACHE  C:\runpod_data\fusion_cache_w9 games.npz, wiki_eval.npz,
            tag_labels.npz, _tag_splitM.json, wiki_eval_split.json
  W9_PLAN   the rank plan JSON (default: data/w9/rank_ckpt_plan.json of the
            paper repository, or Pod/rank_ckpt_plan.json next to this file)
Outputs (W9_OUT, default W9_FIG3)
  table1_results.json   per-encoder displacement / margin / ratio / inside,
                        the epochs used and the selection provenance
  fig5_encodings.npz    Zg_<enc> (2020 x d) anchors and Za_<enc> (3256 x d)
                        wiki rewrites, plus gA / variants, for figures/fig5_umap.py
Run with a CUDA interpreter (the gallery is 16.9 GB fp16 on the CPU, anchors
are chunked to the GPU):
    python contrast_experiment/w9/Pod/table1_recompute.py --plan rank
    python contrast_experiment/w9/Pod/table1_recompute.py --plan cvsel --check
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
W9 = HERE.parent
sys.path.insert(0, str(W9))
from Pod.w9_cv_worker import SetPoolN, rown                       # noqa: E402
from VICReg_review.text_variant_eval import (                     # noqa: E402
    train_anchor_ridge, make_or_load_split, micro_prf)

FIG3 = Path(os.environ.get("W9_FIG3", r"C:\runpod_data\w9_fig3_local"))
CACHE = Path(os.environ.get("W9_CACHE", r"C:\runpod_data\fusion_cache_w9"))
OUT = Path(os.environ.get("W9_OUT", str(FIG3)))
CKDIR = FIG3 / "ckpt"
GALPACK = FIG3 / "wscan_gal_rev_g4096.npz"

# encoder label in the paper -> arm label of the worker
ARMS = {
    "I-CE": "i2ce_icetf",
    "CE": "ce_cetf",
    "SimCLR": "bce_cetf",
    "VICReg": "epd_v20i10c20_cetf",
    "BYOL": "byol_bytf",
}
CAP = 4096
FOLD = 0
CV_SEED = 20260711


def default_plan(mode):
    if mode == "cvsel":
        return FIG3 / "ckpt_plan.json"
    cands = []
    if os.environ.get("W9_PLAN"):
        cands.append(Path(os.environ["W9_PLAN"]))
    cands += [HERE / "rank_ckpt_plan.json",
              Path(r"C:\Users\admin\Documents\Aiccc_paper\data\w9\rank_ckpt_plan.json")]
    for cand in cands:
        if cand.is_file():
            return cand
    raise SystemExit("rank plan not found; set W9_PLAN")


def epochs_from_plan(mode, plan_path):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    eps = {}
    for lab, arm in ARMS.items():
        if mode == "cvsel":
            key = f"{arm}_fold{FOLD}"                 # w9_fig3_local/ckpt_plan.json
        else:
            key = f"{arm}_fold{FOLD}_g{CAP}"          # rank_ckpt_plan.json
        v = plan[key]
        eps[lab] = int(v["ep"] if isinstance(v, dict) else v)
    return eps


def ckpt_path(arm, ep):
    return CKDIR / f"ckpt_w9cv_wcle_{arm}_fold{FOLD}_g{CAP}_fp_ep{ep}.pt"


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def geometry(Zg, Za, gA, variants):
    gz, az = l2(Zg), l2(Za)
    noname = np.array([i for i in range(len(variants)) if variants[i] == "noname"])
    qg = gA[noname]
    disp_per = 1.0 - (az[noname] * gz[qg]).sum(1)
    sim = gz @ gz.T
    np.fill_diagonal(sim, -2.0)
    margin_per = 1.0 - sim.max(1)
    return dict(displacement=float(disp_per.mean()),
                margin=float(margin_per.mean()),
                ratio=float(disp_per.mean() / margin_per.mean()),
                inside_margin=float((disp_per < margin_per[qg]).mean()),
                n_queries=int(len(noname)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=("rank", "cvsel"), default="rank")
    ap.add_argument("--plan-file", default=None)
    ap.add_argument("--check", action="store_true",
                    help="compare against the shipped fig3_t3_frozen_results.json")
    ap.add_argument("--frozen-tag", action="store_true",
                    help="also recompute the frozen embedder's five-fold test tag F1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    plan_path = Path(a.plan_file) if a.plan_file else default_plan(a.plan)
    eps = epochs_from_plan(a.plan, plan_path)
    print(f"plan {a.plan} <- {plan_path}")
    for lab, ep in eps.items():
        p = ckpt_path(ARMS[lab], ep)
        print(f"  {lab:7s} ep{ep:5d}  {p.name}  {'ok' if p.exists() else 'MISSING'}")
        if not p.exists():
            raise SystemExit(f"checkpoint missing: {p}")

    dev = torch.device("cuda")
    t0 = time.time()
    names = [str(x) for x in np.load(CACHE / "games.npz", allow_pickle=True)["names"]]
    NG = len(names)
    assert NG == 2020, NG
    A = np.load(CACHE / "wiki_eval.npz", allow_pickle=True)
    SA = rown(torch.tensor(A["S"]).to(dev).float()).half()
    mA = (torch.arange(SA.shape[1], device=dev)[None, :]
          >= torch.tensor(A["S_len"]).to(dev)[:, None])
    variants = [str(x) for x in A["variants"]]
    gA = np.asarray(A["gidx"])
    G = np.load(GALPACK)
    SGal = torch.from_numpy(G["gal"])                       # CPU fp16, already rown
    gal_len = torch.tensor(np.asarray(G["gal_len"], np.int64))
    mGal = (torch.arange(SGal.shape[1])[None, :] >= gal_len[:, None])
    print(f"gallery {tuple(SGal.shape)} on CPU; queries {tuple(SA.shape)} on GPU "
          f"[{time.time()-t0:.0f}s]", flush=True)

    @torch.no_grad()
    def encode_gallery(model, chunk=64):
        outs = []
        for i in range(0, NG, chunk):
            s = SGal[i:i + chunk].to(dev, non_blocking=True)
            m = mGal[i:i + chunk].to(dev, non_blocking=True)
            outs.append(model(s, m).float().cpu())
        return torch.cat(outs).numpy()

    @torch.no_grad()
    def encode_queries(model, chunk=256):
        outs = []
        for i in range(0, SA.shape[0], chunk):
            outs.append(model(SA[i:i + chunk], mA[i:i + chunk]).float().cpu())
        return torch.cat(outs).numpy()

    @torch.no_grad()
    def frozen_vectors(chunk=64):
        Zg = np.zeros((NG, SGal.shape[2]), np.float32)
        for i in range(0, NG, chunk):
            s = SGal[i:i + chunk].to(dev).float()
            w = (~mGal[i:i + chunk].to(dev)).float().unsqueeze(-1)
            Zg[i:i + chunk] = ((s * w).sum(1) / w.sum(1).clamp(min=1)).cpu().numpy()
        w = (~mA).float().unsqueeze(-1)
        Za = ((SA.float() * w).sum(1) / w.sum(1).clamp(min=1)).cpu().numpy()
        return Zg, Za

    def load_tower(path):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
        m = SetPoolN(4).to(dev).eval()
        m.load_state_dict(dict(sd.items()))
        return m

    enc = {}
    print("encode Frozen ...", flush=True)
    enc["Frozen"] = frozen_vectors()
    for lab, arm in ARMS.items():
        t1 = time.time()
        m = load_tower(ckpt_path(arm, eps[lab]))
        enc[lab] = (encode_gallery(m), encode_queries(m))
        del m
        torch.cuda.empty_cache()
        print(f"encode {lab:7s} ep{eps[lab]:5d} done [{time.time()-t1:.0f}s]", flush=True)

    t3 = {lab: geometry(Zg, Za, gA, variants) for lab, (Zg, Za) in enc.items()}
    for lab in t3:
        t3[lab]["epoch"] = None if lab == "Frozen" else eps[lab]
        t3[lab]["arm"] = None if lab == "Frozen" else ARMS[lab]
    print(f"\n=== Table 1 ({a.plan} fold-{FOLD} @{CAP}) ===")
    for lab in ("Frozen", "BYOL", "VICReg", "CE", "SimCLR", "I-CE"):
        r = t3[lab]
        print(f"{lab:7s} ep {str(r['epoch']):>5}  disp {r['displacement']:.4f}  "
              f"margin {r['margin']:.4f}  ratio {r['ratio']:.3f}  "
              f"inside {100*r['inside_margin']:.1f}%")

    res = {"meta": {
        "table": "Table 1 (displacement / margin / ratio) and Figure 5 clearance labels",
        "selection_mode": a.plan,
        "selection": ("rank criterion of Section 5: argmax over epochs 50..2000 of "
                      "sum_i exp(-rank_i) over the validation fold's wiki rewrites of "
                      "both registers against all 2,020 anchors; plan = " + str(plan_path))
                     if a.plan == "rank" else
                     ("five-fold worker online score v_non + v_non5 + 2*val_tag "
                      "(cvsel); plan = " + str(plan_path)),
        "fold": FOLD, "anchor_budget": CAP, "anchor_draw": GALPACK.name,
        "queries": "all name-stripped (noname) wiki rewrites of the 814-game universe, "
                   "train / validation / test alike",
        "gallery": "all 2,020 anchors", "generated": time.strftime("%Y-%m-%d %H:%M"),
        "script": str(Path(__file__).resolve()),
        "cache": str(CACHE), "fig3_dir": str(FIG3)},
        "t3": t3}

    if a.frozen_tag:
        n2i = {n: i for i, n in enumerate(names)}
        appid2name = {n.split("_")[0]: n for n in names}
        y = np.load(CACHE / "tag_labels.npz", allow_pickle=True)["y"]
        targs = SimpleNamespace(tag_text_train_frac=0.7, tag_text_val_frac=0.15,
                                tag_text_split_seed=42, seed=42,
                                tag_text_threshold_steps=33)
        make_or_load_split(CACHE / "_tag_splitM.json", names, targs)
        sp = json.loads((CACHE / "wiki_eval_split.json").read_text())
        universe = sorted(set(sp["test"]) | set(sp["val"]) | set(sp["train"]))
        perm = np.random.default_rng(CV_SEED).permutation(len(universe))
        folds = np.array_split(perm, 5)
        art_games = [str(x) for x in A["names"]]
        Zg0, Za0 = enc["Frozen"]
        tt = []
        for k in range(5):
            te = {universe[i] for i in folds[k]}
            va = {universe[i] for i in folds[(k + 1) % 5]}
            test_g = {appid2name[x] for x in te}
            val_g = {appid2name[x] for x in va}
            excl = test_g | val_g
            ctag = {"train": [names[i] for i in range(NG) if names[i] not in excl],
                    "val": sorted(val_g), "test": sorted(test_g)}
            sc, rg, _, th, _ = train_anchor_ridge(targs, Zg0, y, n2i, ctag)
            ti = [i for i in range(len(art_games))
                  if art_games[i] in test_g and variants[i] == "neutral"]
            s = rg.predict(sc.transform(np.stack([Za0[i] for i in ti]).astype(np.float32)))
            lab = np.stack([y[n2i[art_games[i]]] for i in ti])
            tt.append(float(micro_prf(lab, s, th)["micro_f1"]))
        res["frozen_test_tag"] = tt
        res["frozen_test_tag_mean"] = float(np.mean(tt))
        res["frozen_test_tag_std"] = float(np.std(tt))
        print(f"frozen test tag five-fold {np.round(tt, 3).tolist()} "
              f"mean {np.mean(tt):.3f} ± {np.std(tt):.3f}")

    tag = "" if a.plan == "rank" else f"_{a.plan}"
    jp = out / f"table1_results{tag}.json"
    jp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    npz = out / f"fig5_encodings{tag}.npz"
    np.savez(npz, **{f"Zg_{k}": v[0] for k, v in enc.items()},
             **{f"Za_{k}": v[1] for k, v in enc.items()},
             gA=gA, variants=np.array(variants))
    print(f"\nwrote {jp}\nwrote {npz}  [{time.time()-t0:.0f}s total]")

    if a.check:
        ref = json.loads((FIG3 / "fig3_t3_frozen_results.json").read_text())["t3"]
        worst = 0.0
        for lab, r in ref.items():
            for k in ("displacement", "margin", "ratio", "inside_margin"):
                d = abs(r[k] - t3[lab][k]); worst = max(worst, d)
                flag = "" if d < 1e-4 else "   <-- differs"
                print(f"check {lab:7s} {k:13s} shipped {r[k]:.5f} now {t3[lab][k]:.5f}{flag}")
        print("max |diff| =", f"{worst:.2e}", "PASS" if worst < 1e-4 else "FAIL")


if __name__ == "__main__":
    main()
