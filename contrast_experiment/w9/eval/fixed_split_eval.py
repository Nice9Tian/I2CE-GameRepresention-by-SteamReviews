# -*- coding: utf-8 -*-
r"""The fixed-split towers read as the paper's fixed-split tables and figures read them.

One command, from the published checkpoints (chironYu/latenarray-i2ce-checkpoints) and the data
folder (chironYu/latenarray-i2ce-data):

    python contrast_experiment/w9/eval/fixed_split_eval.py --ckpt-dir <checkpoints> --data-dir <data folder>

It takes every run of fixed_split_ckpt_plan.json at its selected checkpoint and prints, each
beside the number the paper prints,

  Table 3        the decomposition ablation (five rows)
  Section 6.2    the four settings that aim the repulsion at the anchor gallery
  Table A8       the VICReg weight sweep and its two reference rows
  Appendix L     the SimCLR-style tower's test curve from its selected epoch to the end
  Figure A1      the four latent queries of the I-CE run on their first principal component
  Figure 9, Figure A3, Appendix L   six encoders against ten 4,096-sentence anchor draws

The first four are read on ONE 4,096-sentence anchor draw, the pack of the fixed split
(wscan_gal_rev_g4096.npz, all 2,020 games): a tower encodes the pack and the 3,256 wiki rewrites,
and the test set is the 204 test games of wiki_eval_split.json.

  cos        mean pairwise cosine of the 2,020 unit anchors
  margin     nearest-neighbor margin: mean over anchors of 1 - max over the other anchors of cos
  centroid   mean distance of a unit anchor to the mean unit anchor (Section 6.2 quotes it)
  Stripped   Stripped hit@1: the test rewrites without the name, the own anchor ranked against all
             2,020 (an anchor counts as ahead only when strictly nearer); rank-1 is the count of 204
  Name       the same over the name-intact rewrites
  tag F1     the probe of Section 5: a ridge fit on the 1,613 training anchors, its alpha and
             threshold chosen on the 203 validation anchors, micro-F1 on the test games' name-intact
             rewrites

Which checkpoint: the plan file's. A run that reads the anchors in training was selected by the rank
rule of Section 5 on the gallery its worker saved; a run that reads none (SimCLR-style, I + SimCLR,
a+u @ views, VICReg) on this 4,096-sentence pack, over its 40 checkpoints. Only the selected
checkpoints are published, so the selection itself is not repeated here; the plan file records the
rank score.

The readings agree with the printed ones to the third decimal, with two kinds of exception. A
near tie between two anchors can decide one query differently on another machine, which moves a
hit@1 cell by one query of 204 (0.005) and the tag reading in its last digit. And a gallery that
has collapsed onto one direction (a+u @ gallery: cos 1.000) decides Stripped hit@1 by
floating-point ties altogether, so that row moves by about 0.01.

The last line of the list is the ten-draw protocol of the five-fold tables applied to the fixed
split: k10_eval.py draws ten 4,096-sentence packs per game from the full review pool
(full_pool_fp16.npy, built by join_full_pool.py of the dataset) and scores the towers of Figure 9
on the 204 test games. It is run from here (about 40 minutes on one GPU); --skip-draws leaves it
out, and --games N --k K make it a short smoke run that is encoded but not scored. The CE tower
reads 0.676 here where the paper prints 0.675: one of its 2,040 query-and-draw outcomes is a near
tie that the batch a query is encoded in decides.

--plot also draws Figure A1 (needs matplotlib). Everything printed is written to
<out-dir>/fixed_split_readings.json. Needs torch with CUDA, numpy and scikit-learn.
"""
import argparse
import io
import json
import os
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
W9 = HERE.parent
sys.path.insert(0, str(W9)); sys.path.insert(0, str(W9 / "Pod"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
from Pod.w9_cv_worker import SetPoolN, rown  # noqa: E402
from VICReg_review.text_variant_eval import micro_prf, train_anchor_ridge  # noqa: E402

PLAN = HERE / "fixed_split_ckpt_plan.json"
# (row label, key in the plan file, what the paper prints)
TABLE3 = [("I-CE (ours)", "i2ce_icetf_fixed_d512", dict(cos=0.376, marg=0.168, strip=0.725, tag=0.677)),
          ("CE (contrast only)", "ce_cetf_fixed_d512", dict(cos=0.255, marg=0.194, strip=0.667, tag=0.640)),
          ("CE + anchor-in-I", "ai2ce_icetf_fixed_d512", dict(cos=0.381, marg=0.163, strip=0.711, tag=0.695)),
          ("a+u @ gallery", "ai25auni2_icetf_fixed_d512", dict(cos=1.000, marg=0.000, strip=0.172, tag=0.528)),
          ("a+u @ views", "i4uni2_icetf_fixed", dict(cos=0.008, marg=0.063, strip=0.328, tag=0.734))]
# align : uniformity in the paper's units; Section 6.2 quotes the margin, the centroid distance and the rank-1 count
SEC62 = [("12.5 : 1 (ai25auni2, Table 3's a+u @ gallery)", "ai25auni2_icetf_fixed_d512", "distance 0.001, Stripped 0.172"),
         ("2 : 1 (ai4auni2)", "ai4auni2_icetf_fixed_d512", "margin below 0.001, distance 0.002 or less, no query at rank one"),
         ("1 : 1, kernel temperature 25 (ai2auni25)", "ai2auni25_icetf_fixed_d512", "margin below 0.001, distance 0.002 or less, no query at rank one"),
         ("3 : 1 (ai6auni2)", "ai6auni2_icetf_fixed_d512", "distance 0.998, margin 0.045, Stripped 0.275")]
TABLEA7 = [("v/i/c = 25/25/1", "epd_v25i25c1_cetf_fixed", dict(ep=550, name=0.618, strip=0.363, tag=0.708)),
           ("v/i/c = 20/10/20 (recipe weights)", "epd_v20i10c20_cetf_fixed", dict(ep=100, name=0.814, strip=0.451, tag=0.725)),
           ("v/i/c = 20/10/15", "epd_v20i10c15_cetf_fixed", dict(ep=100, name=0.799, strip=0.456, tag=0.718)),
           ("I-CE (Table 3)", "i2ce_icetf_fixed_d512", dict(ep=2000, name=0.946, strip=0.725, tag=0.677)),
           ("CE (Table 3)", "ce_cetf_fixed_d512", dict(ep=1900, name=0.907, strip=0.667, tag=0.640))]
CURVE = "bce_cetf_fixed"                 # Appendix L: no sustained rise after epoch 1,350 (0.59 to 0.65)
QUERIES = "i2ce_icetf_fixed_g2048"       # Figure A1: first principal component, 36% of the variance
# Figure 9, Figure A3, Appendix L: test-set Stripped hit@1 over ten draws (the frozen embedder reads 0.229)
DRAWN = [("I-CE", "i2ce_icetf_fixed_g2048", 0.743), ("CE", "ce_cetf_fixed_g2048", 0.675),
         ("SimCLR-style", "bce_cetf_fixed", 0.640), ("I + SimCLR", "i2bce_icetf_fixed", 0.634),
         ("VICReg", "epd_v20i10c20_cetf_fixed", 0.446), ("a+u @ views", "i4uni2_icetf_fixed", None)]
FROZEN_DRAWN = 0.229


def npz_mmap(path, key):
    """memory-map one stored array of an .npz, so that the 17 GB pack is never read whole"""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(key + ".npy")
    assert info.compress_type == zipfile.ZIP_STORED, f"{path}:{key} is compressed"
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        h = f.read(30)
        assert h[:4] == b"PK\x03\x04"
        n, e = struct.unpack("<HH", h[26:30])
        f.seek(info.header_offset + 30 + n + e)
        ver = np.lib.format.read_magic(f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0(f) if ver == (1, 0)
                                 else np.lib.format.read_array_header_2_0(f))
        off = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=off, order="F" if fortran else "C")


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def tower_state(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    return {k: v for k, v in sd.items() if k == "q0" or k.startswith(("attn.", "head."))}   # VICReg files carry an expander too


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True, help="folder with the ckpt_*.pt files of the checkpoint repository")
    ap.add_argument("--data-dir", default=os.environ.get("W9_DATA_DIR", ""), help="the data folder (or W9_DATA_DIR)")
    ap.add_argument("--out-dir", default="fixed_split_eval")
    ap.add_argument("--skip-draws", action="store_true", help="leave out the ten-draw reading of Figure 9's towers")
    ap.add_argument("--only-draws", action="store_true", help="only the ten-draw reading")
    ap.add_argument("--games", type=int, default=0, help="ten draws: encode the first N games only (smoke run, not scored)")
    ap.add_argument("--k", type=int, default=10, help="ten draws: number of draws")
    ap.add_argument("--chunk", type=int, default=16, help="anchor packs per forward pass")
    ap.add_argument("--plot", action="store_true", help="draw Figure A1 into the out folder")
    a = ap.parse_args()
    data, ckdir, out = Path(a.data_dir), Path(a.ckpt_dir), Path(a.out_dir)
    assert a.data_dir and (data / "games.npz").exists(), "--data-dir <data folder> (or W9_DATA_DIR) is required"
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    res = dict(plan=str(PLAN.name))
    if not a.only_draws:
        res.update(one_draw(a, data, ckdir, out, plan))
        (out / "fixed_split_readings.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    if not a.skip_draws:
        ten = ten_draws(a, data, ckdir, out)
        if ten is not None:
            res["ten_draws"] = ten
    (out / "fixed_split_readings.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\nwrote", out / "fixed_split_readings.json")


def one_draw(a, data, ckdir, out, plan):
    from w9_selection import rank_score, validation_rows
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    galp = data / "wscan_gal_rev_g4096.npz"
    assert galp.exists(), f"{galp} is missing: it is the fixed split's 4,096-sentence anchor pack, a file of the dataset"

    # ---- fixtures: names, queries, the fixed split, tags
    names = [str(x) for x in np.load(data / "games.npz", allow_pickle=True)["names"]]
    NG = len(names); n2i = {n: i for i, n in enumerate(names)}; a2n = {n.split("_")[0]: n for n in names}
    A = np.load(data / "wiki_eval.npz", allow_pickle=True)
    art = [str(x) for x in A["names"]]; variants = [str(x) for x in A["variants"]]; gA = np.asarray(A["gidx"])
    SA = rown(torch.tensor(A["S"]).to(dev).float()).half()
    mA = torch.arange(SA.shape[1], device=dev)[None, :] >= torch.tensor(A["S_len"]).to(dev)[:, None]
    sp = json.loads((data / "wiki_eval_split.json").read_text())
    test_g = {a2n[x] for x in sp["test"]}; val_g = {a2n[x] for x in sp["val"]}
    ctag = {"train": [n for n in names if n not in test_g | val_g], "val": sorted(val_g), "test": sorted(test_g)}
    vrows = validation_rows(names, art, variants, gA, sp)
    q_non = [i for i in range(len(art)) if art[i] in test_g and variants[i] == "noname"]
    q_neu = [i for i in range(len(art)) if art[i] in test_g and variants[i] == "neutral"]
    y = np.load(data / "tag_labels.npz", allow_pickle=True)["y"]
    targs = SimpleNamespace(tag_text_train_frac=0.7, tag_text_val_frac=0.15, tag_text_split_seed=42, seed=42,
                            tag_text_threshold_steps=33)
    print(f"fixed split: train {len(ctag['train'])} / val {len(val_g)} / test {len(test_g)} games")

    def hit1(G, Za, q):
        g, z = l2(G).astype(np.float32), l2(Za[q]).astype(np.float32)
        s = z @ g.T; own = s[np.arange(len(q)), gA[q]][:, None]
        return float(((s > own).sum(1) == 0).mean())

    def columns(G, Za):
        g = l2(G).astype(np.float32); N = len(g); S = g @ g.T
        cos = float((S.sum() - N) / (N * (N - 1)))
        np.fill_diagonal(S, -np.inf)
        marg = float((1 - S.max(1)).mean())
        cen = float(np.linalg.norm(g - g.mean(0, keepdims=True), axis=1).mean())
        st = hit1(G, Za, q_non)
        sc, rg, _, th, _ = train_anchor_ridge(targs, G, y, n2i, ctag)
        Xq = np.stack([Za[i] for i in q_neu]).astype(np.float32); lab = np.stack([y[n2i[art[i]]] for i in q_neu])
        tag = float(micro_prf(lab, rg.predict(sc.transform(Xq)), th)["micro_f1"])
        return dict(cos=cos, marg=marg, centroid=cen, strip=st, n_rank1=int(round(st * len(q_non))),
                    name=hit1(G, Za, q_neu), tag=tag)

    # ---- the towers: every run at its selected checkpoint, and the SimCLR-style run's later checkpoints
    files = {key: v["ckpt"] for key, v in plan.items()}
    curve_files = {o["ep"]: o["ckpt"] for o in plan[CURVE].get("trajectory", {}).get("files", [])}
    files.update({(CURVE, ep): f for ep, f in curve_files.items()})
    missing = [f for f in files.values() if not (ckdir / f).exists()]
    assert not missing, f"{len(missing)} checkpoints are not in {ckdir} (first: {missing[0]})"

    def load(f):
        sd = tower_state(ckdir / f)
        m = SetPoolN(int(sd["q0"].shape[1])).to(dev).eval(); m.load_state_dict(sd)
        return m
    models = {k: load(f) for k, f in files.items()}
    print(f"{len(plan)} runs, {len(models)} checkpoints, one pass over {galp.name} ...")
    G = npz_mmap(galp, "gal")
    with np.load(galp) as z:
        glen = np.asarray(z["gal_len"], np.int64)
    assert G.shape[0] == NG
    Zg = {k: np.zeros((NG, 128), np.float32) for k in models}
    with torch.no_grad():
        Za = {k: torch.cat([m(SA[i:i + 256], mA[i:i + 256]).float() for i in range(0, SA.shape[0], 256)]).cpu().numpy()
              for k, m in models.items()}
        ar = torch.arange(G.shape[1], device=dev)
        for i in range(0, NG, a.chunk):
            s = torch.from_numpy(np.ascontiguousarray(G[i:i + a.chunk])).to(dev)
            msk = ar[None, :] >= torch.from_numpy(glen[i:i + a.chunk]).to(dev)[:, None]
            for k, m in models.items():
                Zg[k][i:i + a.chunk] = m(s, msk).float().cpu().numpy()
            if (i // a.chunk) % 25 == 0:
                print(f"  packs {i + len(s)}/{NG} [{time.time() - t0:.0f}s]")
    print(f"encoded in {time.time() - t0:.0f}s (median pack {int(np.median(glen))} sentences)")
    col = {key: dict(columns(Zg[key], Za[key]), ep=plan[key]["ep"], ckpt=plan[key]["ckpt"],
                     rank_score_here=float(rank_score(Zg[key], Za[key], gA, vrows))) for key in plan}

    f3 = lambda x: "%.3f" % x

    def versus(c, ref, keys):
        return "   ".join("%s %s (%s)" % (k, f3(c[k]), f3(ref[k])) for k in keys)

    print("\nTable 3: the decomposition ablation on the fixed split, computed (paper)")
    for label, key, ref in TABLE3:
        c = col[key]
        print("  %-20s ep%-5d " % (label, c["ep"]) + versus(c, ref, ["cos", "marg", "strip", "tag"]))
    print("\nSection 6.2: repulsion aimed at the anchor gallery, by align : uniformity")
    for label, key, quote in SEC62:
        c = col[key]
        print("  %-46s ep%-5d margin %.5f  centroid %.3f  Stripped %.3f  rank-1 %3d of %d   paper: %s"
              % (label, c["ep"], c["marg"], c["centroid"], c["strip"], c["n_rank1"], len(q_non), quote))
    print("\nTable A8: the VICReg weight sweep, computed (paper)")
    for label, key, ref in TABLEA7:
        c = col[key]
        print("  %-34s epoch %d (%d)   " % (label, c["ep"], ref["ep"]) + versus(c, ref, ["name", "strip", "tag"]))
    curve = {plan[CURVE]["ep"]: col[CURVE]["strip"]}
    curve.update({ep: hit1(Zg[(CURVE, ep)], Za[(CURVE, ep)], q_non) for ep in curve_files})
    print("\nAppendix L: the SimCLR-style tower's test Stripped hit@1 from its selected epoch on (paper: 0.59 to 0.65)")
    print("  " + "  ".join("%d:%.3f" % (ep, curve[ep]) for ep in sorted(curve)))
    print("  lowest %.3f, highest %.3f" % (min(curve.values()), max(curve.values())))
    other = [k for k in plan if k not in {r[1] for r in TABLE3 + SEC62 + TABLEA7}]
    print("\nThe other fixed-split towers on this one draw (the paper reads them on ten draws, below)")
    for key in other:
        c = col[key]
        print("  %-28s ep%-5d cos %.3f  margin %.3f  Stripped %.3f  Name %.3f  tag %.3f"
              % (key, c["ep"], c["cos"], c["marg"], c["strip"], c["name"], c["tag"]))
    res = dict(one_draw=col, simclr_curve={str(e): curve[e] for e in sorted(curve)})

    # ---- Figure A1: the four latent queries of the I-CE run at every saved checkpoint
    traj = {plan[QUERIES]["ep"]: plan[QUERIES]["ckpt"]}
    traj.update({o["ep"]: o["ckpt"] for o in plan[QUERIES].get("trajectory", {}).get("files", [])})
    have = {ep: f for ep, f in traj.items() if (ckdir / f).exists()}
    if len(have) == len(traj):
        E = np.array(sorted(have)); Q = np.stack([tower_state(ckdir / have[e])["q0"][0].float().numpy() for e in E])   # [40, 4, 128]
        flat = Q.reshape(-1, Q.shape[-1]); mu = flat.mean(0)
        _, sv, vt = np.linalg.svd(flat - mu, full_matrices=False)
        share = float(sv[0] ** 2 / (sv ** 2).sum()); pc1 = (Q - mu) @ vt[0]
        print("\nFigure A1: %d checkpoints x %d latent queries; the first principal component holds %.0f%% of their variance (paper: 36%%)"
              % (len(E), Q.shape[1], 100 * share))
        res["figure_a1"] = dict(run=QUERIES, selected=plan[QUERIES]["ep"], pc1_variance_share=share, epochs=E.tolist(),
                                pc1=pc1.T.round(5).tolist())
        if a.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(3.4, 1.8))
            for k, ls in zip(range(Q.shape[1]), ["-", (0, (3.0, 1.6)), (0, (5.0, 1.5, 1.0, 1.5)), (0, (7.0, 2.6))]):
                ax.plot(E, pc1[:, k], lw=1.2, ls=ls, label="slot %d" % (k + 1))
            ax.axvline(plan[QUERIES]["ep"], color="#8a8a8a", lw=0.7, ls=(0, (1.5, 1.5)))
            ax.set_xlim(0, int(E.max())); ax.set_xlabel("epoch"); ax.set_ylabel("latent query on PC1")
            ax.legend(frameon=False, ncol=4, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0))
            fig.tight_layout(pad=0.3); fig.savefig(out / "figure_a1.png", dpi=300)
            print("  drew", out / "figure_a1.png")
    else:
        print("\nFigure A1: %d of the run's %d checkpoints are in %s, skipped" % (len(have), len(traj), ckdir))
    return res


def ten_draws(a, data, ckdir, out):
    """the ten-draw protocol of the five-fold tables, applied to the fixed-split towers of Figure 9"""
    if not (data / "full_pool_fp16.npy").exists():
        print("\nten draws: skipped, %s is missing (join_full_pool.py of the dataset builds it)" % (data / "full_pool_fp16.npy"))
        return None
    k10 = [sys.executable, str(HERE / "k10_eval.py")]
    cells, draws, scores = out / "cells_fixed.json", out / "draws_fixed.npz", out / "scores_fixed.json"
    only = "^(%s)$" % "|".join(key for _, key, _ in DRAWN)
    steps = [k10 + ["cells", str(cells), "--ckpt-dir", str(ckdir), "--only", only, str(PLAN)],
             k10 + ["encode", str(cells), str(draws), "--data-dir", str(data), "--k", str(a.k)] + (["--games", str(a.games)] if a.games else [])]
    if not a.games:
        steps += [k10 + ["score", str(cells), str(draws), str(scores), "--data-dir", str(data)]]
    print("\nFigure 9, Figure A3, Appendix L: the towers against %d draws of 4,096-sentence anchors" % a.k)
    for cmd in steps:
        print("  $ " + " ".join(cmd[1:]))
        subprocess.run(cmd, check=True)
    if a.games:
        print("  smoke run on %d games: encoded, not scored" % a.games)
        return None
    R = json.loads(scores.read_text(encoding="utf-8"))["results"]
    rows = [("frozen embedder", "frozen|fixed", FROZEN_DRAWN)] + DRAWN
    print("\n  test-set Stripped hit@1 over the %d draws, mean ± std over the draws (paper)" % a.k)
    for label, key, ref in rows:
        o = R[key]
        print("  %-16s %.3f ± %.3f %s   Name %.3f   tag F1 %.3f"
              % (label, o["pd_nm_noname"], o["pdsd_nm_noname"], "(%.3f)" % ref if ref is not None else "       ",
                 o["pd_nm_neutral"], o["pd_test_tag"]))
    return {key: R[key] for _, key, _ in rows}


if __name__ == "__main__":
    main()
