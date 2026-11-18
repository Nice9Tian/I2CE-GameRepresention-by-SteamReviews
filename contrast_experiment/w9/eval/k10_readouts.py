# -*- coding: utf-8 -*-
r"""Readouts of the ten-draw encodings that k10_eval.py `score` does not make: Tables A16, A17, S3 and S4.

`k10_eval.py score` reads every five-fold tower on its test-set queries: per-draw retrieval, the
merged-index retrieval and the cross-register tag F1 of Section 5.  This script reads the same
encodings (the file `k10_eval.py encode` writes, the anchors of ten 4,096-sentence draws per game
and the tower's encoding of every wiki rewrite) four more ways:

  in-domain tag F1    the ridge and threshold of the cross-register readout, fitted on one draw's
                      training-pool anchors, scored on the test games' anchors of the same draw
                      instead of their name-intact rewrites (Table A16, Table A17)
  register offset     the norm of the mean anchor-to-query displacement over the training games
                      of the fold that carry a wiki rewrite (name-intact), per draw (Table A16)
  merged-index tag    the ridge fitted on the ten draws averaged into one anchor per game, scored
                      on the test games' name-intact rewrites (Table S3's tag column)
  training queries    per-draw hit@1 and hit@5 of the rewrites of the fold's training games, next
                      to the same reading on its test games (Table S4)

    E = contrast_experiment/w9/eval
    python $E/k10_readouts.py score  <cells.json> <draws.npz> <out.json> --data-dir <data folder>
    python $E/k10_readouts.py tables <out.json> [--margins Pod/table1_results.json] [--all]

`score` needs the data folder (games.npz, wiki_eval.npz, tag_labels.npz, wiki_eval_split.json; or
W9_DATA_DIR) and scikit-learn; it runs on the CPU in a few minutes.  `tables` prints the four
tables from its output.  A table row reads the cells of one run family over the five folds; the
cells are found under the names `k10_eval.py cells` gives them (I-CE|k, CE|k, ...,
mq3072i2ce_icetf_fold<k>_gpool, ...), and a row whose cells are missing is skipped.
Table A17's in-domain cell for the EMA + memory-bank tower is not in the paper ("not measured");
`tables` leaves it out unless --all is given.  paper_readings/k10_readouts_4096.json is this
script's output on the encodings behind the paper, so `tables` runs without a GPU or the pool:

    python $E/k10_readouts.py tables $E/paper_readings/k10_readouts_4096.json
"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
W9 = HERE.parent
sys.path.insert(0, str(W9))
SPLIT_SEED, NF = 20260711, 5
TARGS = SimpleNamespace(tag_text_train_frac=0.7, tag_text_val_frac=0.15, tag_text_split_seed=42, seed=42,
                        tag_text_threshold_steps=33)
# the rows of the tables: printed label, the cell name of fold k
ROWS = [("I-CE (ours)", "I-CE|{k}"),
        ("swin-I-CE (~26% gradient window)", "swin168step84loop2i2ce_icetf_fold{k}_gpool"),
        ("Two-stage: BYOL warm start -> windowed, 600 ep", "swin168step84loop2i2ce_icetf_fold{k}_gpool_bw"),
        ("CE (contrast only)", "CE|{k}"),
        ("SimCLR-style (in-batch views)", "SimCLR|{k}"),
        ("I-CE (EMA + memory bank)", "mq3072i2ce_icetf_fold{k}_gpool"),
        ("VICReg (expander wiring, v/i/c = 20/10/20)", "VICReg|{k}"),
        ("BYOL", "BYOL|{k}"),
        ("Frozen embedder (mean pool)", "frozen|{k}")]
# Table A16: the five towers of Table A1, with the key of each in table1_results.json (Table A18)
A16 = [("CE (contrast only)", "CE"), ("I-CE (ours)", "I-CE"), ("SimCLR-style (in-batch views)", "SimCLR"),
       ("VICReg (v/i/c = 20/10/20)", "VICReg"), ("BYOL", "BYOL")]
A17 = [("BYOL", "BYOL"), ("VICReg", "VICReg"), ("SimCLR-style", "SimCLR"),
       ("EMA + memory bank", "mq3072i2ce_icetf_fold{k}_gpool"), ("CE", "CE")]
NOT_PRINTED = {("EMA + memory bank", "indom")}     # the paper prints "not measured" here; --all shows the reading


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class Fixtures:
    """the games, the wiki rewrites and the folds, as k10_eval.py reads them (no GPU)"""

    def __init__(self, C):
        self.names = [str(x) for x in np.load(C / "games.npz", allow_pickle=True)["names"]]
        self.NG = len(self.names)
        self.n2i = {n: i for i, n in enumerate(self.names)}
        a2n = {n.split("_")[0]: n for n in self.names}
        A = np.load(C / "wiki_eval.npz", allow_pickle=True)
        self.variants = np.array([str(x) for x in A["variants"]])
        self.art = np.array([str(x) for x in A["names"]])
        self.gA = np.asarray(A["gidx"])
        self.y = np.load(C / "tag_labels.npz", allow_pickle=True)["y"]
        sp = json.loads((C / "wiki_eval_split.json").read_text())
        universe = sorted(set(sp["test"]) | set(sp["val"]) | set(sp["train"]))
        folds = np.array_split(np.random.default_rng(SPLIT_SEED).permutation(len(universe)), NF)
        self.fold_sets = [({a2n[universe[i]] for i in folds[k]}, {a2n[universe[i]] for i in folds[(k + 1) % NF]})
                          for k in range(NF)]

    def rows(self, games, var):
        """indices of the rewrites of register `var` whose game is in `games`"""
        return np.array([i for i in range(len(self.art)) if self.variants[i] == var and self.art[i] in games])


def hits(Q, G, own):
    """(hit@1, hit@5) of queries Q against the anchors G (both normalized), own = the true game of each"""
    s = Q @ G.T
    r = (s > s[np.arange(len(own)), own][:, None]).sum(1) + 1
    return float((r == 1).mean()), float((r <= 5).mean())


def tag_readouts(F, Zk, Za, test_g, val_g):
    """per draw: cross-register and in-domain tag F1, register offset; and the merged-index tag F1"""
    from VICReg_review.text_variant_eval import micro_prf, train_anchor_ridge
    ctag = {"train": [n for n in F.names if n not in test_g | val_g], "val": sorted(val_g), "test": sorted(test_g)}
    ti = F.rows(test_g, "neutral")
    train_games = {n for n in F.art if n not in test_g | val_g}
    tri = F.rows(train_games, "neutral")
    tg_idx = np.array([F.n2i[g] for g in sorted(test_g)])
    lab_q, lab_a = F.y[F.gA[ti]], F.y[tg_idx]
    Xq = Za[ti].astype(np.float32)
    o = {"cross": [], "indom": [], "offset": []}
    for d in range(Zk.shape[1]):
        Zg = Zk[:, d].astype(np.float32)
        sc, rg, _, th, _ = train_anchor_ridge(TARGS, Zg, F.y, F.n2i, ctag)
        o["cross"].append(float(micro_prf(lab_q, rg.predict(sc.transform(Xq)), th)["micro_f1"]))
        o["indom"].append(float(micro_prf(lab_a, rg.predict(sc.transform(Zg[tg_idx])), th)["micro_f1"]))
        o["offset"].append(float(np.linalg.norm((Za[tri].astype(np.float32) - Zg[F.gA[tri]]).mean(0))))
    sc, rg, _, th, _ = train_anchor_ridge(TARGS, Zk.mean(1), F.y, F.n2i, ctag)
    o["tag_meanvec"] = float(micro_prf(lab_q, rg.predict(sc.transform(Xq)), th)["micro_f1"])
    o["n_train_q"], o["n_test_q"] = len(tri), len(ti)
    return o


def query_readouts(F, Zk, Za, test_g, val_g):
    """per-draw hit@1 / hit@5 on training-set and test-set rewrites, and the merged-index retrieval"""
    gz, gmean = l2(Zk.astype(np.float32)), l2(Zk.mean(1).astype(np.float32))
    train_games = {n for n in F.art if n not in test_g | val_g}
    o = {}
    for var, tag in (("neutral", "neu"), ("noname", "str")):
        for pop, games in (("train", train_games), ("test", test_g)):
            q = F.rows(games, var)
            az, own = l2(Za[q].astype(np.float32)), F.gA[q]
            per = [hits(az, gz[:, d], own) for d in range(Zk.shape[1])]
            o[f"{tag}_{pop}_h1"] = float(np.mean([p[0] for p in per]))
            o[f"{tag}_{pop}_h1_sd"] = float(np.std([p[0] for p in per]))
            o[f"{tag}_{pop}_h5"] = float(np.mean([p[1] for p in per]))
            if pop == "test":
                o[f"mg_nm_{var}"], o[f"mg_h5_{var}"] = hits(az, gmean, own)
    return o


def score(args):
    C = Path(opt(args, "--data-dir", os.environ.get("W9_DATA_DIR", "")))
    assert str(C) not in ("", ".") and (C / "games.npz").exists(), "--data-dir <data folder> (or W9_DATA_DIR) is required"
    cell = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    d, out = np.load(args[1]), Path(args[2])
    F = Fixtures(C)
    jobs = [(key, cell[key]["fold"]) for key in cell if cell[key]["fold"] is not None]
    jobs += [("frozen", k) for k in sorted({f for _, f in jobs})]       # the frozen embedder on every fold the cells use
    res = {}
    for key, k in jobs:
        Zk, Za = d[key], d["Q|" + key]
        test_g, val_g = F.fold_sets[k]
        o = tag_readouts(F, Zk, Za, test_g, val_g)
        o.update(query_readouts(F, Zk, Za, test_g, val_g))
        for f in ("cross", "indom", "offset"):
            o[f + "_draws"] = o.pop(f)
            o[f] = float(np.mean(o[f + "_draws"]))
        o["transfer_loss"] = o["indom"] - o["cross"]
        name = key if key != "frozen" else "frozen|%d" % k
        res[name] = o
        print("  %-48s fold %d: tag cross %.3f in-domain %.3f offset %.3f merged %.3f | stripped tr %.3f ts %.3f"
              % (name, k, o["cross"], o["indom"], o["offset"], o["tag_meanvec"], o["str_train_h1"], o["str_test_h1"]))
    out.write_text(json.dumps(dict(k=int(d["k"]), cap=int(d["cap"]), results=res), indent=1), encoding="utf-8")
    print("wrote", out)


# ------------------------------------------------------------------ tables
def family(R, pattern, field):
    """the five per-fold values of one run family, or None when a fold is missing"""
    keys = [pattern.format(k=k) for k in range(NF)]
    return np.array([R[key][field] for key in keys], float) if all(key in R for key in keys) else None


def paired_t(a, b):
    d = a - b
    return d.mean(), d.mean() / (d.std(ddof=1) / np.sqrt(len(d))), int((d > 0).sum())


def table_a16_a17(R, margins, show_all=False):
    print("Table A16  (mean ± std over the five folds; transfer loss and offset are five-fold means)")
    print("%-34s %8s  %-15s %-15s %8s %8s" % ("Objective", "margin", "cross-register", "in-domain", "loss", "offset"))
    rows = []
    for label, arm in A16:
        cr, ind, off = (family(R, arm + "|{k}", f) for f in ("cross", "indom", "offset"))
        if cr is None:
            print("%-34s (cells missing)" % label); continue
        rows.append((ind.mean() - cr.mean(), label, arm, cr, ind, off))
    for loss, label, arm, cr, ind, off in sorted(rows, key=lambda r: -r[0]):
        m = margins.get(arm, {}).get("margin", float("nan"))
        print("%-34s %8.3f  %.3f ± %.3f   %.3f ± %.3f   %8.3f %8.3f"
              % (label, m, cr.mean(), cr.std(), ind.mean(), ind.std(), loss, off.mean()))
    print("\nTable A17  (I-CE minus the named tower, paired over the five folds, t with four degrees of freedom)")
    print("%-20s %10s %7s %6s   %10s %7s %6s" % ("I-CE minus", "cross Δ", "t", "won", "in-domain Δ", "t", "won"))
    for label, arm in A17:
        pat = arm if "{k}" in arm else arm + "|{k}"
        out = []
        for f in ("cross", "indom"):
            a, b = family(R, "I-CE|{k}", f), family(R, pat, f)
            if (label, f) in NOT_PRINTED and not show_all:
                a = None
            out.append("%+10.3f %7.2f %3d of 5" % paired_t(a, b) if a is not None and b is not None
                       else "%10s %7s %6s" % ("n/a", "n/a", "n/a"))
        print("%-20s %s   %s" % (label, out[0], out[1]))


def table_s3_s4(R):
    print("\nTable S3  (merged index: the ten draws averaged into one anchor; mean ± std over the five folds)")
    print("%-48s %-15s %-15s %-15s %-15s %-15s" % ("Objective", "Name hit@1", "Name hit@5", "Stripped hit@1", "Stripped hit@5", "tag F1"))
    for label, pat in ROWS:
        cols = [family(R, pat, f) for f in ("mg_nm_neutral", "mg_h5_neutral", "mg_nm_noname", "mg_h5_noname", "tag_meanvec")]
        if cols[0] is not None:
            print("%-48s %s" % (label, "   ".join("%.3f ± %.3f" % (c.mean(), c.std()) for c in cols)))
    print("\nTable S4  (training-set (tr) against test-set (ts) queries, five-fold means of the per-draw reading)")
    print("%-48s %8s %8s %8s   %8s %8s %8s" % ("Objective", "Name tr", "Name ts", "Gap", "Str tr", "Str ts", "Gap"))
    for label, pat in ROWS:
        v = [family(R, pat, f) for f in ("neu_train_h1", "neu_test_h1", "str_train_h1", "str_test_h1")]
        if v[0] is not None:
            m = [x.mean() for x in v]
            print("%-48s %8.3f %8.3f %+8.3f   %8.3f %8.3f %+8.3f" % (label, m[0], m[1], m[0] - m[1], m[2], m[3], m[2] - m[3]))


def tables(args):
    show_all = "--all" in args
    args = [x for x in args if x != "--all"]
    margins_path = Path(opt(args, "--margins", str(W9 / "Pod" / "table1_results.json")))
    margins = json.loads(margins_path.read_text(encoding="utf-8"))["t3"] if margins_path.exists() else {}
    R = json.loads(Path(args[0]).read_text(encoding="utf-8"))["results"]
    table_a16_a17(R, margins, show_all)
    table_s3_s4(R)


def opt(args, name, default):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]
        return type(default)(v)
    return default


if __name__ == "__main__":
    COMMANDS = dict(score=score, tables=tables)
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
    COMMANDS[sys.argv[1]](sys.argv[2:])
