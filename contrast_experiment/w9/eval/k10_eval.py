# -*- coding: utf-8 -*-
r"""Ten-draw evaluation of five-fold towers at one anchor budget (the manuscript's protocol).

An anchor pack is a draw, not the game, so a five-fold retrieval number is the mean over K = 10
independent packs.  Draw k of game g is the store-page prefix (sp_raw_views.npz) followed by whole
reviews taken in the order of default_rng(20260711 + 5 + 1000 k + g).permutation(reviews), a review
entering if it still fits under the cap; sentences are row-standardised like every other input.
A draw depends on the cap, k and g only, never on the towers scored.

Sentences are read from the full pool the full-pool workers already use (full_pool_fp16.npy and
full_pool_meta.npz in the data directory; Pod/w9_a100.ipynb builds them once from
embedding_h5.h5).  When the pool is absent, pass --h5 to read embedding_h5.h5 directly; both hold
the same vectors and offsets and give the same draws.

Which checkpoint a tower is scored at is NOT decided here: five-fold runs are selected by the rank
rule (Pod/w9_selection.py) on the gallery their own worker saved; rank_ckpt_plan.json lists the
selections behind the manuscript, except for its 512-sentence rows: those are the towers retrained
on the corrected 512 packs (run names with _d512), selected in the three docfix512 plan files of the
checkpoint repository, and the plan's own 512-sentence entries are the earlier runs on the packs as
built before 2026-11-07.  `select` re-runs that rule for a run whose worker saved another
gallery, on the gallery the workers of the main runs build (--anchor-cap pool in
Pod/w9_cv_worker.py: store-page prefix plus the whole 2,048-sentence review pool, a median of
2,068 sentences; the manuscript's 2,048-sentence budget, run names ending in _gpool).  I + SimCLR was
selected that way (i2bce_sel_pool.json).  The 110 checkpoints both files name are on Hugging Face,
chironYu/latenarray-i2ce-checkpoints.

Every command takes --data-dir (or the W9_DATA_DIR environment variable): the folder holding
games.npz, wiki_eval.npz, sp_raw_views.npz, tag_labels.npz, wiki_eval_split.json,
wscan_pool_rev*.npy and full_pool_fp16.npy + full_pool_meta.npz.  From the repository root:

    E = contrast_experiment/w9/eval
    python $E/k10_eval.py cells  <cells.json> --ckpt-dir <dir> [$E/i2bce_sel_pool.json ...]
    python $E/k10_eval.py encode <cells.json> <draws.npz> [--cap 4096] [--k 10] [--games N] [--h5 <embedding_h5.h5>]
    python $E/k10_eval.py check  <draws.npz> <reference.npz>       max difference against a reference encoding
    python $E/k10_eval.py score  <cells.json> <draws.npz> <out.json>
    python $E/k10_eval.py table  <out.json>                        five-fold mean ± std per run family
    python $E/k10_eval.py select <sel.json> <checkpoint dir> <run name with {fold}>

`cells` looks a selection file's checkpoint names up in --ckpt-dir (absolute paths are kept).
Needs torch with CUDA, numpy, scikit-learn (the tag probe) and h5py (imported by the tag probe's
module; `encode` reads embedding_h5.h5 only with --h5).
"""
import contextlib
import io
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
sys.path.insert(0, str(W9)); sys.path.insert(0, str(W9 / "Pod"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
from Pod.w9_cv_worker import SetPoolN, rown  # noqa: E402
from w9_selection import rank_score, validation_rows  # noqa: E402

PLAN = HERE / "rank_ckpt_plan.json"
JULY_ARMS = {"I-CE": "i2ce_icetf", "CE": "ce_cetf", "SimCLR": "bce_cetf", "VICReg": "epd_v20i10c20_cetf", "BYOL": "byol_bytf"}
SPLIT_SEED, NF = 20260711, 5
GAP, SPAN = 2048, 65536          # the store is chunked by 2,048 rows; one read holds at most 65,536
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
C = None                         # the data directory, set in main


def opt(args, name, default):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]
        return type(default)(v)
    return default


def load_tower(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    sd = {k: v for k, v in sd.items() if k == "q0" or k.startswith(("attn.", "head."))}    # VICReg checkpoints carry their expander too
    m = SetPoolN(sd["q0"].shape[1]).to(dev).eval(); m.load_state_dict(sd)    # q0 is [1, Q, 128]: Q latent queries (slot<Q> arms)
    return m


class Fixtures:
    """what every command reads: the games, the wiki queries, the folds"""

    def __init__(self):
        self.names = [str(x) for x in np.load(C / "games.npz", allow_pickle=True)["names"]]
        self.NG = len(self.names)
        self.n2i = {n: i for i, n in enumerate(self.names)}
        self.appid2name = {n.split("_")[0]: n for n in self.names}
        A = np.load(C / "wiki_eval.npz", allow_pickle=True)
        self.SA = rown(torch.tensor(A["S"]).to(dev).float()).half()
        self.mA = torch.arange(self.SA.shape[1], device=dev)[None, :] >= torch.tensor(A["S_len"]).to(dev)[:, None]
        self.variants = [str(x) for x in A["variants"]]
        self.art = [str(x) for x in A["names"]]
        self.gA = np.asarray(A["gidx"])
        self.split = json.loads((C / "wiki_eval_split.json").read_text())
        self.universe = sorted(set(self.split["test"]) | set(self.split["val"]) | set(self.split["train"]))
        self.folds = np.array_split(np.random.default_rng(SPLIT_SEED).permutation(len(self.universe)), NF)
        ST = np.load(C / "sp_raw_views.npz", allow_pickle=True)
        self.sp_row = {int(ST["gidx"][i]): i for i in range(len(ST["gidx"]))}
        self.S_sp, self.L_sp = ST["S"], ST["S_len"]

    def doc(self, g):
        """the store-page prefix of game g, row-standardised, or None"""
        if g not in self.sp_row:
            return None
        i = self.sp_row[g]; dl = int(self.L_sp[i])
        return rown(torch.tensor(self.S_sp[i, :dl]).float()).numpy().astype(np.float16)

    @torch.no_grad()
    def queries(self, model):
        return torch.cat([model(self.SA[i:i + 256], self.mA[i:i + 256]).float().cpu()
                          for i in range(0, self.SA.shape[0], 256)]).numpy()

    def frozen_queries(self):
        w = (~self.mA).float().unsqueeze(-1)
        return ((self.SA.float() * w).sum(1) / w.sum(1).clamp(min=1)).cpu().numpy()

    def fold_games(self, k):
        test = {self.appid2name[self.universe[i]] for i in self.folds[k]}
        val = {self.appid2name[self.universe[i]] for i in self.folds[(k + 1) % NF]}
        return test, val


def read_rows(V, uni):
    """rows `uni` (sorted, unique) of the store; rows closer than one storage chunk share a read"""
    out = np.empty((len(uni), V.shape[1]), np.float16)
    brk = np.flatnonzero(np.diff(uni) > GAP) + 1
    for i0, j0 in zip(np.concatenate([[0], brk]), np.concatenate([brk, [len(uni)]])):
        while i0 < j0:
            a = int(uni[i0]); j1 = int(np.searchsorted(uni[i0:j0], a + SPAN)) + i0
            b = int(uni[j1 - 1]) + 1
            out[i0:j1] = V[a:b][uni[i0:j1] - a]
            i0 = j1
    return out


# ------------------------------------------------------------------ select
def pool_plans(F, cap=None):
    """the anchors a five-fold worker builds from the review pool (Pod/w9_cv_worker.py): the
    store-page prefix plus whole reviews up to `cap` sentences, or, with cap None, the whole pool
    (--anchor-cap pool, the main runs)"""
    cap = 10 ** 9 if cap is None else cap
    RID = np.load(C / "wscan_pool_rev_rid.npy"); plen = np.load(C / "wscan_pool_rev_len.npy")
    rng = np.random.default_rng(SPLIT_SEED + 5); plans = []
    for g in range(F.NG):
        row = int(F.L_sp[F.sp_row[g]]) if g in F.sp_row else 0
        r_lens = np.bincount(RID[g, :plen[g]]); r_starts = np.zeros(len(r_lens), np.int64); r_starts[1:] = np.cumsum(r_lens)[:-1]
        segs = []
        for j in rng.permutation(len(r_lens)):
            L = int(r_lens[j])
            if row + L <= cap:
                segs.append((int(r_starts[j]), L)); row += L
        plans.append((segs, row))
    return plans


def select(args):
    out, ckdir, pattern = Path(args[0]), Path(args[1]), args[2]
    F = Fixtures(); plans = pool_plans(F)
    lens = np.array([p[1] for p in plans]); print("pool-built packs: median %d sentences, longest %d" % (np.median(lens), lens.max()))
    POOL = np.load(C / "wscan_pool_rev.npy", mmap_mode="r")
    runs = {}
    for k in range(NF):
        base = pattern.format(fold=k)
        for p in sorted(ckdir.glob("ckpt_%s_ep*.pt" % base), key=lambda q: int(q.stem.split("_ep")[-1])):
            runs[(k, int(p.stem.split("_ep")[-1]))] = p
    models = {key: load_tower(p) for key, p in runs.items()}
    print("%d checkpoints of %s" % (len(runs), pattern)); t0 = time.time()
    Z = {key: [] for key in models}
    with torch.no_grad():
        for i in range(0, F.NG, 32):
            part = plans[i:i + 32]; LM = max(p[1] for p in part)
            S = np.zeros((len(part), LM, 1024), np.float16)
            for b, (segs, row) in enumerate(part):
                d = F.doc(i + b); r = 0; blk = np.asarray(POOL[i + b])        # one sequential read per game
                if d is not None:
                    S[b, :len(d)] = d; r = len(d)
                for s, L in segs:
                    S[b, r:r + L] = blk[s:s + L]; r += L
                assert r == row
            Sf = torch.from_numpy(S).to(dev).float()
            m = torch.arange(LM, device=dev)[None, :] >= torch.as_tensor(lens[i:i + 32], device=dev)[:, None]
            for key, mdl in models.items():
                Z[key].append(mdl(Sf, m).float().cpu())
    Z = {key: torch.cat(v).numpy() for key, v in Z.items()}
    print("gallery encoded in %.0fs" % (time.time() - t0))
    sel = {}
    for k in range(NF):
        rows = validation_rows(F.names, F.art, F.variants, F.gA, F.split, k, NF, SPLIT_SEED)
        scores = {ep: rank_score(Z[(kk, ep)], F.queries(models[(kk, ep)]), F.gA, rows) for (kk, ep) in runs if kk == k}
        ep = max(scores, key=lambda e: (scores[e], -e))
        key = pattern.format(fold=k) + "@pool"
        sel[key] = dict(ckpt=str(runs[(k, ep)]), fold=k, ep=ep, score=scores[ep], scores={str(e): scores[e] for e in sorted(scores)})
        print("  fold %d: ep%d (score %.3f of %d validation queries)" % (k, ep, scores[ep], len(rows)))
    out.write_text(json.dumps(sel, indent=1), encoding="utf-8"); print("wrote", out)


# ------------------------------------------------------------------ cells
def cells(args):
    ckdir = Path(opt(args, "--ckpt-dir", ""))
    assert str(ckdir) not in ("", "."), "--ckpt-dir <folder with the ckpt_*.pt files> is required"
    out = Path(args[0]); plan = json.loads(PLAN.read_text(encoding="utf-8")); c = {}
    for key, v in plan.items():
        if v["cap"] != "_gpool":                                   # packs capped at 512, 1,024 or 2,048 sentences
            c[key] = dict(ckpt=str(ckdir / v["ckpt"]), fold=v["fold"], ep=v["ep"])
    for label, arm in JULY_ARMS.items():                           # the five main arms: prefix + whole pool (_gpool)
        for k in range(NF):
            v = plan["%s_fold%d_gpool" % (arm, k)]; c["%s|%d" % (label, k)] = dict(ckpt=str(ckdir / v["ckpt"]), fold=k, ep=v["ep"])
    for f in args[1:]:
        for key, v in json.loads(Path(f).read_text(encoding="utf-8")).items():
            p = Path(v["ckpt"])                                    # a name is looked up in --ckpt-dir
            c[key] = dict(ckpt=str(p if p.is_absolute() else ckdir / p), fold=v["fold"], ep=v["ep"])
    missing = [k for k, v in c.items() if not Path(v["ckpt"]).exists()]
    if missing:
        print("warning: %d checkpoints not found (first: %s); `encode` needs them all" % (len(missing), c[missing[0]]["ckpt"]))
    out.write_text(json.dumps(c, indent=1), encoding="utf-8"); print("%d cells -> %s" % (len(c), out))


# ------------------------------------------------------------------ encode
def open_store(h5_path):
    """the sentence store: the full pool (npy + meta) by default, embedding_h5.h5 with --h5"""
    if h5_path:
        import h5py
        print("reading", h5_path)
        return h5py.File(h5_path, "r", rdcc_nbytes=1 << 30, rdcc_nslots=1_000_003)
    meta = np.load(C / "full_pool_meta.npz", allow_pickle=True)
    print("reading", C / "full_pool_fp16.npy")
    return contextlib.nullcontext({"game_names": meta["game_names"], "game_review_offsets": meta["game_review_offsets"],
                                   "review_offsets": meta["review_offsets"],
                                   "vectors": np.load(C / "full_pool_fp16.npy", mmap_mode="r")})


def encode(args):
    cap = opt(args, "--cap", 4096); K = opt(args, "--k", 10); first = opt(args, "--games", 0); batch = opt(args, "--batch", 48)
    h5_path = opt(args, "--h5", "")
    cellf, out = Path(args[0]), Path(args[1]); out.parent.mkdir(parents=True, exist_ok=True)
    cell = json.loads(cellf.read_text(encoding="utf-8"))
    F = Fixtures(); NG = first or F.NG
    models = {key: load_tower(v["ckpt"]) for key, v in cell.items()}
    Z = {key: np.zeros((F.NG, K, 128), np.float32) for key in models}; Z["frozen"] = np.zeros((F.NG, K, 1024), np.float32)
    part = out.with_name(out.stem + ".part.npz"); g0 = 0
    if part.exists():
        d = np.load(part, allow_pickle=True)
        if list(d["keys"]) == list(Z) and int(d["cap"]) == cap and int(d["k"]) == K:
            g0 = int(d["done"])
            for key in Z:
                Z[key] = d["Z|" + key]
            print("resuming at game %d" % g0)
    print("%d towers + frozen | cap %d | K %d | games %d..%d" % (len(models), cap, K, g0, NG))
    buf_v, buf_l, buf_s = [], [], []; used = []

    @torch.no_grad()
    def flush():
        if not buf_v:
            return
        S = torch.zeros((len(buf_v), cap, 1024), dtype=torch.float16)
        for b, v in enumerate(buf_v):
            S[b, :len(v)] = torch.from_numpy(v)
        m_ = (torch.arange(cap)[None, :] >= torch.tensor(buf_l)[:, None]).to(dev)
        Sf = S.to(dev).float(); w = (~m_).float().unsqueeze(-1)
        fz = ((Sf * w).sum(1) / w.sum(1).clamp(min=1)).cpu().numpy()
        for b, (g, k) in enumerate(buf_s):
            Z["frozen"][g, k] = fz[b]
        for key, mdl in models.items():
            z = mdl(Sf, m_).float().cpu().numpy()
            for b, (g, k) in enumerate(buf_s):
                Z[key][g, k] = z[b]
        buf_v.clear(); buf_l.clear(); buf_s.clear()

    t0 = time.time()
    with open_store(h5_path) as h:
        hn = [x.decode() if isinstance(x, bytes) else str(x) for x in h["game_names"][:]]
        e2i = {n: i for i, n in enumerate(hn)}
        gro, ro, V = h["game_review_offsets"][:], h["review_offsets"][:], h["vectors"]
        for g in range(g0, NG):
            fi = e2i[F.names[g]]; r0, r1 = int(gro[fi]), int(gro[fi + 1])
            rs = ro[r0:r1].astype(np.int64); rl = (ro[r0 + 1:r1 + 1] - ro[r0:r1]).astype(np.int64)
            doc = F.doc(g); dl = 0 if doc is None else len(doc)
            plans = []
            for k in range(K):
                rk = np.random.default_rng(SPLIT_SEED + 5 + 1000 * k + g)
                row, segs = dl, []
                for j in rk.permutation(len(rl)):
                    L = int(rl[j])
                    if row + L <= cap:
                        segs.append((int(rs[j]), L)); row += L
                        if row == cap:                             # nothing longer than zero sentences can follow
                            break
                plans.append((segs, row))
            rows = [np.concatenate([np.arange(s, s + L) for s, L in segs]) if segs else np.zeros(0, np.int64) for segs, _ in plans]
            uni = np.unique(np.concatenate(rows))
            raw = read_rows(V, uni) if len(uni) else None
            for k, (segs, row) in enumerate(plans):
                parts = [doc] if doc is not None else []
                if len(rows[k]):
                    parts.append(rown(torch.from_numpy(raw[np.searchsorted(uni, rows[k])]).float()).numpy().astype(np.float16))
                buf_v.append(np.concatenate(parts) if len(parts) > 1 else parts[0])
                buf_l.append(row); buf_s.append((g, k)); used.append(row)
                if len(buf_v) >= batch:
                    flush()
            if (g + 1) % 100 == 0:
                el = time.time() - t0
                print("  %d/%d [%.1f min, %.0f min to go] median pack %d sentences"
                      % (g + 1, NG, el / 60, el / (g + 1 - g0) * (NG - g - 1) / 60, np.median(used)))
            if (g + 1) % 400 == 0 and g + 1 < NG:
                flush()
                np.savez(part, keys=np.array(list(Z)), cap=cap, k=K, done=g + 1, **{"Z|" + key: v for key, v in Z.items()})
        flush()
    Q = {"Q|" + key: F.queries(mdl) for key, mdl in models.items()}; Q["Q|frozen"] = F.frozen_queries()
    np.savez(out, cap=cap, k=K, games=NG, **Z, **Q)
    part.unlink(missing_ok=True)
    print("encoded in %.1f min -> %s" % ((time.time() - t0) / 60, out))


# ------------------------------------------------------------------ check
def check(args):
    d = np.load(args[0]); ref = np.load(args[1]); n = int(d["games"]); worst = 0.0
    for key in d.files:
        if key in ref.files and d[key].ndim == 3:
            a, b = d[key][:n], ref[key][:n]; diff = float(np.abs(a - b).max()); worst = max(worst, diff)
            print("%-34s %d games x %d draws: max |now - reference| = %.2e (vector norm about %.2f)"
                  % (key, n, a.shape[1], diff, np.linalg.norm(b, axis=-1).mean()))
    print("largest difference %.2e" % worst)


# ------------------------------------------------------------------ score
def score(args):
    from VICReg_review.text_variant_eval import micro_prf, train_anchor_ridge
    cell = json.loads(Path(args[0]).read_text(encoding="utf-8")); d = np.load(args[1]); out = Path(args[2])
    F = Fixtures(); y = np.load(C / "tag_labels.npz", allow_pickle=True)["y"]
    targs = SimpleNamespace(tag_text_train_frac=0.7, tag_text_val_frac=0.15, tag_text_split_seed=42, seed=42, tag_text_threshold_steps=33)
    l2 = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    res = {}
    for key in list(cell) + ["frozen"]:
        Zk, Za = d[key], d["Q|" + key]; K = Zk.shape[1]; gzk = l2(Zk); flat = gzk.reshape(F.NG * K, -1); gmean = l2(Zk.mean(1))
        for f in (range(NF) if key == "frozen" else [cell[key]["fold"]]):
            test_g, val_g = F.fold_games(f)
            ctag = {"train": [n for n in F.names if n not in test_g | val_g], "val": sorted(val_g), "test": sorted(test_g)}
            o = {}
            for var in ("neutral", "noname"):
                q = [i for i in range(len(F.art)) if F.art[i] in test_g and F.variants[i] == var]
                az = l2(Za[q]); tgt = F.gA[q]; own = np.arange(len(q))

                def ranks(s):
                    return (s > s[own, tgt][:, None]).sum(1) + 1
                per = [ranks(az @ gzk[:, kk].T) for kk in range(K)]
                h1 = [float((r == 1).mean()) for r in per]; h5 = [float((r <= 5).mean()) for r in per]
                o["pd_nm_" + var], o["pdsd_nm_" + var] = float(np.mean(h1)), float(np.std(h1))
                o["pd_h5_" + var], o["pdsd_h5_" + var] = float(np.mean(h5)), float(np.std(h5))
                o["draws_nm_" + var] = h1
                r = ranks((az @ flat.T).reshape(len(q), F.NG, K).max(2))                    # the nearest of a game's K packs
                o["mx_nm_" + var], o["mx_h5_" + var] = float((r == 1).mean()), float((r <= 5).mean())
                r = ranks(az @ gmean.T)                                                     # the K packs averaged into one anchor
                o["nm_" + var], o["h5_" + var] = float((r == 1).mean()), float((r <= 5).mean())
            ti = [i for i in range(len(F.art)) if F.art[i] in test_g and F.variants[i] == "neutral"]
            Xq = np.stack([Za[i] for i in ti]).astype(np.float32); lab = np.stack([y[F.n2i[F.art[i]]] for i in ti])
            tags = []
            for kk in range(K):
                sc, rg, _, th, _ = train_anchor_ridge(targs, Zk[:, kk], y, F.n2i, ctag)
                tags.append(float(micro_prf(lab, rg.predict(sc.transform(Xq)), th)["micro_f1"]))
            o["pd_test_tag"], o["pdsd_test_tag"] = float(np.mean(tags)), float(np.std(tags))
            res[key if key != "frozen" else "frozen|%d" % f] = o
            print("  %-34s fold %d: name %.3f  stripped %.3f (max over packs %.3f)  tag %.3f"
                  % (key, f, o["pd_nm_neutral"], o["pd_nm_noname"], o["mx_nm_noname"], o["pd_test_tag"]))
    out.write_text(json.dumps(dict(cap=int(d["cap"]), k=int(d["k"]), cells=cell, results=res), indent=1), encoding="utf-8")
    print("wrote", out)


# ------------------------------------------------------------------ table
def table(args):
    """five-fold mean ± population std of the per-draw means, a row per run family (the tables' cells)"""
    import re
    R = json.loads(Path(args[0]).read_text(encoding="utf-8"))["results"]; fam = {}
    for key, o in R.items():
        fam.setdefault(re.sub(r"_fold\d|\|\d", "", key), []).append(o)
    cols = ["pd_nm_neutral", "pd_h5_neutral", "pd_nm_noname", "pd_h5_noname", "pd_test_tag"]
    print("%-36s folds  %-15s %-15s %-15s %-15s %-15s" % ("run family", "Name@1", "Name@5", "Stripped@1", "Stripped@5", "tag F1"))
    for name in sorted(fam):
        v = fam[name]
        print("%-36s %d      %s" % (name, len(v), "   ".join("%.3f ± %.3f" % (np.mean([o[c] for o in v]), np.std([o[c] for o in v])) for c in cols)))


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd not in ("check", "table"):
        C = Path(opt(rest, "--data-dir", os.environ.get("W9_DATA_DIR", "")))
        assert str(C) not in ("", ".") and (C / "games.npz").exists(), "--data-dir <fusion cache folder> (or W9_DATA_DIR) is required"
    dict(select=select, cells=cells, encode=encode, check=check, score=score, table=table)[cmd](rest)
