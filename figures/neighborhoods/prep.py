# -*- coding: utf-8 -*-
r"""The fixed-split vectors behind Figure 9, Figure A3 and the neighborhood statistics of Appendix L.

For the 814 games with a wiki article and the six fixed-split towers the paper draws (plus a+u @ views),
at the checkpoints of contrast_experiment/w9/eval/fixed_split_ckpt_plan.json:

  queries   three versions of the article per game: the wiki text itself (wiki_clean_views.npz), the LLM
            rewrite with names kept and the rewrite with names stripped (wiki_eval.npz), encoded here
  anchors   the ten 4,096-sentence draws of every one of the 2,020 games, taken from the file
            fixed_split_eval.py writes (<its out folder>/draws_fixed.npz); nothing is drawn again here
  hits      per query and draw, whether its nearest of the 2,020 anchors is its own game; a query's
            score is its hit rate over the ten draws
  frozen    the frozen embedder: a game is the mean of its anchor pack's sentence embeddings, a query
            the mean of its own sentences

Written into <HOOD_WORK>/fixed: allgames.npz, frozen.npz and ckpt/ref_anchors_<key>{,_mean,_draws}.npy
(the first draw, the mean of the ten unit anchors, all ten), the files fig_triplet.py reads.

    python figures/neighborhoods/prep.py --ckpt-dir <checkpoints> --draws <fixed_split_eval out>/draws_fixed.npz

Needs torch with CUDA and numpy; W9_DATA_DIR names the data folder.
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W9 = ROOT / "contrast_experiment" / "w9"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(W9)); sys.path.insert(0, str(W9 / "Pod"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import hood_env  # noqa: E402
from Pod.w9_cv_worker import SetPoolN  # noqa: E402

PLAN = W9 / "eval" / "fixed_split_ckpt_plan.json"
# the order of the panels' towers in allgames.npz, and their keys in the plan file and in draws_fixed.npz
MODELS = [("SimCLR-style", "bce_cetf_fixed"), ("I + SimCLR", "i2bce_icetf_fixed"), ("A+U @ views", "i4uni2_icetf_fixed"),
          ("VICReg", "epd_v20i10c20_cetf_fixed"), ("CE", "ce_cetf_fixed_g2048"), ("I-CE", "i2ce_icetf_fixed_g2048")]
VARS = ["wiki", "neutral", "noname"]
dev = torch.device("cuda")


def rown(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)


def nrm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def load_tower(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    sd = {k: v for k, v in sd.items() if k == "q0" or k.startswith(("attn.", "head."))}
    m = SetPoolN(int(sd["q0"].shape[1])).to(dev); m.load_state_dict(sd)
    return m.eval()


@torch.no_grad()
def encode_rows(model, rows, chunk=64):
    out = []
    for i in range(0, len(rows), chunk):
        part = rows[i:i + chunk]; LM = max(r.shape[0] for r in part)
        X = np.zeros((len(part), LM, 1024), np.float16)
        for k, r in enumerate(part):
            X[k, :r.shape[0]] = r
        L = torch.as_tensor([r.shape[0] for r in part], device=dev)
        out.append(model(torch.from_numpy(X).to(dev), torch.arange(LM, device=dev)[None, :] >= L[:, None]).float().cpu())
    return torch.cat(out).numpy()


def hits(Q, G, own):
    """Q [games, versions, d] unit queries, G [2020, d] unit anchors -> [games, versions]: own game nearest"""
    n, v, d = Q.shape
    return (np.argmax(Q.reshape(-1, d) @ G.T, axis=1) == np.repeat(own, v)).reshape(n, v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--draws", required=True, help="draws_fixed.npz of contrast_experiment/w9/eval/fixed_split_eval.py")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    C = hood_env.need_data_dir(); ckdir = Path(a.ckpt_dir); OUT = hood_env.BIN; (OUT / "ckpt").mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    D = np.load(a.draws); K = int(D["k"])
    names = [str(x) for x in np.load(C / "games.npz", allow_pickle=True)["names"]]; NG = len(names)
    assert int(D["cap"]) == 4096 and int(D["games"]) == NG, "the draws file must hold every game at 4,096 sentences (no --games)"
    titles_all = json.loads(hood_env.TITLES.read_text(encoding="utf-8")); assert len(titles_all) == NG
    a2n = {n.split("_")[0]: n for n in names}; n2i = {n: i for i, n in enumerate(names)}
    sp = json.loads((C / "wiki_eval_split.json").read_text())
    status = np.array(["train"] * NG, dtype=object)
    for x in sp["test"]:
        status[n2i[a2n[x]]] = "test"
    for x in sp["val"]:
        status[n2i[a2n[x]]] = "val"
    A = np.load(C / "wiki_eval.npz", allow_pickle=True)
    art = [str(x) for x in A["names"]]; variants = [str(x) for x in A["variants"]]; AS, AL = A["S"], A["S_len"]
    qidx = {}
    for i, (n, v) in enumerate(zip(art, variants)):
        qidx.setdefault(n, {})[v] = i
    WC = np.load(C / "wiki_clean_views.npz", allow_pickle=True); wc = {str(n): i for i, n in enumerate(WC["names"])}; WS, WL = WC["S"], WC["S_len"]
    games = sorted(n2i[n] for n in qidx if all(v in qidx[n] for v in VARS[1:]) and n in wc); NQ = len(games); own = np.asarray(games)
    print("%d wiki games: %s" % (NQ, {s: int((status[games] == s).sum()) for s in ("train", "val", "test")}))
    raw = []                                                   # per game: the three versions' sentence rows, as stored
    for g in games:
        n = names[g]
        raw.append([WS[wc[n], :int(WL[wc[n]])].astype(np.float32)] + [AS[qidx[n][v], :int(AL[qidx[n][v]])].astype(np.float32) for v in VARS[1:]])
    qrows = [rown(r).astype(np.float16) for rows in raw for r in rows]

    qs = np.zeros((len(MODELS), NQ, len(VARS), 128), np.float32); HD = np.zeros((len(MODELS), NQ, len(VARS), K), bool); keys = []
    test = status[games] == "test"; J = VARS.index("noname")
    print("%-14s %-28s test-set Stripped hit@1 over the %d draws, mean ± std" % ("tower", "checkpoint", K))
    for i, (label, key) in enumerate(MODELS):
        tower = load_tower(ckdir / plan[key]["ckpt"]); keys.append(key)
        qs[i] = encode_rows(tower, qrows).reshape(NQ, len(VARS), 128)
        G = nrm(D[key].astype(np.float32)); Q = nrm(qs[i])
        for d in range(K):
            HD[i, :, :, d] = hits(Q, G[:, d], own)
        np.save(OUT / "ckpt" / f"ref_anchors_{key}.npy", G[:, 0]); np.save(OUT / "ckpt" / f"ref_anchors_{key}_mean.npy", nrm(G.mean(1)))
        np.save(OUT / "ckpt" / f"ref_anchors_{key}_draws.npy", G)
        s = HD[i][test][:, J].mean(0); print("%-14s %-28s %.3f ± %.3f" % (label, plan[key]["ckpt"][8:-3], s.mean(), s.std()))
    np.savez(OUT / "allgames.npz", models=np.array([m for m, _ in MODELS]), keys=np.array(keys), variants=np.array(VARS), game_index=own,
             appids=np.array([names[g].split("_")[0] for g in games]), titles=np.array([titles_all[g] for g in games]),
             status=np.array([status[g] for g in games]), resample=np.zeros((len(MODELS), NQ, 0, 128), np.float32),
             hit_resample=np.zeros((len(MODELS), NQ, 0), bool), queries=qs, hit_queries=HD[..., 0], hit_rate=HD.mean(-1).astype(np.float32), hit_draws=HD)

    # the frozen embedder: a query is the mean of its row-standardized sentences
    fq = np.zeros((NQ, len(VARS), 1024), np.float32)
    for k, rows in enumerate(raw):
        for j, r in enumerate(rows):
            fq[k, j] = rown(r).mean(0)
    fq = fq / (np.linalg.norm(fq, axis=-1, keepdims=True) + 1e-8)
    G = nrm(D["frozen"].astype(np.float32)); Q = nrm(fq)
    HF = np.stack([hits(Q, G[:, d], own) for d in range(K)], -1)
    np.save(OUT / "ckpt" / "ref_anchors_frozen.npy", G[:, 0]); np.save(OUT / "ckpt" / "ref_anchors_frozen_mean.npy", nrm(G.mean(1)))
    np.save(OUT / "ckpt" / "ref_anchors_frozen_draws.npy", G)
    np.savez(OUT / "frozen.npz", queries=fq, hit_queries=HF[..., 0], hit_rate=HF.mean(-1).astype(np.float32), hit_draws=HF,
             game_index=own, variants=np.array(VARS))
    s = HF[test][:, J].mean(0); print("%-14s %-28s %.3f ± %.3f" % ("Frozen", "(mean of the pack)", s.mean(), s.std()))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
