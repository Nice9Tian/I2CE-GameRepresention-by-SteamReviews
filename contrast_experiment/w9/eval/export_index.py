# -*- coding: utf-8 -*-
r"""Write the 128-d game index: one L2-normalised vector per game.

    python contrast_experiment/w9/eval/export_index.py --ckpt-dir checkpoints --data-dir /path/to/data

A game's vector is the tower's encoding of its 4,096-sentence anchor pack,
the single draw `wscan_gal_rev_g4096.npz` of the data folder (store-page
prefix, then whole reviews). The tower is one run of the checkpoint plan
`fixed_split_ckpt_plan.json`; the default, `i2ce_icetf_fixed_g2048`, is the
fixed-split I-CE tower of Figure 9, trained at the 2,048-sentence budget and
taken at the checkpoint the rank criterion of Section 5 selects (epoch
1,750). Against this pack it places 0.956 of the 204 test games' name-intact
rewrites and 0.716 of their name-stripped rewrites at rank one (over ten
fresh anchor draws the name-stripped reading is 0.743, Appendix L); `--check`
prints those two readings from the vectors just written.

The index covers all 2,020 games, the 407 validation and test games of the
fixed split among them: their packs are encoded by a tower that never saw
their text, which is the cold-start case the paper measures.

Output (`--out`, default `index/i2ce_index_128d.npz` at the repository root):
    names   (2020,)      game identifiers, `<appid>_<n>`, in the order of games.npz
    appids  (2020,)      the Steam app id of each game
    Z       (2020, 128)  float32, unit rows; cosine similarity is `Z @ q`
    tower, checkpoint, sha256, anchors   what the vectors were made from

A query is encoded by the same tower: `SetPoolN(4)` with the checkpoint's
weights, applied to the query's row-standardised sentence embeddings
(Qwen3-Embedding-0.6B, 1,024-d), as `fixed_split_eval.py` does.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fixed_split_eval import PLAN, SetPoolN, l2, npz_mmap, rown, tower_state  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True, help="folder with the ckpt_*.pt files of the checkpoint repository")
    ap.add_argument("--data-dir", default=os.environ.get("W9_DATA_DIR", ""), help="the data folder (or W9_DATA_DIR)")
    ap.add_argument("--run", default="i2ce_icetf_fixed_g2048", help="a key of fixed_split_ckpt_plan.json")
    ap.add_argument("--out", default=str(HERE.parents[2] / "index" / "i2ce_index_128d.npz"))
    ap.add_argument("--chunk", type=int, default=32, help="anchor packs per forward pass")
    ap.add_argument("--check", action="store_true", help="also read test-set hit@1 from the vectors written")
    a = ap.parse_args()
    data, ckdir = Path(a.data_dir), Path(a.ckpt_dir)
    assert a.data_dir and (data / "games.npz").exists(), "--data-dir <data folder> (or W9_DATA_DIR) is required"
    entry = json.loads(PLAN.read_text(encoding="utf-8"))[a.run]
    ckpt, galp = ckdir / entry["ckpt"], data / "wscan_gal_rev_g4096.npz"
    assert ckpt.exists(), f"{ckpt} is missing"
    assert galp.exists(), f"{galp} is missing: the fixed split's 4,096-sentence anchor pack, a file of the dataset"
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert sha == entry["sha256"], f"{ckpt.name} is not the file the plan names (sha256 {sha})"

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    names = [str(x) for x in np.load(data / "games.npz", allow_pickle=True)["names"]]
    sd = tower_state(ckpt)
    model = SetPoolN(int(sd["q0"].shape[1])).to(dev).eval()
    model.load_state_dict(sd)
    G = npz_mmap(galp, "gal")
    with np.load(galp) as z:
        glen = np.asarray(z["gal_len"], np.int64)
    assert G.shape[0] == len(names)
    Z = np.zeros((len(names), 128), np.float32)
    with torch.no_grad():
        ar = torch.arange(G.shape[1], device=dev)
        for i in range(0, len(names), a.chunk):
            s = torch.from_numpy(np.ascontiguousarray(G[i:i + a.chunk])).to(dev)
            msk = ar[None, :] >= torch.from_numpy(glen[i:i + a.chunk]).to(dev)[:, None]
            Z[i:i + a.chunk] = model(s, msk).float().cpu().numpy()
    Z = l2(Z).astype(np.float32)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=np.array(names), appids=np.array([n.split("_")[0] for n in names]), Z=Z,
             tower=a.run, checkpoint=entry["ckpt"], sha256=sha, anchors=galp.name)
    print(f"wrote {out}: {Z.shape[0]} games x {Z.shape[1]} dimensions "
          f"(tower {a.run}, epoch {entry['ep']}, anchors {galp.name}, median pack {int(np.median(glen))} sentences)")

    if a.check:
        A = np.load(data / "wiki_eval.npz", allow_pickle=True)
        art = [str(x) for x in A["names"]]
        variants = [str(x) for x in A["variants"]]
        gA = np.asarray(A["gidx"])
        test = {n.split("_")[0] for n in names} & set(json.loads((data / "wiki_eval_split.json").read_text())["test"])
        S, S_len, cols = A["S"], torch.tensor(A["S_len"]), torch.arange(A["S"].shape[1], device=dev)
        with torch.no_grad():                    # queries go to the GPU a batch at a time
            Za = l2(torch.cat([model(rown(torch.tensor(S[i:i + 256]).to(dev).float()).half(),
                                     cols[None, :] >= S_len[i:i + 256].to(dev)[:, None]).float().cpu()
                               for i in range(0, S.shape[0], 256)]).numpy())
        for var, label in (("neutral", "name-intact"), ("noname", "name-stripped")):
            q = [i for i in range(len(art)) if variants[i] == var and art[i].split("_")[0] in test]
            hit = float((np.argmax(Za[q] @ Z.T, axis=1) == gA[q]).mean())
            print(f"  test-set hit@1, {label} rewrites ({len(q)} queries): {hit:.3f}")


if __name__ == "__main__":
    main()
