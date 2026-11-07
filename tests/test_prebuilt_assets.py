"""The packaged path on prebuilt assets, without the review h5.

build_assets.py step 3 rebuilds the anchor packs at a new budget from the
review pool and full_pool_meta.npz alone. These tests build a tiny synthetic
corpus, run the h5 build at two budgets, drop the h5, and require the
pool-only rebuild to reproduce the h5 build's packs exactly, with every
store-page prefix row-standardised.

Run: python -m unittest discover -s tests -p test_prebuilt_assets.py -v
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "dataset_builder" / "build_assets.py"


def write_corpus(assets: Path, h5_path: Path, seed=0, sp_scale=1.0, sp_shift=0.0):
    """Six games, two of them with more review sentences than the 2,048 pool holds.
    The store-page vectors are N(sp_shift, sp_scale^2) per component, raw like sp_raw_views.npz."""
    rng = np.random.default_rng(seed)
    names = [f"{100 + g}_Game {g}" for g in range(6)]
    n_reviews = [40, 150, 12, 220, 3, 60]
    lens = [rng.integers(1, 30, n) for n in n_reviews]
    review_offsets = np.concatenate([[0], np.cumsum(np.concatenate(lens))]).astype(np.int64)
    game_review_offsets = np.concatenate([[0], np.cumsum(n_reviews)]).astype(np.int64)
    vectors = rng.standard_normal((int(review_offsets[-1]), 1024)).astype(np.float16)
    with h5py.File(h5_path, "w") as h:
        h["game_names"] = np.array(names, dtype=h5py.string_dtype())
        h["game_review_offsets"] = game_review_offsets
        h["review_offsets"] = review_offsets
        h["vectors"] = vectors
    assets.mkdir(parents=True, exist_ok=True)
    np.savez(assets / "games.npz", names=np.array(names, object))
    has_doc = [0, 1, 3, 4]                                   # games 2 and 5 have no store page
    S_len = rng.integers(5, 40, len(has_doc)).astype(np.int32)
    S = np.zeros((len(has_doc), int(S_len.max()), 1024), np.float16)
    for i, L in enumerate(S_len):
        S[i, :L] = rng.standard_normal((L, 1024)) * sp_scale + sp_shift
    np.savez(assets / "sp_raw_views.npz", S=S, S_len=S_len, gidx=np.array(has_doc, np.int64),
             names=np.array([names[g] for g in has_doc], object))
    return names, game_review_offsets, review_offsets


class PrebuiltAssets(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prebuilt-assets-")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def build(self, assets, h5_path, cap):
        env = dict(os.environ, LARICE_ASSETS=str(assets), LARICE_EMBED_H5=str(h5_path),
                   PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
        env.pop("LARICE_ANCHOR_CAP", None)
        r = subprocess.run([sys.executable, "-B", str(BUILD), "--step", "3", "--anchor-cap", str(cap)],
                           env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout

    def test_pool_rebuild_matches_the_h5_build(self):
        small, big, h5_path = self.work / "small", self.work / "big", self.work / "embedding_h5.h5"
        names, gro, ro = write_corpus(small, h5_path)
        write_corpus(big, self.work / "unused.h5")            # same games and store pages
        self.build(small, h5_path, 64)                       # what a 512 download is to the real data
        self.build(big, h5_path, 256)                        # the reference at the new budget
        for f in ("wscan_pool_rev.npy", "wscan_pool_rev_rid.npy"):
            np.testing.assert_array_equal(np.load(small / f), np.load(big / f))

        np.savez(small / "full_pool_meta.npz", game_names=np.array(names, object),
                 game_review_offsets=gro, review_offsets=ro)
        h5_path.unlink()                                     # from here on there is no h5
        out = self.build(small, h5_path, 256)
        self.assertIn("rebuilding 256-sentence packs from the review pool", out)
        got, want = np.load(small / "wscan_gal_rev_pool256.npz"), np.load(big / "wscan_gal_rev.npz")
        for key in ("gal", "gal_len", "gal_doc_len"):
            np.testing.assert_array_equal(got[key], want[key])
        self.assertEqual(np.load(small / "wscan_gal_rev.npz")["gal"].shape[1], 64)   # left as it was

        out = self.build(small, h5_path, 256)                # second run: found, not rebuilt
        self.assertIn("wscan_gal_rev_pool256.npz — skip", out)

        env = dict(os.environ, LARICE_ASSETS=str(small))
        code = ("from dataset_builder.paths import gallery_path; "
                "print(gallery_path(64).name, gallery_path(256).name, gallery_path(1024).name)")
        r = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, env=env,
                           capture_output=True, text=True)
        self.assertEqual(r.stdout.split(),
                         ["wscan_gal_rev.npz", "wscan_gal_rev_pool256.npz", "wscan_gal_rev_pool1024.npz"],
                         r.stderr)

    def test_store_page_prefix_is_row_standardised(self):
        """Store-page vectors far from mean 0 / std 1 enter the h5-built packs, the
        pool-rebuilt packs and the pseudo-queries row-standardised, the way the w9
        workers standardise sp_raw_views.npz (unbiased std, eps 1e-6)."""
        assets, h5_path = self.work / "a", self.work / "embedding_h5.h5"
        names, gro, ro = write_corpus(assets, h5_path, sp_scale=3.0, sp_shift=0.5)
        self.build(assets, h5_path, 64)
        np.savez(assets / "full_pool_meta.npz", game_names=np.array(names, object),
                 game_review_offsets=gro, review_offsets=ro)
        h5_path.unlink()
        self.build(assets, h5_path, 256)                     # the pool rebuild

        SPV = np.load(assets / "sp_raw_views.npz", allow_pickle=True)
        raw = {int(g): SPV["S"][i, :SPV["S_len"][i]].astype(np.float32)
               for i, g in enumerate(SPV["gidx"])}

        def rown(x):
            return (x - x.mean(-1, keepdims=True)) / (x.std(-1, ddof=1, keepdims=True) + 1e-6)

        for pack in ("wscan_gal_rev.npz", "wscan_gal_rev_pool256.npz"):
            G = np.load(assets / pack)
            for g, x in raw.items():
                self.assertEqual(int(G["gal_doc_len"][g]), len(x))
                np.testing.assert_allclose(G["gal"][g, :len(x)].astype(np.float32), rown(x),
                                           atol=4e-3, err_msg=f"{pack}, game {g}")
        Q = np.load(assets / "ss_queries_rev.npz")
        S = np.load(assets / "ss_queries_rev_S.npy")
        checked = 0
        for j, g in enumerate(Q["gidx"]):
            if int(g) in raw:
                x = raw[int(g)]
                np.testing.assert_allclose(S[Q["off"][j]:Q["off"][j] + len(x)].astype(np.float32),
                                           rown(x), atol=4e-3, err_msg=f"pseudo-query {j}")
                checked += 1
        self.assertEqual(checked, 4 * len(raw))

    def test_no_h5_and_no_meta_stops_with_the_options(self):
        assets, h5_path = self.work / "a", self.work / "embedding_h5.h5"
        write_corpus(assets, h5_path)
        self.build(assets, h5_path, 64)
        h5_path.unlink()
        env = dict(os.environ, LARICE_ASSETS=str(assets), LARICE_EMBED_H5=str(h5_path))
        r = subprocess.run([sys.executable, "-B", str(BUILD), "--step", "3", "--anchor-cap", "256"],
                           env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("full_pool_meta.npz", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
