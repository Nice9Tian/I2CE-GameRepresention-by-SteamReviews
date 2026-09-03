# -*- coding: utf-8 -*-
"""gallery_sample.py: how the anchor gallery of the published recipe is set up
for larice / I-CE (the full standing gallery; pfc_sample.py is the deployable
sharded-window variant built on the same pieces).

The anchor pack is the recipe that turns a game's raw material into the fixed
pack the CE term classifies against:

    anchor pack of game g = [store-page (or document) sentences first]
                          + [whole reviews, in one fixed random permutation,
                             appended until the sentence budget m_a is full;
                             a review that does not fit is skipped, never cut]

The pack is drawn ONCE (a fixed random pack) and then re-encoded by the tower
at EVERY step with gradients on, for every training item, so the standing
gallery of anchors moves with the tower ("batch = all" negatives, no EMA
teacher, no memory queue, no mining). The strong views are the opposite: a
fresh draw of whole reviews per item per step (rejection-sampled by review
length until at least 16 sentences), plus one document view.

This file is a self-contained, CPU-runnable illustration on synthetic
embeddings. The production path is:

    dataset_builder/build_assets.py      builds the packs -> wscan_gal_rev.npz
                                          (gal, gal_len, gal_doc_len) and the
                                          review pool + per-review id table
    steam_reviews_framework/anchors.py    gallery_train(): the standing gallery
    steam_reviews_framework/sampler.py    accept_prob() / sample_views()
    steam_reviews_framework/train.py      the step: gallery -> views -> ice_loss

The windowed teacher of the appendix (swin: a ring over the catalog, two
micro-passes of 168 fresh packs per step) is a different anchor supply and
lives in contrast_experiment/w9/Pod/w9_a100_worker.py as the arm
`wcle_swin168step84loop2i2ce_icetf` (five-fold: w9_cv_worker.py). It is not
a LariceConfig option because the tower and loss do not care where the
gallery rows come from.

    py main_model/gallery_sample.py          # runs ~20 steps on random data
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main_model import LariceConfig, LariceTower, ice_loss  # noqa: E402

# ----------------------------------------------------------------- settings
D_IN = 1024          # upstream sentence-embedding width (Qwen3-Embedding-0.6B)
M_A = 512            # anchor-pack budget in sentences (paper: 512 to 4,096)
VIEW_MIN = 16        # a strong view holds whole reviews until >= 16 sentences
V = 4                # views per item: V-1 review views + 1 document view


# ------------------------------------------------- 1. the anchor packs
def build_anchor_pack(doc_sents: np.ndarray | None, reviews: list[np.ndarray],
                      cap: int, rng: np.random.Generator):
    """One fixed anchor pack: document prefix first, then whole reviews in a
    random permutation until the cap is full (a review that does not fit is
    skipped). Returns (pack [cap, D], length, doc_prefix_length)."""
    pack = np.zeros((cap, doc_sents.shape[-1] if doc_sents is not None
                     else reviews[0].shape[-1]), np.float16)
    row = 0
    doc_len = 0
    if doc_sents is not None:
        dl = min(len(doc_sents), cap)
        pack[:dl] = doc_sents[:dl]
        row, doc_len = dl, dl
    for j in rng.permutation(len(reviews)):
        L = len(reviews[j])
        if row + L <= cap:                         # whole review or nothing
            pack[row:row + L] = reviews[j]
            row += L
    return pack, row, doc_len


def build_gallery_assets(games, cap, rng):
    """Mirror of wscan_gal_rev.npz: gal [G, cap, D], gal_len [G], gal_doc_len [G]."""
    packs, lens, dlens = [], [], []
    for g in games:
        p, n, dl = build_anchor_pack(g["doc"], g["reviews"], cap, rng)
        packs.append(p); lens.append(n); dlens.append(dl)
    gal = torch.from_numpy(np.stack(packs))
    gal_len = torch.tensor(lens)
    pad = torch.arange(cap)[None, :] >= gal_len[:, None]   # True = padding
    return gal, pad, torch.tensor(dlens)


# ---------------------------------------------- 2. strong views, fresh draws
def accept_prob(lens):
    """Pr(accept | r) = 0.2 + 0.7 (r - r_min)/(r_max - r_min); flat 0.9 if all tie."""
    lens = np.asarray(lens, np.float64)
    lo, hi = lens.min(), lens.max()
    if hi <= lo:
        return np.full_like(lens, 0.9)
    return 0.2 + 0.7 * (lens - lo) / (hi - lo)


def sample_review_view(reviews, rng, m_min=VIEW_MIN):
    """Rejection-draw WHOLE reviews until >= m_min sentences; the last one is never truncated."""
    lens = np.array([len(r) for r in reviews])
    a = accept_prob(lens)
    taken = np.zeros(len(reviews), bool)
    chosen, tot = [], 0
    while tot < m_min and not taken.all():
        i = int(rng.integers(len(reviews)))
        if taken[i]:
            continue
        if rng.random() < a[i]:
            taken[i] = True
            chosen.append(i)
            tot += int(lens[i])
    return np.concatenate([reviews[i] for i in chosen])


def pad_batch(blocks):
    L = max(len(b) for b in blocks)
    out = np.zeros((len(blocks), L, blocks[0].shape[-1]), np.float16)
    lens = np.zeros(len(blocks), np.int64)
    for k, b in enumerate(blocks):
        out[k, :len(b)] = b
        lens[k] = len(b)
    x = torch.from_numpy(out)
    mask = torch.arange(L)[None, :] >= torch.tensor(lens)[:, None]
    return x, mask


def sample_views(games, gids, rng):
    """V views per item: V-1 review views plus one document view (a review
    view again when the item has no document)."""
    views = []
    for _ in range(V - 1):
        views.append(pad_batch([sample_review_view(games[g]["reviews"], rng) for g in gids]))
    doc_blocks = [games[g]["doc"] if games[g]["doc"] is not None
                  else sample_review_view(games[g]["reviews"], rng) for g in gids]
    views.append(pad_batch(doc_blocks))
    return views


# --------------------------------------------------- 3. the standing gallery
def encode_gallery(tower, gal, pad, rows, chunk=256):
    """gallery_train(): encode the anchor packs of the TRAINING items every
    step, gradients on. CE targets are positions in this row order."""
    outs = []
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        outs.append(tower(gal[r], pad[r]))
    return torch.cat(outs)


# ------------------------------------------------------------------- demo
def make_synthetic_games(n_games, rng, d=D_IN):
    """Random stand-ins for the sentence embeddings: each game has a latent
    centre; reviews and the document are noisy sentence sets around it."""
    games = []
    for _ in range(n_games):
        c = rng.standard_normal(d)
        n_rev = int(rng.integers(20, 60))
        reviews = [(c + 3.0 * rng.standard_normal((int(rng.integers(1, 12)), d))).astype(np.float16)
                   for _ in range(n_rev)]
        doc = (c + 1.5 * rng.standard_normal((int(rng.integers(8, 30)), d))).astype(np.float16) \
            if rng.random() < 0.9 else None
        games.append({"reviews": reviews, "doc": doc})
    return games


def main(steps=20, n_games=64, batch=16, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    games = make_synthetic_games(n_games, rng)

    # (1) fixed packs, drawn once
    gal, pad, doc_len = build_gallery_assets(games, M_A, rng)
    train_rows = np.arange(n_games)                     # here every item trains
    pos_of_g = {g: i for i, g in enumerate(train_rows)}  # gallery column per item

    cfg = LariceConfig(num_queries=4, dim_model=128, input_dim=D_IN,
                       num_views=V, tau=0.02, inv_weight=2.0, readout="pool")
    tower = LariceTower(cfg)
    opt = torch.optim.AdamW(tower.parameters(), lr=5e-4, weight_decay=1e-4)

    for step in range(steps):
        gids = rng.choice(train_rows, batch, replace=False)
        targets = torch.tensor([pos_of_g[int(g)] for g in gids])
        # (3) standing gallery: every training pack re-encoded, gradients on
        gallery = encode_gallery(tower, gal, pad, train_rows)
        # (2) fresh strong views for the items of this step
        z = torch.stack([tower(x, m) for x, m in sample_views(games, gids, rng)], dim=1)  # [B, V, D]
        loss = ice_loss(z, gallery, targets, cfg)         # CE against the gallery + inv_weight * I
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 5 == 0 or step == steps - 1:
            with torch.no_grad():
                hit = (z[:, 0] @ gallery.T).argmax(-1).eq(targets).float().mean().item()
            print(f"step {step:3d}  loss {loss.item():.3f}  view->own-anchor hit@1 {hit:.2f}")
    print(f"packs: budget {M_A}, mean fill {int(gal.shape[1]) - int(pad.sum(1).float().mean())}, "
          f"doc-bearing {(doc_len > 0).sum().item()}/{n_games}")


if __name__ == "__main__":
    main()
