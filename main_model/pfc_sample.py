# -*- coding: utf-8 -*-
"""pfc_sample.py — the deployable anchor supply: sharded sliding window
(Partial-FC style) over a memory-mapped pack store.

The published objective classifies every strong view against the standing
gallery of ALL training anchors, re-encoded with gradients at every step
(see gallery_sample.py). That is affordable at n = 1,694 games but its device
memory grows linearly with (number of items) x (pack size). The variant we
propose for deployment keeps the same loss and the same tower and changes
only where the negatives come from:

  ring        the catalog laid out on a ring in a fixed order, split into
              K shards (κ in Appendix H of the paper); shard k owns every
              K-th ring position
  window      per micro-pass, shard k fresh-encodes the next w packs after
              its own pointer (gradients on), then advances the pointer by
              S; T micro-passes per optimizer step
  partition   the CE softmax of a view runs over [own anchors of the batch
              | window of shard 1 | ... | window of shard K]; a batch item
              that also sits inside a window is masked there so it is a
              positive exactly once
  memory      each window encode is gradient-checkpointed, so only the
              [w, dim] outputs stay live and the [w, m_a, D_in] activation
              is recomputed one window at a time in backward; the pack
              store itself is a memory-mapped fp16 array on disk and only
              the rows of the current windows are paged in
  backward    the CE of each micro-pass is back-propagated immediately, so
              window activations are freed between passes; the invariance
              term over the strong views is back-propagated once at the end

Device memory is therefore constant in the catalog size (own anchors plus
one window at a time), per-step compute is Θ(T·(G + K·w)·m_a·Q·d), and a
K-shard single process reproduces exactly what K workers would compute
with an all-reduce of their window logits. With K = 1 this is the windowed
teacher of the appendix (w = 168, S = 84, T = 2); the research
implementation is the `--pfc-shards K --pfc-window w` path of
contrast_experiment/w9/Pod/w9_cv_worker.py on a swin arm such as
`wcle_swin168step84loop2i2ce_icetf`, with `--full-pool-path` supplying the
memory-mapped store.

This file is a self-contained CPU-runnable illustration on synthetic
embeddings:

    py main_model/pfc_sample.py          # ~20 steps, K = 2 shards
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main_model import LariceConfig, LariceTower, invariance_loss  # noqa: E402
from main_model.gallery_sample import (build_gallery_assets, make_synthetic_games,  # noqa: E402
                                   sample_views, D_IN, M_A, V)


# ------------------------------------------------ the memory-mapped store
def write_pack_store(gal: torch.Tensor, path: Path) -> np.memmap:
    """Persist the fixed packs as an fp16 memmap; training pages in rows on demand."""
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=tuple(gal.shape))
    arr[:] = gal.numpy()
    arr.flush()
    return np.load(path, mmap_mode="r")


def load_rows(store: np.memmap, pad: torch.Tensor, rows: np.ndarray):
    """Page the packs of `rows` into memory (the streamed store's per-window I/O)."""
    x = torch.from_numpy(np.ascontiguousarray(store[np.sort(rows)]))
    order = np.argsort(np.argsort(rows))            # restore the requested order
    return x[order], pad[rows]


# ------------------------------------------------- the sharded ring supply
class ShardedRing:
    """K shards over the ring of training rows; each keeps its own pointer."""

    def __init__(self, train_rows: np.ndarray, K: int, w: int, S: int, T: int):
        self.shards = [np.sort(train_rows[k::K]) for k in range(K)]
        self.ptr = [(k * 997) % max(len(s), 1) for k, s in enumerate(self.shards)]  # staggered sweeps
        self.w, self.S, self.T = w, S, T

    def windows(self):
        """One window of rows per shard for the current micro-pass; advance the pointers by S."""
        out = []
        for k, s in enumerate(self.shards):
            wl = min(self.w, len(s))
            out.append(s[(self.ptr[k] + np.arange(wl)) % len(s)])
            self.ptr[k] = int((self.ptr[k] + self.S) % len(s))
        return out


def encode_window(tower, store, pad, rows):
    """Gradient-checkpointed window encode: keeps only the [w, dim] output live."""
    x, m = load_rows(store, pad, rows)

    def _enc(x_, m_):
        return tower(x_, m_)

    return checkpoint(_enc, x, m, use_reentrant=False)


def pfc_step(tower, opt, cfg, store, pad, ring, games, gids, rng, inv_tau):
    """One optimizer step of the sharded sliding-window objective."""
    opt.zero_grad()
    gids_t = torch.as_tensor(gids)
    # own anchors of the batch: fresh, gradients on, always in the field
    e_now = tower(*load_rows(store, pad, gids)).float()
    # strong views of the batch items (V - 1 review views + 1 document view)
    zs = [tower(x, m) for x, m in sample_views(games, gids, rng)]
    own = torch.arange(len(gids))                    # each item's column in e_now
    ce_total = 0.0
    for _ in range(ring.T):                          # T micro-passes
        cols_all = [[z.float() @ e_now.T * inv_tau] for z in zs]
        for rows in ring.windows():                  # K shard windows this pass
            e_win = encode_window(tower, store, pad, rows).float()
            dup = torch.isin(torch.as_tensor(rows), gids_t)   # an item already positive in e_now
            for c, z in zip(cols_all, zs):
                c.append((z.float() @ e_win.T * inv_tau).masked_fill(dup[None, :], -1e4))
        loss_ce = sum(F.cross_entropy(torch.cat(c, 1), own) for c in cols_all)
        loss_ce.backward(retain_graph=True)          # window activations freed here
        ce_total += float(loss_ce)
    loss_inv = cfg.inv_weight * invariance_loss(zs)  # view axis, once per step
    loss_inv.backward()
    opt.step()
    return ce_total / ring.T, float(loss_inv)


def main(steps=20, n_games=96, batch=16, K=2, w=12, S=6, T=2, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    games = make_synthetic_games(n_games, rng)
    gal, pad, _ = build_gallery_assets(games, M_A, rng)
    tmp = Path(tempfile.mkdtemp()) / "packs.npy"
    store = write_pack_store(gal, tmp)               # the memory-mapped store
    train_rows = np.arange(n_games)
    ring = ShardedRing(train_rows, K=K, w=w, S=S, T=T)

    cfg = LariceConfig(num_queries=4, dim_model=128, input_dim=D_IN,
                       num_views=V, tau=0.02, inv_weight=2.0, readout="pool")
    tower = LariceTower(cfg)
    opt = torch.optim.AdamW(tower.parameters(), lr=5e-4, weight_decay=1e-4)
    inv_tau = 1.0 / cfg.tau
    for step in range(steps):
        gids = rng.choice(train_rows, batch, replace=False)
        ce, inv = pfc_step(tower, opt, cfg, store, pad, ring, games, gids, rng, inv_tau)
        if step % 5 == 0 or step == steps - 1:
            with torch.no_grad():                    # eval: every pack encoded once, no gradient
                z = tower(*load_rows(store, pad, gids))
                full = torch.cat([tower(*load_rows(store, pad, train_rows[i:i + 32]))
                                  for i in range(0, n_games, 32)])
                hit = (z @ full.T).argmax(-1).eq(torch.as_tensor(gids)).float().mean().item()
            print(f"step {step:3d}  CE/pass {ce:.3f}  inv {inv:.3f}  anchor self-retrieval hit@1 {hit:.2f}")
    print(f"ring: {n_games} packs in {K} shards, window {w}, stride {S}, {T} micro-passes; "
          f"negatives per view per pass = {batch} own + {K * w} window rows; store {store.shape} fp16 on disk")


if __name__ == "__main__":
    main()
