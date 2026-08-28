# -*- coding: utf-8 -*-
"""Contrast tower: BYOL (alignment-only philosophy).

Predictor + EMA target + stop-grad; no CE, no negatives — so the inductive
gallery restriction does not apply (nothing sees the gallery). Full budget,
checkpoints only; the head stage is shared with every other arm.
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from main_model import LariceConfig, LariceTower
from steam_reviews_framework.sampler import sample_views
from steam_reviews_framework.train import (ArmSpec, assemble_doc_view,
                                           make_doc_tiers, _amp_scaler)

TAU_EMA = 0.996


def train_byol(B, spec: ArmSpec, seed=0, W=16, bs=192, per_epoch=3072,
               log_cb=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = LariceConfig(readout="pool", num_views=spec.num_views, inv_weight=0.0)
    model = LariceTower(cfg).to(B.dev)
    DM = cfg.dim_model
    pred = nn.Sequential(nn.Linear(DM, 256), nn.GELU(),
                         nn.Linear(256, DM)).to(B.dev)
    target = copy.deepcopy(model).to(B.dev)
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(list(model.parameters()) + list(pred.parameters()),
                            lr=5e-4, weight_decay=1e-4)
    amp = _amp_scaler()
    P = lambda z: F.normalize(pred(z.float()), dim=-1)
    tiers, _, _ = make_doc_tiers(B, spec)
    NV = spec.num_views

    ckpts = {}
    t0 = time.time()
    for ep in range(spec.epochs):
        model.train()
        pred.train()
        for _ in range(per_epoch // bs):
            gids = rng.choice(B.train_pool_games, bs, replace=False)
            with torch.amp.autocast("cuda"):
                views = [sample_views(B.pool, B.rev_tab, gids, W, rng, B.dev)
                         for _ in range(NV - 1)]
                Zo = [model(S, m) for S, m in views]
                with torch.no_grad():
                    Zt = [target(S, m) for S, m in views]
                Zo.append(assemble_doc_view(model, B, tiers, gids, W, rng, bs, cfg.out_dim))
                with torch.no_grad():
                    Zt.append(assemble_doc_view(target, B, tiers, gids, W, rng, bs, cfg.out_dim))
                loss, npairs = 0.0, 0
                for i in range(NV):
                    for j in range(NV):
                        if i == j:
                            continue
                        loss = loss + (1 - (P(Zo[i]) * Zt[j].float().detach()
                                            ).sum(-1)).mean()
                        npairs += 1
                loss = loss / npairs
            opt.zero_grad()
            amp.scale(loss).backward()
            amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(pred.parameters()), 5.0)
            amp.step(opt)
            amp.update()
            with torch.no_grad():
                for pt, po in zip(target.parameters(), model.parameters()):
                    pt.data.mul_(TAU_EMA).add_(po.data, alpha=1 - TAU_EMA)
        if (ep + 1) % spec.ckpt_every == 0:
            ckpts[ep + 1] = {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()}
        if ep % 100 == 99:
            log_cb(f"  [byol {spec.name} ep{ep + 1}] {time.time() - t0:.0f}s")
    model.eval()
    return model, ckpts
