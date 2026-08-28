# -*- coding: utf-8 -*-
"""LariceConfig — one dataclass builds one larice tower
(larice = Latent Represent I-CE).

The model package is task-agnostic: it knows nothing about Steam, corpora,
or evaluation protocols. Everything a downstream task can tune lives here.
"""
from dataclasses import dataclass


@dataclass
class LariceConfig:
    # ---- architecture ----
    num_queries: int = 4       # N latent query slots
    dim_model: int = 128       # DM: query/attention/output width per slot
    num_heads: int = 4         # attention heads
    input_dim: int = 1024      # D_in: dimensionality of upstream embeddings
    hidden: int = 256          # readout MLP hidden width
    readout: str = "pool"      # "pool" (default, mean over slots; better for name-recall)
    #                          # | "concat" (general representation)

    # ---- the I-CE objective ----
    num_views: int = 4         # NV: views per data item fed to the loss
    tau_mode: str = "frozen"   # "frozen" | "learnable"
    tau: float = 0.02          # CE temperature (init value when learnable)
    inv_weight: float = 2.0    # I: invariance weight across the view axis
    #                          # Both terms fire on every item in the step.
    #                          # The CE gate is NOT configured here: it is an
    #                          # ablation-only per-item mask passed at loss
    #                          # call time and defaults to off. See model.py.

    @property
    def out_dim(self) -> int:
        return (self.num_queries * self.dim_model if self.readout == "concat"
                else self.dim_model)
