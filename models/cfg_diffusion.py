import math
import torch
import torch.nn as nn
from typing import Optional


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for diffusion timestep.

    Args:
        t: Timestep tensor, shape [B] or [B, 1].
        dim: Embedding dimension.

    Returns:
        Embedding tensor, shape [B, dim].
    """
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device))
    args = t * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
    return emb

class FiLMResBlock(nn.Module):
    """Residual MLP block with adaLN (adaptive Layer Normalization) conditioning.

    Architecture (adaLN):
        gamma, beta = FiLM(ctx)
        h = LayerNorm(x) * (1 + gamma) + beta
        h = Linear(SiLU(h))  ->  Dropout
        return x + h
    """

    def __init__(self, hidden: int, context_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.film = nn.Linear(context_dim, hidden * 2)
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),  # Project back to original hidden dim.
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(ctx).chunk(2, dim=-1)
        h = self.norm(x) * (1.0 + gamma) + beta
        h = self.net(h)
        return x + h


class ResBlock(nn.Module):
    """Pre-norm residual block with two linear layers.

    Architecture:
        LayerNorm -> Linear -> SiLU -> Dropout -> LayerNorm -> Linear -> Dropout, with skip.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PostureEncoder(nn.Module):
    """MLP that encodes a flattened posture into a bottleneck context embedding.

    Flattens posture (100x3 -> 300), projects to a wide hidden dim, passes
    through a stack of ResBlocks, then projects down to a smaller out_dim.
    The bottleneck (out_dim < hidden) produces a compact conditioning vector
    for the score network.

    Architecture:
        Linear(input_dim, hidden) -> ResBlock x num_layers
        -> LayerNorm -> SiLU -> Linear(hidden, out_dim)
    """

    def __init__(
        self,
        input_dim: int = 300,
        hidden: int = 256,
        out_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.inp = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout=dropout) for _ in range(num_layers)])
        self.out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, posture: torch.Tensor) -> torch.Tensor:
        if posture.dim() == 3:
            posture = posture.reshape(posture.size(0), -1)
        h = self.inp(posture)
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)


class GeometryDiffusionNetwork(nn.Module):
    """Posture-conditioned score network that predicts noise in geometry space.

    Data flow:
        t   [B]     -->  sinusoidal  -->  [B, time_dim] --+
                                                           |--> cat --> MLP --> ctx [B, ctx_dim]
        cond [B, cond_dim]  ------------------------------+

        x_t [B, geom_dim]  -->  g_proj  -->  [B, hidden]
                -->  FiLM ResBlock x num_blocks (conditioned on ctx)
                -->  LayerNorm -> SiLU -> Linear --> noise [B, geom_dim]
    """

    def __init__(
        self,
        geom_dim: int = 6,
        hidden: int = 128,
        time_dim: int = 64,
        cond_dim: int = 64,
        num_blocks: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.geom_dim = geom_dim
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        ctx_dim = time_dim + cond_dim

        self.g_proj = nn.Linear(geom_dim, hidden)

        self.ctx_mlp = nn.Sequential(
            nn.Linear(ctx_dim, ctx_dim),
            nn.SiLU(),
            nn.Linear(ctx_dim, ctx_dim),
        )

        self.blocks = nn.ModuleList(
            [
                FiLMResBlock(hidden=hidden, context_dim=ctx_dim, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, geom_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t[None]
        time_emb = timestep_embedding(t, self.time_dim)
        ctx = self.ctx_mlp(torch.cat([time_emb, cond_emb], dim=-1))

        h = self.g_proj(x_t)
        for block in self.blocks:
            h = block(h, ctx)
        return self.head(h)


class ConditionalDiffusionNet(nn.Module):
    """Top-level CFG diffusion model: posture encoder + score network.

    Implements Classifier-Free Guidance (CFG) via a learnable null embedding.
    During training, cond_mask randomly replaces real posture embeddings with
    the null embedding so the network learns both conditional and unconditional
    score estimation.

    Data flow:
        posture [B, 100, 3]  -->  PostureEncoder  -->  cond [B, cond_embed_dim]
                                                          |
        (CFG: randomly replace with null_embedding)       |
                                                          v
        x_t [B, 6], t [B]  -->  GeometryDiffusionNetwork  -->  noise [B, 6]
    """

    def __init__(
        self,
        geom_dim: int = 6,
        cond_in_dim: int = 300,
        cond_embed_dim: int = 64,
        hidden: int = 128,
        num_layers: int = 6,
        time_embed_dim: int = 64,
        dropout: float = 0.0,
        posture_layers: int = 3,
        learn_null_embedding: bool = True,
    ):
        super().__init__()
        self.posture_encoder = PostureEncoder(
            input_dim=cond_in_dim,
            hidden=hidden,
            out_dim=cond_embed_dim,
            num_layers=posture_layers,
            dropout=dropout,
        )
        self.diffusion_net = GeometryDiffusionNetwork(
            geom_dim=geom_dim,
            hidden=hidden,
            time_dim=time_embed_dim,
            cond_dim=cond_embed_dim,
            num_blocks=num_layers,
            dropout=dropout,
        )
        if learn_null_embedding:
            self.null_embedding = nn.Parameter(torch.zeros(cond_embed_dim))
        else:
            self.register_buffer("null_embedding", torch.zeros(cond_embed_dim), persistent=False)

    def _null_condition(self, batch_size: int) -> torch.Tensor:
        return self.null_embedding.unsqueeze(0).expand(batch_size, -1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        posture: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
        force_unconditional: bool = False,
    ) -> torch.Tensor:
        batch_size = x_t.size(0)
        if force_unconditional or posture is None:
            cond = self._null_condition(batch_size)
        elif cond_mask is None:
            cond = self.posture_encoder(posture)
        else:
            cond_mask = cond_mask.view(-1).to(dtype=torch.bool)
            if cond_mask.all():
                cond = self.posture_encoder(posture)
            elif (~cond_mask).all():
                cond = self._null_condition(batch_size)
            else:
                cond = self._null_condition(batch_size).clone()
                cond_enc = self.posture_encoder(posture[cond_mask])
                cond[cond_mask] = cond_enc
        return self.diffusion_net(x_t, t, cond)

