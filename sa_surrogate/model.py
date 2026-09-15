"""
Sigma-conditioned graph attention transformer for SA prediction.

Design constraints, all of them imposed by stage 2 rather than by stage 1:

1. Atom types enter as a float [N, K] matrix through `nn.Linear`, never through
   `nn.Embedding` and never via `argmax`.  At stage 2 that matrix is the output
   of a Gumbel straight-through estimator, so it must stay differentiable and
   must tolerate relaxed (non-one-hot) rows.

2. Edges come from interatomic distances, never from RDKit bonds -- at stage 2
   there is no bond graph, only coordinates.  Edge *selection* (the cutoff) is
   not differentiable, but edge *features* are, and a cosine envelope drives
   each edge's contribution smoothly to zero at the cutoff.  Without it,
   d(score)/d(coords) is discontinuous every time an atom crosses the boundary.

3. Only pairwise distances are used, so the model is E(3)-invariant.  SA is an
   invariant property; equivariance would buy nothing here.

4. Conditioning is on log10(sigma/alpha), not on the integer timestep, so the
   surrogate stays valid whatever stage 2 feeds it and across schedule changes.

Pooling concatenates mean, scaled sum and attention pooling, plus log(N)
explicitly: SA's size penalty is `nAtoms**1.005 - nAtoms`, an explicit function
of atom count, so handing the head that number directly costs nothing and saves
it from having to recover the count from a mean-pooled vector.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

from sa_surrogate.diffusion_bridge import NoiseSchedule


# -----------------------------------------------------------------------------
# segment reductions
#
# Written with index_add_ rather than torch_geometric.utils.scatter because the
# latter dispatches to aten::scatter_reduce, which MPS does not implement -- and
# `device: auto` picks MPS on Apple silicon.  index_add_ is native everywhere.
# -----------------------------------------------------------------------------

def segment_sum(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    out = torch.zeros(num_segments, src.size(-1), device=src.device, dtype=src.dtype)
    return out.index_add_(0, index, src)


def segment_count(index: torch.Tensor, num_segments: int,
                  dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    out = torch.zeros(num_segments, 1, device=device, dtype=dtype)
    return out.index_add_(0, index, torch.ones(index.size(0), 1, device=device, dtype=dtype))


def segment_softmax(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Softmax within each segment.

    Stabilised by subtracting the *global* max rather than a per-segment max: the
    constant cancels inside every segment, so the result is identical, and it
    avoids a segment-max reduction that MPS also lacks.
    """
    e = (src - src.max().detach()).exp()
    denom = segment_sum(e, index, num_segments)[index].clamp_min(1e-12)
    return e / denom


# -----------------------------------------------------------------------------
# building blocks
# -----------------------------------------------------------------------------

def build_edges(
    pos: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float,
    max_neighbors: int = 32,
) -> torch.Tensor:
    """Within-graph radius graph via cdist, mirroring EGNNDynamics.get_edges
    (equivariant_diffusion/dynamics.py:163).  Deliberately avoids torch_cluster,
    which is not in DiffSBDD's environment.

    Returns [2, E] with edges pointing source -> target.
    """
    with torch.no_grad():
        same_graph = batch[:, None] == batch[None, :]
        dist = torch.cdist(pos.detach(), pos.detach())
        adj = same_graph & (dist <= cutoff)
        adj.fill_diagonal_(False)

        n = pos.size(0)
        k = int(min(max_neighbors, max(1, n)))
        masked = dist.masked_fill(~adj, float("inf"))
        nn_dist, nn_idx = torch.topk(masked, k=k, dim=1, largest=False)
        keep = torch.isfinite(nn_dist)

        row = torch.arange(n, device=pos.device).unsqueeze(1).expand(-1, k)[keep]
        col = nn_idx[keep]

    return torch.stack([col, row], dim=0)  # message flows col -> row


class RadialBasis(nn.Module):
    """Gaussian smearing of |r_ij| with a smooth cosine cutoff envelope."""

    def __init__(self, num_rbf: int = 32, cutoff: float = 5.0, envelope: bool = True):
        super().__init__()
        self.cutoff = float(cutoff)
        self.envelope = bool(envelope)
        self.register_buffer("centers", torch.linspace(0.0, cutoff, num_rbf))
        self.width = cutoff / max(num_rbf, 1)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        d = dist.unsqueeze(-1)
        out = torch.exp(-(((d - self.centers) / self.width) ** 2))
        if self.envelope:
            env = 0.5 * (torch.cos(math.pi * dist.clamp(max=self.cutoff) / self.cutoff) + 1.0)
            out = out * env.unsqueeze(-1)
        return out


class SigmaEmbedding(nn.Module):
    """Sinusoidal features of the normalised log noise level, then an MLP."""

    def __init__(self, dim: int = 128, num_freqs: int = 32, max_freq: float = 64.0):
        super().__init__()
        self.register_buffer(
            "freqs", torch.exp(torch.linspace(0.0, math.log(max_freq), num_freqs))
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_freqs + 1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s = s.view(-1, 1)
        ang = s * self.freqs.view(1, -1)
        return self.mlp(torch.cat([s, torch.sin(ang), torch.cos(ang)], dim=-1))


class GraphTransformerBlock(nn.Module):
    """Pre-LN attention + FFN, both FiLM-conditioned on the noise level.

    FiLM is what lets the model say "at this sigma the types are unreliable,
    lean on geometry" -- behaviour a model without noise conditioning simply
    cannot express.
    """

    def __init__(self, dim: int, heads: int, edge_dim: int,
                 dropout: float = 0.1, ffn_mult: int = 2, cond_dim: int = 128):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"hidden_dim ({dim}) must be divisible by num_heads ({heads})")
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TransformerConv(dim, dim // heads, heads=heads,
                                    edge_dim=edge_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_mult * dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(ffn_mult * dim, dim),
        )
        self.film = nn.Linear(cond_dim, 4 * dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)  # starts as the identity modulation
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_attr, cond, batch):
        g1, b1, g2, b2 = self.film(cond)[batch].chunk(4, dim=-1)
        h = h + self.dropout(self.attn(self.norm1(h) * (1 + g1) + b1,
                                       edge_index, edge_attr))
        h = h + self.dropout(self.ffn(self.norm2(h) * (1 + g2) + b2))
        return h


# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------

class SAGraphTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_mult: int = 2,
        edge_cutoff: float = 5.0,
        max_neighbors: int = 32,
        num_rbf: int = 32,
        envelope: bool = True,
        condition_on_sigma: bool = True,
        sigma_embed_dim: int = 128,
        pool: Sequence[str] = ("sum", "mean", "attn"),
        head_hidden: int = 256,
        head: str = "regression",
        num_bins: int = 10,
        aux_classification: bool = False,
        size_scale: float = 25.0,
        noise_schedule: str = "polynomial_2",
        timesteps: int = 500,
        noise_precision: float = 5.0e-4,
    ):
        super().__init__()
        if head not in ("regression", "classification"):
            raise ValueError(f"unknown head: {head!r}")

        self.num_classes = num_classes
        self.edge_cutoff = float(edge_cutoff)
        self.max_neighbors = int(max_neighbors)
        self.condition_on_sigma = bool(condition_on_sigma)
        self.pool = tuple(pool)
        self.head_type = head
        self.num_bins = int(num_bins)
        self.size_scale = float(size_scale)

        # Owned rather than passed in, so the sigma normalisation used at
        # training time cannot drift from the one used at guidance time.
        self.schedule = NoiseSchedule(noise_schedule, timesteps, noise_precision)

        # float Linear, NOT Embedding -- the input may be a relaxed simplex row
        self.node_in = nn.Linear(num_classes, hidden_dim)
        self.rbf = RadialBasis(num_rbf, edge_cutoff, envelope)
        self.edge_in = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.sigma_embed = SigmaEmbedding(sigma_embed_dim) if condition_on_sigma else None
        cond_dim = sigma_embed_dim if condition_on_sigma else 1

        self.blocks = nn.ModuleList([
            GraphTransformerBlock(hidden_dim, num_heads, hidden_dim,
                                  dropout, ffn_mult, cond_dim)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.attn_gate = nn.Linear(hidden_dim, 1) if "attn" in self.pool else None

        pooled_dim = hidden_dim * len(self.pool) + 1  # +1 for log(N)
        out_dim = 1 if head == "regression" else self.num_bins
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, head_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden), nn.SiLU(),
            nn.Linear(head_hidden, out_dim),
        )
        self.aux_head = (
            nn.Sequential(nn.Linear(pooled_dim, head_hidden), nn.SiLU(),
                          nn.Linear(head_hidden, self.num_bins))
            if (aux_classification and head == "regression") else None
        )

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        one_hot: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        r: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            one_hot: [N, K] atom types.  May be a hard one-hot, a relaxed
                     simplex row, or a Gumbel-ST output; must be differentiable.
            pos:     [N, 3] coordinates in Angstrom.
            batch:   [N] graph index per node (DiffSBDD's `lig_mask`).
            r:       [G] noise level sigma_t/alpha_t per graph.  None == clean.
        """
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0

        if r is None:
            r = torch.full((num_graphs,), float(self.schedule.r_t[0]),
                           device=pos.device, dtype=pos.dtype)
        if self.condition_on_sigma:
            cond = self.sigma_embed(self.schedule.normalize_log_r(r))
        else:
            cond = torch.zeros(num_graphs, 1, device=pos.device, dtype=pos.dtype)

        edge_index = build_edges(pos, batch, self.edge_cutoff, self.max_neighbors)
        # Recomputed from `pos` (not from the detached copy used for selection)
        # so gradients reach the coordinates.
        d = (pos[edge_index[0]] - pos[edge_index[1]]).pow(2).sum(-1).clamp_min(1e-12).sqrt()
        edge_attr = self.edge_in(self.rbf(d))

        h = self.node_in(one_hot)
        for blk in self.blocks:
            h = blk(h, edge_index, edge_attr, cond, batch)
        h = self.norm_out(h)

        counts = segment_count(batch, num_graphs, h.dtype, h.device)  # [G, 1]
        h_sum = segment_sum(h, batch, num_graphs)

        parts: List[torch.Tensor] = []
        if "mean" in self.pool:
            parts.append(h_sum / counts.clamp_min(1.0))
        if "sum" in self.pool:
            parts.append(h_sum / self.size_scale)
        if "attn" in self.pool:
            w = segment_softmax(self.attn_gate(h), batch, num_graphs)
            parts.append(segment_sum(h * w, batch, num_graphs))
        parts.append(torch.log(counts.clamp_min(1.0)))
        pooled = torch.cat(parts, dim=-1)

        out = self.head(pooled)
        result = {"pred": out.squeeze(-1) if self.head_type == "regression" else out,
                  "pooled": pooled}
        if self.aux_head is not None:
            result["aux_logits"] = self.aux_head(pooled)
        return result

    # -- convenience --------------------------------------------------------
    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: dict, num_classes: int) -> SAGraphTransformer:
    m = cfg["model"]
    c = cfg["corruption"]
    return SAGraphTransformer(
        num_classes=num_classes,
        hidden_dim=m["hidden_dim"], num_layers=m["num_layers"],
        num_heads=m["num_heads"], dropout=m["dropout"], ffn_mult=m["ffn_mult"],
        edge_cutoff=m["edge_cutoff"], max_neighbors=m["max_neighbors"],
        num_rbf=m["num_rbf"], envelope=m["envelope"],
        condition_on_sigma=m["condition_on_sigma"],
        sigma_embed_dim=m["sigma_embed_dim"],
        pool=m["pool"], head_hidden=m["head_hidden"], head=m["head"],
        num_bins=m["num_bins"],
        aux_classification=m["aux_classification"]["enabled"],
        noise_schedule=c["noise_schedule"], timesteps=c["timesteps"],
        noise_precision=c["noise_precision"],
    )
