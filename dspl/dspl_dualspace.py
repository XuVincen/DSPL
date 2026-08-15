"""
DSPL-DualSpace: Dual-Space Hierarchical Equivariant Network
===========================================================

Core Innovation:
  - Euclidean Tower (3D coord space, distance edges) + Spherical Harmonic Tower
    (SH feature space, SH-similarity KNN graph)
  - Bidirectional Cross-Space Message Passing at every layer
  - Residue Context Modulation for Task 1 (preserves atomic resolution)

Compatible with train_ablation.py via --architecture dualspace.
Self-contained: no e3nn/torch_cluster/torch_scatter required (pure PyTorch).

Reference: DualEquiNet (J. Xu et al.) — adapted for atom-level protein B-factor.

Author: DSPL Project
Date:   2026-06-22
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================
# e3nn Detection & Pure PyTorch Fallbacks
# ==============================

_HAS_E3NN = False
try:
    import e3nn
    # Check that e3nn has the o3 submodule (e3nn >= 0.5.0)
    if hasattr(e3nn, 'o3'):
        _HAS_E3NN = True
    else:
        print("[dspl_dualspace] e3nn found but no o3 module — using pure PyTorch fallback")
except ImportError:
    print("[dspl_dualspace] e3nn not installed — using pure PyTorch fallback")
except Exception as e:
    print(f"[dspl_dualspace] e3nn check failed ({e}) — using pure PyTorch fallback")


# ---- Pure-PyTorch spherical harmonics up to l=2 ----

def _spherical_harmonics_l2(x):
    """Compute real spherical harmonics Y_l^m for l=0,1,2 in PyTorch.

    Args:
        x: (E, 3) unit direction vectors (normalized relative positions)
    Returns:
        sh: (E, 9) — [Y_0^0, Y_1^{-1}, Y_1^0, Y_1^1, Y_2^{-2}, ..., Y_2^2]
    """
    # x: (E, 3) with x,y,z components, already normalized
    x_n, y_n, z_n = x[:, 0], x[:, 1], x[:, 2]

    # l=0: Y_0^0 = 1/sqrt(4π)
    y00 = torch.full_like(x_n, 0.5 / math.sqrt(math.pi))  # (E,)

    # l=1
    sqrt3_over_4pi = math.sqrt(3.0 / (4.0 * math.pi))
    y1m1 = sqrt3_over_4pi * y_n   # Y_1^{-1}
    y10  = sqrt3_over_4pi * z_n   # Y_1^0
    y11  = sqrt3_over_4pi * x_n   # Y_1^1

    # l=2
    sqrt15_over_4pi = math.sqrt(15.0 / (4.0 * math.pi))
    sqrt5_over_16pi  = math.sqrt(5.0 / (16.0 * math.pi))
    y2m2 = sqrt15_over_4pi * x_n * y_n               # Y_2^{-2}
    y2m1 = sqrt15_over_4pi * y_n * z_n               # Y_2^{-1}
    y20  = sqrt5_over_16pi  * (3.0 * z_n**2 - 1.0)   # Y_2^0
    y21  = sqrt15_over_4pi * x_n * z_n               # Y_2^1
    y22  = sqrt15_over_4pi * 0.5 * (x_n**2 - y_n**2) # Y_2^2

    sh = torch.stack([y00, y1m1, y10, y11, y2m2, y2m1, y20, y21, y22], dim=-1)  # (E, 9)
    return sh


def _sh_dim(lmax: int) -> int:
    """Number of spherical harmonic coefficients up to lmax."""
    return (lmax + 1) ** 2


# ---- scatter operations (torch_scatter fallback) ----

def _scatter_sum(src, index, dim, dim_size):
    """Pure PyTorch scatter sum — fallback for torch_scatter.

    Handles multi-dimensional src by flattening extra dims into dim.
    """
    assert index.ndim == 1, f"index must be 1D, got shape={index.shape}"
    N = src.shape[0]
    assert index.shape[0] == N, f"index length {index.shape[0]} != src dim0 {N}"

    if src.ndim == 1:
        return torch.zeros(dim_size, device=src.device, dtype=src.dtype) \
            .scatter_add(0, index, src)

    # For multi-dim src (N, d1, d2, ...): flatten extra dims
    extra_shape = src.shape[1:]
    src_flat = src.reshape(N, -1)
    n_extra = src_flat.shape[1]

    index_expanded = index.unsqueeze(-1).expand(-1, n_extra).reshape(-1)
    src_expanded = src_flat.reshape(-1)

    out_flat = torch.zeros(dim_size * n_extra, device=src.device, dtype=src.dtype)
    out_flat.scatter_add_(0, index_expanded, src_expanded)
    return out_flat.reshape(dim_size, *extra_shape)


def _scatter_mean(src, index, dim, dim_size):
    """Pure PyTorch scatter mean."""
    summed = _scatter_sum(src, index, dim, dim_size)
    ones = torch.ones_like(src[..., :1]) if src.ndim > 1 else torch.ones_like(src)
    count = _scatter_sum(ones, index, dim, dim_size)
    # Expand count to match summed shape for broadcasting
    while count.ndim < summed.ndim:
        count = count.unsqueeze(-1)
    return summed / count.clamp(min=1)


# ==============================
# BaseMLP
# ==============================

class BaseMLP(nn.Module):
    """Two-layer MLP with optional skip connection."""
    def __init__(self, input_dim, hidden_dim, output_dim,
                 activation=nn.SiLU(), residual=False, last_act=False, bias=True):
        super().__init__()
        self.residual = residual
        if residual:
            assert output_dim == input_dim, \
                f"Residual requires output_dim==input_dim, got {output_dim} vs {input_dim}"
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=bias),
            activation,
            nn.Linear(hidden_dim, output_dim, bias=bias),
            activation if last_act else nn.Identity(),
        )

    def forward(self, x):
        return x + self.mlp(x) if self.residual else self.mlp(x)


# ==============================
# InnerProductSH
# ==============================

class InnerProductSH(nn.Module):
    """Compute spherical harmonic inner product between node SH features on edges.

    For each edge (i,j), computes the per-l inner product of node_sh[i] * node_sh[j],
    yielding (lmax+1) scalar values per edge.
    """
    def __init__(self, lmax: int = 2):
        super().__init__()
        self.lmax = lmax
        self.sh_dim = _sh_dim(lmax)
        # Compute start indices for each l
        self.l_offsets = []
        offset = 0
        for ell in range(lmax + 1):
            self.l_offsets.append((ell, offset, 2 * ell + 1))
            offset += 2 * ell + 1

    def forward(self, edge_index, node_sh):
        """Args: edge_index (2,E), node_sh (N, sh_dim). Returns: (E, lmax+1)"""
        row, col = edge_index
        temp = node_sh[row] * node_sh[col]  # (E, sh_dim)

        in_prod = []
        for ell, start, dim_ell in self.l_offsets:
            ip_ell = temp[:, start:start + dim_ell].sum(dim=-1)  # (E,)
            in_prod.append(ip_ell)
        in_prod = torch.stack(in_prod, dim=-1)  # (E, lmax+1)

        # Normalize
        in_prod = in_prod / (torch.norm(in_prod, dim=-1, keepdim=True).detach() + 1.0)
        return in_prod

    @property
    def irreps_dim(self):
        return self.sh_dim


# ==============================
# SH_INIT — initialize node SH features
# ==============================

class SH_INIT(nn.Module):
    """Initialize node spherical harmonic features from (h, pos, edge_index).

    Uses e3nn if available, else pure PyTorch SH + MLP projection.
    """
    def __init__(self, hidden_dim: int, lmax: int = 2, activation=nn.SiLU()):
        super().__init__()
        self.lmax = lmax
        self.hidden_dim = hidden_dim
        self.sh_dim = _sh_dim(lmax)

        if _HAS_E3NN:
            self.sh_irreps = e3nn.o3.Irreps.spherical_harmonics(lmax)
            self.spherical_harmonics = e3nn.o3.SphericalHarmonics(
                self.sh_irreps, normalize=True, normalization="norm")
            self.sh_coff = e3nn.o3.FullyConnectedTensorProduct(
                self.sh_irreps, '1x0e', self.sh_irreps, shared_weights=False)
            self.sh_mlp = BaseMLP(
                input_dim=2 * hidden_dim + 1,
                hidden_dim=hidden_dim,
                output_dim=self.sh_coff.weight_numel,
                activation=activation, last_act=True)
            self._use_e3nn = True
        else:
            # Pure PyTorch path
            self.sh_mlp = BaseMLP(
                input_dim=2 * hidden_dim + 1 + self.sh_dim,
                hidden_dim=hidden_dim,
                output_dim=self.sh_dim,
                activation=activation, last_act=False)
            self.node_proj = nn.Linear(self.sh_dim, self.sh_dim)
            self._use_e3nn = False

    def forward(self, h, pos, edge_index):
        row, col = edge_index
        rel_pos = pos[row] - pos[col]  # (E, 3)
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # (E, 1)

        if self._use_e3nn:
            msg = torch.cat([dist, h[row], h[col]], dim=-1)  # (E, 2*hidden_dim + 1)
            msg = self.sh_mlp(msg)
            rel_sh = self.spherical_harmonics(rel_pos).detach()
            one = torch.ones([rel_sh.size(0), 1], device=rel_sh.device).detach()
            rel_sh = self.sh_coff(rel_sh, one, msg)
            node_sh = _scatter_mean(rel_sh, index=row, dim=0, dim_size=h.size(0))
        else:
            # Pure PyTorch: compute SH, then MLP message
            vec_dir = rel_pos / (dist + 1e-8)
            rel_sh = _spherical_harmonics_l2(vec_dir).detach()  # (E, 9) — only l≤2
            msg = torch.cat([dist, h[row], h[col], rel_sh], dim=-1)
            msg = self.sh_mlp(msg)  # (E, sh_dim)
            node_sh = _scatter_mean(msg, index=row, dim=0, dim_size=h.size(0))
            node_sh = self.node_proj(node_sh)

        return node_sh


# ==============================
# SphericalHarmonicGraph — KNN similarity graph in SH space
# ==============================

class SphericalHarmonicGraph(nn.Module):
    """Build a KNN graph based on spherical harmonic feature similarity.

    Pure PyTorch — no torch_cluster dependency.
    Strategy: For each node, find top-k neighbors by SH cosine similarity.
    """
    def __init__(self, threshold: float = 0.5, max_neighbors: int = 32):
        super().__init__()
        self.threshold = threshold
        self.max_neighbors = max_neighbors

    def forward(self, node_sh, batch=None):
        """
        Args:
            node_sh: (N, sh_dim) SH features
            batch: (N,) batch indices (optional, for multi-graph batching)
        Returns:
            edge_index: (2, E_sh)
        """
        N = node_sh.size(0)
        node_sh_norm = F.normalize(node_sh, p=2, dim=-1)  # (N, sh_dim)

        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=node_sh.device)

        # For single-protein graphs (batch_size=1), compute full similarity
        # and take top-k per node
        unique_batches = batch.unique()

        all_edges = []
        for b in unique_batches:
            mask = (batch == b)
            idx = mask.nonzero(as_tuple=False).squeeze(-1)  # (N_b,)
            n_b = idx.size(0)

            if n_b <= 1:
                continue

            sh_b = node_sh_norm[idx]  # (N_b, sh_dim)
            sim = torch.mm(sh_b, sh_b.t())  # (N_b, N_b)

            # Zero self-similarity
            sim.fill_diagonal_(float('-inf'))

            # Top-k per node
            k = min(self.max_neighbors, n_b - 1)
            top_vals, top_idx = sim.topk(k, dim=-1)  # (N_b, k)

            # Filter by threshold
            valid = top_vals > self.threshold  # (N_b, k)

            src_local = torch.arange(n_b, device=node_sh.device).unsqueeze(1).expand(-1, k)
            src_global = idx[src_local[valid]]  # map to global indices
            dst_global = idx[top_idx[valid]]

            if src_global.numel() > 0:
                ei = torch.stack([src_global, dst_global], dim=0)  # (2, E_b)
                all_edges.append(ei)

        if all_edges:
            edge_index = torch.cat(all_edges, dim=1)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=node_sh.device)

        return edge_index


# ==============================
# EuclideanSpaceBlock
# ==============================

class EuclideanSpaceBlock(nn.Module):
    """Single block in the Euclidean tower.

    Features:
      - Main MP: h_i, h_j, dist → msg, attention, pos_scale
      - Cross-space MP: SH inner_product on SH edges + dist → cross_msg
      - Multi-head attention aggregation
      - Coordinate update with learnable scale
      - DSPl state_corr injection point
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, lmax: int = 2,
                 inner_product=None, activation=nn.SiLU()):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, \
            f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"

        self.inner_product = inner_product
        self.pos_scale_factor = nn.Parameter(torch.tensor([0.1]))
        self.lmax = lmax

        # Main message MLP: [h_i, h_j, dist, (state_corr)] → msg + attn + pos_scale
        n_msg_input = 2 * hidden_dim + 1 + 5  # +5 for K=5 DSPL state_corr
        self.msg_mlp = BaseMLP(
            input_dim=n_msg_input,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim * num_heads + num_heads + 1,  # msg + attention + pos_scale
            activation=activation, last_act=False)

        # Cross-space MLP: [SH inner_product on SH edges, dist] → cross_msg + cross_attn + pos_cross_scale
        self.cross_mlp = BaseMLP(
            input_dim=(lmax + 1) + 1,  # inner_prod (lmax+1) + dist
            hidden_dim=hidden_dim,
            output_dim=hidden_dim * num_heads + num_heads + 1,  # msg + attention + cross_pos_scale
            activation=activation, last_act=False)

        # Node update MLP
        self.h_mlp = BaseMLP(
            input_dim=2 * hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            activation=activation, last_act=False)

    def forward(self, h, pos, edge_index_eu, edge_index_sh, node_sh, state_corr=None):
        """
        Args:
            h: (N, hidden_dim) node features
            pos: (N, 3) coordinates
            edge_index_eu: (2, E_eu) Euclidean distance edges
            edge_index_sh: (2, E_sh) SH similarity edges
            node_sh: (N, sh_dim) SH node features
            state_corr: (E_eu, K) or None — DSPL state-correlation for Euclidean edges
        Returns:
            h_update, pos_update
        """
        num_nodes = h.size(0)
        row_eu, col_eu = edge_index_eu

        # ---- Euclidean Message Passing ----
        rel_pos_eu = pos[row_eu] - pos[col_eu]
        dist_eu = torch.norm(rel_pos_eu, dim=-1, keepdim=True)  # (E_eu, 1)

        msg_inputs = [h[row_eu], h[col_eu], dist_eu]

        # ── DSPL state_corr injection ──
        if state_corr is not None and state_corr.shape[0] == edge_index_eu.shape[1]:
            msg_inputs.append(state_corr)  # (E_eu, K)
        else:
            # Zero-pad when no state_corr available
            msg_inputs.append(torch.zeros(
                edge_index_eu.shape[1], 5, device=h.device, dtype=h.dtype))

        msg_inputs = torch.cat(msg_inputs, dim=-1)  # (E_eu, 2*hidden + 1 + K)
        msg_outputs = self.msg_mlp(msg_inputs)  # (E_eu, hidden*heads + heads + 1)

        msg, attention_weights, pos_scale = torch.split(
            msg_outputs,
            [self.hidden_dim * self.num_heads, self.num_heads, 1],
            dim=-1)
        attention_weights = torch.sigmoid(attention_weights)  # (E_eu, heads)

        # ---- Cross-Space Message (SH→Euclidean) via SH edges ----
        if edge_index_sh.shape[1] > 0:
            row_sh, col_sh = edge_index_sh
            rel_pos_sh = pos[row_sh] - pos[col_sh]
            dist_sh = torch.norm(rel_pos_sh, dim=-1, keepdim=True)  # (E_sh, 1)
            in_prod = self.inner_product(edge_index_sh, node_sh)  # (E_sh, lmax+1)
            cross_inputs = torch.cat([in_prod, dist_sh], dim=-1)
            cross_outputs = self.cross_mlp(cross_inputs)
            cross_msg, cross_attention, pos_scale_cross = torch.split(
                cross_outputs,
                [self.hidden_dim * self.num_heads, self.num_heads, 1],
                dim=-1)
            cross_attention = torch.sigmoid(cross_attention)

            cross_msg = cross_msg.view(-1, self.num_heads, self.hidden_dim)
            attended_cross_msg = cross_msg * cross_attention.unsqueeze(-1)
            h_update_cross = _scatter_sum(
                attended_cross_msg, row_sh, dim=0, dim_size=num_nodes)

            # Cross-space position update
            pos_update_cross = (
                rel_pos_sh *
                torch.tanh(pos_scale_cross) *
                cross_attention.mean(dim=1, keepdim=True))
            pos_update_cross = _scatter_sum(
                pos_update_cross, row_sh, dim=0, dim_size=num_nodes)
        else:
            h_update_cross = torch.zeros(num_nodes, self.num_heads, self.hidden_dim,
                                         device=h.device)
            pos_update_cross = torch.zeros(num_nodes, 3, device=h.device)

        # ---- Aggregate Euclidean + Cross messages ----
        msg = msg.view(-1, self.num_heads, self.hidden_dim)  # (E_eu, heads, hidden)
        attended_msg = msg * attention_weights.unsqueeze(-1)
        h_update_eu = _scatter_sum(attended_msg, row_eu, dim=0, dim_size=num_nodes)

        h_update = (h_update_eu + h_update_cross).mean(dim=1)  # (N, hidden)
        h_update = self.h_mlp(torch.cat([h, h_update], dim=-1))  # (N, hidden)

        # ---- Position update ----
        pos_update_eu = (
            rel_pos_eu *
            torch.tanh(pos_scale) *
            attention_weights.mean(dim=1, keepdim=True))
        pos_update_eu = _scatter_sum(pos_update_eu, row_eu, dim=0, dim_size=num_nodes)

        pos_update = (pos_update_eu + pos_update_cross) * self.pos_scale_factor

        return h_update, pos_update


# ==============================
# SHSpaceBlock
# ==============================

class SHSpaceBlock(nn.Module):
    """Single block in the Spherical Harmonic tower.

    Features:
      - Main MP: SH inner_prod, h_i, h_j → msg, attention, node_sh_scale
      - Cross-space MP: SH inner_prod on Euclidean edges + dist → cross_msg
      - SH feature update with cross-space contribution
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, lmax: int = 2,
                 inner_product=None, spherical_harmonics_fn=None,
                 activation=nn.SiLU()):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.lmax = lmax
        self.sh_dim = _sh_dim(lmax)

        self.inner_product = inner_product
        self.spherical_harmonics_fn = spherical_harmonics_fn  # function or e3nn module
        self.node_sh_scale_factor = nn.Parameter(torch.tensor([0.1]))

        # Main message MLP: [inner_prod_SH, h_i, h_j] → msg + attention + node_sh_scale
        self.msg_mlp = BaseMLP(
            input_dim=(lmax + 1) + 2 * hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim * num_heads + num_heads + 1,
            activation=activation, last_act=False)

        # Cross-space MLP: [inner_prod on EU edges, dist] → cross_msg + cross_attn + cross_sh_scale
        self.cross_mlp = BaseMLP(
            input_dim=(lmax + 1) + 1,  # inner_prod + dist
            hidden_dim=hidden_dim,
            output_dim=hidden_dim * num_heads + num_heads + 1,
            activation=activation, last_act=False)

        # Node update MLP
        self.h_mlp = BaseMLP(
            input_dim=2 * hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            activation=activation, last_act=False)

    def forward(self, h, pos, edge_index_eu, edge_index_sh, node_sh):
        """Args: h (N,hidden), pos (N,3), edge_index_eu (2,E_eu), edge_index_sh (2,E_sh), node_sh (N,sh_dim).
        Returns: h_update, node_sh_update"""
        num_nodes = h.size(0)

        # ---- SH Message Passing ----
        if edge_index_sh.shape[1] == 0:
            return torch.zeros_like(h), torch.zeros_like(node_sh)

        row_sh, col_sh = edge_index_sh
        in_prod_sh = self.inner_product(edge_index_sh, node_sh)  # (E_sh, lmax+1)

        msg_inputs = torch.cat([in_prod_sh, h[row_sh], h[col_sh]], dim=-1)
        msg_outputs = self.msg_mlp(msg_inputs)
        msg, attention_logits, node_sh_scale = torch.split(
            msg_outputs,
            [self.hidden_dim * self.num_heads, self.num_heads, 1],
            dim=-1)
        attention_weights = torch.sigmoid(attention_logits)  # (E_sh, heads)

        # H update in SH space
        msg = msg.view(-1, self.num_heads, self.hidden_dim)
        attended_msg = msg * attention_weights.unsqueeze(-1)
        h_update_sh = _scatter_sum(attended_msg, row_sh, dim=0, dim_size=num_nodes)

        # SH feature update
        node_sh_update = (
            node_sh[col_sh].unsqueeze(1) *                      # (E_sh, 1, sh_dim)
            torch.tanh(node_sh_scale).unsqueeze(-1) *           # (E_sh, 1, 1)
            attention_weights.mean(dim=1, keepdim=True).unsqueeze(-1)  # (E_sh, 1, 1)
        )
        node_sh_update = _scatter_sum(node_sh_update, row_sh, dim=0, dim_size=num_nodes)

        # ---- Cross-Space Message (Euclidean→SH) via Euclidean edges ----
        if edge_index_eu.shape[1] > 0:
            row_eu, col_eu = edge_index_eu
            rel_pos_eu = pos[row_eu] - pos[col_eu]
            dist_eu = torch.norm(rel_pos_eu, dim=-1, keepdim=True)
            in_prod_eu = self.inner_product(edge_index_eu, node_sh)
            cross_inputs = torch.cat([in_prod_eu, dist_eu], dim=-1)
            cross_outputs = self.cross_mlp(cross_inputs)
            cross_msg, cross_attention, rel_sh_cross_scale = torch.split(
                cross_outputs,
                [self.hidden_dim * self.num_heads, self.num_heads, 1],
                dim=-1)
            cross_attention = torch.sigmoid(cross_attention)

            cross_msg = cross_msg.view(-1, self.num_heads, self.hidden_dim)
            attended_cross_msg = cross_msg * cross_attention.unsqueeze(-1)
            h_update_cross = _scatter_sum(attended_cross_msg, row_eu, dim=0, dim_size=num_nodes)

            # Cross-space SH update: local geometry → SH features
            if self.spherical_harmonics_fn is not None:
                if _HAS_E3NN and isinstance(self.spherical_harmonics_fn,
                                             e3nn.o3.SphericalHarmonics):
                    rel_sh = self.spherical_harmonics_fn(rel_pos_eu).detach()
                else:
                    # Pure PyTorch path
                    vec_dir = rel_pos_eu / (dist_eu + 1e-8)
                    rel_sh = _spherical_harmonics_l2(vec_dir).detach()
            else:
                vec_dir = rel_pos_eu / (dist_eu + 1e-8)
                rel_sh = _spherical_harmonics_l2(vec_dir).detach()

            rel_sh_update_cross = (
                rel_sh.unsqueeze(1) *
                torch.tanh(rel_sh_cross_scale).unsqueeze(-1) *
                cross_attention.mean(dim=1, keepdim=True).unsqueeze(-1))
            rel_sh_update_cross = _scatter_sum(
                rel_sh_update_cross, row_eu, dim=0, dim_size=num_nodes)
            node_sh_update_cross = rel_sh_update_cross.mean(dim=1)  # (N, sh_dim)
        else:
            h_update_cross = torch.zeros(num_nodes, self.num_heads, self.hidden_dim,
                                        device=h.device)
            node_sh_update_cross = torch.zeros(num_nodes, self.sh_dim, device=h.device)

        # ---- Combine ----
        h_update = (h_update_sh + h_update_cross).mean(dim=1)  # (N, hidden)
        h_update = self.h_mlp(torch.cat([h, h_update], dim=-1))  # (N, hidden)

        node_sh_update = (node_sh_update.mean(dim=1) + node_sh_update_cross) * self.node_sh_scale_factor

        return h_update, node_sh_update


# ==============================
# DualEquiLayer — one complete dual-space layer
# ==============================

class DualEquiLayer(nn.Module):
    """One complete layer: EuclideanSpaceBlock + SHSpaceBlock with residual.

    Output: h' = LayerNorm(h + scale_eu * h_eu + scale_sh * h_sh),
            pos' = pos + pos_update,
            node_sh' = node_sh + sh_update

    Numerical stability fixes (v2):
      - LayerNorm after h residual to prevent std explosion (5.5x per layer → ~1x)
      - Learnable per-branch scale factors initialized to 0.1 (was implicit 1.0)
      - Separate LayerNorm for pos to prevent coordinate drift
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, lmax: int = 2,
                 activation=nn.SiLU(), residual: bool = True):
        super().__init__()
        self.residual = residual
        self.sh_dim = _sh_dim(lmax)
        self.inner_product = InnerProductSH(lmax)

        self.eu_layer = EuclideanSpaceBlock(
            hidden_dim=hidden_dim, num_heads=num_heads, lmax=lmax,
            inner_product=self.inner_product, activation=activation)
        self.sh_layer = SHSpaceBlock(
            hidden_dim=hidden_dim, num_heads=num_heads, lmax=lmax,
            inner_product=self.inner_product,
            spherical_harmonics_fn=(
                e3nn.o3.SphericalHarmonics(
                    e3nn.o3.Irreps.spherical_harmonics(lmax),
                    normalize=True, normalization="norm") if _HAS_E3NN else None),
            activation=activation)

        # ── Numerical stability: LayerNorm + learnable update scales ──
        self.h_norm = nn.LayerNorm(hidden_dim)
        self.h_update_scale_eu = nn.Parameter(torch.tensor(0.1))
        self.h_update_scale_sh = nn.Parameter(torch.tensor(0.1))
        # Separate norm for pos doesn't make sense (3D coords have physical meaning),
        # but we can gate the pos update magnitude
        self.pos_update_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, h, pos, edge_index_eu, edge_index_sh, node_sh, state_corr=None):
        h_update_eu, pos_update = self.eu_layer(
            h, pos, edge_index_eu, edge_index_sh, node_sh, state_corr=state_corr)
        h_update_sh, node_sh_update = self.sh_layer(
            h, pos, edge_index_eu, edge_index_sh, node_sh)

        if self.residual:
            # Scaled residual: h + α·h_eu + β·h_sh, then LayerNorm
            h_new = h + self.h_update_scale_eu * h_update_eu + self.h_update_scale_sh * h_update_sh
            h_new = self.h_norm(h_new)

            # Pos update: gated by learnable scale (pos_scale_factor already in block,
            # but this provides per-layer gating)
            pos_new = pos + pos_update * self.pos_update_scale

            # SH update: node_sh + node_sh_update (SH features are self-normalizing)
            node_sh_new = node_sh + node_sh_update

            return h_new, pos_new, node_sh_new
        else:
            h_new = self.h_update_scale_eu * h_update_eu + self.h_update_scale_sh * h_update_sh
            return (self.h_norm(h_new),
                    pos_update * self.pos_update_scale,
                    node_sh_update)


# ==============================
# ResidueContextModulation (Task 1)
# ==============================

class ResidueContextModulation(nn.Module):
    """Residue-level context modulation for atom-level prediction.

    Instead of pooling atoms to residues (which loses atomic resolution),
    this module:
    1. Aggregates atom features to residue-level context via scatter_mean
    2. Broadcasts residue context back to each atom
    3. Applies gated modulation: h_out = h + σ(MLP([h || h_ctx])) * h_ctx

    For v1, atom_to_residue is identity (each atom = its own group).
    When true residue mapping is available, this provides genuine per-residue context.
    """
    def __init__(self, hidden_dim: int, activation=nn.SiLU()):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, h, atom_to_residue):
        """
        Args:
            h: (N, hidden_dim) atom features
            atom_to_residue: (N,) LongTensor — residue index for each atom
        Returns:
            h_modulated: (N, hidden_dim) context-modulated features
        """
        N = h.size(0)
        M = int(atom_to_residue.max().item()) + 1

        # Aggregate to residue level
        h_res = _scatter_mean(h, atom_to_residue, dim=0, dim_size=M)  # (M, hidden)

        # Broadcast back to atoms
        h_ctx = h_res[atom_to_residue]  # (N, hidden)

        # Gated modulation
        gate = self.gate_mlp(torch.cat([h, h_ctx], dim=-1))  # (N, hidden)
        return h + gate * h_ctx


# ==============================
# DSPLDualSpaceEncoder — main encoder
# ==============================

class DSPLDualSpaceEncoder(nn.Module):
    """DSPL-DualSpace encoder: Euclidean + Spherical Harmonic dual towers with
    bidirectional cross-space MP and residue context modulation.

    Args:
        in_dim: Input node feature dimension (48 for atom graph)
        hidden_dim: Hidden dimension
        num_layers: Number of DualEquiLayer blocks
        lmax: Maximum spherical harmonic order (2 = 0e + 1o + 2e)
        sh_threshold: Cosine similarity threshold for SH graph edges
        max_sh_neighbors: Max KNN neighbors in SH graph
        num_heads: Multi-head attention heads
        activation: Activation function
        residual: Use residual connections in DualEquiLayer
        task_type: 'task1' (atom-level regression) or 'task2' (residue-level classification)
    """
    def __init__(self,
                 in_dim: int = 48,
                 hidden_dim: int = 64,
                 num_layers: int = 5,
                 lmax: int = 2,
                 sh_threshold: float = 0.5,
                 max_sh_neighbors: int = 32,
                 num_heads: int = 4,
                 activation=nn.SiLU(),
                 residual: bool = True,
                 task_type: str = 'task1'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lmax = lmax
        self.sh_dim = _sh_dim(lmax)
        self.task_type = task_type

        # Embedding
        self.atom_embedding = nn.Linear(in_dim, hidden_dim)

        # SH initialization
        self.sh_init = SH_INIT(hidden_dim, lmax, activation)

        # SH graph builder
        self.sh_graph = SphericalHarmonicGraph(
            threshold=sh_threshold, max_neighbors=max_sh_neighbors)

        # Dual-space layers
        self.dual_blocks = nn.ModuleList([
            DualEquiLayer(hidden_dim=hidden_dim, num_heads=num_heads,
                          lmax=lmax, activation=activation, residual=residual)
            for _ in range(num_layers)
        ])

        # Residue context modulation (Task 1) — preserves atom resolution
        if task_type == 'task1':
            self.residue_context = ResidueContextModulation(hidden_dim, activation)
        else:
            self.residue_context = None

        # Predictor
        if task_type == 'task1':
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1),
            )

    def forward(self, x, pos, edge_index, batch=None,
                state_corr=None, atom_to_residue=None):
        """
        Args:
            x: (N, in_dim) atom features
            pos: (N, 3) coordinates
            edge_index: (2, E) Euclidean distance edges (precomputed, KD-tree 4.5A)
            batch: (N,) batch indices (all zeros for single protein)
            state_corr: (E, K) DSPL state-conditioned correlation (optional)
            atom_to_residue: (N,) residue membership (if None: identity)
        Returns:
            out: (N, 1) per-atom predictions (Task 1)
            aux: dict with (h_euclidean, h_spherical) for external fusion
        """
        N = x.size(0)
        device = x.device
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)
        if atom_to_residue is None:
            atom_to_residue = torch.arange(N, dtype=torch.long, device=device)
        else:
            # Ensure device consistency (atom_to_residue may come from CPU cache)
            atom_to_residue = atom_to_residue.to(device=device)

        # Embedding
        h = self.atom_embedding(x)  # (N, hidden_dim)

        # SH initialization
        node_sh = self.sh_init(h, pos, edge_index)  # (N, sh_dim)

        # SH graph
        edge_index_sh = self.sh_graph(node_sh, batch)

        # Dual-space layers
        for block in self.dual_blocks:
            h, pos, node_sh = block(h, pos, edge_index, edge_index_sh, node_sh,
                                    state_corr=state_corr)
            # Belt-and-suspenders: clip extreme h values that escape LayerNorm
            if h.abs().max() > 50.0:
                h = torch.tanh(h / h.abs().max().detach() * 10.0) * 10.0

        # Residue context modulation (Task 1)
        if self.residue_context is not None:
            h = self.residue_context(h, atom_to_residue)

        # Predict
        out = self.predictor(h)  # (N, 1)

        return out, {'h_final': h, 'pos_final': pos, 'node_sh_final': node_sh}


# ==============================
# DSPLDualSpace — LightningModule wrapper
# ==============================
# Compatible with DSPL_Ablation interface: forward returns (out, meta, towers)

import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch.nn.functional as TorchF


class DSPLDualSpace(nn.Module):
    """Lightning-compatible wrapper for DSPLDualSpaceEncoder.

    Provides the same interface as DSPL_Ablation:
      forward(data) → (predictions, meta_dict, towers_tuple)

    towers_tuple = (h_euclidean, h_spherical, gate) — for compatibility with DSPL_Ablation.
    Here h_spherical is derived from node_sh features; gate is always None.
    """

    def __init__(self, dspl_state_file=None, **kwargs):
        super().__init__()
        self.dspl_state_file = dspl_state_file
        self.encoder = DSPLDualSpaceEncoder(**kwargs)

    def forward(self, data, state_corr=None, atom_to_residue=None):
        """Forward pass.

        Args:
            data: PyG Data with x, pos, edge_index, pdb_id
            state_corr: (E, K) pre-loaded state_corr (from caller)
            atom_to_residue: (N,) residue mapping
        Returns:
            out: (N, 1) predictions
            meta: dict with metrics
            towers: (h, None, None) for compatibility
        """
        x = data.x
        pos = data.pos if hasattr(data, 'pos') else torch.zeros(
            x.shape[0], 3, device=x.device)
        edge_index = data.edge_index

        N = x.shape[0]
        if atom_to_residue is None:
            atom_to_residue = torch.arange(N, dtype=torch.long, device=x.device)

        enc_out, aux = self.encoder(
            x, pos, edge_index, state_corr=state_corr,
            atom_to_residue=atom_to_residue)

        out = torch.nan_to_num(enc_out, nan=0.0, posinf=2.0, neginf=0.0)
        meta = {'dyn_coverage': torch.tensor(0.0)}

        # Towers tuple for backward compat: (h_euclidean, None, None)
        return out, meta, (aux['h_final'], None, None)


# ==============================
# Module exports
# ==============================

__all__ = [
    'DSPLDualSpace',
    'DSPLDualSpaceEncoder',
    'DualEquiLayer',
    'EuclideanSpaceBlock',
    'SHSpaceBlock',
    'SH_INIT',
    'SphericalHarmonicGraph',
    'InnerProductSH',
    'BaseMLP',
    'ResidueContextModulation',
]
