"""Vendored E(n) Equivariant Graph Convolutional Layer (E_GCL).

This is a self-contained re-implementation of the `E_GCL` layer from the
original "Boosting Protein Graph Representations through Static-Dynamic Fusion"
repository (``src/models/regnn/regnn_ensemble.py``), which itself follows the
EGNN formulation of Satorras et al. (2021).

We vendor it here so that the DSPL package has **no hard dependency** on the
upstream ``src`` package, making the repository clone-and-run reproducible.

The interface is kept identical to the upstream ``E_GCL`` so it is a drop-in
replacement:

    class E_GCL(nn.Module):
        def __init__(self, input_nf, output_nf, hidden_nf,
                     edges_in_d=0, act_fn=nn.SiLU(),
                     residual=True, attention=False, normalize=False,
                     coords_agg='mean', tanh=False, coords_range=15,
                     update_coords=False, norm_diff=False):
        def forward(self, h, edge_index, coord=None, edge_attr=None, node_attr=None):
            return h_new, coord  # pos updated only if update_coords=True

Notes on the fields actually used by DSPL:

* ``edges_in_d`` -- dimension of scalar edge features (distance weight for the
  static tower, K-dim state-conditioned correlation for the dynamic tower).
* ``residual`` -- whether ``h`` residuals to the input; DSPL passes
  ``residual=False`` on the input (dim-changing) layer and ``residual=True`` on
  hidden layers.
* ``coord`` -- the atom coordinates (N, 3); may be ``None`` in which case the
  coordinate term is skipped (not used by DSPL, which keeps ``update_coords=False``).

The forward pass returns ``(h, coord)``; DSPL only uses ``[0]``.
"""

import torch
import torch.nn as nn


class E_GCL(nn.Module):
    """E(n) Equivariant Convolutional Layer (Satorras et al. 2021).

    Duplicates the ``E_GCL`` interface from the upstream reproducibility repo so
    this package is fully self-contained.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        act_fn=nn.SiLU(),
        residual=True,
        attention=False,
        normalize=False,
        coords_agg="mean",
        tanh=False,
        coords_range=15,
        update_coords=False,
        norm_diff=False,
    ):
        super().__init__()

        self.input_nf = input_nf
        self.output_nf = output_nf
        self.hidden_nf = hidden_nf
        self.edges_in_d = edges_in_d
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.update_coords = update_coords
        self.tanh = tanh
        self.coords_range = coords_range
        self.norm_diff = norm_diff

        input_edge = input_nf * 2 + edges_in_d
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        coord_mlp_layers = [nn.Linear(hidden_nf, hidden_nf), act_fn, nn.Linear(hidden_nf, 1)]
        if tanh:
            coord_mlp_layers.append(nn.Tanh())
            self.coords_range = coords_range
        self.coord_mlp = nn.Sequential(*coord_mlp_layers)

        if attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

        if normalize:
            self.normalize_module = nn.LayerNorm(output_nf)

    def _edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # unused, kept for interface parity
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def _node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = _unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        if self.normalize:
            out = self.normalize_module(out)
        return out, agg

    def _coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = _unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        trans = _clamp(trans, self.coords_range)
        agg = _unsorted_segment_mean(coord, row, num_segments=coord.size(0))
        return coord + trans, agg

    def forward(self, h, edge_index, coord=None, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial = torch.zeros((edge_index.size(1), 1), device=h.device)
        if coord is not None:
            coord_diff = coord[row] - coord[col]
            radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
            if self.norm_diff:
                radial = torch.sqrt(radial + 1e-8) / torch.max(radial + 1e-8)

        # Edge features always carry the scalar inputs found in edges_in_d
        edge_feat = self._edge_model(h[row], h[col], radial, edge_attr)
        h, agg = self._node_model(h, edge_index, edge_feat, node_attr)

        coord_out = coord
        if coord is not None and self.update_coords:
            coord_out, _ = self._coord_model(
                coord, edge_index, coord_diff, edge_feat
            )

        return h, coord_out


# ------------------------------------------------------------------------------
# Scatter helpers (avoid torch_scatter dependency)
# ------------------------------------------------------------------------------
def _unsorted_segment_sum(data, segment_ids, num_segments):
    """Equivalent of torch_scatter.scatter_add(..., dim=0, reduce='sum')."""
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    return result


def _unsorted_segment_mean(data, segment_ids, num_segments):
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    count = data.new_zeros((num_segments, data.size(1)))
    count.index_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def _clamp(x, lim):
    return torch.clamp(x, -lim, lim)
