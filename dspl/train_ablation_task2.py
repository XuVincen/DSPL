"""
DSPL Phase 2 — Task 2 (Binding Site Detection) Ablation Training

基于 train_ablation.py, 修改为 Task 2:
  - 残基级图 (21-dim one-hot, Cα coordinates)
  - 分类头 (2-class + CrossEntropyLoss + class_weights)
  - 评估指标: F1/Precision/Recall/AUC/Accuracy
  - state_corr: dspl_state_corr_task2.h5 (预处理时生成)

Usage:
  python dspl/phase2_crossmodal/train_ablation_task2.py --ablation full --architecture egnn --n-proteins 1000 --epochs 100
"""
import os, sys, argparse

# Root of the project (where data/ and dspl/ live).
# Defaults to this file's repo root; override with DSPL_ROOT env var.
ROOT = os.environ.get('DSPL_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, ROOT)

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GPSConv, RGCNConv
from vendor.egnn_layer import E_GCL  # vendored replacement for src.models.regnn.regnn_ensemble

# ==============================
# Point 2: State-Weighted E_GCL Wrapper
# ==============================
class StateWeightedE_GCL(nn.Module):
    """Point 2: State-Aware Message Passing with multi-state edge broadcasting.
    (Identical to train_ablation.py version)
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_states=5,
                 edges_in_d=1, act_fn=nn.ReLU(), residual=True,
                 shared_state_logits=None):
        super().__init__()
        self.num_states = num_states
        self.hidden_dim = hidden_dim
        if shared_state_logits is not None:
            self.state_logits = shared_state_logits
        else:
            self.state_logits = nn.Parameter(torch.ones(num_states))
        self.e_gcl = E_GCL(in_dim, hidden_dim, out_dim,
                           edges_in_d=edges_in_d, act_fn=act_fn, residual=residual)

    def forward(self, h, edge_index, coord, edge_attr=None, edge_type=None,
                state_corr_mask=None, node_attr=None):
        N = h.shape[0]
        device = h.device
        state_weights = F.softmax(self.state_logits, dim=0)
        h_sum = torch.zeros(N, self.hidden_dim, device=device)
        coord_sum = torch.zeros(N, 3, device=device)
        total_h_count = torch.zeros(N, device=device)
        total_coord_count = torch.zeros(N, 1, device=device)
        edge_feat_s = None

        if state_corr_mask is not None and edge_attr is not None:
            edge_attr_k = edge_attr
            for k in range(self.num_states):
                state_mask = state_corr_mask[:, k]
                if not state_mask.any():
                    continue
                ei_k = edge_index[:, state_mask]
                ea_k = edge_attr_k[state_mask, k:k+1]
                h_k, coord_k, _ = self.e_gcl(h, ei_k, coord,
                                             edge_attr=ea_k, node_attr=node_attr)
                h_sum += state_weights[k] * h_k
                coord_sum += state_weights[k] * coord_k
                total_h_count += state_weights[k]
                total_coord_count += state_weights[k]
        elif edge_type is not None:
            static_mask = (edge_type == 0)
            if static_mask.any():
                ei_static = edge_index[:, static_mask]
                ea_static = edge_attr[static_mask] if edge_attr is not None else None
                h_s, coord_s, edge_feat_s = self.e_gcl(
                    h, ei_static, coord, edge_attr=ea_static, node_attr=node_attr)
                h_sum += h_s
                coord_sum += coord_s
                total_h_count += 1.0
                total_coord_count += 1.0
            for k in range(self.num_states):
                state_k = k + 1
                if state_k > (edge_type.max().item() if edge_type.numel() > 0 else 0):
                    break
                state_mask = (edge_type == state_k)
                if not state_mask.any():
                    continue
                ei_k = edge_index[:, state_mask]
                ea_k = edge_attr[state_mask] if edge_attr is not None else None
                h_k, coord_k, _ = self.e_gcl(h, ei_k, coord,
                                             edge_attr=ea_k, node_attr=node_attr)
                h_sum += state_weights[k] * h_k
                coord_sum += state_weights[k] * coord_k
                total_h_count += state_weights[k]
                total_coord_count += state_weights[k]
        else:
            return self.e_gcl(h, edge_index, coord, edge_attr=edge_attr, node_attr=node_attr)

        h_out = h_sum / total_h_count.unsqueeze(-1).clamp(min=1e-8)
        coord_out = coord_sum / total_coord_count.clamp(min=1e-8)
        edge_feat = edge_feat_s if edge_feat_s is not None else torch.zeros(0, device=device)
        return h_out, coord_out, edge_feat

    def get_state_distribution(self):
        with torch.no_grad():
            weights = F.softmax(self.state_logits, dim=0)
        return weights.detach().cpu().numpy().tolist()

from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================
# Worker-level HDF5 handle cache
# ==============================
_worker_h5_cache = {}

def _get_worker_h5(h5_path):
    pid_key = (h5_path, os.getpid())
    if pid_key not in _worker_h5_cache:
        _worker_h5_cache[pid_key] = h5py.File(h5_path, 'r')
    return _worker_h5_cache[pid_key]

# ==============================
# Task 2: Residue Dataset
# ==============================
class ResidueDataset(torch.utils.data.Dataset):
    """Residue-level dataset for Task 2 (Binding Site Detection).
    Reads Cα-based graphs with 21-dim residue one-hot features, distance edges,
    and binding site binary labels.
    """
    def __init__(self, residue_h5_path, pdb_ids):
        self.h5_path = residue_h5_path
        self.pdb_ids = list(pdb_ids)

    def __len__(self):
        return len(self.pdb_ids)

    def __getitem__(self, idx):
        pid = self.pdb_ids[idx]
        f = _get_worker_h5(self.h5_path)
        g = f[pid]
        x = torch.tensor(g['node_features'][:], dtype=torch.float)        # (N, 21)
        y = torch.tensor(g['node_labels'][:], dtype=torch.long)           # (N,) binary
        pos = torch.tensor(g['ca_coords'][:], dtype=torch.float)          # (N, 3)
        ei = torch.tensor(g['edge_index_distance'][:], dtype=torch.long)  # (2, E)
        ew = torch.tensor(g['edge_weight_distance'][:], dtype=torch.float)  # (E,)
        return Data(
            x=x, edge_index=ei, edge_attr=ew.unsqueeze(-1),
            y=y, pos=pos, pdb_id=pid,
        )

# ==============================
# Constants
# ==============================
N_PROTOTYPES = 5
CORR_THRESHOLD = 0.1
GNN_HIDDEN_DIM = 64
GN_NUM_LAYERS = 5
CROSS_ATTENTION_LAYERS = 3

# Task 2 specific
TASK2_IN_DIM = 21         # residue type one-hot
TASK2_OUT_DIM = 2         # binary classification
TASK2_CLASS_WEIGHTS = [0.608, 2.805]  # balanced for 1:5.8 negative:positive ratio

# ==============================
# Cross-Modal Attention Layer (identical to train_ablation.py)
# ==============================
class CrossModalAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_out_s = nn.Linear(hidden_dim, hidden_dim)
        self.W_out_d = nn.Linear(hidden_dim, hidden_dim)
        self.gate_s = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
        )
        self.gate_d = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_dim ** -0.5

    def forward(self, h_static, h_dynamic):
        N = h_static.shape[0]
        s_pool = h_static.mean(dim=0)
        d_pool = h_dynamic.mean(dim=0)
        q_s = self.W_q(s_pool)
        k_d = self.W_k(d_pool)
        v_d = self.W_v(d_pool)
        cross_s_pool = torch.sigmoid((q_s * k_d).sum() * self.scale) * v_d
        cross_s_pool = self.dropout(self.W_out_s(cross_s_pool))
        q_d = self.W_q(d_pool)
        k_s = self.W_k(s_pool)
        v_s = self.W_v(s_pool)
        cross_d_pool = torch.sigmoid((q_d * k_s).sum() * self.scale) * v_s
        cross_d_pool = self.dropout(self.W_out_d(cross_d_pool))
        cross_s = cross_s_pool.unsqueeze(0).expand(N, -1)
        cross_d = cross_d_pool.unsqueeze(0).expand(N, -1)
        gate_s = self.gate_s(torch.cat([h_static, cross_s], dim=-1))
        h_s_out = gate_s * h_static + (1 - gate_s) * cross_s
        gate_d = self.gate_d(torch.cat([h_dynamic, cross_d], dim=-1))
        h_d_out = gate_d * h_dynamic + (1 - gate_d) * cross_d
        return h_s_out, h_d_out

# ==============================
# Task 2 Model: DSPL_Ablation adapted for classification
# ==============================
# Import the original DSPL_Ablation but override with Task 2 head
# We create a thin wrapper to minimize code duplication

class DSPL_Ablation_Task2(LightningModule):
    """Task 2 variant of DSPL_Ablation — Binding Site Classification.

    Shares the same architecture (Dual Tower + Cross-Attention + Gate Fusion)
    but with classification head and task-specific metrics.
    """
    def __init__(
        self,
        in_dim=TASK2_IN_DIM,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        num_gnn_layers=GN_NUM_LAYERS,
        num_cross_attn_layers=CROSS_ATTENTION_LAYERS,
        lr=1e-4,
        ablation='full',
        dspl_state_file=None,
        architecture='egnn',
        use_class_weights=True,
        lambda_align=0.0,
        n_prototypes=5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_gnn_layers = num_gnn_layers
        self.num_cross_attn_layers = num_cross_attn_layers
        self.ablation = ablation
        self.dspl_state_file = dspl_state_file
        self.architecture = architecture
        self.lambda_align = lambda_align
        self.num_prototypes = n_prototypes

        use_static = ablation in ('full', 'static_only', 'dual_tower', 'e2_combined')
        use_dynamic = ablation in ('full', 'dynamic_only', 'dual_tower', 'e2_combined')
        use_cross_attn = ablation == 'full'

        print(f"\nTask 2 Ablation: {ablation}")
        print(f"  Architecture: {architecture}")
        print(f"  Static Tower: {use_static}")
        print(f"  Dynamic Tower: {use_dynamic}")
        print(f"  Cross-Attention: {use_cross_attn}")
        print(f"  Class Weights: {use_class_weights}")
        if lambda_align > 0:
            print(f"  Alignment Loss (lambda={lambda_align}): cosine_embedding_loss(h_s, h_d)")
        if dspl_state_file:
            print(f"  State Corr: OFFLINE ({dspl_state_file})")

        # Shared state logits (Point 2)
        self.shared_state_logits = nn.Parameter(torch.ones(self.num_prototypes))

        # ========================================
        # Static Tower
        # ========================================
        self.use_static = use_static
        if use_static:
            if architecture == 'gcn':
                self.static_conv_in = GCNConv(in_dim, gnn_hidden_dim, add_self_loops=False)
            elif architecture == 'gat':
                self.static_conv_in = GATConv(in_dim, gnn_hidden_dim, heads=4, concat=False, edge_dim=1)
            elif architecture == 'egnn':
                self.static_conv_in = E_GCL(in_dim, gnn_hidden_dim, gnn_hidden_dim,
                                            edges_in_d=1, act_fn=nn.ReLU(), residual=False)
            elif architecture == 'gps':
                # GPSConv(channels, conv): channels = input = output dim
                # Need Linear projection to map in_dim → gnn_hidden_dim first
                self.static_proj_in = nn.Linear(in_dim, gnn_hidden_dim)
                self.gps_num_relations = self.num_prototypes + 1  # 0=distance, 1..K=states
                self.static_conv_in = GPSConv(
                    gnn_hidden_dim,
                    RGCNConv(gnn_hidden_dim, gnn_hidden_dim, num_relations=self.gps_num_relations),
                    heads=4, dropout=0.1,
                )
            self.static_norm_in = nn.LayerNorm(gnn_hidden_dim)
            self.static_dropout_in = nn.Dropout(0.1)
            self.static_convs = nn.ModuleList()
            self.static_norms = nn.ModuleList()
            self.static_dropouts = nn.ModuleList()
            for _ in range(num_gnn_layers - 1):
                if architecture == 'gcn':
                    self.static_convs.append(GCNConv(gnn_hidden_dim, gnn_hidden_dim, add_self_loops=False))
                elif architecture == 'gat':
                    self.static_convs.append(GATConv(gnn_hidden_dim, gnn_hidden_dim, heads=4, concat=False, edge_dim=1))
                elif architecture == 'egnn':
                    self.static_convs.append(E_GCL(gnn_hidden_dim, gnn_hidden_dim, gnn_hidden_dim,
                                                    edges_in_d=1, act_fn=nn.ReLU(), residual=True))
                elif architecture == 'gps':
                    self.static_convs.append(GPSConv(
                        gnn_hidden_dim,
                        RGCNConv(gnn_hidden_dim, gnn_hidden_dim, num_relations=self.gps_num_relations),
                        heads=4, dropout=0.1,
                    ))
                self.static_norms.append(nn.LayerNorm(gnn_hidden_dim))
                self.static_dropouts.append(nn.Dropout(0.1))

        # ========================================
        # Dynamic Tower
        # ========================================
        self.use_dynamic = use_dynamic
        if use_dynamic:
            if architecture == 'gcn':
                self.dynamic_edge_encoder = nn.Sequential(
                    nn.Linear(self.num_prototypes, gnn_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
                )
            else:
                self.dynamic_edge_encoder = None

            if architecture == 'gcn':
                self.dynamic_conv_in = GCNConv(in_dim, gnn_hidden_dim, add_self_loops=False)
            elif architecture == 'gat':
                self.dynamic_conv_in = GATConv(in_dim, gnn_hidden_dim, heads=4, concat=False, edge_dim=self.num_prototypes)
            elif architecture == 'egnn':
                self.dynamic_conv_in = StateWeightedE_GCL(in_dim, gnn_hidden_dim, gnn_hidden_dim,
                                                          num_states=self.num_prototypes, edges_in_d=1,
                                                          act_fn=nn.ReLU(), residual=False,
                                                          shared_state_logits=self.shared_state_logits)
            elif architecture == 'gps':
                # GPS dynamic tower: Linear projection + GPSConv
                self.dynamic_proj_in = nn.Linear(in_dim, gnn_hidden_dim)
                self.dynamic_conv_in = GPSConv(
                    gnn_hidden_dim,
                    RGCNConv(gnn_hidden_dim, gnn_hidden_dim, num_relations=self.gps_num_relations),
                    heads=4, dropout=0.1,
                )
            self.dynamic_norm_in = nn.LayerNorm(gnn_hidden_dim)
            self.dynamic_dropout_in = nn.Dropout(0.1)
            self.dynamic_convs = nn.ModuleList()
            self.dynamic_norms = nn.ModuleList()
            self.dynamic_dropouts = nn.ModuleList()
            for _ in range(num_gnn_layers - 1):
                if architecture == 'gcn':
                    self.dynamic_convs.append(GCNConv(gnn_hidden_dim, gnn_hidden_dim, add_self_loops=False))
                elif architecture == 'gat':
                    self.dynamic_convs.append(GATConv(gnn_hidden_dim, gnn_hidden_dim, heads=4, concat=False, edge_dim=self.num_prototypes))
                elif architecture == 'egnn':
                    self.dynamic_convs.append(StateWeightedE_GCL(gnn_hidden_dim, gnn_hidden_dim, gnn_hidden_dim,
                                                                 num_states=self.num_prototypes, edges_in_d=1,
                                                                 act_fn=nn.ReLU(), residual=True,
                                                                 shared_state_logits=self.shared_state_logits))
                elif architecture == 'gps':
                    self.dynamic_convs.append(GPSConv(
                        gnn_hidden_dim,
                        RGCNConv(gnn_hidden_dim, gnn_hidden_dim, num_relations=self.gps_num_relations),
                        heads=4, dropout=0.1,
                    ))
                self.dynamic_norms.append(nn.LayerNorm(gnn_hidden_dim))
                self.dynamic_dropouts.append(nn.Dropout(0.1))

        # ========================================
        # Cross-Modal Attention
        # ========================================
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn_layers = nn.ModuleList([
                CrossModalAttentionLayer(gnn_hidden_dim)
                for _ in range(num_cross_attn_layers)
            ])

        # ========================================
        # Fusion & Predictor (Task 2: binary classification head)
        # ========================================
        fusion_in = gnn_hidden_dim * 2 if use_static and use_dynamic else gnn_hidden_dim
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_in, gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
            nn.Sigmoid(),
        )
        # Task 2 head: 2-class binary classification
        self.predictor = nn.Sequential(
            nn.Linear(gnn_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, TASK2_OUT_DIM),
        )

        # Loss function
        self.use_class_weights = use_class_weights
        if use_class_weights:
            self.register_buffer('class_weights_tensor',
                                torch.tensor(TASK2_CLASS_WEIGHTS, dtype=torch.float))
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
            print(f"  Using class weights: {TASK2_CLASS_WEIGHTS}")
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # MD cache (for e2_combined / online mode)
        self.md_cache = {}

    # ---- MD preload (for e2_combined / online) ----
    def preload_md(self, pdb_ids):
        """Preload Cα MD trajectories for online PCA+k-means computation."""
        import h5py
        from scipy.spatial import cKDTree
        from vendor.kabsch import kabsch_align

        print(f"Preloading MD for {len(pdb_ids)} proteins...")
        with h5py.File('data/downloaded/MD.hdf5', 'r') as md_f, \
             h5py.File('data/downloaded/adaptability_MD.hdf5', 'r') as adapt_f:
            for i, pid in enumerate(pdb_ids):
                if pid not in md_f or pid not in adapt_f:
                    continue
                try:
                    # Extract Cα trajectory (simplified version)
                    adapt_grp = adapt_f[pid]
                    atoms_coords_ref = adapt_grp['atoms_coordinates_ref'][:]
                    atoms_residue = adapt_grp['atoms_residue'][:]
                    mbi = adapt_grp['molecules_begin_atom_index'][:].astype(int)
                    protein_end = int(mbi[1])
                    n_atoms = atoms_coords_ref.shape[0]

                    ca_indices = []
                    for j in range(n_atoms):
                        if j >= protein_end:
                            break
                        if j == 0 or atoms_residue[j] != atoms_residue[j - 1]:
                            ca_indices.append(j)
                    ca_indices = np.array(ca_indices, dtype=int)
                    ca_coords_ref = atoms_coords_ref[ca_indices]

                    md_grp = md_f[pid]
                    md_traj = md_grp['trajectory_coordinates'][:]
                    md_frame0 = md_traj[0]

                    # Match Cα to MD
                    md_mbi = md_grp.get('molecules_begin_atom_index')
                    search_limit = min(int(md_mbi[1]) if md_mbi is not None else md_traj.shape[1] // 2,
                                      md_traj.shape[1])
                    tree = cKDTree(md_frame0[:search_limit])
                    _, nn_idx = tree.query(ca_coords_ref, k=1)

                    ca_traj = md_traj[:, nn_idx].copy().astype(np.float32)
                    ref = ca_traj[0].copy()
                    for f in range(1, ca_traj.shape[0]):
                        ca_traj[f] = kabsch_align(ca_traj[f], ref)

                    self.md_cache[pid] = torch.from_numpy(ca_traj)
                except Exception as e:
                    pass
                if (i + 1) % 500 == 0:
                    print(f"  MD preload: {i+1}/{len(pdb_ids)}")
        print(f"  MD preload complete: {len(self.md_cache)} entries")

    # ---- State Corr computation (online mode) ----
    def _compute_state_corr(self, ca_traj, dist_edges):
        """PCA + k-means + state-conditioned Pearson — identical to train_ablation.py"""
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        T, N = ca_traj.shape[0], ca_traj.shape[1]
        f_flat = ca_traj.reshape(T, N * 3)
        n_comp = min(20, T - 1, f_flat.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        f_pca = pca.fit_transform(f_flat.numpy())
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10, max_iter=300)
        frame_labels = kmeans.fit_predict(f_pca)
        alpha = np.zeros((T, 5), dtype=np.float32)
        alpha[np.arange(T), frame_labels] = 1.0
        alpha_t = torch.from_numpy(alpha).to(self.device)
        alpha_entropy = -(alpha_t * torch.log(alpha_t + 1e-8)).sum(dim=-1).mean()

        E = dist_edges.shape[1]
        src_idx = dist_edges[0].long()
        dst_idx = dist_edges[1].long()
        xi = ca_traj[:, src_idx, :].permute(1, 0, 2)
        xj = ca_traj[:, dst_idx, :].permute(1, 0, 2)

        eps = 1e-8
        state_corr = torch.zeros(E, 5, device=self.device)
        state_corr_mask = torch.zeros(E, 5, dtype=torch.bool, device=self.device)

        for k in range(5):
            w = torch.tensor((frame_labels == k).astype(np.float32), device=self.device)
            w_sum = w.sum()
            if w_sum < 10:
                continue
            corr_k = torch.zeros(E, device=self.device)
            for dim in range(3):
                x_i = xi[:, :, dim]
                x_j = xj[:, :, dim]
                x_i_mean = (x_i * w[None, :]).sum(dim=1) / w_sum
                x_j_mean = (x_j * w[None, :]).sum(dim=1) / w_sum
                x_i_c = x_i - x_i_mean[:, None]
                x_j_c = x_j - x_j_mean[:, None]
                cov = (w[None, :] * x_i_c * x_j_c).sum(dim=1) / w_sum
                x_i_var = (w[None, :] * x_i_c ** 2).sum(dim=1) / w_sum
                x_j_var = (w[None, :] * x_j_c ** 2).sum(dim=1) / w_sum
                corr_dim = cov / (torch.sqrt(x_i_var + eps) * torch.sqrt(x_j_var + eps) + eps)
                corr_dim = torch.clamp(corr_dim, -1.0, 1.0)
                corr_k += corr_dim / 3.0
            state_corr[:, k] = corr_k
            state_corr_mask[:, k] = (~torch.isnan(corr_k)) & (torch.abs(corr_k) >= 0.1)

        state_corr = torch.nan_to_num(state_corr, nan=0.0)
        return state_corr, state_corr_mask, alpha_t, alpha_entropy

    # ---- Graph building ----
    def _build_dual_graphs(self, data):
        """Build static + dynamic graphs from data + offline state_corr."""
        pdb_id = data.pdb_id
        if isinstance(pdb_id, list):
            pdb_id = pdb_id[0]

        dist_edges = data.edge_index
        dist_weights = data.edge_attr

        static_data = {
            'x': data.x,
            'edge_index': dist_edges,
            'edge_weight': dist_weights.reshape(-1),
            'pos': data.pos if hasattr(data, 'pos') else None,
        }

        if self.dspl_state_file:
            f = _get_worker_h5(self.dspl_state_file)
            if pdb_id not in f:
                return None, None, None
            grp = f[pdb_id]
            state_corr_raw = grp['state_corr'][:]
            state_corr_mask_raw = grp['state_corr_mask'][:]

            E_expected = dist_edges.shape[1]
            if state_corr_raw.shape[0] != E_expected:
                if state_corr_raw.shape[0] > E_expected:
                    state_corr_raw = state_corr_raw[:E_expected]
                    state_corr_mask_raw = state_corr_mask_raw[:E_expected]
                else:
                    pad = E_expected - state_corr_raw.shape[0]
                    state_corr_raw = np.pad(state_corr_raw, ((0, pad), (0, 0)), mode='constant')
                    state_corr_mask_raw = np.pad(state_corr_mask_raw, ((0, pad), (0, 0)), mode='constant')

            state_corr = torch.from_numpy(state_corr_raw).to(self.device)
            state_corr_mask = torch.from_numpy(state_corr_mask_raw).to(self.device)
            alpha = torch.from_numpy(grp['alpha'][:]).to(self.device)
            alpha_entropy = -(alpha * torch.log(alpha + 1e-8)).sum(dim=-1).mean()
            dyn_edge_attr = state_corr * state_corr_mask.float()
            dyn_coverage = state_corr_mask.any(dim=-1).float().mean()

            dynamic_data = {
                'x': data.x,
                'edge_index': dist_edges.clone(),
                'edge_attr': dyn_edge_attr,
                'state_corr_mask': state_corr_mask,
                'pos': data.pos if hasattr(data, 'pos') else None,
            }
            return static_data, dynamic_data, {
                'alpha': alpha, 'alpha_entropy': alpha_entropy,
                'dyn_coverage': dyn_coverage, 'state_corr': state_corr,
            }

        # Online mode
        if pdb_id not in self.md_cache:
            return None, None, None
        ca_traj = self.md_cache[pdb_id].to(self.device)
        state_corr, state_corr_mask, alpha, alpha_entropy = self._compute_state_corr(ca_traj, dist_edges)
        dyn_edge_attr = state_corr * state_corr_mask.float()
        dyn_coverage = state_corr_mask.any(dim=-1).float().mean()

        dynamic_data = {
            'x': data.x, 'edge_index': dist_edges.clone(),
            'edge_attr': dyn_edge_attr, 'state_corr_mask': state_corr_mask,
        }
        return static_data, dynamic_data, {
            'alpha': alpha, 'alpha_entropy': alpha_entropy,
            'dyn_coverage': dyn_coverage, 'state_corr': state_corr,
        }

    def _build_e2_graphs(self, data):
        """E2: distance + global Pearson correlation edges (paper baseline)."""
        pdb_id = data.pdb_id
        if isinstance(pdb_id, list):
            pdb_id = pdb_id[0]
        if pdb_id not in self.md_cache:
            return None, None, None
        ca_traj = self.md_cache[pdb_id].to(self.device)
        dist_edges = data.edge_index
        dist_weights = data.edge_attr

        static_data = {
            'x': data.x, 'edge_index': dist_edges,
            'edge_weight': dist_weights.reshape(-1),
            'pos': data.pos if hasattr(data, 'pos') else None,
        }

        # Global Pearson
        src_idx = dist_edges[0].long()
        dst_idx = dist_edges[1].long()
        xi = ca_traj[:, src_idx, :].permute(1, 0, 2)
        xj = ca_traj[:, dst_idx, :].permute(1, 0, 2)
        T = ca_traj.shape[0]
        eps = 1e-8
        global_corr = torch.zeros(dist_edges.shape[1], device=self.device)
        for dim in range(3):
            x_i = xi[:, :, dim]
            x_j = xj[:, :, dim]
            x_i_c = x_i - x_i.mean(dim=1, keepdim=True)
            x_j_c = x_j - x_j.mean(dim=1, keepdim=True)
            cov = (x_i_c * x_j_c).sum(dim=1) / (T - 1)
            xi_var = (x_i_c.pow(2)).sum(dim=1) / (T - 1)
            xj_var = (x_j_c.pow(2)).sum(dim=1) / (T - 1)
            corr_dim = cov / (torch.sqrt(xi_var + eps) * torch.sqrt(xj_var + eps) + eps)
            corr_dim = torch.clamp(corr_dim, -1.0, 1.0)
            global_corr += corr_dim / 3.0

        corr_mask = (~torch.isnan(global_corr)) & (torch.abs(global_corr) >= 0.3)
        corr_attr = global_corr * corr_mask.float()
        dyn_coverage = corr_mask.float().mean()

        corr_data = {
            'x': data.x,
            'edge_index': dist_edges.clone(),
            'edge_weight': corr_attr,
            'pos': data.pos if hasattr(data, 'pos') else None,
        }
        return static_data, corr_data, {'dyn_coverage': dyn_coverage}

    # ---- Static Tower forward ----
    def _static_forward(self, static_data, x):
        edge_attr_s = static_data['edge_weight']
        if self.architecture == 'gcn':
            h_s = self.static_conv_in(x, static_data['edge_index'], edge_weight=edge_attr_s)
        elif self.architecture == 'gat':
            if isinstance(edge_attr_s, torch.Tensor) and edge_attr_s.dim() > 1:
                edge_attr_s = edge_attr_s.unsqueeze(-1) if edge_attr_s.dim() == 1 else edge_attr_s
            h_s = self.static_conv_in(x, static_data['edge_index'], edge_attr=edge_attr_s)
        elif self.architecture == 'egnn':
            pos = static_data.get('pos', torch.zeros(x.shape[0], 3, device=x.device))
            edge_attr_s = edge_attr_s.unsqueeze(-1) if edge_attr_s.dim() == 1 else edge_attr_s
            h_s = self.static_conv_in(x, static_data['edge_index'], pos, edge_attr=edge_attr_s)[0]
        elif self.architecture == 'gps':
            # GPS: project in_dim → hidden_dim, then GPSConv with edge_type=0 (distance)
            h_s = self.static_proj_in(x)
            E = static_data['edge_index'].shape[1]
            edge_type_s = torch.zeros(E, dtype=torch.long, device=x.device)
            batch_vec = torch.zeros(h_s.shape[0], dtype=torch.long, device=x.device)
            h_s = self.static_conv_in(h_s, static_data['edge_index'],
                                       batch=batch_vec, edge_type=edge_type_s)
        h_s = self.static_norm_in(h_s)
        h_s = F.relu(h_s)
        h_s = self.static_dropout_in(h_s)
        for layer_idx in range(len(self.static_convs)):
            if self.architecture == 'gcn':
                h_s = self.static_convs[layer_idx](h_s, static_data['edge_index'],
                                                    edge_weight=edge_attr_s)
            elif self.architecture == 'gat':
                h_s = self.static_convs[layer_idx](h_s, static_data['edge_index'],
                                                    edge_attr=edge_attr_s)
            elif self.architecture == 'egnn':
                h_s = self.static_convs[layer_idx](h_s, static_data['edge_index'], pos,
                                                    edge_attr=edge_attr_s)[0]
            elif self.architecture == 'gps':
                h_s = self.static_convs[layer_idx](h_s, static_data['edge_index'],
                                                    batch=batch_vec, edge_type=edge_type_s)
            h_s = self.static_norms[layer_idx](h_s)
            h_s = F.relu(h_s)
            h_s = self.static_dropouts[layer_idx](h_s)
        return h_s

    # ---- Dynamic Tower forward ----
    def _dynamic_forward(self, dynamic_data, x):
        edge_attr_d = dynamic_data.get('edge_attr')
        state_corr_mask = dynamic_data.get('state_corr_mask')
        pos = dynamic_data.get('pos', torch.zeros(x.shape[0], 3, device=x.device))

        # Precompute GCN edge_weight once (outside the loop)
        if self.architecture == 'gcn' and edge_attr_d is not None:
            # GCNConv accepts edge_weight as a 1D scalar per edge
            # Simply average the 5-dim state_corr vector → scalar, then pass through a
            # small MLP to get edge weights in a reasonable range
            edge_attr_d_f = edge_attr_d.float()
            if self.dynamic_edge_encoder is not None:
                # Encoder: [E,5] → [E,64] → mean → [E] → sigmoid → [0,1] range
                edge_attr_encoded = self.dynamic_edge_encoder(edge_attr_d_f)
                edge_weight = edge_attr_encoded.mean(dim=1)  # [E]
            else:
                edge_weight = edge_attr_d_f.mean(dim=1)
            # Clamp to prevent extreme values
            edge_weight = torch.sigmoid(edge_weight)  # [0,1] range, numerically stable

        if self.architecture == 'gcn' and edge_attr_d is not None:
            h_d = self.dynamic_conv_in(x, dynamic_data['edge_index'], edge_weight=edge_weight)
        elif self.architecture == 'gat':
            h_d = self.dynamic_conv_in(x, dynamic_data['edge_index'],
                                       edge_attr=edge_attr_d.float())
        elif self.architecture == 'egnn':
            h_d = self.dynamic_conv_in(x, dynamic_data['edge_index'], pos,
                                       edge_attr=edge_attr_d,
                                       state_corr_mask=state_corr_mask)[0]

        # Construct edge_type for GPS dynamic tower
        if self.architecture == 'gps':
            edge_attr_raw = dynamic_data.get('edge_attr')
            state_corr_mask = dynamic_data.get('state_corr_mask')
            E = dynamic_data['edge_index'].shape[1]
            edge_type_d = torch.zeros(E, dtype=torch.long, device=x.device)
            if state_corr_mask is not None:
                # Use state_corr_mask to determine active states per edge
                has_active = state_corr_mask.any(dim=-1)  # (E,)
                if has_active.any():
                    # argmax of abs(corr) among active states, +1 (reserve 0 for distance-only)
                    active_corr = edge_attr_raw * state_corr_mask.float()
                    dominant = active_corr.abs().argmax(dim=-1) + 1
                    edge_type_d[has_active] = dominant[has_active]
            else:
                # Fallback: threshold-based
                threshold = 0.1
                has_active = (edge_attr_raw.abs() >= threshold).any(dim=-1)
                if has_active.any():
                    dominant = edge_attr_raw[has_active].abs().argmax(dim=-1) + 1
                    edge_type_d[has_active] = dominant
            # Safety: clamp to valid range
            edge_type_d = edge_type_d.clamp(0, self.gps_num_relations - 1)

        if self.architecture == 'gps':
            h_d_proj = self.dynamic_proj_in(x)
            batch_vec_d = torch.zeros(h_d_proj.shape[0], dtype=torch.long, device=x.device)
            h_d = self.dynamic_conv_in(h_d_proj, dynamic_data['edge_index'],
                                        batch=batch_vec_d, edge_type=edge_type_d)

        h_d = self.dynamic_norm_in(h_d)
        h_d = F.relu(h_d)
        h_d = self.dynamic_dropout_in(h_d)
        for layer_idx in range(len(self.dynamic_convs)):
            if self.architecture == 'gcn' and edge_attr_d is not None:
                h_d = self.dynamic_convs[layer_idx](h_d, dynamic_data['edge_index'],
                                                     edge_weight=edge_weight)
            elif self.architecture == 'gat':
                h_d = self.dynamic_convs[layer_idx](h_d, dynamic_data['edge_index'],
                                                     edge_attr=edge_attr_d.float())
            elif self.architecture == 'egnn':
                h_d = self.dynamic_convs[layer_idx](h_d, dynamic_data['edge_index'], pos,
                                                     edge_attr=edge_attr_d,
                                                     state_corr_mask=state_corr_mask)[0]
            elif self.architecture == 'gps':
                h_d = self.dynamic_convs[layer_idx](h_d, dynamic_data['edge_index'],
                                                     batch=batch_vec_d, edge_type=edge_type_d)
            h_d = self.dynamic_norms[layer_idx](h_d)
            h_d = F.relu(h_d)
            h_d = self.dynamic_dropouts[layer_idx](h_d)
        return h_d

    # ---- E2 Correlation Tower forward ----
    def _e2_corr_forward(self, corr_data, x):
        edge_attr_c = corr_data['edge_weight']
        if isinstance(edge_attr_c, torch.Tensor) and edge_attr_c.dim() == 1:
            edge_attr_c = edge_attr_c.unsqueeze(-1)
        pos = corr_data.get('pos', torch.zeros(x.shape[0], 3, device=x.device))

        if self.architecture == 'gcn':
            h_c = self.e2_corr_conv_in(x, corr_data['edge_index'],
                                        edge_weight=edge_attr_c.reshape(-1))
        elif self.architecture == 'gat':
            h_c = self.e2_corr_conv_in(x, corr_data['edge_index'], edge_attr=edge_attr_c)
        elif self.architecture == 'egnn':
            h_c = self.e2_corr_conv_in(x, corr_data['edge_index'], pos, edge_attr=edge_attr_c)[0]
        elif self.architecture == 'gps':
            h_c_proj = self.e2_corr_proj_in(x)
            Ec = corr_data['edge_index'].shape[1]
            edge_type_c = torch.zeros(Ec, dtype=torch.long, device=x.device)
            batch_vec_c = torch.zeros(h_c_proj.shape[0], dtype=torch.long, device=x.device)
            h_c = self.e2_corr_conv_in(h_c_proj, corr_data['edge_index'],
                                        batch=batch_vec_c, edge_type=edge_type_c)
        h_c = self.e2_corr_norm_in(h_c)
        h_c = F.relu(h_c)
        for layer_idx in range(len(self.e2_corr_convs)):
            if self.architecture == 'gcn':
                h_c = self.e2_corr_convs[layer_idx](h_c, corr_data['edge_index'],
                                                     edge_weight=edge_attr_c.reshape(-1))
            elif self.architecture == 'gat':
                h_c = self.e2_corr_convs[layer_idx](h_c, corr_data['edge_index'],
                                                     edge_attr=edge_attr_c)
            elif self.architecture == 'egnn':
                h_c = self.e2_corr_convs[layer_idx](h_c, corr_data['edge_index'], pos,
                                                     edge_attr=edge_attr_c)[0]
            elif self.architecture == 'gps':
                h_c = self.e2_corr_convs[layer_idx](h_c, corr_data['edge_index'],
                                                     batch=batch_vec_c, edge_type=edge_type_c)
            h_c = self.e2_corr_norms[layer_idx](h_c)
            h_c = F.relu(h_c)
        return h_c

    def forward(self, data):
        x = data.x
        meta = {}

        if self.ablation == 'static_only':
            static_data = {
                'x': x, 'edge_index': data.edge_index,
                'edge_weight': data.edge_attr.reshape(-1),
                'pos': data.pos if hasattr(data, 'pos') else None,
            }
            h_s = self._static_forward(static_data, x)
            out = self.predictor(h_s)
            return out, meta, (h_s, None)

        if self.ablation == 'e2_combined':
            static_data, corr_data, corr_meta = self._build_e2_graphs(data)
            if static_data is None:
                return None, meta, (None, None)
            meta.update(corr_meta)
            if not hasattr(self, 'e2_corr_conv_in'):
                # Build E2 tower on first call
                self._build_e2_tower()
            h_s = self._static_forward(static_data, x)
            h_c = self._e2_corr_forward(corr_data, x)
            h_cat = torch.cat([h_s, h_c], dim=-1)
            gate = self.final_fusion(h_cat)
            h_fused = gate * h_s + (1 - gate) * h_c
            out = self.predictor(h_fused)
            return out, meta, (h_s, h_c, gate)

        static_data, dynamic_data, corr_meta = self._build_dual_graphs(data)
        if static_data is None:
            return None, meta, (None, None)
        meta.update(corr_meta)

        if self.ablation == 'dynamic_only':
            h_d = self._dynamic_forward(dynamic_data, x)
            out = self.predictor(h_d)
            return out, meta, (None, h_d)

        # Static forward
        h_s = self._static_forward(static_data, x)

        if self.ablation == 'static_only':
            out = self.predictor(h_s)
            return out, meta, (h_s, None)

        # Dynamic forward
        h_d = self._dynamic_forward(dynamic_data, x)

        if self.ablation == 'dual_tower':
            h_cat = torch.cat([h_s, h_d], dim=-1)
            gate = self.final_fusion(h_cat)
            h_fused = gate * h_s + (1 - gate) * h_d
            out = self.predictor(h_fused)
            return out, meta, (h_s, h_d, gate)

        elif self.ablation == 'full':
            cross_idx = 0
            for layer_idx in range(self.num_gnn_layers):
                if layer_idx >= self.num_gnn_layers - self.num_cross_attn_layers:
                    h_s, h_d = self.cross_attn_layers[cross_idx](h_s, h_d)
                    cross_idx += 1
            h_cat = torch.cat([h_s, h_d], dim=-1)
            gate = self.final_fusion(h_cat)
            h_fused = gate * h_s + (1 - gate) * h_d
            out = self.predictor(h_fused)
            return out, meta, (h_s, h_d, gate)

    def _build_e2_tower(self):
        """Build E2 correlation tower on first call (lazy init)."""
        self.e2_corr_conv_in = None
        if self.architecture == 'gcn':
            self.e2_corr_conv_in = GCNConv(TASK2_IN_DIM, GNN_HIDDEN_DIM, add_self_loops=False)
        elif self.architecture == 'gat':
            self.e2_corr_conv_in = GATConv(TASK2_IN_DIM, GNN_HIDDEN_DIM, heads=4, concat=False, edge_dim=1)
        elif self.architecture == 'egnn':
            self.e2_corr_conv_in = E_GCL(TASK2_IN_DIM, GNN_HIDDEN_DIM, GNN_HIDDEN_DIM,
                                         edges_in_d=1, act_fn=nn.ReLU(), residual=False)
        elif self.architecture == 'gps':
            self.e2_corr_proj_in = nn.Linear(TASK2_IN_DIM, GNN_HIDDEN_DIM)
            self.e2_corr_conv_in = GPSConv(
                GNN_HIDDEN_DIM,
                RGCNConv(GNN_HIDDEN_DIM, GNN_HIDDEN_DIM, num_relations=self.gps_num_relations),
                heads=4, dropout=0.1,
            )
        self.e2_corr_norm_in = nn.LayerNorm(GNN_HIDDEN_DIM)
        self.e2_corr_convs = nn.ModuleList()
        self.e2_corr_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers - 1):
            if self.architecture == 'gcn':
                self.e2_corr_convs.append(GCNConv(GNN_HIDDEN_DIM, GNN_HIDDEN_DIM, add_self_loops=False))
            elif self.architecture == 'gat':
                self.e2_corr_convs.append(GATConv(GNN_HIDDEN_DIM, GNN_HIDDEN_DIM, heads=4, concat=False, edge_dim=1))
            elif self.architecture == 'egnn':
                self.e2_corr_convs.append(E_GCL(GNN_HIDDEN_DIM, GNN_HIDDEN_DIM, GNN_HIDDEN_DIM,
                                                edges_in_d=1, act_fn=nn.ReLU(), residual=True))
            elif self.architecture == 'gps':
                self.e2_corr_convs.append(GPSConv(
                    GNN_HIDDEN_DIM,
                    RGCNConv(GNN_HIDDEN_DIM, GNN_HIDDEN_DIM, num_relations=self.gps_num_relations),
                    heads=4, dropout=0.1,
                ))
            self.e2_corr_norms.append(nn.LayerNorm(GNN_HIDDEN_DIM))

    # ---- Training & Evaluation ----
    def _get_state_distribution(self):
        if self.architecture != 'egnn' or not self.use_dynamic:
            return None
        if not hasattr(self, 'shared_state_logits'):
            return None
        with torch.no_grad():
            weights = F.softmax(self.shared_state_logits, dim=0)
        return weights.detach().cpu().numpy().tolist()

    def _log_state_distribution(self, prefix='train'):
        state_dist = self._get_state_distribution()
        if state_dist is None:
            return
        K = len(state_dist)
        for k in range(K):
            self.log(f'{prefix}/state_pi_{k+1}', float(state_dist[k]),
                    on_step=False, on_epoch=True, batch_size=1)
        probs = np.array(state_dist)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        self.log(f'{prefix}/state_entropy', float(entropy),
                on_step=False, on_epoch=True, batch_size=1)
        if prefix == 'train':
            pi_str = ' | '.join([f'π{k+1}={state_dist[k]:.4f}' for k in range(K)])
            print(f"[StateAttn] epoch={self.current_epoch} {pi_str}  H={entropy:.4f}")

    def training_step(self, batch, batch_idx):
        data = batch if not isinstance(batch, list) else batch[0]
        out, meta, intermediates = self(data)
        if out is None:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        target = data.y.squeeze()
        loss_task = self.loss_fn(out, target)
        self.log('train/loss', loss_task, on_epoch=True, batch_size=1)
        if meta and 'dyn_coverage' in meta:
            self.log('train/dyn_coverage', meta['dyn_coverage'], on_epoch=True, batch_size=1)

        # Alignment Loss (State-Structure)
        if self.lambda_align > 0 and isinstance(intermediates, tuple) and len(intermediates) >= 2:
            h_s, h_d = intermediates[0], intermediates[1]
            if h_s is not None and h_d is not None and self.ablation == 'full':
                target_sim = torch.ones(h_s.size(0), device=h_s.device)
                loss_align = F.cosine_embedding_loss(h_s, h_d, target_sim)
                self.log('train/align_loss', loss_align, on_epoch=True, batch_size=1)
                loss = loss_task + self.lambda_align * loss_align
            else:
                loss = loss_task
        else:
            loss = loss_task

        self.log('train/total_loss', loss, on_epoch=True, batch_size=1)
        if batch_idx == 0 and self.architecture == 'egnn':
            self._log_state_distribution('train')
        return loss

    def _shared_eval_step(self, batch, batch_idx, prefix):
        data = batch if not isinstance(batch, list) else batch[0]
        out, meta, _ = self(data)
        if out is None:
            return None
        target = data.y.squeeze()
        loss = self.loss_fn(out, target)
        self.log(f'{prefix}/loss', loss, on_step=False, on_epoch=True, batch_size=1)
        outputs_attr = f'{prefix}_outputs'
        if not hasattr(self, outputs_attr):
            setattr(self, outputs_attr, [])
        getattr(self, outputs_attr).append((out.detach().cpu(), target.detach().cpu()))
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, 'test')

    def _shared_epoch_end(self, prefix):
        outputs_attr = f'{prefix}_outputs'
        if not hasattr(self, outputs_attr):
            return
        outputs = getattr(self, outputs_attr)
        if len(outputs) == 0:
            return

        logits_list, targets_list = zip(*outputs)
        logits = torch.cat(logits_list)
        y = torch.cat(targets_list)
        preds = torch.argmax(logits, dim=1)

        # Task 2 metrics: Accuracy, Precision, Recall, F1, AUC
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                      f1_score, roc_auc_score, average_precision_score)

        y_np = y.numpy()
        preds_np = preds.numpy()
        probs_np = F.softmax(logits, dim=1)[:, 1].numpy()  # positive class probability

        acc = accuracy_score(y_np, preds_np)
        prec = precision_score(y_np, preds_np, zero_division=0)
        rec = recall_score(y_np, preds_np, zero_division=0)
        f1 = f1_score(y_np, preds_np, zero_division=0)

        metrics = {'acc': acc, 'precision': prec, 'recall': rec, 'f1': f1}

        # AUC (may fail if only one class present)
        try:
            if len(np.unique(y_np)) > 1:
                auc_roc = roc_auc_score(y_np, probs_np)
                auc_pr = average_precision_score(y_np, probs_np)
                metrics['auc_roc'] = auc_roc
                metrics['auc_pr'] = auc_pr
        except:
            pass

        for k, v in metrics.items():
            self.log(f'{prefix}/{k}', float(v), on_epoch=True, batch_size=1)

        if prefix == 'test':
            print(f"\nTest Results: F1={f1:.4f}, Precision={prec:.4f}, "
                  f"Recall={rec:.4f}, Accuracy={acc:.4f}", end='')
            if 'auc_roc' in metrics:
                print(f", AUC-ROC={metrics['auc_roc']:.4f}, AUC-PR={metrics['auc_pr']:.4f}", end='')
            print()

        setattr(self, outputs_attr, [])

    def on_validation_epoch_end(self):
        self._shared_epoch_end('val')
        if self.architecture == 'egnn' and self.use_dynamic:
            self._log_state_distribution('val')
        # Manual best-model saving (track best val/f1)
        if not hasattr(self, '_best_val_f1'):
            self._best_val_f1 = -float('inf')
        current = self.trainer.callback_metrics.get('val/f1', -float('inf'))
        if isinstance(current, torch.Tensor):
            current = current.item()
        if current > self._best_val_f1:
            self._best_val_f1 = current
            ckpt_dir = None
            for cb in self.trainer.callbacks:
                if isinstance(cb, ModelCheckpoint):
                    ckpt_dir = cb.dirpath
                    break
            if ckpt_dir is None:
                ckpt_dir = f'dspl_checkpoints/ablation_{self.ablation}_{self.architecture}_task2'
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.state_dict(),
                'val_f1': current,
                'ablation': self.ablation,
                'architecture': self.architecture,
            }, os.path.join(ckpt_dir, f'best_epoch_{self.current_epoch:03d}_f1_{current:.4f}.pt'))
            # Keep top 3
            ckpts = sorted(
                [f for f in os.listdir(ckpt_dir) if f.startswith('best_epoch_')],
                key=lambda x: float(x.split('_')[-1].replace('.pt', '')),
                reverse=True,
            )
            for old in ckpts[3:]:
                os.remove(os.path.join(ckpt_dir, old))

    def on_test_epoch_end(self):
        self._shared_epoch_end('test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        return [optimizer], [scheduler]


# ==============================
# Task 2 DataModule
# ==============================
class Task2DataModule(LightningDataModule):
    def __init__(self, residue_h5_path, train_ids, val_ids, test_ids, num_workers=0):
        super().__init__()
        self.h5_path = residue_h5_path
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_set = ResidueDataset(self.h5_path, self.train_ids)
        self.val_set = ResidueDataset(self.h5_path, self.val_ids)
        self.test_set = ResidueDataset(self.h5_path, self.test_ids)
        _ = self.train_set[0]
        print(f"Task 2 datasets ready: train={len(self.train_set)}, "
              f"val={len(self.val_set)}, test={len(self.test_set)}")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=1, shuffle=True,
                         num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, shuffle=False,
                         num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, shuffle=False,
                         num_workers=self.num_workers, pin_memory=True)


# ==============================
# Main
# ==============================
def main():
    parser = argparse.ArgumentParser(description='DSPL Task 2 — Binding Site Detection')
    parser.add_argument('--architecture', type=str, default='egnn',
                       choices=['gcn', 'gat', 'egnn', 'gps'])
    parser.add_argument('--ablation', type=str, required=True,
                       choices=['full', 'static_only', 'dual_tower', 'dynamic_only', 'e2_combined'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n-proteins', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fast-dev-run', action='store_true')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--cross-attn-layers', type=int, default=CROSS_ATTENTION_LAYERS)
    parser.add_argument('--dspl-state-file', type=str,
                       default='data/data_files/dspl_state_corr_task2.h5')
    parser.add_argument('--lambda-align', type=float, default=0.0)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--no-class-weights', action='store_true',
                       help='Disable class weights')
    parser.add_argument('--n-prototypes', type=int, default=5,
                       help='Number of state prototypes (K)')
    parser.add_argument('--config', type=str, default=None,
                       help='Optional YAML config file (see configs/*.yaml). CLI args override YAML values.')

    args = parser.parse_args()

    # Optional YAML config loading: defaults from config file, CLI overrides.
    if args.config:
        try:
            import yaml
        except ImportError:
            raise SystemExit('PyYAML is required to use --config. Install with: pip install pyyaml')
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        cli_defaults = {
            'architecture': 'egnn', 'ablation': None, 'epochs': 100, 'lr': 1e-4,
            'n_proteins': 1000, 'seed': 42, 'gpu_id': 0,
            'cross_attn_layers': CROSS_ATTENTION_LAYERS,
            'dspl_state_file': 'data/data_files/dspl_state_corr_task2.h5',
            'lambda_align': 0.0, 'num_workers': 0, 'n_prototypes': 5,
            'no_class_weights': False, 'fast_dev_run': False, 'config': None,
        }
        for key, value in cfg.items():
            arg_name = key.replace('-', '_')
            if arg_name in cli_defaults and getattr(args, arg_name, None) == cli_defaults[arg_name]:
                setattr(args, arg_name, value)
        print(f"Loaded config from {args.config}")

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    residue_h5 = 'data/data_files/residue_graph_data_distance_10.0_corr_aligned_0.3.h5'

    def load_split(p):
        with open(p) as f:
            return [l.strip() for l in f if l.strip()]

    train_ids = load_split('data/splits/binding_site/train.txt')
    val_ids = load_split('data/splits/binding_site/val.txt')
    test_ids = load_split('data/splits/binding_site/test.txt')

    if args.n_proteins:
        train_ids = train_ids[:args.n_proteins]
        val_ids = val_ids[:max(1, args.n_proteins // 5)]
        test_ids = test_ids[:max(1, args.n_proteins // 5)]

    ablation = args.ablation
    print("=" * 60)
    print(f"DSPL Phase 2 (Task 2) — Ablation Study: {ablation}")
    print(f"Task 2: Binding Site Detection (Residue-level Classification)")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Architecture: {args.architecture}, K: {args.n_prototypes}")
    print("=" * 60)

    # Data
    dm = Task2DataModule(residue_h5, train_ids, val_ids, test_ids,
                         num_workers=args.num_workers)
    dm.setup()

    # Model
    dspl_state_file = args.dspl_state_file
    if dspl_state_file and not os.path.exists(dspl_state_file):
        print(f"WARNING: state_corr file not found: {dspl_state_file}")
        print("  Will use ONLINE mode (PCA+k-means per forward) — MUCH SLOWER")
        dspl_state_file = None

    if dspl_state_file:
        print(f"Using offline state_corr: {dspl_state_file}")

    model = DSPL_Ablation_Task2(
        in_dim=TASK2_IN_DIM,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        num_gnn_layers=GN_NUM_LAYERS,
        num_cross_attn_layers=args.cross_attn_layers,
        lr=args.lr,
        ablation=ablation,
        dspl_state_file=dspl_state_file,
        architecture=args.architecture,
        use_class_weights=not args.no_class_weights,
        lambda_align=args.lambda_align,
        n_prototypes=args.n_prototypes,
    )

    # MD preload (for online mode / e2_combined)
    needs_md_online = (ablation == 'e2_combined') or (not dspl_state_file and ablation != 'static_only')
    if needs_md_online:
        all_ids = train_ids + val_ids + test_ids
        model.preload_md(list(set(all_ids)))

    # Checkpoint & Logger
    exp_name = f'ablation_{ablation}_{args.architecture}_task2'
    suffix_parts = []
    if dspl_state_file:
        suffix_parts.append('offline')
    if not args.n_proteins:
        suffix_parts.append('fulldata')
    elif args.n_proteins != 1000:
        suffix_parts.append(f'n{args.n_proteins}')
    if args.lambda_align > 0:
        suffix_parts.append(f'align{args.lambda_align}')
    if args.n_prototypes != 5:
        suffix_parts.append(f'k{args.n_prototypes}')
    if suffix_parts:
        exp_name = exp_name + '_' + '_'.join(suffix_parts)

    ckpt_dir = f'dspl_checkpoints/{exp_name}'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir, filename='epoch_{epoch:03d}',
        save_top_k=-1, every_n_epochs=25,
        save_on_train_epoch_end=True,
    )

    tb_logger = TensorBoardLogger(
        save_dir='dspl/phase2_crossmodal/outputs',
        name=f'ablation_{ablation}_{args.architecture}_task2'
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[ckpt_cb],
        logger=tb_logger,
        log_every_n_steps=5,
        val_check_interval=1.0,
        num_sanity_val_steps=0,
        accumulate_grad_batches=4,
        accelerator="gpu", devices=[args.gpu_id],
        gradient_clip_val=1.0,
        fast_dev_run=args.fast_dev_run,
    )

    print(f"\nStarting Task 2 training: {ablation}")
    trainer.fit(model, dm)
    trainer.test(model, dm)


if __name__ == '__main__':
    main()
