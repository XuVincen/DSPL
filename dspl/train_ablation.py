# """
# DSPL Phase 2 — Ablation Study for Cross-Modal Attention
#   + State-Structure Alignment Loss (cosine embedding between h_static & h_dynamic)
#
# Ablation variants (--ablation flag):
#   full          = Static Tower + Dynamic Tower + Cross-Attention + Gate Fusion (Phase 2 original)
#   static_only   = Static Tower only (distance edges, single tower) — B
#   dual_tower    = Static + Dynamic Tower, concat fusion, NO cross-attention — C
#   dynamic_only  = Dynamic Tower only (K-dim state-corr edges, single tower) — D
#   e2_combined   = Paper E2 baseline (Distance + Global Pearson Corr, dual-tower + gate fusion)
#
# All variants: same 1000 proteins, same hyperparameters, same random seed.
#
# Usage:
#   python dspl/phase2_crossmodal/train_ablation.py --ablation full --n-proteins 1000 --epochs 100
#   python dspl/phase2_crossmodal/train_ablation.py --ablation static_only --n-proteins 1000 --epochs 100
#   python dspl/phase2_crossmodal/train_ablation.py --ablation dual_tower --n-proteins 1000 --epochs 100
#   python dspl/phase2_crossmodal/train_ablation.py --ablation dynamic_only --n-proteins 1000 --epochs 100
#   python dspl/phase2_crossmodal/train_ablation.py --ablation e2_combined --n-proteins 1000 --epochs 100
#   python dspl/phase2_crossmodal/train_ablation.py --ablation full --lambda-align 0.1 --n-proteins 1000 --epochs 100
# """
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
from dspl.phase2_crossmodal.dspl_dualspace import DSPLDualSpace

from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================
# Worker-level HDF5 handle cache
# ==============================
# Each DataLoader worker process opens each HDF5 file ONCE and reuses it.
# Keyed by (file_path, os.getpid()) so forked workers get their own handle.
# With num_workers=0, only the main process has one entry per file.
_worker_h5_cache = {}

def _get_worker_h5(h5_path):
    """Return a cached h5py.File handle for the current worker process.
    Opens the file once per process, never closes (OS reclaims on exit)."""
    pid_key = (h5_path, os.getpid())
    if pid_key not in _worker_h5_cache:
        _worker_h5_cache[pid_key] = h5py.File(h5_path, 'r')
    return _worker_h5_cache[pid_key]

# ==============================
# Lazy Dataset — reads single protein per __getitem__
# ==============================
class LazyAtomDataset(torch.utils.data.Dataset):
    """Lazy-loading atom graph dataset.
    Each __getitem__ reads exactly ONE protein from HDF5, so peak memory
    is O(1) instead of O(N). HDF5 handles are cached per worker process
    via _get_worker_h5, avoiding per-step file open/close overhead."""

    def __init__(self, atom_h5_path, pdb_ids):
        self.atom_h5_path = atom_h5_path
        self.pdb_ids = list(pdb_ids)

    def __len__(self):
        return len(self.pdb_ids)

    def __getitem__(self, idx):
        pid = self.pdb_ids[idx]
        f = _get_worker_h5(self.atom_h5_path)
        ag = f[pid]
        x = torch.tensor(ag['node_features'][:], dtype=torch.float)
        y = torch.tensor(ag['node_labels'][:], dtype=torch.float)
        pos = torch.tensor(ag['atoms_coordinates'][:], dtype=torch.float)
        ei = torch.tensor(ag['edge_index_distance'][:], dtype=torch.long)
        ew = torch.tensor(ag['edge_weight_distance'][:], dtype=torch.float)
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
CROSS_ATTENTION_LAYERS = 3  # only used in 'full' variant


# ==============================
# Cross-Modal Attention Layer
# ==============================
class CrossModalAttentionLayer(nn.Module):
    """Bidirectional cross-attention using mean-pooled protein-level representations."""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_out_s = nn.Linear(hidden_dim, hidden_dim)
        self.W_out_d = nn.Linear(hidden_dim, hidden_dim)

        self.gate_s = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.gate_d = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
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
# Main Model: DSPL Ablation
# ==============================
class DSPL_Ablation(LightningModule):
    def __init__(
        self,
        in_dim=48,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        num_gnn_layers=GN_NUM_LAYERS,
        num_cross_attn_layers=CROSS_ATTENTION_LAYERS,
        n_prototypes=N_PROTOTYPES,
        lr=1e-4,
        ablation='full',
        dspl_state_file=None,  # Path to precomputed state_corr H5 (offline mode)
        architecture='gcn',    # 'gcn' | 'gat' | 'egnn'
        lambda_align=0.0,      # State-Structure Alignment Loss weight (0=off)
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
        self.n_prototypes = n_prototypes

        use_static = ablation in ('full', 'static_only', 'dual_tower', 'e2_combined')
        use_dynamic = ablation in ('full', 'dynamic_only', 'dual_tower', 'e2_combined')
        use_cross_attn = ablation == 'full'

        print(f"\nAblation config: {ablation}")
        print(f"  Architecture: {architecture}")
        print(f"  Static Tower: {use_static}")
        print(f"  Dynamic Tower: {use_dynamic}")
        print(f"  Cross-Attention: {use_cross_attn}")
        if lambda_align > 0:
            print(f"  Alignment Loss (lambda={lambda_align}): cosine_embedding_loss(h_s, h_d)")
        if dspl_state_file:
            print(f"  State Corr Source: OFFLINE ({dspl_state_file})")
        else:
            print(f"  State Corr Source: ONLINE (PCA+k-means per forward)")

        # ========================================
        # DualSpace Architecture (standalone, skip dual-tower setup)
        # ========================================
        if architecture == 'dualspace':
            print(f"  DualSpace: Euclidean + Spherical Harmonic dual towers")
            print(f"  DualSpace params: hidden=64, layers=5, lmax=2, heads=4")
            self.dualspace_model = DSPLDualSpace(
                dspl_state_file=dspl_state_file,
                in_dim=in_dim,
                hidden_dim=gnn_hidden_dim,
                num_layers=num_gnn_layers,
                lmax=2,
                num_heads=4,
                task_type='task1')
            # Cache for atom_to_residue mappings (lazy-loaded from MD HDF5)
            self._residue_cache = {}
            # MD file handle for reading residue data on-demand
            self._md_file_for_residue = None
            # Skip all tower/attention/fusion/predictor setup
            return

        # ========================================
        # Static Tower
        # ========================================
        self.use_static = use_static
        if use_static:
            if architecture == 'gcn':
                self.static_conv_in = GCNConv(in_dim, gnn_hidden_dim, add_self_loops=False)
            elif architecture == 'gat':
                # edge_dim=1: scalar distance weight becomes edge_attr
                self.static_conv_in = GATConv(in_dim, gnn_hidden_dim, heads=4, concat=False, edge_dim=1)
            elif architecture == 'egnn':
                self.static_conv_in = E_GCL(in_dim, gnn_hidden_dim, gnn_hidden_dim,
                                            edges_in_d=1, act_fn=nn.ReLU(), residual=False)  # dim change, no residual
            elif architecture == 'gps':
                self.static_proj_in = nn.Linear(in_dim, gnn_hidden_dim)
                self.gps_num_relations = self.n_prototypes + 1
                self.static_conv_in = GPSConv(
                    gnn_hidden_dim,
                    RGCNConv(gnn_hidden_dim, gnn_hidden_dim, num_relations=self.gps_num_relations),
                    heads=4, dropout=0.1,
                )
            else:
                raise ValueError(f"Unknown architecture: {architecture}")
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
                                                    edges_in_d=1, act_fn=nn.ReLU(), residual=True))  # same dim, residual OK
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
            # Edge encoder: only needed for GCN (flatten K-dim edge_attr to scalar)
            if architecture == 'gcn':
                self.dynamic_edge_encoder = nn.Sequential(
                    nn.Linear(self.n_prototypes, gnn_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
                )
            else:
                self.dynamic_edge_encoder = None  # GAT uses K-dim directly; EGNN uses edges_in_d

            if architecture == 'gcn':
                self.dynamic_conv_in = GCNConv(in_dim, gnn_hidden_dim, add_self_loops=False)
            elif architecture == 'gat':
                # edge_dim=K: directly consume K-dim state_corr as edge features
                self.dynamic_conv_in = GATConv(in_dim, gnn_hidden_dim, heads=4, concat=False, edge_dim=self.n_prototypes)
            elif architecture == 'egnn':
                self.dynamic_conv_in = E_GCL(in_dim, gnn_hidden_dim, gnn_hidden_dim,
                                             edges_in_d=self.n_prototypes, act_fn=nn.ReLU(), residual=False)  # dim change
            elif architecture == 'gps':
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
                    self.dynamic_convs.append(GATConv(gnn_hidden_dim, gnn_hidden_dim, heads=4, concat=False, edge_dim=self.n_prototypes))
                elif architecture == 'egnn':
                    self.dynamic_convs.append(E_GCL(gnn_hidden_dim, gnn_hidden_dim, gnn_hidden_dim,
                                                    edges_in_d=self.n_prototypes, act_fn=nn.ReLU(), residual=True))
                elif architecture == 'gps':
                    self.dynamic_convs.append(GPSConv(
                        gnn_hidden_dim,
                        RGCNConv(gnn_hidden_dim, gnn_hidden_dim, num_relations=self.gps_num_relations),
                        heads=4, dropout=0.1,
                    ))
                self.dynamic_norms.append(nn.LayerNorm(gnn_hidden_dim))
                self.dynamic_dropouts.append(nn.Dropout(0.1))

        # ========================================
        # E2 Correlation Tower (scalar edge_weight GCNConv for global Pearson)
        # ========================================
        self.is_e2 = (ablation == 'e2_combined')
        if self.is_e2:
            if architecture == 'gcn':
                self.e2_corr_conv_in = GCNConv(in_dim, gnn_hidden_dim, add_self_loops=False)
            elif architecture == 'gat':
                self.e2_corr_conv_in = GATConv(in_dim, gnn_hidden_dim, heads=4, concat=False, edge_dim=1)
            elif architecture == 'egnn':
                self.e2_corr_conv_in = E_GCL(in_dim, gnn_hidden_dim, gnn_hidden_dim,
                                             edges_in_d=1, act_fn=nn.ReLU(), residual=False)  # dim change
            elif architecture == 'gps':
                self.e2_corr_proj_in = nn.Linear(in_dim, gnn_hidden_dim)
                self.e2_corr_conv_in = GPSConv(
                    gnn_hidden_dim,
                    RGCNConv(gnn_hidden_dim, gnn_hidden_dim, num_relations=self.gps_num_relations),
                    heads=4, dropout=0.1,
                )
            self.e2_corr_norm_in = nn.LayerNorm(gnn_hidden_dim)
            self.e2_corr_dropout_in = nn.Dropout(0.1)
            self.e2_corr_convs = nn.ModuleList()
            self.e2_corr_norms = nn.ModuleList()
            self.e2_corr_dropouts = nn.ModuleList()
            for _ in range(num_gnn_layers - 1):
                if architecture == 'gcn':
                    self.e2_corr_convs.append(GCNConv(gnn_hidden_dim, gnn_hidden_dim, add_self_loops=False))
                elif architecture == 'gat':
                    self.e2_corr_convs.append(GATConv(gnn_hidden_dim, gnn_hidden_dim, heads=4, concat=False, edge_dim=1))
                elif architecture == 'egnn':
                    self.e2_corr_convs.append(E_GCL(gnn_hidden_dim, gnn_hidden_dim, gnn_hidden_dim,
                                                    edges_in_d=1, act_fn=nn.ReLU(), residual=True))
                elif architecture == 'gps':
                    self.e2_corr_convs.append(GPSConv(
                        gnn_hidden_dim,
                        RGCNConv(gnn_hidden_dim, gnn_hidden_dim, num_relations=self.gps_num_relations),
                        heads=4, dropout=0.1,
                    ))
                self.e2_corr_norms.append(nn.LayerNorm(gnn_hidden_dim))
                self.e2_corr_dropouts.append(nn.Dropout(0.1))

        # ========================================
        # Cross-Modal Attention
        # ========================================
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn_layers = nn.ModuleList()
            for _ in range(num_cross_attn_layers):
                self.cross_attn_layers.append(CrossModalAttentionLayer(gnn_hidden_dim))
            self.cross_start_layer = num_gnn_layers - num_cross_attn_layers

        # ========================================
        # Final Fusion & Predictor
        # ========================================
        if use_static and use_dynamic:
            # Gate fusion for dual-tower
            self.final_fusion = nn.Sequential(
                nn.Linear(gnn_hidden_dim * 2, gnn_hidden_dim),
                nn.ReLU(),
                nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
                nn.Sigmoid(),
            )
            self._fusion_type = 'gate'
        elif use_static:
            self._fusion_type = 'static_only'
        elif use_dynamic:
            self._fusion_type = 'dynamic_only'
        else:
            raise ValueError("Invalid ablation: need at least one tower")

        self.predictor = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gnn_hidden_dim // 2, 1),
        )

        # MD cache
        self.md_cache = {}

    def preload_md(self, pdb_ids, md_path='data/downloaded/MD.hdf5'):
        """Preload MD trajectories — match graph atom nodes to MD protein atoms."""
        from vendor.kabsch import kabsch_align
        print(f"Preloading MD trajectories for {len(pdb_ids)} proteins...")
        atom_h5 = 'data/data_files/atom_graph_OnlyProtein_distance_4.5_planA.h5'
        loaded = 0
        with h5py.File(md_path, 'r') as md_file, h5py.File(atom_h5, 'r') as atom_h5_f:
            for pid in pdb_ids:
                if pid not in md_file or pid not in atom_h5_f:
                    continue
                try:
                    grp = md_file[pid]
                    mbi = grp['molecules_begin_atom_index'][:]
                    p_start, p_end = mbi[0], mbi[1]
                    full_traj = grp['trajectory_coordinates'][:]
                    f0 = full_traj[0, p_start:p_end]

                    graph_coords = atom_h5_f[pid]['atoms_coordinates'][:]
                    from scipy.spatial import cKDTree
                    tree = cKDTree(f0)
                    _, nn_idx = tree.query(graph_coords, k=1)
                    atom_idx = nn_idx + p_start

                    atom_traj = full_traj[:, atom_idx]
                    ref = atom_traj[0].copy()
                    for f in range(1, atom_traj.shape[0]):
                        atom_traj[f] = kabsch_align(atom_traj[f], ref)

                    self.md_cache[pid] = torch.tensor(atom_traj, dtype=torch.float32)
                    loaded += 1
                except Exception as e:
                    print(f"  [SKIP] {pid}: {e}")
        print(f"Preloaded {loaded}/{len(pdb_ids)} MD trajectories")

    def _load_atom_to_residue(self, pdb_id, data, md_path='data/downloaded/MD.hdf5'):
        """Map graph atoms to residue indices using MD HDF5 atoms_residue field.

        Uses KD-tree matching of graph atom coordinates to MD frame-0 protein atom
        coordinates (same as preload_md), then reads atoms_residue from MD HDF5
        and remaps to consecutive 0..M-1 indices.

        Args:
            pdb_id: protein identifier
            data: PyG Data with pos (N, 3) atom coordinates
            md_path: path to MD HDF5
        Returns:
            atom_to_residue: LongTensor (N,) — consecutive residue index per atom,
                             or identity (0..N-1) if matching fails
        """
        N = data.x.shape[0]
        # Default: identity mapping (fallback if MD not available or match fails)
        identity = torch.arange(N, dtype=torch.long)
        try:
            import h5py
            from scipy.spatial import cKDTree

            if self._md_file_for_residue is None:
                self._md_file_for_residue = h5py.File(md_path, 'r')

            md_f = self._md_file_for_residue
            if pdb_id not in md_f:
                return identity

            grp = md_f[pdb_id]
            mbi = grp['molecules_begin_atom_index'][:]
            p_start, p_end = int(mbi[0]), int(mbi[1])

            # KD-tree: MD frame-0 protein atoms
            f0 = grp['trajectory_coordinates'][0, p_start:p_end]
            graph_coords = data.pos.cpu().numpy() if hasattr(data, 'pos') else None
            if graph_coords is None:
                return identity

            tree = cKDTree(f0)
            _, nn_idx = tree.query(graph_coords, k=1)
            atom_idx = nn_idx + p_start  # absolute index in MD

            # Read residue IDs and remap
            residues = grp['atoms_residue'][:]
            matched_residues = residues[atom_idx]
            unique_res = sorted(set(matched_residues.tolist()))
            res_map = {old: new for new, old in enumerate(unique_res)}
            remapped = np.array([res_map[r] for r in matched_residues], dtype=np.int64)
            return torch.from_numpy(remapped)

        except Exception as e:
            # Silently fall back to identity — won't break training
            return identity


    def _compute_state_corr(self, ca_traj, dist_edges):
        """Per-protein PCA + k-means → K-dim edge features."""
        T, N, _ = ca_traj.shape
        device = ca_traj.device

        with torch.no_grad():
            f_flat = ca_traj.reshape(T, -1).cpu().numpy()
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
            n_comp = min(20, T - 1, f_flat.shape[1])
            pca = PCA(n_components=n_comp, random_state=42)
            f_pca_np = pca.fit_transform(f_flat)
            kmeans = KMeans(n_clusters=self.n_prototypes, random_state=42, n_init=10, max_iter=300)
            labels = kmeans.fit_predict(f_pca_np)
            alpha = F.one_hot(torch.tensor(labels, dtype=torch.long, device=device),
                             num_classes=self.n_prototypes).float()

        alpha_entropy = -(alpha * torch.log(alpha + 1e-8)).sum(dim=-1).mean()
        n_pairs = dist_edges.shape[1]
        residue_pairs = dist_edges.T
        xi = ca_traj[:, residue_pairs[:, 0], :].transpose(0, 1)
        xj = ca_traj[:, residue_pairs[:, 1], :].transpose(0, 1)
        eps = 1e-8

        state_corr = torch.zeros(n_pairs, self.n_prototypes, device=device)
        for k in range(self.n_prototypes):
            w = alpha[:, k]
            w_sum = w.sum().clamp(min=eps)
            if w_sum < eps:
                continue
            corr_k = torch.zeros(n_pairs, device=device)
            for dim in range(3):
                x_i = xi[:, :, dim]
                x_j = xj[:, :, dim]
                x_i_mean = (x_i * w.unsqueeze(0)).sum(dim=1) / w_sum
                x_j_mean = (x_j * w.unsqueeze(0)).sum(dim=1) / w_sum
                x_i_c = x_i - x_i_mean.unsqueeze(1)
                x_j_c = x_j - x_j_mean.unsqueeze(1)
                cov = (w.unsqueeze(0) * x_i_c * x_j_c).sum(dim=1) / w_sum
                x_i_var = (w.unsqueeze(0) * x_i_c.pow(2)).sum(dim=1) / w_sum
                x_j_var = (w.unsqueeze(0) * x_j_c.pow(2)).sum(dim=1) / w_sum
                corr_dim = cov / (torch.sqrt(x_i_var + eps) * torch.sqrt(x_j_var + eps) + eps)
                corr_dim = torch.clamp(corr_dim, -1.0, 1.0)
                corr_k += corr_dim / 3.0
            state_corr[:, k] = corr_k

        state_corr_mask = (~torch.isnan(state_corr)) & (torch.abs(state_corr) >= CORR_THRESHOLD)
        return state_corr, state_corr_mask, alpha, alpha_entropy

    def _compute_global_corr(self, ca_traj, dist_edges):
        """Global Pearson correlation across ALL frames (paper E2, no state discovery)."""
        T, N, _ = ca_traj.shape
        device = ca_traj.device
        n_pairs = dist_edges.shape[1]
        residue_pairs = dist_edges.T
        xi = ca_traj[:, residue_pairs[:, 0], :].transpose(0, 1)  # (E, T, 3)
        xj = ca_traj[:, residue_pairs[:, 1], :].transpose(0, 1)  # (E, T, 3)
        eps = 1e-8

        with torch.no_grad():
            global_corr = torch.zeros(n_pairs, device=device)
            for dim in range(3):
                x_i = xi[:, :, dim]  # (E, T)
                x_j = xj[:, :, dim]  # (E, T)
                xi_m = x_i.mean(dim=1, keepdim=True)
                xj_m = x_j.mean(dim=1, keepdim=True)
                x_ic = x_i - xi_m
                x_jc = x_j - xj_m
                cov = (x_ic * x_jc).sum(dim=1) / (T - 1)
                xi_var = (x_ic.pow(2)).sum(dim=1) / (T - 1)
                xj_var = (x_jc.pow(2)).sum(dim=1) / (T - 1)
                corr_dim = cov / (torch.sqrt(xi_var + eps) * torch.sqrt(xj_var + eps) + eps)
                corr_dim = torch.clamp(corr_dim, -1.0, 1.0)
                global_corr += corr_dim / 3.0

        # Threshold at 0.6 (paper Task 1 atom-level corr threshold)
        GLOBAL_CORR_THRESHOLD = 0.6
        corr_mask = (~torch.isnan(global_corr)) & (torch.abs(global_corr) >= GLOBAL_CORR_THRESHOLD)
        corr_attr = global_corr * corr_mask.float()  # (E,)
        dyn_coverage = corr_mask.float().mean()

        return corr_attr, corr_mask, dyn_coverage

    def _build_dual_graphs(self, data):
        """Build static + dynamic graphs from data + MD cache or offline state_corr."""
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

        # ---- OFFLINE mode: read precomputed state_corr from H5 ----
        if self.dspl_state_file:
            f = _get_worker_h5(self.dspl_state_file)
            if pdb_id not in f:
                return None, None, None
            grp = f[pdb_id]
            state_corr_raw = grp['state_corr'][:]
            CORR_THRESHOLD = 0.1  # same as precompute

            # Derive state_corr_mask from state_corr if not in H5
            if 'state_corr_mask' in grp:
                state_corr_mask_raw = grp['state_corr_mask'][:]
            else:
                state_corr_mask_raw = (np.abs(state_corr_raw) >= CORR_THRESHOLD)

            # Derive alpha from frame_labels if not in H5
            if 'alpha' in grp:
                alpha_raw = grp['alpha'][:]
            elif 'frame_labels' in grp:
                frame_labels = grp['frame_labels'][:]
                n_frames = len(frame_labels)
                n_states = state_corr_raw.shape[1]
                alpha_raw = np.zeros((n_frames, n_states), dtype=np.float32)
                alpha_raw[np.arange(n_frames), frame_labels] = 1.0
            else:
                # Fallback: uniform alpha
                n_states = state_corr_raw.shape[1]
                # Assume 100 frames (standard MD)
                alpha_raw = np.ones((100, n_states), dtype=np.float32) / n_states

            # Safety: verify edge count matches
            E_expected = dist_edges.shape[1]
            if state_corr_raw.shape[0] != E_expected:
                print(f"[WARN] {pdb_id}: state_corr has {state_corr_raw.shape[0]} edges "
                      f"but graph has {E_expected} distance edges. Truncating/padding.")
                if state_corr_raw.shape[0] > E_expected:
                    state_corr_raw = state_corr_raw[:E_expected]
                    state_corr_mask_raw = state_corr_mask_raw[:E_expected]
                else:
                    # Pad with zeros
                    pad = E_expected - state_corr_raw.shape[0]
                    state_corr_raw = np.pad(state_corr_raw, ((0, pad), (0, 0)), mode='constant')
                    state_corr_mask_raw = np.pad(state_corr_mask_raw, ((0, pad), (0, 0)), mode='constant')

            state_corr = torch.from_numpy(state_corr_raw).to(self.device)
            state_corr_mask = torch.from_numpy(state_corr_mask_raw).to(self.device)
            alpha = torch.from_numpy(alpha_raw).to(self.device)
            alpha_entropy = -(alpha * torch.log(alpha + 1e-8)).sum(dim=-1).mean()
            dyn_edge_attr = state_corr * state_corr_mask.float()
            dyn_coverage = state_corr_mask.any(dim=-1).float().mean()

            dynamic_data = {
                'x': data.x,
                'edge_index': dist_edges.clone(),
                'edge_attr': dyn_edge_attr,
                'pos': data.pos if hasattr(data, 'pos') else None,
            }
            return static_data, dynamic_data, {
                'alpha': alpha,
                'alpha_entropy': alpha_entropy,
                'dyn_coverage': dyn_coverage,
                'state_corr': state_corr,
            }

        # ---- ONLINE mode: PCA + k-means per forward (legacy) ----
        if pdb_id not in self.md_cache:
            return None, None, None

        ca_traj = self.md_cache[pdb_id].to(self.device)

        state_corr, state_corr_mask, alpha, alpha_entropy = self._compute_state_corr(ca_traj, dist_edges)
        dyn_edge_attr = state_corr * state_corr_mask.float()
        dyn_coverage = state_corr_mask.any(dim=-1).float().mean()

        dynamic_data = {
            'x': data.x,
            'edge_index': dist_edges.clone(),
            'edge_attr': dyn_edge_attr,
        }

        return static_data, dynamic_data, {
            'alpha': alpha,
            'alpha_entropy': alpha_entropy,
            'dyn_coverage': dyn_coverage,
            'state_corr': state_corr,
        }

    def _lazy_load_md(self, pdb_id, md_path='data/downloaded/MD.hdf5',
                      atom_h5='data/data_files/atom_graph_OnlyProtein_distance_4.5_planA.h5'):
        """Lazy-load MD trajectory for a single protein (used by E2 to avoid OOM)."""
        from vendor.kabsch import kabsch_align
        import h5py
        with h5py.File(md_path, 'r') as md_f, h5py.File(atom_h5, 'r') as atom_f:
            if pdb_id not in md_f or pdb_id not in atom_f:
                return None
            grp = md_f[pdb_id]
            mbi = grp['molecules_begin_atom_index'][:]
            p_start, p_end = mbi[0], mbi[1]
            full_traj = grp['trajectory_coordinates'][:]
            f0 = full_traj[0, p_start:p_end]
            graph_coords = atom_f[pdb_id]['atoms_coordinates'][:]
            from scipy.spatial import cKDTree
            tree = cKDTree(f0)
            _, nn_idx = tree.query(graph_coords, k=1)
            atom_idx = nn_idx + p_start
            atom_traj = full_traj[:, atom_idx]
            ref = atom_traj[0].copy()
            for f in range(1, atom_traj.shape[0]):
                atom_traj[f] = kabsch_align(atom_traj[f], ref)
            return torch.tensor(atom_traj, dtype=torch.float32)

    def _build_e2_graphs(self, data):
        """E2: distance + global Pearson correlation edges (paper baseline)."""
        pdb_id = data.pdb_id
        if isinstance(pdb_id, list):
            pdb_id = pdb_id[0]

        # Lazy load MD data per-protein to avoid OOM on full data
        ca_traj = self.md_cache.get(pdb_id)
        if ca_traj is None:
            ca_traj = self._lazy_load_md(pdb_id)
            if ca_traj is None:
                return None, None, None
            # Cache for potential reuse across epochs
            self.md_cache[pdb_id] = ca_traj
        ca_traj = ca_traj.to(self.device)

        ca_traj = self.md_cache[pdb_id].to(self.device)
        dist_edges = data.edge_index
        dist_weights = data.edge_attr

        static_data = {
            'x': data.x,
            'edge_index': dist_edges,
            'edge_weight': dist_weights.reshape(-1),
            'pos': data.pos if hasattr(data, 'pos') else None,
        }

        global_corr, corr_mask, dyn_coverage = self._compute_global_corr(ca_traj, dist_edges)
        # global_corr: (E,) scalar per edge, use as edge_weight for corr tower
        corr_data = {
            'x': data.x,
            'edge_index': dist_edges.clone(),
            'edge_weight': global_corr.abs().clamp(min=0.0),  # GCNConv needs non-negative
            'pos': data.pos if hasattr(data, 'pos') else None,
        }

        return static_data, corr_data, {
            'dyn_coverage': dyn_coverage,
            'global_corr': global_corr,
        }

    def _forward_static_tower(self, x, static_data):
        """Static tower: 5-layer on distance edges.
        GCN: edge_weight=scalar. GAT: edge_attr=scalar.unsqueeze(-1). EGNN: pos + edge_attr."""
        if self.architecture == 'gcn':
            h_s = self.static_conv_in(x, static_data['edge_index'],
                                      edge_weight=static_data['edge_weight'])
        elif self.architecture == 'gat':
            # GATConv uses edge_attr (not edge_weight)
            edge_attr_s = static_data['edge_weight'].unsqueeze(-1)  # (E, 1)
            h_s = self.static_conv_in(x, static_data['edge_index'], edge_attr=edge_attr_s)
        elif self.architecture == 'egnn':
            # E_GCL returns (h, agg) with update_coords=False
            pos = static_data.get('pos', torch.zeros(x.shape[0], 3, device=x.device))
            edge_attr_s = static_data['edge_weight'].unsqueeze(-1)
            h_s = self.static_conv_in(x, static_data['edge_index'], pos, edge_attr=edge_attr_s)[0]
        elif self.architecture == 'gps':
            h_s = self.static_proj_in(x)
            E = static_data['edge_index'].shape[1]
            edge_type_s = torch.zeros(E, dtype=torch.long, device=x.device)
            batch_vec = torch.zeros(h_s.shape[0], dtype=torch.long, device=x.device)
            h_s = self.static_conv_in(h_s, static_data['edge_index'],
                                       batch=batch_vec, edge_type=edge_type_s)
        h_s = self.static_norm_in(h_s)
        h_s = F.relu(h_s)
        h_s = self.static_dropout_in(h_s)
        for layer_idx in range(self.num_gnn_layers - 1):
            if self.architecture == 'gcn':
                h_s = self.static_convs[layer_idx](h_s, static_data['edge_index'],
                                                   edge_weight=static_data['edge_weight'])
            elif self.architecture == 'gat':
                h_s = self.static_convs[layer_idx](h_s, static_data['edge_index'], edge_attr=edge_attr_s)
            elif self.architecture == 'egnn':
                h_s = self.static_convs[layer_idx](h_s, static_data['edge_index'], pos, edge_attr=edge_attr_s)[0]
            elif self.architecture == 'gps':
                h_s = self.static_convs[layer_idx](h_s, static_data['edge_index'],
                                                    batch=batch_vec, edge_type=edge_type_s)
            h_s = self.static_norms[layer_idx](h_s)
            h_s = F.relu(h_s)
            h_s = self.static_dropouts[layer_idx](h_s)
        return h_s

    def _forward_dynamic_tower(self, x, dynamic_data):
        """Dynamic tower: 5-layer with K-dim state-corr edge features.
        GCN: encode K-dim→scalar edge_weight. GAT: use K-dim to edge_attr directly.
        EGNN: pos + edges_in_d=K."""
        if self.architecture == 'gcn':
            dyn_edge_encoded = self.dynamic_edge_encoder(dynamic_data['edge_attr'])
            dyn_edge_weight0 = dyn_edge_encoded.abs().max(dim=-1).values.clamp(min=0.0)
            h_d = self.dynamic_conv_in(x, dynamic_data['edge_index'],
                                       edge_weight=dyn_edge_weight0)
        elif self.architecture == 'gat':
            # GATConv: pass K-dim state_corr directly (no encoder, no scalar reduction)
            dyn_edge_attr = dynamic_data['edge_attr']  # (E, K)
            h_d = self.dynamic_conv_in(x, dynamic_data['edge_index'], edge_attr=dyn_edge_attr)
        elif self.architecture == 'egnn':
            pos = dynamic_data.get('pos', torch.zeros(x.shape[0], 3, device=x.device))
            dyn_edge_attr = dynamic_data['edge_attr']  # (E, K)
            h_d = self.dynamic_conv_in(x, dynamic_data['edge_index'], pos, edge_attr=dyn_edge_attr)[0]
        elif self.architecture == 'gps':
            # Construct edge_type from state_corr (argmax of K-dim correlation)
            edge_attr_raw = dynamic_data['edge_attr']  # (E, K)
            E = dynamic_data['edge_index'].shape[1]
            edge_type_d = torch.zeros(E, dtype=torch.long, device=x.device)
            # Task 1 dynamic_data may have state_corr_mask
            state_corr_mask = dynamic_data.get('state_corr_mask', None)
            if state_corr_mask is not None:
                has_active = state_corr_mask.any(dim=-1)
                if has_active.any():
                    active_corr = edge_attr_raw * state_corr_mask.float()
                    dominant = active_corr.abs().argmax(dim=-1) + 1
                    edge_type_d[has_active] = dominant[has_active]
            else:
                threshold = 0.1
                has_active = (edge_attr_raw.abs() >= threshold).any(dim=-1)
                if has_active.any():
                    dominant = edge_attr_raw[has_active].abs().argmax(dim=-1) + 1
                    edge_type_d[has_active] = dominant
            edge_type_d = edge_type_d.clamp(0, self.gps_num_relations - 1)
            h_d_proj = self.dynamic_proj_in(x)
            batch_vec_d = torch.zeros(h_d_proj.shape[0], dtype=torch.long, device=x.device)
            h_d = self.dynamic_conv_in(h_d_proj, dynamic_data['edge_index'],
                                        batch=batch_vec_d, edge_type=edge_type_d)
        h_d = self.dynamic_norm_in(h_d)
        h_d = F.relu(h_d)
        h_d = self.dynamic_dropout_in(h_d)
        for layer_idx in range(self.num_gnn_layers - 1):
            if self.architecture == 'gcn':
                h_d = self.dynamic_convs[layer_idx](h_d, dynamic_data['edge_index'],
                                                    edge_weight=dyn_edge_encoded.abs().mean(dim=-1).clamp(min=0.0))
            elif self.architecture == 'gat':
                h_d = self.dynamic_convs[layer_idx](h_d, dynamic_data['edge_index'], edge_attr=dyn_edge_attr)
            elif self.architecture == 'egnn':
                h_d = self.dynamic_convs[layer_idx](h_d, dynamic_data['edge_index'], pos, edge_attr=dyn_edge_attr)[0]
            elif self.architecture == 'gps':
                h_d = self.dynamic_convs[layer_idx](h_d, dynamic_data['edge_index'],
                                                     batch=batch_vec_d, edge_type=edge_type_d)
            h_d = self.dynamic_norms[layer_idx](h_d)
            h_d = F.relu(h_d)
            h_d = self.dynamic_dropouts[layer_idx](h_d)
        return h_d

    def _forward_e2_corr_tower(self, x, corr_data):
        """E2 correlation tower: 5-layer with scalar global Pearson on edges."""
        if self.architecture == 'gcn':
            h_c = self.e2_corr_conv_in(x, corr_data['edge_index'],
                                        edge_weight=corr_data['edge_weight'])
        elif self.architecture == 'gat':
            edge_attr_c = corr_data['edge_weight'].unsqueeze(-1)  # (E, 1)
            h_c = self.e2_corr_conv_in(x, corr_data['edge_index'], edge_attr=edge_attr_c)
        elif self.architecture == 'egnn':
            pos = corr_data.get('pos', torch.zeros(x.shape[0], 3, device=x.device))
            edge_attr_c = corr_data['edge_weight'].unsqueeze(-1)
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
        h_c = self.e2_corr_dropout_in(h_c)
        for layer_idx in range(self.num_gnn_layers - 1):
            if self.architecture == 'gcn':
                h_c = self.e2_corr_convs[layer_idx](h_c, corr_data['edge_index'],
                                                      edge_weight=corr_data['edge_weight'])
            elif self.architecture == 'gat':
                h_c = self.e2_corr_convs[layer_idx](h_c, corr_data['edge_index'], edge_attr=edge_attr_c)
            elif self.architecture == 'egnn':
                h_c = self.e2_corr_convs[layer_idx](h_c, corr_data['edge_index'], pos, edge_attr=edge_attr_c)[0]
            elif self.architecture == 'gps':
                h_c = self.e2_corr_convs[layer_idx](h_c, corr_data['edge_index'],
                                                     batch=batch_vec_c, edge_type=edge_type_c)
            h_c = self.e2_corr_norms[layer_idx](h_c)
            h_c = F.relu(h_c)
            h_c = self.e2_corr_dropouts[layer_idx](h_c)
        return h_c

    def _build_static_data(self, data):
        """Build static-only data from distance edges (no MD needed)."""
        return {
            'x': data.x,
            'edge_index': data.edge_index,
            'edge_weight': data.edge_attr.reshape(-1),
            'pos': data.pos if hasattr(data, 'pos') else None,
        }

    def forward(self, data, return_intermediate=False):
        # ---- DualSpace: delegate to standalone dual-space encoder ----
        if self.architecture == 'dualspace':
            pdb_id = data.pdb_id
            if isinstance(pdb_id, list):
                pdb_id = pdb_id[0]

            # Load state_corr from H5 (same logic as _build_dual_graphs offline mode)
            if self.dspl_state_file:
                f_sc = _get_worker_h5(self.dspl_state_file)
                if pdb_id not in f_sc:
                    return None, None, None
                grp_sc = f_sc[pdb_id]
                state_corr_raw = grp_sc['state_corr'][:]
                E_exp = data.edge_index.shape[1]
                if state_corr_raw.shape[0] != E_exp:
                    if state_corr_raw.shape[0] > E_exp:
                        state_corr_raw = state_corr_raw[:E_exp]
                    else:
                        state_corr_raw = np.pad(state_corr_raw, ((0, E_exp - state_corr_raw.shape[0]), (0, 0)), mode='constant')
                state_corr = torch.from_numpy(state_corr_raw).to(self.device)
            else:
                state_corr = None

            # Lazy-load atom_to_residue from MD HDF5 (cached per protein)
            if pdb_id not in self._residue_cache:
                atom_to_residue = self._load_atom_to_residue(pdb_id, data)
                self._residue_cache[pdb_id] = atom_to_residue
            else:
                atom_to_residue = self._residue_cache[pdb_id]

            return self.dualspace_model(data, state_corr=state_corr,
                                        atom_to_residue=atom_to_residue)

        if self.ablation == 'static_only':
            # Static-only: no MD needed, build data directly from PyG Data
            static_data = self._build_static_data(data)
            x = static_data['x']
            h_s = self._forward_static_tower(x, static_data)
            out = self.predictor(h_s)
            out = torch.nan_to_num(out, nan=0.0, posinf=2.0, neginf=0.0)
            return out, {}, (h_s, None, None)

        # E2 Combined: need MD but use global Pearson (not state-conditioned)
        if self.ablation == 'e2_combined':
            static_data, corr_data, meta = self._build_e2_graphs(data)
            if static_data is None:
                return None, None, None
            x = static_data['x']
            h_s = self._forward_static_tower(x, static_data)
            h_c = self._forward_e2_corr_tower(x, corr_data)
            h_cat = torch.cat([h_s, h_c], dim=-1)
            gate = self.final_fusion(h_cat)
            h_fused = gate * h_s + (1 - gate) * h_c
            out = self.predictor(h_fused)
            out = torch.nan_to_num(out, nan=0.0, posinf=2.0, neginf=0.0)
            return out, meta, (h_s, h_c, gate)

        # For dynamic/dual/full: need MD for dynamic graph
        static_data, dynamic_data, meta = self._build_dual_graphs(data)
        if static_data is None:
            return None, None, None

        x = static_data['x']

        if self.ablation == 'dynamic_only':
            # ---- Single Dynamic Tower ----
            h_d = self._forward_dynamic_tower(x, dynamic_data)
            out = self.predictor(h_d)
            out = torch.nan_to_num(out, nan=0.0, posinf=2.0, neginf=0.0)
            return out, meta, (None, h_d, None)

        elif self.ablation == 'dual_tower':
            # ---- Dual Tower, NO cross-attention ----
            h_s = self._forward_static_tower(x, static_data)
            h_d = self._forward_dynamic_tower(x, dynamic_data)
            # Gate fusion
            h_cat = torch.cat([h_s, h_d], dim=-1)
            gate = self.final_fusion(h_cat)
            h_fused = gate * h_s + (1 - gate) * h_d
            out = self.predictor(h_fused)
            out = torch.nan_to_num(out, nan=0.0, posinf=2.0, neginf=0.0)
            return out, meta, (h_s, h_d, gate)

        elif self.ablation == 'full':
            # ---- Full: Dual Tower + Cross-Attention (Phase 2 original) ----
            cross_start = self.num_gnn_layers - self.num_cross_attn_layers

            # Pre-compute edge tensors for efficiency
            if self.architecture == 'gcn':
                dyn_edge_encoded = self.dynamic_edge_encoder(dynamic_data['edge_attr'])
                dyn_edge_weight = dyn_edge_encoded.abs().mean(dim=-1).clamp(min=0.0)
                dyn_edge_weight0 = dyn_edge_encoded.abs().max(dim=-1).values.clamp(min=0.0)
            elif self.architecture == 'gat':
                edge_attr_s = static_data['edge_weight'].unsqueeze(-1)  # (E, 1)
                dyn_edge_attr = dynamic_data['edge_attr']  # (E, K)
            elif self.architecture == 'egnn':
                pos = static_data.get('pos', torch.zeros(x.shape[0], 3, device=x.device))
                edge_attr_s = static_data['edge_weight'].unsqueeze(-1)  # (E, 1)
                dyn_edge_attr = dynamic_data['edge_attr']  # (E, K)
            elif self.architecture == 'gps':
                # Pre-compute edge_type tensors
                E = static_data['edge_index'].shape[1]
                edge_type_s = torch.zeros(E, dtype=torch.long, device=x.device)
                batch_vec = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                # Dynamic edge_type from state_corr
                edge_attr_raw = dynamic_data['edge_attr']
                edge_type_d = torch.zeros(E, dtype=torch.long, device=x.device)
                state_corr_mask = dynamic_data.get('state_corr_mask', None)
                if state_corr_mask is not None:
                    has_active = state_corr_mask.any(dim=-1)
                    if has_active.any():
                        active_corr = edge_attr_raw * state_corr_mask.float()
                        dominant = active_corr.abs().argmax(dim=-1) + 1
                        edge_type_d[has_active] = dominant[has_active]
                else:
                    threshold = 0.1
                    has_active = (edge_attr_raw.abs() >= threshold).any(dim=-1)
                    if has_active.any():
                        dominant = edge_attr_raw[has_active].abs().argmax(dim=-1) + 1
                        edge_type_d[has_active] = dominant
                edge_type_d = edge_type_d.clamp(0, self.gps_num_relations - 1)
                # Input projections for GPS
                x_proj_s = self.static_proj_in(x)
                x_proj_d = self.dynamic_proj_in(x)

            # Layer 0 — Static
            if self.architecture == 'gcn':
                h_s = self.static_conv_in(x, static_data['edge_index'],
                                          edge_weight=static_data['edge_weight'])
            elif self.architecture == 'gat':
                h_s = self.static_conv_in(x, static_data['edge_index'], edge_attr=edge_attr_s)
            elif self.architecture == 'egnn':
                h_s = self.static_conv_in(x, static_data['edge_index'], pos, edge_attr=edge_attr_s)[0]
            elif self.architecture == 'gps':
                h_s = self.static_conv_in(x_proj_s, static_data['edge_index'],
                                           batch=batch_vec, edge_type=edge_type_s)
            h_s = self.static_norm_in(h_s)
            h_s = F.relu(h_s)
            h_s = self.static_dropout_in(h_s)

            # Layer 0 — Dynamic
            if self.architecture == 'gcn':
                h_d = self.dynamic_conv_in(x, dynamic_data['edge_index'],
                                           edge_weight=dyn_edge_weight0)
            elif self.architecture == 'gat':
                h_d = self.dynamic_conv_in(x, dynamic_data['edge_index'], edge_attr=dyn_edge_attr)
            elif self.architecture == 'egnn':
                h_d = self.dynamic_conv_in(x, dynamic_data['edge_index'], pos, edge_attr=dyn_edge_attr)[0]
            elif self.architecture == 'gps':
                h_d = self.dynamic_conv_in(x_proj_d, dynamic_data['edge_index'],
                                            batch=batch_vec, edge_type=edge_type_d)
            h_d = self.dynamic_norm_in(h_d)
            h_d = F.relu(h_d)
            h_d = self.dynamic_dropout_in(h_d)

            # Layers 1..L-1
            cross_idx = 0
            for layer_idx in range(self.num_gnn_layers - 1):
                if self.architecture == 'gcn':
                    h_s = self.static_convs[layer_idx](h_s, static_data['edge_index'],
                                                       edge_weight=static_data['edge_weight'])
                elif self.architecture == 'gat':
                    h_s = self.static_convs[layer_idx](h_s, static_data['edge_index'], edge_attr=edge_attr_s)
                elif self.architecture == 'egnn':
                    h_s = self.static_convs[layer_idx](h_s, static_data['edge_index'], pos, edge_attr=edge_attr_s)[0]
                elif self.architecture == 'gps':
                    h_s = self.static_convs[layer_idx](h_s, static_data['edge_index'],
                                                        batch=batch_vec, edge_type=edge_type_s)
                h_s = self.static_norms[layer_idx](h_s)
                h_s = F.relu(h_s)
                h_s = self.static_dropouts[layer_idx](h_s)

                if self.architecture == 'gcn':
                    h_d = self.dynamic_convs[layer_idx](h_d, dynamic_data['edge_index'],
                                                        edge_weight=dyn_edge_weight)
                elif self.architecture == 'gat':
                    h_d = self.dynamic_convs[layer_idx](h_d, dynamic_data['edge_index'], edge_attr=dyn_edge_attr)
                elif self.architecture == 'egnn':
                    h_d = self.dynamic_convs[layer_idx](h_d, dynamic_data['edge_index'], pos, edge_attr=dyn_edge_attr)[0]
                elif self.architecture == 'gps':
                    h_d = self.dynamic_convs[layer_idx](h_d, dynamic_data['edge_index'],
                                                         batch=batch_vec, edge_type=edge_type_d)
                h_d = self.dynamic_norms[layer_idx](h_d)
                h_d = F.relu(h_d)
                h_d = self.dynamic_dropouts[layer_idx](h_d)

                if (layer_idx + 1) >= cross_start:
                    h_s, h_d = self.cross_attn_layers[cross_idx](h_s, h_d)
                    cross_idx += 1

            # Final fusion
            h_cat = torch.cat([h_s, h_d], dim=-1)
            gate = self.final_fusion(h_cat)
            h_fused = gate * h_s + (1 - gate) * h_d
            out = self.predictor(h_fused)
            out = torch.nan_to_num(out, nan=0.0, posinf=2.0, neginf=0.0)
            # Return pre-fusion h_s/h_d when alignment loss is needed
            intermediates = (h_s, h_d, gate)
            return out, meta, intermediates

    def training_step(self, batch, batch_idx):
        data = batch if not isinstance(batch, list) else batch[0]
        out, meta, intermediates = self(data)
        if out is None:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        target = data.y.squeeze()
        loss_task = F.mse_loss(out.squeeze(), target)
        self.log('train/task_loss', loss_task, on_epoch=True, batch_size=1)

        # ---- Alignment Loss (State-Structure) ----
        if self.lambda_align > 0 and isinstance(intermediates, tuple) and len(intermediates) >= 2:
            h_s, h_d = intermediates[0], intermediates[1]
            if h_s is not None and h_d is not None and self.ablation == 'full':
                # Cosine embedding loss: encourage static & dynamic representations to align
                target_sim = torch.ones(h_s.size(0), device=h_s.device)
                loss_align = F.cosine_embedding_loss(h_s, h_d, target_sim)
                self.log('train/align_loss', loss_align, on_epoch=True, batch_size=1)
                loss = loss_task + self.lambda_align * loss_align
            else:
                loss = loss_task
        else:
            loss = loss_task

        self.log('train/loss', loss, on_epoch=True, batch_size=1)
        if meta and 'dyn_coverage' in meta:
            self.log('train/dyn_coverage', meta['dyn_coverage'], on_epoch=True, batch_size=1)
        return loss

    def _shared_eval_step(self, batch, batch_idx, prefix):
        data = batch if not isinstance(batch, list) else batch[0]
        out, meta, _ = self(data)
        if out is None:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        target = data.y.squeeze()
        loss = F.mse_loss(out.squeeze(), target)
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
        outs, targets = zip(*outputs)
        y_hat = torch.cat(outs).squeeze()
        y = torch.cat(targets).squeeze()
        from scipy.stats import pearsonr, spearmanr
        y_hat_np = y_hat.numpy()
        y_np = y.numpy()
        mask = np.isfinite(y_hat_np) & np.isfinite(y_np)
        if mask.sum() < 2:
            for m in ['pearson', 'spearman', 'rmse', 'mae']:
                self.log(f'{prefix}/{m}', 0.0, on_epoch=True, batch_size=1)
            setattr(self, outputs_attr, [])
            return
        y_hat_np = y_hat_np[mask]
        y_np = y_np[mask]
        pearson, _ = pearsonr(y_hat_np, y_np)
        spearman, _ = spearmanr(y_hat_np, y_np)
        rmse = np.sqrt(np.mean((y_hat_np - y_np) ** 2))
        mae = np.mean(np.abs(y_hat_np - y_np))
        self.log(f'{prefix}/pearson', float(pearson), on_epoch=True, batch_size=1)
        self.log(f'{prefix}/spearman', float(spearman), on_epoch=True, batch_size=1)
        self.log(f'{prefix}/rmse', float(rmse), on_epoch=True, batch_size=1)
        self.log(f'{prefix}/mae', float(mae), on_epoch=True, batch_size=1)
        if prefix == 'test':
            print(f"\nTest Results: Pearson={pearson:.4f}, Spearman={spearman:.4f}, "
                  f"RMSE={rmse:.4f}, MAE={mae:.4f}")
        setattr(self, outputs_attr, [])

    def on_validation_epoch_end(self):
        self._shared_epoch_end('val')
        # Manual best-model saving (avoid ModelCheckpoint race condition)
        if not hasattr(self, '_best_val_pearson'):
            self._best_val_pearson = -float('inf')
        current = self.trainer.callback_metrics.get('val/pearson', -float('inf'))
        if isinstance(current, torch.Tensor):
            current = current.item()
        if current > self._best_val_pearson:
            self._best_val_pearson = current
            ckpt_dir = None
            for cb in self.trainer.callbacks:
                if isinstance(cb, ModelCheckpoint):
                    ckpt_dir = cb.dirpath
                    break
            if ckpt_dir is None:
                ckpt_dir = f'dspl_checkpoints/ablation_{self.ablation}'
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.state_dict(),
                'val_pearson': current,
                'ablation': self.ablation,
            }, os.path.join(ckpt_dir, f'best_epoch_{self.current_epoch:03d}_pearson_{current:.4f}.pt'))
            # Remove old checkpoints, keep top 3
            ckpts = sorted(
                [f for f in os.listdir(ckpt_dir) if f.startswith('best_epoch_')],
                key=lambda x: float(x.split('_')[-1].replace('.pt','')),
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
# DataModule
# ==============================
class Task1DataModule(LightningDataModule):
    def __init__(self, atom_h5_path, train_ids, val_ids, test_ids, num_workers=0):
        super().__init__()
        self.atom_h5_path = atom_h5_path
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_set = LazyAtomDataset(self.atom_h5_path, self.train_ids)
        self.val_set = LazyAtomDataset(self.atom_h5_path, self.val_ids)
        self.test_set = LazyAtomDataset(self.atom_h5_path, self.test_ids)
        # Touch one sample to pre-warm the worker H5 cache
        _ = self.train_set[0]
        print(f"Lazy datasets ready: train={len(self.train_set)}, val={len(self.val_set)}, test={len(self.test_set)}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='gcn',
                       choices=['gcn', 'gat', 'egnn', 'gps', 'dualspace'],
                       help='Conv layer type: gcn (GCNConv), gat (GATConv heads=4), egnn (E_GCL), gps (GPSConv+RGCN), dualspace (DSPL-DualSpace)')
    parser.add_argument('--ablation', type=str, required=True,
                       choices=['full', 'static_only', 'dual_tower', 'dynamic_only', 'e2_combined'],
                       help='Ablation variant')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n-proteins', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fast-dev-run', action='store_true')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--cross-attn-layers', type=int, default=CROSS_ATTENTION_LAYERS)
    parser.add_argument('--dspl-state-file', type=str, default=None,
                       help='Path to offline precomputed state_corr H5 (skips online PCA+k-means)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='DataLoader workers (0=single-process, safest with h5py)')
    parser.add_argument('--n-prototypes', type=int, default=N_PROTOTYPES,
                       help='Number of state prototypes K (default 5, must match dspl-state-file)')
    parser.add_argument('--lambda-align', type=float, default=0.0,
                       help='Weight for cosine embedding alignment loss (default 0 = off)')
    parser.add_argument('--split', type=str, default='adaptability',
                       choices=['adaptability', 'adaptability_cath'],
                       help='Data split scheme: adaptability (random) or adaptability_cath (CATH homology-disjoint)')
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
        # Map YAML keys (kebab-case -> arg name) onto args, but only for keys the
        # CLI did NOT explicitly override (i.e. keys still at their default).
        defaults = {a.dest: getattr(args, a.dest, None) for a in parser._actions if a.dest}
        cli_defaults = {
            'architecture': 'gcn', 'ablation': None, 'epochs': 100, 'lr': 1e-4,
            'n_proteins': 1000, 'seed': 42, 'gpu_id': 0,
            'cross_attn_layers': CROSS_ATTENTION_LAYERS, 'dspl_state_file': None,
            'num_workers': 0, 'n_prototypes': N_PROTOTYPES, 'lambda_align': 0.0,
            'split': 'adaptability', 'fast_dev_run': False, 'config': None,
        }
        for key, value in cfg.items():
            arg_name = key.replace('-', '_')
            if arg_name in cli_defaults and getattr(args, arg_name, None) == cli_defaults[arg_name]:
                setattr(args, arg_name, value)
        print(f"Loaded config from {args.config}")

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    atom_h5 = 'data/data_files/atom_graph_OnlyProtein_distance_4.5_planA.h5'

    def load_split(p):
        with open(p) as f:
            return [l.strip() for l in f if l.strip()]

    split_dir = f'data/splits/{args.split}'
    train_ids = load_split(f'{split_dir}/train.txt')
    val_ids = load_split(f'{split_dir}/val.txt')
    test_ids = load_split(f'{split_dir}/test.txt')

    if args.n_proteins:
        train_ids = train_ids[:args.n_proteins]
        val_ids = val_ids[:max(1, args.n_proteins // 5)]
        test_ids = test_ids[:max(1, args.n_proteins // 5)]

    ablation = args.ablation
    if args.architecture == 'dualspace':
        ablation = 'dualspace'  # Override: DualSpace uses its own encoder, not ablation variants
    print("=" * 60)
    print(f"DSPL Phase 2 — Ablation Study: {ablation}")
    print(f"Task 1: B-factor Regression (Atom-level)")
    if args.lambda_align > 0:
        print(f"Alignment Loss: lambda={args.lambda_align} (cosine embedding)")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print("=" * 60)

    # Data
    dm = Task1DataModule(atom_h5, train_ids, val_ids, test_ids,
                         num_workers=args.num_workers)
    dm.setup()

    # Model
    dspl_state_file = args.dspl_state_file
    if dspl_state_file:
        print(f"Using offline state_corr from: {dspl_state_file}")
    model = DSPL_Ablation(
        in_dim=48, gnn_hidden_dim=GNN_HIDDEN_DIM,
        num_gnn_layers=GN_NUM_LAYERS,
        num_cross_attn_layers=args.cross_attn_layers,
        n_prototypes=args.n_prototypes,
        lr=args.lr,
        ablation=ablation,
        dspl_state_file=dspl_state_file,
        architecture=args.architecture,
        lambda_align=args.lambda_align,
    )

    # Preload MD
    # - e2_combined: always needs MD for global Pearson (no offline equivalent yet)
    # - full/dual_tower/dynamic_only: skip if using offline state_corr
    needs_md_online = ablation == 'e2_combined'
    if not dspl_state_file and ablation in ('full', 'dual_tower', 'dynamic_only'):
        needs_md_online = True
    if args.architecture == 'dualspace' and dspl_state_file:
        needs_md_online = False

    if needs_md_online:
        # Use lazy loading per-batch instead of preloading all 13k proteins
        # (preload_md causes OOM with full dataset)
        print("  [E2 lazy-load] MD trajectories loaded per-batch (no mass preload)")
        model.md_cache = {}  # empty cache, filled on demand by _build_e2_graphs
    elif ablation in ('full', 'dual_tower', 'dynamic_only'):
        print("  [Skip preload_md] Using offline state_corr, no MD preload needed")

    # Checkpoint & Logger
    exp_name = f'ablation_{ablation}_{args.architecture}'
    suffix_parts = []
    if args.lambda_align > 0:
        suffix_parts.append(f'align{args.lambda_align}')
    if dspl_state_file:
        suffix_parts.append('offline')
    if args.n_prototypes != N_PROTOTYPES:
        suffix_parts.append(f'k{args.n_prototypes}')
    if not args.n_proteins:
        suffix_parts.append('fulldata')
    elif args.n_proteins != 1000:
        suffix_parts.append(f'n{args.n_proteins}')
    if suffix_parts:
        exp_name = exp_name + '_' + '_'.join(suffix_parts)
    ckpt_dir = f'dspl_checkpoints/{exp_name}'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir, filename='epoch_{epoch:03d}',
        save_top_k=-1, every_n_epochs=25,
        save_on_train_epoch_end=True,  # save at epoch end to ensure metrics available
    )

    tb_logger = TensorBoardLogger(
        save_dir='dspl/phase2_crossmodal/outputs',
        name=f'ablation_{ablation}_{args.architecture}'
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[ckpt_cb],
        logger=tb_logger,
        log_every_n_steps=5,
        val_check_interval=1.0,
        num_sanity_val_steps=0,  # skip sanity check to avoid ModelCheckpoint crash with save_on_train_epoch_end=False
        accumulate_grad_batches=4,
        accelerator="gpu", devices=[args.gpu_id],
        gradient_clip_val=1.0,
        fast_dev_run=args.fast_dev_run,
    )

    print(f"\nStarting ablation training: {ablation}")
    trainer.fit(model, dm)

    print(f"\nTesting {ablation}...")
    try:
        result = trainer.test(model, dataloaders=dm.test_dataloader())
        print(f"Test result: {result}")
    except Exception as e:
        print(f"Test evaluation failed: {e}")

    # Save final state
    torch.save({
        'model_state_dict': model.state_dict(),
        'ablation': ablation,
    }, os.path.join(ckpt_dir, 'final_model.pt'))
    print(f"Model saved to: {ckpt_dir}")


if __name__ == '__main__':
    main()
