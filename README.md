# DSPL — Dynamic State Prototype Learning for Protein Graph Representations

## Overview

DSPL instead:

1. **Discovers states** from each MD trajectory via `PCA → k-means` on the per-frame displacement vectors (`K` state prototypes).
2. Computes **state-conditioned** Pearson correlation for every atom pair within each state, producing a `(E, K)` correlation tensor instead of a single scalar.
3. Builds a **dual-tower** architecture — a *static* tower on KD-tree distance edges and a *dynamic* tower on the `K`-dimensional state-conditioned edges — fused with cross-modal attention and a gate, optionally regularized by a cosine *state–structure alignment* loss.

The result is +0.018–0.078 Pearson over the paper's strongest baselines on atomic-adaptability (B-factor) regression, with the gain most pronounced on the cross-family (CATH homology-disjoint) split.

---

## Repository Layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── configs/                    # example YAML configs (task1_dspl_full / task1_demo / task2_dspl_full)
├── vendor/                     # Self-contained copies of external dependencies
│   ├── egnn_layer.py           #   E_GCL (E(n)-equivariant GCL), replaces src.models.regnn.regnn_ensemble
│   └── kabsch.py               #   Kabsch alignment, replaces scripts.build_residue_graph_v3.kabsch_align
└── dspl/
    ├── shared/
    │   ├── state_utils.py         # state-visualization / analysis helpers
    │   └── weighted_correlation.py# differentiable weighted-Pearson (for learnable-prototype variant)
    └── phase2_crossmodal/
        ├── train_ablation.py           # Task 1 (B-factor regression) DSPL trainer
        ├── train_ablation_task2.py     # Task 2 (binding-site classification) DSPL trainer
        ├── precompute_dspl_states.py          # Generate state_corr H5 for Task 1
        ├── precompute_dspl_states_task2.py    # Generate state_corr H5 for Task 2
        └── dspl_dualspace.py           # Standalone DSPL-DualSpace architecture (--architecture dualspace)
```

Every path below is expressed **relative to the repository root** (`DSPL_ROOT`). The scripts resolve the root automatically and also accept the `DSPL_ROOT` environment variable to point at a different data location.

---

## Dependencies

The code is built on PyTorch, PyTorch Geometric, and PyTorch Lightning. A minimal environment:

```bash
conda create -n dspl python=3.10
conda activate dspl

# PyTorch (choose the channel matching your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric (version must match torch; here torch 2.x)
pip install torch_geometric

# Training stack
pip install pytorch-lightning tensorboard

# Scientific / data
pip install numpy h5py scipy scikit-learn
```

---

## Data

DSPL consumes the **MISATO** dataset (Siebenmorgen et al., 2024), a large benchmark of protein MD trajectories and downstream tasks built from PDBbind. The raw data is **not** included in this repo — download it from the MISATO Zenodo page and place files under `data/` as shown below.

The MISATO h5 files can be downloaded like this:

```
wget -O data/MD.hdf5 https://zenodo.org/record/7711953/files/MD.hdf5
wget -O data/QM.hdf5 https://zenodo.org/record/7711953/files/QM.hdf5
```

 You can download a preprocessed h5 file containing the MD adaptability and reference coordinates from here: https://drive.google.com/drive/folders/1pMff605gAUVV174P-3btYaDlnujEFjvq

The preprocessed graphs for the dataloader can also be downloaded.

Invariant graph: https://drive.google.com/drive/folders/1pMff605gAUVV174P-3btYaDlnujEFjvq

Alternatively, generate a h5 file containing the  adaptability values from the MD.hdf5 file by running the preprocessing.  To this end follow the instructions from the MISATO repository https://github.com/t7morgen/misato-dataset .

The preprocessing scripts for the graphs can be found in src/data/processing/.

```bash
data/
├── downloaded/               # Direct downloads from MISATO
│   ├── adaptability_MD.hdf5
│   └── preprocessed_graph_invariant_combined.h5
├── processed/                # Processed HDF5 files
├── trajectories/             # MD trajectory files (.nc, .top)
├── correlations/
│   └── aligned_atomiccorr_no_hydrogen/
├── distances/
│   └── ca-ligand_distance/
├── maps/                     # Atom/residue type mappings
└── affinity_data.csv         # Binding affinity data
```



## Workflow

### Config files (optional)

Instead of long CLI invocations, pass a YAML config via `--config` (CLI values still win):

```bash
# Quick smoke-test on 1000 proteins
python dspl/phase2_crossmodal/train_ablation.py --config configs/task1_demo.yaml

# Task 2
python dspl/phase2_crossmodal/train_ablation_task2.py --config configs/task2_dspl_full.yaml
```

## Citation

If you use this code, please cite the original paper and the MISATO dataset:

```bibtex
@inproceedings{guo2025boosting,
  title     = {Boosting Protein Graph Representations through Static-Dynamic Fusion},
  author    = {Guo, Pengkai and others},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025}
}

@article{siebenmorgen2024misato,
  title   = {MISATO: Machine learning dataset of protein–ligand complexes for structure-based drug discovery},
  author  = {Siebenmorgen, Till and others},
  journal = {Nature Computational Science},
  year    = {2024}
}
```

