"""Vendored Kabsch alignment utility.

Self-contained re-implementation of ``kabsch_align`` from the upstream
``scripts/build_residue_graph_v3.py`` so that the DSPL package does not depend
on the original repository's ``scripts`` package.

Kabsch alignment solves the orthogonal Procrustes problem: given two point sets
P and Q, find the rotation R that minimizes ||P @ R - Q||_F, then return the
rotated P. It is used by DSPL to remove rigid-body motion from each MD frame
before computing per-frame displacements for state-conditioned correlation.
"""

import numpy as np


def kabsch_align(P, Q):
    """Align P onto Q via Kabsch (optimal rotation + translation).

    Args:
        P: (N, 3) source point set (will be rotated to best match Q).
        Q: (N, 3) reference point set.

    Returns:
        (N, 3) aligned copy of P.
    """
    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    P_centered = P - p_mean
    Q_centered = Q - q_mean

    # Covariance matrix H = P^T Q
    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection (ensures det(R) = +1)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    aligned = P_centered @ R.T + q_mean
    return aligned
