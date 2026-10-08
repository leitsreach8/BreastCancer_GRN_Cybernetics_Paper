#!/usr/bin/env python3
"""
03_network_inference.py - Infer a sparse gene co-regulation network from the
processed expression matrix using the graphical lasso (Friedman et al. 2008).

The inverse covariance (precision) matrix Theta is estimated with an l1
penalty; partial correlations are derived from Theta and define an undirected
conditional-dependence network among the most variable genes.

Outputs (results/):
  - grn_adjacency.npy      : adjacency matrix of the inferred GRN (binary)
  - grn_weights.npy        : |partial correlation| matrix (weighted network)
  - grn_precision.npy      : precision matrix
  - grn_edges.csv          : edge list with partial correlation weights
  - grn_bootstrap.csv      : edge stability across bootstrap resamples
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from sklearn.exceptions import ConvergenceWarning

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RES = os.path.join(BASE, "results")
SEED = 42
N_BOOT = 25          # bootstrap resamples for edge stability
BOOT_FRAC = 0.8      # fraction of samples per bootstrap
P_CORR_MIN = 0.05    # partial correlation threshold for "present" edge


def to_partial_corr(prec: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(prec))
    pcorr = -prec / np.outer(d, d)
    np.fill_diagonal(pcorr, 0.0)
    return pcorr


def main():
    rng = np.random.default_rng(SEED)
    expr = pd.read_csv(os.path.join(DATA, "metabric_processed.csv"), index_col=0)
    X = expr.values   # samples x genes (METABRIC matrix is already T x G)
    genes = list(expr.columns)
    n, p = X.shape
    print(f"X: {n} samples x {p} genes")

    # ---- graphical lasso with CV-selected penalty ----
    print("running GraphicalLassoCV ...", flush=True)
    # cv=3 uses sklearn's deterministic KFold (no shuffling), so the
    # selected penalty is reproducible for a fixed input matrix and a
    # fixed sklearn version. alphas=6 is sklearn's integer semantics:
    # the estimator builds a data-adaptive grid between
    # alpha_max = max|S_offdiag| and alpha_min = alpha_max*1e-2, and
    # re-samples 6 candidate values (not a fixed logspace grid). The
    # CV-selected alpha for this dataset is 0.34533760: it is printed
    # above and written to results/grn_meta.json, and the manuscript
    # reports it as alpha = 0.3453.
    glcv = GraphicalLassoCV(cv=3, alphas=6, n_jobs=2,
                            max_iter=500, tol=1e-6)
    glcv.fit(X)
    alpha = glcv.alpha_
    prec = glcv.precision_
    pcorr = to_partial_corr(prec)
    adj = (np.abs(pcorr) > P_CORR_MIN).astype(int)
    np.fill_diagonal(adj, 0)
    n_edges = int(adj.sum() // 2)
    print(f"alpha={alpha:.4f}, edges={n_edges}, density={2 * n_edges / (p * (p - 1)):.4f}",
          flush=True)

    np.save(os.path.join(RES, "grn_precision.npy"), prec)
    np.save(os.path.join(RES, "grn_weights.npy"), np.abs(pcorr))
    np.save(os.path.join(RES, "grn_adjacency.npy"), adj)

    # ---- edge list ----
    rows = []
    for i in range(p):
        for j in range(i + 1, p):
            if adj[i, j]:
                rows.append((genes[i], genes[j], float(pcorr[i, j])))
    edges = pd.DataFrame(rows, columns=["gene_i", "gene_j", "partial_corr"])
    edges.to_csv(os.path.join(RES, "grn_edges.csv"), index=False)
    print(f"edge list: {len(edges)} rows")

    # ---- bootstrap edge stability (two criteria) ----
    # Candidate edges are those present in the FULL weighted network
    # W = |pcorr| (> 1e-12; 317 undirected edges), consistent with the
    # network object used for all topology, control and dynamics analyses.
    # An edge's stability is the fraction of bootstrap resamples in which the
    # re-estimated |partial correlation| exceeds a threshold.  Two criteria
    # are reported side by side: (i) the numerical floor W_b > 1e-12 (edge
    # re-appears with any non-zero weight); (ii) the same threshold used to
    # define the sparse network in the manuscript, |pi| > P_CORR_MIN = 0.05.
    boot_floor = np.zeros((p, p))
    boot_thr = np.zeros((p, p))
    n_warn = 0
    for b in range(N_BOOT):
        idx = rng.choice(n, size=int(n * BOOT_FRAC), replace=False)
        gl = GraphicalLasso(alpha=alpha, max_iter=500, tol=1e-6)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            gl.fit(X[idx])
        n_warn += sum(issubclass(w.category, ConvergenceWarning)
                      for w in wlist)
        pcorr_b = to_partial_corr(gl.precision_)
        W_b = np.abs(pcorr_b)
        boot_floor += (W_b > 1e-12).astype(float)
        boot_thr += (W_b > P_CORR_MIN).astype(float)
    boot_floor /= N_BOOT
    boot_thr /= N_BOOT
    stability = pd.DataFrame(
        [(genes[i], genes[j],
          float(boot_floor[i, j]), float(boot_thr[i, j]))
         for i in range(p) for j in range(i + 1, p)
         if np.abs(pcorr[i, j]) > 1e-12],
        columns=["gene_i", "gene_j", "stability_floor",
                 "stability_threshold"])
    stability.to_csv(os.path.join(RES, "grn_bootstrap.csv"), index=False)
    sf, st = stability["stability_floor"], stability["stability_threshold"]
    print(f"bootstrap (floor criterion): mean={sf.mean():.3f}, "
          f"edges>0.8: {(sf > 0.8).sum()}")
    print(f"bootstrap (|pi|>0.05 criterion): mean={st.mean():.3f}, "
          f"edges>0.8: {(st > 0.8).sum()}, median={st.median():.3f}")
    print(f"GraphicalLasso convergence warnings across {N_BOOT} "
          f"bootstrap fits: {n_warn}")

    n_edges_w = int(np.sum(np.abs(pcorr) > 1e-12) // 2)  # full weighted network
    meta = {"n_samples": n, "p_genes": p, "alpha": float(alpha),
            # thresholded (|partial correlation| > 0.05) network: visualisation
            # and sensitivity only; no reported topology statistic uses it
            "n_edges_thresholded": n_edges,
            "density_thresholded": float(2 * n_edges / (p * (p - 1))),
            "pcorr_threshold": P_CORR_MIN,
            # full weighted network W = |pcorr| (> 1e-12): the unique object
            # used for all topology, control and dynamics analyses
            "n_edges_weighted": n_edges_w,
            "density_weighted": float(2 * n_edges_w / (p * (p - 1))),
            # stability under the numerical-floor criterion (weighted network)
            "mean_stability_floor": float(sf.mean()),
            "n_edges_stable_80_floor": int((sf > 0.8).sum()),
            "frac_edges_stable_80_floor": float((sf > 0.8).mean()),
            # stability under the manuscript's own edge threshold |pi| > 0.05
            "mean_stability_threshold": float(st.mean()),
            "median_stability_threshold": float(st.median()),
            "n_edges_stable_80_threshold": int((st > 0.8).sum()),
            "frac_edges_stable_80_threshold": float((st > 0.8).mean()),
            "n_boot": N_BOOT,
            "n_convergence_warnings_boot": n_warn}
    with open(os.path.join(RES, "grn_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
