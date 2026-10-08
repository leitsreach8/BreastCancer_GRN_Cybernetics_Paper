#!/usr/bin/env python3
"""
14_tcga_network_replication.py - Cross-cohort replication of the inferred
network in TCGA-BRCA (Section 3.8 and Fig. 11).

The METABRIC graphical-lasso network (results/grn_weights.npy, 99 genes) is
re-estimated on the TCGA-BRCA tumour expression matrix with the SAME
inference pipeline (sklearn GraphicalLassoCV, cv = 3, alphas = 6).  Because
CXCL8 is absent from the TCGA expression matrix, the comparison uses the
shared 98-gene panel: the METABRIC edge counts are recomputed on that panel
(removing CXCL8), and the TCGA network is estimated on the 98 genes.

Edge sets are compared at two thresholds, identical to the definitions used
in the main analysis:
  dense  : |partial correlation| > 1e-12 (full weighted network)
  sparse : |partial correlation| > 0.05   (thresholded network)
The Jaccard index is computed on the shared gene-pair universe (4,753 pairs
for the 98-gene panel).

Outputs: results/tcga_network_replication.json,
         results/tcga_grpcorr.npy
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV
from sklearn.exceptions import ConvergenceWarning

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "TCGA-BRCA")
RESULTS = os.path.join(ROOT, "results")

P_CORR_MIN = 0.05
FLOOR = 1e-12


def to_partial_corr(precision):
    d = np.sqrt(np.diag(precision))
    return -precision / np.outer(d, d)


def edge_stats(pcorr, names):
    """Edge counts and the pair set at two thresholds."""
    p = pcorr.shape[0]
    W = np.abs(pcorr)
    dense = W > FLOOR
    sparse = W > P_CORR_MIN
    pairs_d = [(names[i], names[j])
               for i in range(p) for j in range(i + 1, p) if dense[i, j]]
    pairs_s = [(names[i], names[j])
               for i in range(p) for j in range(i + 1, p) if sparse[i, j]]
    return (len(pairs_d), len(pairs_s),
            set(pairs_d), set(pairs_s))


def main():
    # METABRIC network (results/grn_weights.npy = |partial correlation|)
    genes_mb = pd.read_csv(os.path.join(ROOT, "data", "gene_list.csv"),
                           header=None)[0].tolist()
    W_mb = np.load(os.path.join(RESULTS, "grn_weights.npy"))
    assert W_mb.shape == (len(genes_mb), len(genes_mb))

    # TCGA expression (rows = genes, columns = samples), 98-gene panel
    genes = pd.read_csv(os.path.join(ROOT, "supplementary",
                                     "Table_S1_genes.csv"))
    genes_tcga = genes["gene_symbol"].tolist()
    wanted = set(genes_tcga)
    expr = {}
    header = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import gzip
        with gzip.open(os.path.join(DATA, "HiSeqV2.gz"), "rt") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if header is None:
                    header = parts
                    continue
                g = parts[0].upper()
                if g in wanted:
                    expr[g] = np.array(parts[1:], dtype=float)
    if not expr:
        raise RuntimeError("no TCGA genes found")
    df = pd.DataFrame(expr).T
    df.columns = header[1:]
    # tumour samples only: sample barcodes end in -01 (primary tumour);
    # normal and control samples (-11, -06, ...) are excluded so that the
    # network is estimated on the same tumour cohort as the clinical
    # validation (n = 1,097 tumour samples with expression)
    is_tumour = [c.endswith("-01") for c in df.columns]
    df = df.loc[:, is_tumour]
    E = df.T
    print(f"TCGA expression: {E.shape[1]} genes x {E.shape[0]} tumour samples")

    # shared 98-gene panel: genes present in BOTH cohorts
    shared = [g for g in genes_mb if g in E.columns]
    print(f"shared genes (CXCL8 absent from TCGA): {len(shared)}")
    missing = sorted(set(genes_mb) - set(shared))
    if missing:
        print("METABRIC genes absent in TCGA:", missing)

    # METABRIC edge counts on the shared 98-gene panel
    idx_mb = [genes_mb.index(g) for g in shared]
    W_mb98 = W_mb[np.ix_(idx_mb, idx_mb)]
    n_d_mb, n_s_mb, pairs_d_mb, pairs_s_mb = edge_stats(W_mb98, shared)

    # TCGA: z-score per gene across the tumour samples, then graphical lasso
    # with the same CV protocol as the METABRIC inference (cv = 3, alphas = 6)
    Ez = (E - E.mean(axis=0)) / E.std(axis=0)
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        glcv = GraphicalLassoCV(cv=3, alphas=6, n_jobs=2,
                                max_iter=500, tol=1e-6)
        glcv.fit(Ez.values)
    n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in wlist)
    alpha = float(glcv.alpha_)
    pcorr_tcga = to_partial_corr(glcv.precision_)
    n_d_tcga, n_s_tcga, pairs_d_tcga, pairs_s_tcga = edge_stats(
        pcorr_tcga, shared)
    print(f"TCGA alpha = {alpha:.6f}; dense edges = {n_d_tcga}; "
          f"sparse edges = {n_s_tcga}; convergence warnings = {n_warn}")

    # Jaccard on the shared gene-pair universe
    jac_d = (len(pairs_d_mb & pairs_d_tcga)
             / len(pairs_d_mb | pairs_d_tcga)) if pairs_d_mb or pairs_d_tcga \
        else 0.0
    jac_s = (len(pairs_s_mb & pairs_s_tcga)
             / len(pairs_s_mb | pairs_s_tcga)) if pairs_s_mb or pairs_s_tcga \
        else 0.0

    out = {
        "n_tcga_samples": int(E.shape[0]),
        "shared_genes": len(shared),
        "tcga_alpha": alpha,
        "dense": {
            "metabric_98gene_panel": n_d_mb,
            "tcga": n_d_tcga,
            "overlap": len(pairs_d_mb & pairs_d_tcga),
            "jaccard": jac_d},
        "sparse": {
            "metabric_98gene_panel": n_s_mb,
            "tcga": n_s_tcga,
            "overlap": len(pairs_s_mb & pairs_s_tcga),
            "jaccard": jac_s},
        "thresholds": {"dense_floor": FLOOR, "sparse_floor": P_CORR_MIN},
        "n_convergence_warnings": n_warn,
        "note": ("TCGA network estimated with the same GraphicalLassoCV "
                 "protocol (cv=3, alphas=6) on 98 shared genes; edge "
                 "thresholds identical to the METABRIC analysis")}
    with open(os.path.join(RESULTS, "tcga_network_replication.json"),
              "w") as fh:
        json.dump(out, fh, indent=2)
    np.save(os.path.join(RESULTS, "tcga_grpcorr.npy"), pcorr_tcga)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
