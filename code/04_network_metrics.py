#!/usr/bin/env python3
"""
04_network_metrics.py - Cybernetic and topological signatures of the inferred
gene regulatory network (GRN).

Measures:
  Topology      : degree / betweenness / eigenvector centrality, clustering,
                  community structure (greedy modularity).
  Feedback      : cyclomatic number M = E - V + C (Mason-gain sense), which
                  quantifies the number of independent feedback loops.
  Stability     : spectral radius, leading eigenvalue (Perron root) of the
                  weighted adjacency matrix A (linear dynamics x' = Ax).
  Controllability (exact, undirected networks): for a symmetric system
                  matrix the minimum number of independent input channels
                  equals the largest geometric multiplicity over all
                  eigenvalues, which for symmetric A equals the nullity
                  N - rank(A). On the inferred |pcorr| matrix this is 24
                  (a 24-dimensional zero eigenspace); the 23 isolated nodes
                  of the weighted graph require direct inputs for full
                  controllability.
  Control kernel: top-ranked genes by combined centrality score (a
                  structure-based proxy for the control kernel; distinct
                  from the minimum driver set above).

Outputs:
  results/network_metrics.json   : network-level cybernetic metrics
  results/node_metrics.csv       : per-gene topology + controllability scores
  results/driver_genes.csv       : ranked control-kernel (driver) genes
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
N_DRIVER_REPORT = 60  # number of top driver genes reported


def load_network():
    genes = pd.read_csv(os.path.join(DATA := os.path.join(BASE, "data"),
                                     "gene_list.csv"), header=None)[0].tolist()
    adj = np.load(os.path.join(RES, "grn_adjacency.npy"))
    W = np.load(os.path.join(RES, "grn_weights.npy"))
    G = nx.Graph()
    G.add_nodes_from(genes)
    p = len(genes)
    # one network object: the full weighted matrix W = |pcorr| (edges where
    # W > 1e-12; 317 undirected edges). The thresholded adjacency adj is
    # retained for visualisation/sensitivity only and is not used for any
    # reported topology statistic.
    for i in range(p):
        for j in range(i + 1, p):
            if W[i, j] > 1e-12:
                G.add_edge(genes[i], genes[j], weight=float(W[i, j]))
    return genes, adj, W, G


def spectral_metrics(W: np.ndarray):
    """Return spectral radius, leading eigenvalue, eigenvector centrality vec."""
    # W symmetric; use eigvalsh (only values) and eigh for Perron vector
    vals, vecs = np.linalg.eigh(W)
    lmax = float(vals[-1])
    lmin = float(vals[0])
    rho = float(np.max(np.abs(vals)))
    # For the non-negative matrix W = |pcorr| the Perron-Frobenius theorem
    # guarantees that the eigenvector of the largest eigenvalue is non-negative;
    # eigh returns eigenvectors with arbitrary sign, so the absolute value only
    # removes this sign arbitrariness while leaving the component magnitudes
    # (and hence the normalised projection weights) unchanged.
    perron = np.abs(vecs[:, -1])
    perron /= perron.sum()
    mult_lmax = int(np.sum(np.isclose(vals, lmax, atol=1e-8)))
    return lmax, lmin, rho, perron, mult_lmax, vals


def minimal_driver_channels(vals: np.ndarray, N: int, tol: float = 1e-8):
    """Minimum number of independent input channels for a symmetric system
    matrix A: the largest geometric multiplicity over all eigenvalues
    (equal to the nullity N - rank(A) for symmetric A; Yuan et al. 2013,
    Corollary for undirected networks)."""
    _, counts = np.unique(np.round(vals, 8), return_counts=True)
    return int(counts.max())


def main():
    genes, adj, W, G = load_network()
    N = len(genes)
    E = int(G.number_of_edges())
    C = nx.number_connected_components(G)
    M = E - N + C                      # cyclomatic number (feedback loops)
    print(f"nodes={N}, edges={E}, components={C}, cyclomatic={M}")

    # topology metrics
    deg = dict(G.degree())
    between = nx.betweenness_centrality(G)
    clust = nx.clustering(G)
    try:
        communities = greedy_modularity_communities(G)
        com_map = {node: ci for ci, comm in enumerate(communities) for node in comm}
        n_com = len(communities)
    except Exception:  # pragma: no cover
        com_map = {g: 0 for g in genes}
        n_com = 1

    # spectral metrics
    lmax, lmin, rho, perron, mult_lmax, vals = spectral_metrics(W)
    rank_W = int(np.linalg.matrix_rank(W, tol=1e-10))
    n_driver = minimal_driver_channels(vals, N)
    nullity = N - rank_W
    # the minimum-input count equals the largest geometric multiplicity over
    # all eigenvalues; for the symmetric weighted matrix this is the nullity,
    # i.e. the multiplicity of the zero eigenvalue. Guard the identity
    # explicitly (Yuan et al. 2013, exact controllability).
    n_zero = int(np.sum(np.abs(vals) < 1e-8))
    assert n_zero == n_driver, (n_zero, n_driver)
    print(f"lambda_max={lmax:.4f}, zero-eig multiplicity={n_driver}, "
          f"rank={rank_W}, nullity={nullity}")

    # node-level table
    rows = []
    for i, g in enumerate(genes):
        rows.append({
            "gene": g,
            "degree": int(deg.get(g, 0)),
            "betweenness": float(between.get(g, 0.0)),
            "clustering": float(clust.get(g, float("nan"))),
            "eigenvector_centrality": float(perron[i]),
            "community": int(com_map.get(g, 0)),
        })
    node_df = pd.DataFrame(rows)

    # control-kernel score: combine eigenvector centrality, degree and
    # betweenness (rank-normalized) - a structure-based proxy for the nodes
    # most relevant to steering the network state.
    for col in ["degree", "betweenness", "eigenvector_centrality"]:
        node_df[col + "_rank"] = node_df[col].rank(ascending=True, pct=True)
    node_df["control_score"] = (
        node_df["degree_rank"] + node_df["betweenness_rank"] +
        node_df["eigenvector_centrality_rank"]
    ) / 3.0
    node_df = node_df.sort_values("control_score", ascending=False)
    node_df.to_csv(os.path.join(RES, "node_metrics.csv"), index=False)
    node_df.head(N_DRIVER_REPORT)[["gene", "degree", "betweenness",
                                   "eigenvector_centrality", "control_score",
                                   "community"]].to_csv(
        os.path.join(RES, "driver_genes.csv"), index=False)
    print(f"saved node_metrics.csv ({len(node_df)} rows), driver_genes.csv "
          f"(top {N_DRIVER_REPORT})")

    # largest connected component (size in genes, edge count)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    gc_nodes = len(comps[0])
    gc_edges = sum(1 for u, v in G.edges() if u in comps[0] and v in comps[0])
    assert gc_edges == E, (gc_edges, E)   # largest component carries all edges
    # network-level JSON
    metrics = {
        "n_nodes": N, "n_edges": E, "n_components": C,
        "largest_component_size": gc_nodes,
        "largest_component_edges": gc_edges,
        "cyclomatic_number": M, "n_communities": n_com,
        "mean_degree": float(np.mean(list(deg.values()))),
        "mean_clustering": float(np.nanmean(list(clust.values()))),
        "lambda_max": lmax, "lambda_min": lmin, "spectral_radius": rho,
        "lambda_max_multiplicity": mult_lmax,
        # the largest algebraic multiplicity over all eigenvalues; on the
        # symmetric weighted matrix this is the zero-eigenvalue multiplicity,
        # which by exact controllability equals the minimum independent input
        # count (Yuan et al. 2013)
        "max_algebraic_multiplicity": n_driver,
        "zero_eigenvalue_multiplicity": n_zero,
        "rank_weighted_A": rank_W,
        "nullity_weighted_A": nullity,
        "n_min_driver_channels": n_driver,
        "control_kernel_size": N_DRIVER_REPORT,
    }
    with open(os.path.join(RES, "network_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    np.save(os.path.join(RES, "grn_spectrum.npy"), vals)
    np.save(os.path.join(RES, "grn_perron.npy"), perron)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
