#!/usr/bin/env python3
"""
05_control_energy.py - Control energy analysis of the inferred GRN.

Model: stabilised linear dynamics  x'(t) = (A - eta*I) x(t) + B u(t), eta = 1.2
  A : weighted adjacency matrix of the GRN (|partial correlation|)
  B : input matrix selecting the control-kernel (driver) genes
The controllability Gramian Wc solves the continuous Lyapunov equation
      A Wc + Wc A^T = - B B^T
and the minimum energy needed to steer the system from x0 to xf is
      E_min(xf) = xf^T Wc^{-1} xf   (for x0 = 0).

We report:
  - the eigenvalue spectrum of Wc (control energy spectrum; fat-tail
    behaviour in the sense of Yan et al., Nat. Phys. 2015),
  - per-driver energy contribution (each driver gene alone vs. full kernel),
  - the condition number / effective rank of Wc.

Outputs: results/control_energy.json, results/energy_spectrum.csv,
         results/driver_energy.csv
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.linalg import solve_continuous_lyapunov, eigh

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")

# eta > lambda_max(A) (~0.90) keeps A - eta*I stable, so Wc is positive definite.
ETA = 1.2


def main():
    genes = pd.read_csv(os.path.join(os.path.join(BASE, "data"),
                                     "gene_list.csv"), header=None)[0].tolist()
    W = np.load(os.path.join(RES, "grn_weights.npy"))
    driver = pd.read_csv(os.path.join(RES, "driver_genes.csv"))
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    driver_idx = [gene_to_idx[g] for g in driver["gene"] if g in gene_to_idx]
    N = len(genes)
    K = len(driver_idx)
    print(f"N={N}, control kernel size K={K}")

    # ---- full-kernel Gramian (stabilised dynamics) ----
    B = np.zeros((N, K))
    B[driver_idx, np.arange(K)] = 1.0
    Wd = W - ETA * np.eye(N)
    # Solve A Wc + Wc A^T = -B B^T (unique since A - eta*I is stable).
    Wc = solve_continuous_lyapunov(Wd, -B @ B.T)
    # Tikhonov regularisation (delta = 1e-6 * tr(Wc) / N), quoted in Section 2.4;
    # used by the steering experiments.
    reg = 1e-6 * np.trace(Wc) / N

    evals = eigh(Wc, eigvals_only=True)
    # Clip Lyapunov noise (< 1e-14) to zero. Null-subspace zeros are excluded:
    # the dynamic range uses largest / smallest NON-ZERO eigenvalue (> 1e-12);
    # the floored ratio is kept as an upper bound only.
    evals_raw = np.maximum(evals, 0.0)
    nonzero = evals_raw[evals_raw > 1e-12]
    evals_floored = np.maximum(evals_raw, 1e-12)
    cond = float(np.max(evals_floored) / np.min(evals_floored))
    dyn_range = (float(evals_raw[-1] / nonzero[0]) if nonzero.size else
                 float("nan"))
    # 1e-12 floor lies below the 1e-6*max threshold; truncation cannot alter the count.
    eff_rank = float(np.sum(evals_raw > 1e-6 * np.max(evals_raw)))

    # ---- per-driver marginal energy (single-driver Gramian trace) ----
    rows = []
    failed_drivers = []   # (gene, reason) for any singular Gramian solve
    for k, g in zip(driver_idx, driver["gene"]):
        Bk = np.zeros((N, 1))
        Bk[k, 0] = 1.0
        try:
            Wck = solve_continuous_lyapunov(Wd, -Bk @ Bk.T)
            # trace = total energy over unit impulses; larger = easier to control.
            trace = float(np.trace(Wck))
        except Exception as exc:  # pragma: no cover
            trace = float("nan")
            failed_drivers.append((g, str(exc)))
        rows.append({"driver_gene": g, "gramian_trace": trace})
    driver_energy = pd.DataFrame(rows).sort_values(
        "gramian_trace", ascending=False)
    driver_energy.to_csv(os.path.join(RES, "driver_energy.csv"), index=False)
    if failed_drivers:
        print("WARNING: per-driver Gramian solve failed for:",
              [g for g, _ in failed_drivers])

    # ---- energy spectrum ----
    pd.DataFrame({"eigenvalue": evals_raw,
                  "rank": np.arange(1, len(evals_raw) + 1)}).to_csv(
        os.path.join(RES, "energy_spectrum.csv"), index=False)

    # share of total energy carried by the top 10% of modes (fat-tail proxy;
    # Yan et al., Nat. Phys. 2015).
    k_top = max(1, int(np.ceil(0.1 * len(evals_raw))))
    tot_energy = float(evals_raw.sum())
    # guard against a vanishing denominator
    top10_share = float(evals_raw[-k_top:].sum() / tot_energy) if tot_energy > 0 else 0.0
    controllable_dim = int(np.sum(evals_raw > 1e-12))
    n_min_drivers = int(json.load(open(os.path.join(
        RES, "network_metrics.json")))["n_min_driver_channels"])
    summary = {
        "system_size_N": N, "kernel_size_K": K,
        "n_min_driver_channels": n_min_drivers,
        "controllable_subspace_dim": controllable_dim,
        "gramian_rank": controllable_dim,
        "gramian_regularisation": float(reg),
        "gramian_cond": cond, "gramian_effective_rank": eff_rank,
        "gramian_max_eig": float(evals_raw[-1]),
        # dynamic range = largest / smallest NON-ZERO eigenvalue
        # (null-subspace zeros excluded).
        "gramian_min_eig_reachable": float(nonzero[0]) if nonzero.size else float("nan"),
        "gramian_dynamic_range_reachable": dyn_range,
        # floored spectrum, used for the truncation convention in the figure only.
        "gramian_min_eig_floored": float(evals_floored[0]),
        "gramian_cond_floor": cond,
        "gramian_cond_floor_threshold": 1e-12,
        # share of the Gramian TRACE carried by the top 10% of modes;
        # control cost scales with the inverse eigenvalue (Yan et al. 2015).
        "energy_top10_trace_share": top10_share,
        # same as gramian_dynamic_range_reachable, kept for the spectrum JSON.
        "energy_spectrum_dynamic_range": dyn_range,
    }
    with open(os.path.join(RES, "control_energy.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
