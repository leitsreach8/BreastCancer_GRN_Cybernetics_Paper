#!/usr/bin/env python3
"""
06_dynamics_simulation.py - Dynamical simulation on the inferred GRN as a
complex adaptive system.

Two complementary views:
  (i)  Linear dissipative dynamics    x'(t) = (A - eta I) x(t) + B u(t)
       - transient projections from many random initial states; because the
         system is stable there is a single equilibrium, and the projections
         at the simulation horizon decay to it as t -> inf.
       - minimum-energy steering between the low- and high-projection poles
         (control kernel applied as input u).
  (ii) Nonlinear saturating dynamics  x'(t) = -x(t) + tanh(A x(t)) + B u(t)
       - transient spread at finite horizon, plus an explicit fixed-point
         analysis (multi-start root finding) that tests whether the network
         is genuinely multistable.

The two response "poles" are defined by the sample-averaged control-kernel
activity: patients with low/high kernel activity define the two target poles;
we emulate them as low/high projections of the state onto the Perron vector.

Outputs: results/dynamics_attractors.csv, results/dynamics_trajectory.csv,
         results/dynamics_steering.csv, results/dynamics_meta.json
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
SEED = 42
# eta > lambda_max(A) (~0.90) keeps the linear system stable, so the
# Gramian and control energies are well defined.
ETA = 1.2            # dissipation for stable linear system
T_SPAN = (0.0, 12.0)
N_IC = 40             # random initial conditions for trajectory scan


def linear_dynamics(t, x, A, B, u):
    return (A - ETA * np.eye(A.shape[0])) @ x + B @ u


def nonlinear_dynamics(t, x, A, B, u):
    return -x + np.tanh(A @ x) + B @ u


def main():
    rng = np.random.default_rng(SEED)
    genes = pd.read_csv(os.path.join(os.path.join(BASE, "data"),
                                     "gene_list.csv"), header=None)[0].tolist()
    W = np.load(os.path.join(RES, "grn_weights.npy"))
    perron = np.load(os.path.join(RES, "grn_perron.npy"))
    driver = pd.read_csv(os.path.join(RES, "driver_genes.csv"))
    driver_idx = [genes.index(g) for g in driver["gene"] if g in genes]
    N = len(genes)
    K = len(driver_idx)

    B = np.zeros((N, K))
    B[driver_idx, np.arange(K)] = 1.0

    # ---- (i) linear system: transient projection scan ----
    A = W
    ics = rng.normal(0, 0.5, size=(N_IC, N))
    trajs = []
    for ic in ics:
        sol = solve_ivp(linear_dynamics, T_SPAN, ic, args=(A, B, np.zeros(K)),
                        dense_output=False, rtol=1e-6, atol=1e-8)
        trajs.append(sol.y[:, -1])
    att = np.array(trajs)
    # project onto Perron direction -> scalar phenotype coordinate
    proj = att @ perron
    # median split into low/high response poles (not attractors)
    med = np.median(proj)
    basin = (proj > med).astype(int)
    attr_df = pd.DataFrame({
        "rep": np.arange(N_IC),
        "projection_perron": proj,
        "basin": basin,
    })
    attr_df.to_csv(os.path.join(RES, "dynamics_attractors.csv"), index=False)

    # ---- (ii) constant-input perturbation, linear system ----
    # Heuristic: from the lowest-projection state x_low, apply constant input
    # u_const = B^T Wc^{-1} delta and compare with the free run.
    # Not a minimum-energy or phenotype-transfer claim.
    x_low = att[np.argmin(proj)]
    x_high = att[np.argmax(proj)]
    delta = x_high - x_low
    try:
        Wc = solve_continuous_lyapunov(A - ETA * np.eye(N), -B @ B.T)
        # Tikhonov-regularised Gramian; reported energies are the regularised values.
        reg = 1e-6 * np.trace(Wc) / N
        Wc_r = Wc + reg * np.eye(N)
        Wc_inv = np.linalg.inv(Wc_r)
        E_unit = float(max(delta_u := delta / np.linalg.norm(delta)
                           @ Wc_inv @ (delta / np.linalg.norm(delta)), 0.0))
        E_total = float(max(delta @ Wc_inv @ delta, 0.0))
        u_norm = float(np.linalg.norm(B.T @ Wc_inv @ delta))
        u_const = B.T @ Wc_inv @ delta
    except Exception as exc:
        print("steering energy failed:", exc)
        sys.exit(1)

    sol_steer = solve_ivp(linear_dynamics, (0.0, 8.0), x_low,
                          args=(A, B, u_const), rtol=1e-6, atol=1e-8)
    sol_free = solve_ivp(linear_dynamics, (0.0, 8.0), x_low,
                         args=(A, B, np.zeros(K)), rtol=1e-6, atol=1e-8)
    proj_steer = sol_steer.y.T @ perron
    proj_free = sol_free.y.T @ perron
    # actual integrated input energy (constant input): ||u_const||^2 * T
    input_energy_T8 = float(np.dot(u_const, u_const) * (8.0 - 0.0))
    end_proj = float(proj_steer[-1])
    start_proj = float(x_low @ perron)
    target_proj = float(x_high @ perron)
    # achieved fraction of the target displacement on the Perron projection
    displacement_fraction = float(
        (end_proj - start_proj) / (target_proj - start_proj))

    full_state_error = float(np.linalg.norm(sol_steer.y[:, -1] - x_high))
    relative_error = float(full_state_error / np.linalg.norm(x_high))
    # free-run median-crossing time (informative control)
    free_cross = None
    free_above = proj_free > med
    if np.any(free_above):
        idx = int(np.argmax(free_above))
        if idx > 0:
            free_cross = float(sol_free.t[idx])
    rows = []
    for t_, p_ in zip(sol_steer.t, proj_steer):
        rows.append((t_, float(p_), 1))
    for t_, p_ in zip(sol_free.t, proj_free):
        rows.append((t_, float(p_), 0))
    steer_df = pd.DataFrame(rows, columns=["t", "projection_perron",
                                           "control_active"])
    steer_df.to_csv(os.path.join(RES, "dynamics_steering.csv"), index=False)

    # ---- (iii) nonlinear dynamics: transient spread and convergence ----
    ics_nl = rng.normal(0, 0.5, size=(N_IC, N))
    att_nl = []
    for ic in ics_nl:
        sol = solve_ivp(nonlinear_dynamics, (0.0, 20.0), ic,
                        args=(A, B, np.zeros(K)), rtol=1e-6, atol=1e-8)
        att_nl.append(sol.y[:, -1])
    att_nl = np.array(att_nl)
    proj_nl = att_nl @ perron
    # spread at t=20 by tertiles (descriptive; see fixed-point analysis below)
    bands = np.digitize(proj_nl, np.percentile(proj_nl, [33, 67]))
    attr_nl_df = pd.DataFrame({"rep": np.arange(N_IC),
                               "projection_perron": proj_nl, "band": bands})
    attr_nl_df.to_csv(os.path.join(RES, "dynamics_attractors_nl.csv"), index=False)

    # extended horizon t=50: verify convergence to fixed point(s)
    att_nl50 = []
    for ic in ics_nl:
        sol = solve_ivp(nonlinear_dynamics, (0.0, 50.0), ic,
                        args=(A, B, np.zeros(K)), rtol=1e-7, atol=1e-9)
        att_nl50.append(sol.y[:, -1])
    att_nl50 = np.array(att_nl50)
    proj_nl50 = att_nl50 @ perron
    attr_nl50_df = pd.DataFrame({"rep": np.arange(N_IC),
                                 "projection_perron": proj_nl50})
    attr_nl50_df.to_csv(os.path.join(RES, "dynamics_attractors_nl50.csv"),
                        index=False)

    # fixed-point analysis: x* = tanh(A x*); multi-start root finding
    from scipy.optimize import root
    fps = {}
    for ic in ics_nl:
        sol = root(lambda x: -x + np.tanh(A @ x), ic, method="hybr", tol=1e-9)
        if sol.success:
            key = tuple(np.round(sol.x, 5))
            fps.setdefault(key, 0)
            fps[key] += 1
    fp_proj = [np.array(k) @ perron for k in fps]
    fp_max_re = []
    for k in fps:
        x = np.array(k)
        ev = np.linalg.eigvals(-np.eye(N) + A * (1 - np.tanh(A @ x) ** 2)[:, None])
        fp_max_re.append(float(np.real(ev).max()))

    meta = {
        "eta": ETA, "t_span": list(T_SPAN), "n_ic": N_IC,
        "n_linear_halves": 2,          # median split of transient projections
        "n_nonlinear_quantile_bands": int(len(np.unique(bands))),
        "n_fixed_points": len(fps),
        "fixed_point_projections": [float(p) for p in fp_proj],
        "fixed_point_max_re_eig": fp_max_re,
        "perturbation_regularisation": float(reg),
        "perturbation_unit_energy_regularised": E_unit,
        "perturbation_total_energy_regularised": E_total,
        "perturbation_input_norm": u_norm,
        "perturbation_integrated_input_energy_T8": input_energy_T8,
        "perturbation_start_projection": start_proj,
        "perturbation_end_projection": end_proj,
        "perturbation_target_projection": target_proj,
        "perturbation_displacement_fraction": displacement_fraction,
        "perturbation_full_state_relative_error": relative_error,
        "uncontrolled_end_projection": float(proj_free[-1]),
        "uncontrolled_median_cross_time": free_cross,
        "perron_projection_low": float(proj.min()),
        "perron_projection_high": float(proj.max()),
    }
    with open(os.path.join(RES, "dynamics_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
