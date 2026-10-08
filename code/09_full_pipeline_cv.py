#!/usr/bin/env python3
"""
09_full_pipeline_cv.py - nested (full-pipeline) cross-validation.

The main analysis (07) constructs the network, the control-kernel gene set,
the control-kernel score (CS) and the clinical covariate encoding once on the
full cohort, and then cross-validates only the Cox fitting step. That
protocol measures discrimination conditional on fixed, full-cohort features.

This script re-runs the COMPLETE feature-construction pipeline inside each
training fold - expression standardisation, Graphical-Lasso network
inference, centrality computation, kernel-gene selection, CS weights,
CS computation and clinical covariate encoding are all refit on the training
fold only - and scores each held-out test fold. The resulting C-indices
therefore measure generalisation of the whole analysis pipeline to unseen
patients.

Per fold (stratified 5-fold by event):
  train: z-score expression (fit on train) -> GraphicalLassoCV (penalty
         selected on train) -> W = |partial correlation| -> network
         centralities (degree, betweenness, eigenvector) -> top-60 kernel
         genes -> CS weights -> CS on train -> standardise CS and clinical
         covariates (fit on train) -> Cox M0/M1/M2 (fit on train)
  test : apply train-fitted transforms -> CS -> risk -> Harrell C-index

Outputs:
  results/full_pipeline_cv.csv : per-fold C-index per model
  results/full_pipeline_cv.json : mean and SD per model
"""
import os

import json
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.covariance import GraphicalLassoCV
from sklearn.model_selection import StratifiedKFold

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RES = os.path.join(BASE, "results")

# import the helpers from 07_statistical_models.py (module name starts with
# a digit, so load it explicitly by path)
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "stat_models", os.path.join(BASE, "code", "07_statistical_models.py"))
_statmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_statmod)
cox_fit = _statmod.cox_fit
c_index = _statmod.c_index
load_analysis = _statmod.load_analysis

SEED = 2025
N_FOLDS = 5
N_KERNEL_USED = 40
KERNEL_SIZE = 60


def infer_network_train(expr, tr_sid):
    """Graphical-lasso network + centralities, trained on the fold only."""
    Xtr = expr.loc[tr_sid]
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-8
    Xz = (Xtr - mu) / sd
    glcv = GraphicalLassoCV(cv=3, alphas=6, n_jobs=2, max_iter=500, tol=1e-6)
    glcv.fit(Xz.values)
    prec = glcv.precision_
    d = np.sqrt(np.diag(prec))
    # sign convention identical to 03_network_inference.py: partial
    # correlations are -prec / outer(d, d); the absolute value used
    # below is unaffected, but keeping the same definition avoids a
    # silent convention divergence between the two pipelines
    pcorr = -prec / np.outer(d, d)
    np.fill_diagonal(pcorr, 0.0)
    W = np.abs(pcorr)

    genes = list(expr.columns)
    G = nx.Graph()
    G.add_nodes_from(genes)
    p = len(genes)
    for i in range(p):
        for j in range(i + 1, p):
            if W[i, j] > 1e-12:
                G.add_edge(genes[i], genes[j], weight=float(W[i, j]))

    deg = dict(G.degree())
    between = nx.betweenness_centrality(G)
    vals, vecs = np.linalg.eigh(W)
    perron = np.abs(vecs[:, -1])
    perron /= perron.sum()
    evc = {genes[i]: float(perron[i]) for i in range(p)}

    node = pd.DataFrame({
        "gene": genes,
        "degree": [deg[g] for g in genes],
        "betweenness": [between[g] for g in genes],
        "eigenvector_centrality": [evc[g] for g in genes],
    })
    for col in ["degree", "betweenness", "eigenvector_centrality"]:
        node[col + "_rank"] = node[col].rank(ascending=True, pct=True)
    node["control_score"] = (
        node["degree_rank"] + node["betweenness_rank"]
        + node["eigenvector_centrality_rank"]) / 3.0
    node = node.sort_values("control_score", ascending=False)
    driver = node.head(KERNEL_SIZE)[["gene", "control_score"]]
    return driver, node


def build_features(expr, sub, driver, node, rows):
    """CS + clinical covariates for a set of positional rows of sub.

    Standardisation parameters are fit on the supplied rows (training fold);
    the same fitted transforms are applied to test rows by the caller via
    the returned dictionary.
    """
    sid = sub["sampleId"].iloc[rows].values
    genes = driver["gene"].tolist()
    present = [g for g in genes if g in expr.columns]
    w = node.set_index("gene").loc[present, "control_score"].values

    mu = expr.loc[sid, present].mean(0)
    sd = expr.loc[sid, present].std(0) + 1e-8
    Z = (expr.loc[sid, present] - mu) / sd

    CS = (Z.values @ w) / w.sum()
    cs_mu, cs_sd = float(CS.mean()), float(CS.std()) + 1e-8
    CS = (CS - cs_mu) / cs_sd

    grade = sub["grade"].iloc[rows].values
    age = sub["age"].iloc[rows].values
    g_mu, g_sd = float(np.nanmean(grade)), float(np.nanstd(grade)) + 1e-8
    a_mu, a_sd = float(np.nanmean(age)), float(np.nanstd(age)) + 1e-8
    clin = np.column_stack([
        sub["er"].iloc[rows].values, sub["pr"].iloc[rows].values,
        sub["her2"].iloc[rows].values,
        (grade - g_mu) / g_sd, (age - a_mu) / a_sd])
    clin = np.nan_to_num(clin, nan=0.0)

    X1 = np.column_stack([CS, clin])
    Zk = Z.values[:, :min(N_KERNEL_USED, Z.shape[1])]
    zk_mu, zk_sd = Zk.mean(0), Zk.std(0) + 1e-8
    Zk = (Zk - zk_mu) / zk_sd
    X2 = np.column_stack([Zk, clin])

    trans = {"present": present, "w": w, "cs_mu": cs_mu, "cs_sd": cs_sd,
             "g_mu": g_mu, "g_sd": g_sd, "a_mu": a_mu, "a_sd": a_sd,
             "expr_mu": mu, "expr_sd": sd, "zk_mu": zk_mu, "zk_sd": zk_sd}
    return X1, X2, clin, trans


def apply_features(expr, sub, rows, trans):
    """Apply train-fitted transforms to a test set of rows."""
    sid = sub["sampleId"].iloc[rows].values
    present, w = trans["present"], trans["w"]
    Z = (expr.loc[sid, present] - trans["expr_mu"]) / trans["expr_sd"]
    CS = (Z.values @ w) / w.sum()
    CS = (CS - trans["cs_mu"]) / trans["cs_sd"]
    grade = sub["grade"].iloc[rows].values
    age = sub["age"].iloc[rows].values
    clin = np.column_stack([
        sub["er"].iloc[rows].values, sub["pr"].iloc[rows].values,
        sub["her2"].iloc[rows].values,
        (grade - trans["g_mu"]) / trans["g_sd"],
        (age - trans["a_mu"]) / trans["a_sd"]])
    clin = np.nan_to_num(clin, nan=0.0)
    X1 = np.column_stack([CS, clin])
    Zk = (Z.values[:, :min(N_KERNEL_USED, Z.shape[1])]
          - trans["zk_mu"]) / trans["zk_sd"]
    X2 = np.column_stack([Zk, clin])
    return X1, X2, clin


def main():
    expr, anal, _, _ = load_analysis()
    expr = expr.copy()
    anal = anal.copy()
    anal["rfs_time"] = pd.to_numeric(anal["p_RFS_MONTHS"], errors="coerce")
    anal["rfs_event"] = pd.to_numeric(
        anal["p_RFS_STATUS"].astype(str).str.split(":").str[0],
        errors="coerce").fillna(0).astype(int)
    anal["age"] = pd.to_numeric(anal["p_AGE_AT_DIAGNOSIS"], errors="coerce")
    anal["grade"] = pd.to_numeric(anal["s_GRADE"], errors="coerce")
    anal["er"] = anal["s_ER_STATUS"].astype(str).str.lower().str.contains(
        "pos").astype(int)
    anal["pr"] = anal["s_PR_STATUS"].astype(str).str.lower().str.contains(
        "pos").astype(int)
    anal["her2"] = anal["s_HER2_STATUS"].astype(str).str.lower().str.contains(
        "pos").astype(int)

    sub = anal.dropna(subset=["rfs_time", "rfs_event"])
    sub = sub[sub["sampleId"].isin(expr.index)].reset_index(drop=True)
    t = sub["rfs_time"].values
    e = sub["rfs_event"].values
    print(f"full-pipeline CV: n={len(sub)}, events={int(e.sum())}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(sub, e)):
        tr_sid = sub["sampleId"].iloc[tr].values
        driver, node = infer_network_train(expr, tr_sid)
        X1_tr, X2_tr, clin_tr, trans = build_features(
            expr, sub, driver, node, tr)
        X1_te, X2_te, clin_te = apply_features(expr, sub, te, trans)

        b0, _ = cox_fit(clin_tr, t[tr], e[tr], compute_se=False)
        b1, _ = cox_fit(X1_tr, t[tr], e[tr], compute_se=False)
        b2, _ = cox_fit(X2_tr, t[tr], e[tr], penalty=0.08, compute_se=False)
        c0 = c_index(t[te], e[te], clin_te @ b0)
        c1 = c_index(t[te], e[te], X1_te @ b1)
        c2 = c_index(t[te], e[te], X2_te @ b2)
        rows.append({"fold": fold, "M0": c0, "M1": c1, "M2": c2})
        print(f"fold {fold}: M0={c0:.4f} M1={c1:.4f} M2={c2:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "full_pipeline_cv.csv"), index=False)
    summary = {m: {"mean": float(df[m].mean()),
                   "sd": float(df[m].std())} for m in ["M0", "M1", "M2"]}
    with open(os.path.join(RES, "full_pipeline_cv.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
