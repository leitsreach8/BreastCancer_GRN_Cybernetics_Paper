#!/usr/bin/env python3
"""
12_supplementary_checks.py - additional robustness analyses requested at
review.

(1) Kernel-size sensitivity.  The control kernel is defined as the top K
    ranked genes by the structure-based control score; K = 60 is used
    throughout the main analysis.  To show that the prognostic results are
    not a coincidence of that particular cutoff, we repeat the univariate
    Cox and median-split log-rank analysis for K in {20, 40, 60},
    using both the centrality-weighted score (CS_K) and the equal-weight
    mean (EW_K) of the top K kernel genes.

(2) Correlation of CS with clinical covariates (ER, PR, HER2, grade, age).
    Quantifies how much of the prognostic signal may overlap with
    established molecular/clinical phenotypes.

(3) Time-dependent (IPCW) AUC of the clinical model M0 and of M0 + CS at
    12, 36, 60 and 120 months, using Uno's inverse-probability-of-censoring
    estimator with the censoring distribution estimated by Kaplan-Meier.

Outputs (results/):
  kernel_size_sensitivity.csv / .json
  cs_clinical_correlations.json
  time_auc.csv / time_auc.json
"""
import os
import json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RES = os.path.join(BASE, "results")

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "stat_models", os.path.join(BASE, "code", "07_statistical_models.py"))
_statmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_statmod)
cox_fit = _statmod.cox_fit
c_index = _statmod.c_index
logrank = _statmod.logrank
km_estimator = _statmod.km_estimator
load_analysis = _statmod.load_analysis
parse_status = _statmod.parse_status

KERNEL_SIZES = [20, 40, 60]  # the kernel contains 60 genes; K >= 60 is identical
TIME_POINTS = [12, 36, 60, 120]


def build_data():
    expr, anal, driver, node = load_analysis()
    anal = anal.copy()
    anal["rfs_time"] = pd.to_numeric(anal["p_RFS_MONTHS"], errors="coerce")
    anal["rfs_event"] = parse_status(anal["p_RFS_STATUS"])
    anal["age"] = pd.to_numeric(anal["p_AGE_AT_DIAGNOSIS"], errors="coerce")
    anal["grade"] = pd.to_numeric(anal["s_GRADE"], errors="coerce")
    er = anal["s_ER_STATUS"].astype(str).str.lower().str.contains("pos").astype(int)
    pr = anal["s_PR_STATUS"].astype(str).str.lower().str.contains("pos").astype(int)
    her2 = anal["s_HER2_STATUS"].astype(str).str.lower().str.contains("pos").astype(int)
    sub = anal.dropna(subset=["rfs_time", "rfs_event"])
    sub = sub[sub["sampleId"].isin(expr.index)]
    kernel_genes = driver["gene"].tolist()
    present = [g for g in kernel_genes if g in expr.columns]
    w = node.set_index("gene").loc[present, "control_score"].values
    Z = expr.loc[sub["sampleId"], present].values
    t = sub["rfs_time"].values
    e = sub["rfs_event"].values
    return present, w, Z, t, e, sub, er, pr, her2


def ipcw_auc(t, e, risk, tau):
    """Uno's time-dependent AUC at horizon tau (IPCW).

    AUC(tau) = [ sum_{i,j} I(eta_i > eta_j) I(T_i<=tau, d_i=1)/G(T_i)
                     I(T_j>tau) ] /
               [ sum_i I(T_i<=tau, d_i=1)/G(T_i) * sum_j I(T_j>tau) ]
    with G the KM estimate of the censoring survival distribution.
    """
    n = len(t)
    # censoring distribution: treat censored (= no event) as the "event"
    uniq, g_surv, _ = km_estimator(t, 1 - e)
    g = np.interp(t, uniq, g_surv, left=1.0, right=g_surv[-1] if len(g_surv) else 1.0)
    g = np.maximum(g, 1e-6)
    case = (t <= tau) & (e == 1)
    control = t > tau
    if case.sum() == 0 or control.sum() == 0:
        return float("nan")
    eta = risk
    # vectorised sum over case x control pairs
    num = 0.0
    den = 0.0
    w_c = 1.0 / g[case]
    for k, i in enumerate(np.where(case)[0]):
        gt = np.sum(eta[i] > eta[control])
        eq = np.sum(eta[i] == eta[control])
        num += w_c[k] * (gt + 0.5 * eq)   # ties contribute 0.5
        den += w_c[k] * control.sum()
    return float(num / den) if den > 0 else float("nan")


def main():
    present, w, Z, t, e, sub, er, pr, her2 = build_data()
    grade = sub["grade"].values
    age = sub["age"].values
    gz = (grade - np.nanmean(grade)) / (np.nanstd(grade) + 1e-8)
    agez = (age - np.nanmean(age)) / (np.nanstd(age) + 1e-8)
    er_v = er.values
    pr_v = pr.values
    her2_v = her2.values

    # ---- (1) kernel-size sensitivity ----
    rows = []
    for K in KERNEL_SIZES:
        Zk = Z[:, :K]
        wk = w[:K]
        csk = (Zk @ wk) / wk.sum()
        ewk = Zk.mean(axis=1)
        for label, s in [("CS", csk), ("EW", ewk)]:
            s = (s - s.mean()) / (s.std() + 1e-8)
            b, se = cox_fit(s.reshape(-1, 1), t, e)
            from scipy.stats import norm
            p_wald = float(2 * (1 - norm.cdf(abs(b[0] / max(se[0], 1e-9)))))
            chi2, p_lr = logrank(t, e, (s > np.median(s)).astype(int))
            ci = c_index(t, e, s)
            rows.append({"K": K, "score": label,
                         "HR": float(np.exp(b[0])),
                         "HR_low": float(np.exp(b[0] - 1.96 * se[0])),
                         "HR_high": float(np.exp(b[0] + 1.96 * se[0])),
                         "p_wald": p_wald, "logrank_p": p_lr,
                         "cindex": ci})
    kdf = pd.DataFrame(rows)
    kdf.to_csv(os.path.join(RES, "kernel_size_sensitivity.csv"), index=False)
    with open(os.path.join(RES, "kernel_size_sensitivity.json"), "w") as fh:
        json.dump({"kernel_sizes": KERNEL_SIZES,
                   "rows": rows,
                   "kernel_size_rule": ("top-K by the structure-based control "
                                        "score; K=60 pre-specified in the "
                                        "analysis plan before model fitting")},
                  fh, indent=2)

    # ---- (2) CS-clinical correlations ----
    cs = (Z @ w) / w.sum()
    cs = (cs - cs.mean()) / cs.std()
    ok_g = ~np.isnan(gz)
    corr = {
        "CS_vs_ER": float(np.corrcoef(er_v, cs)[0, 1]),
        "CS_vs_PR": float(np.corrcoef(pr_v, cs)[0, 1]),
        "CS_vs_HER2": float(np.corrcoef(her2_v, cs)[0, 1]),
        "CS_vs_grade_z": float(np.corrcoef(gz[ok_g], cs[ok_g])[0, 1]),
        "CS_vs_age_z": float(np.corrcoef(agez, cs)[0, 1]),
    }
    with open(os.path.join(RES, "cs_clinical_correlations.json"), "w") as fh:
        json.dump(corr, fh, indent=2)

    # ---- (3) time-dependent (IPCW) AUC ----
    clin = np.column_stack([er_v, pr_v, her2_v,
                            np.nan_to_num(gz, nan=0.0),
                            np.nan_to_num(agez, nan=0.0)])
    X1 = np.column_stack([cs, clin])
    b0, _ = cox_fit(clin, t, e)
    b1, _ = cox_fit(X1, t, e)
    r0 = clin @ b0
    r1 = X1 @ b1
    auc_rows = []
    for tau in TIME_POINTS:
        auc_rows.append({"t_months": tau, "model": "M0_clinical",
                         "ipcw_auc": ipcw_auc(t, e, r0, tau)})
        auc_rows.append({"t_months": tau, "model": "M0_plus_CS",
                         "ipcw_auc": ipcw_auc(t, e, r1, tau)})
    adf = pd.DataFrame(auc_rows)
    adf.to_csv(os.path.join(RES, "time_auc.csv"), index=False)
    with open(os.path.join(RES, "time_auc.json"), "w") as fh:
        json.dump({"time_points_months": TIME_POINTS,
                   "rows": auc_rows,
                   "estimator": ("Uno IPCW time-dependent AUC; censoring "
                                 "distribution estimated by KM"),
                   "note": ("Risk models are fitted on the full analysis "
                            "cohort (in-sample); the AUCs are therefore "
                            "descriptive, not cross-validated, and ties in "
                            "the risk score contribute 0.5")},
                  fh, indent=2)

    print("Kernel-size sensitivity (HR, 95% CI, log-rank p):")
    print(kdf.to_string(index=False))
    print("CS-clinical correlations:", json.dumps(corr, indent=1))
    print("Time-dependent AUC:")
    print(adf.to_string(index=False))


if __name__ == "__main__":
    main()
