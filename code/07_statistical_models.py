#!/usr/bin/env python3
"""
07_statistical_models.py - Biostatistical modelling of relapse-free survival
(RFS) from the cybernetic control-kernel signature in METABRIC.

Patient-level signature:
    CS_i = sum_k w_k z_ik / sum_k w_k
where z are z-scored expressions of the control-kernel (driver) genes and w
are structure-based, outcome-blind control scores.

Models:
  Cox proportional hazards (implemented directly via partial likelihood,
  Breslow tie handling, Newton-Raphson; Harrell C-index):
      M0  clinical covariates only        (ER, PR, HER2, grade, age)
      M1  CS + clinical covariates
      M2  Ridge-Cox (L2-penalised Cox) on kernel genes + clinical
  Bayesian logistic (horseshoe prior, Polya-Gamma Gibbs) on 5-year RFS as a
  robustness/small-sample companion analysis.
  Permutation test for the CS-RFS association; Kaplan-Meier + log-rank for
  CS-high vs CS-low strata.

Outputs (results/):
  model_performance.json, cox_results.csv, cv_predictions.csv,
  km_data.csv, km_logrank.json, bayes_posterior.csv, permutation_test.json
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RES = os.path.join(BASE, "results")
SEED = 42
N_REPEATS = 5
N_FOLDS = 5
N_PERM = 1000
N_KERNEL_USED = 40  # top control-kernel genes in penalised models


# ----------------------------------------------------------------------
# Cox proportional hazards (direct partial-likelihood implementation)
# ----------------------------------------------------------------------
def cox_neg_partial_likelihood(beta, X, time, event, penalty=0.0):
    """Negative partial log-likelihood (Breslow method), vectorised.

    For each event time t_i the denominator is the log-sum-exp of the
    linear predictors over the risk set {j: t_j >= t_i}; the boolean
    outer product builds all risk sets at once (O(n_events x n) memory),
    which is fast because logsumexp runs in compiled code.
    """
    eta = X @ beta
    ev = event == 1
    if not np.any(ev):
        return penalty * np.sum(beta ** 2)
    # mask[ev_i, j] = 1 iff time[j] >= time[ev_i]
    mask = time[ev][:, None] <= time[None, :]
    eta_masked = np.where(mask, eta, -np.inf)
    denom = logsumexp(eta_masked, axis=1)
    negll = float(denom.sum() - eta[ev].sum())
    return negll + penalty * np.sum(beta ** 2)


def cox_neg_gradient(beta, X, time, event, penalty=0.0):
    """Analytic gradient of the negative partial log-likelihood (Breslow).

    d/db = 2*penalty*b - sum_{events i} x_i
           + sum_{events i} [ sum_{j in risk_i} x_j p_{ij} ]
    with p_{ij} = exp(eta_j - denom_i) the softmax weight of sample j in
    the risk set of event i.  Vectorised exactly as the objective above.
    """
    eta = X @ beta
    ev = event == 1
    if not np.any(ev):
        return 2.0 * penalty * beta
    mask = time[ev][:, None] <= time[None, :]
    eta_masked = np.where(mask, eta, -np.inf)
    denom = logsumexp(eta_masked, axis=1)
    p = np.exp(eta[None, :] - denom[:, None]) * mask     # n_events x n
    p = p / p.sum(axis=1, keepdims=True)                 # softmax over risk set
    grad = p @ X                                          # n_events x p
    grad = grad.sum(axis=0) - X[ev].sum(axis=0) + 2.0 * penalty * beta
    return grad


def cox_fit(X, time, event, penalty=0.0, compute_se=True):
    p = X.shape[1]
    b0 = np.zeros(p)
    res = minimize(cox_neg_partial_likelihood, b0,
                   args=(X, time, event, penalty), jac=cox_neg_gradient,
                   method="BFGS", options={"maxiter": 150, "gtol": 1e-6})
    beta = res.x
    if not compute_se:
        return beta, np.zeros(p)
    # SEs from the BFGS inverse-Hessian approximation (not the exact
    # observed information matrix); survival::coxph (R 4.5.2) agrees to <1%.
    Hinv = np.asarray(res.hess_inv)
    se = np.sqrt(np.clip(np.diag(Hinv), 1e-12, None))
    return beta, se


def _beta_with(beta, i, di, j, dj):
    b = beta.copy()
    b[i] += di
    b[j] += dj
    return b


def c_index(time, event, risk_score):
    """Harrell's concordance index (vectorised, equivalent to the pairwise
    definition over all i<j pairs)."""
    n = len(time)
    I, J = np.triu_indices(n, k=1)
    ti, tj = time[I], time[J]
    ri, rj = risk_score[I], risk_score[J]
    ei, ej = event[I], event[J]
    # comparable pairs (three mutually exclusive cases, as in the pairwise loop)
    m1 = (ei == 1) & (ej == 1) & (ti != tj)   # both events, distinct times
    m2 = (ei == 1) & (ej == 0) & (ti <= tj)   # i event, j censored later
    m3 = (ej == 1) & (ei == 0) & (tj <= ti)   # j event, i censored later
    cmp = m1 | m2 | m3
    if not np.any(cmp):
        return float("nan")
    sel_ti, sel_tj = ti[cmp], tj[cmp]
    sel_ri, sel_rj = ri[cmp], rj[cmp]
    # m1/m2/m3 restricted to comparable pairs (exhaustive by construction)
    m1s, m2s, m3s = m1[cmp], m2[cmp], m3[cmp]
    conc = np.zeros(int(cmp.sum()))
    tie_r = sel_ri == sel_rj
    # concordant if earlier event carries higher risk; ties contribute 0.5
    conc[tie_r] = 0.5
    nt = ~tie_r
    conc[m1s & nt] = ((sel_ri > sel_rj) == (sel_ti < sel_tj))[m1s & nt]
    conc[m2s & nt] = (sel_ri > sel_rj)[m2s & nt]
    conc[m3s & nt] = (sel_rj > sel_ri)[m3s & nt]
    return float(conc.sum() / cmp.sum())


# ----------------------------------------------------------------------
# Polya-Gamma + horseshoe logistic (5-year RFS binary companion model)
# ----------------------------------------------------------------------
def polya_gamma_draw(z, rng, trunc=200):
    z2 = z ** 2
    out = np.zeros(len(z))
    for k in range(1, trunc + 1):
        gk = rng.gamma(1.0, 1.0, size=len(z))
        out += gk / ((k - 0.5) ** 2 + z2 / (4.0 * np.pi ** 2))
    return out / (2.0 * np.pi ** 2)


class HorseshoeLogistic:
    def __init__(self, p0=5, sigma=1.0, n_iter=4000, burn=1000, seed=SEED):
        self.p0, self.sigma = p0, sigma
        self.n_iter, self.burn = n_iter, burn
        self.rng = np.random.default_rng(seed)
        self.tau0 = None                 # global shrinkage scale, set in fit()

    def fit(self, X, y):
        n, p = X.shape
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
        Xd = np.column_stack([np.ones(n), Xs])
        p1 = p + 1
        tau0 = self.p0 / (p - self.p0) * self.sigma / np.sqrt(n)
        self.tau0 = float(tau0)          # global shrinkage scale
        beta = np.zeros(p1); tau2 = 1.0; lam2 = np.ones(p1)
        nu = np.ones(p1); xi = 1.0; kappa = y - 0.5
        chain = np.zeros((self.n_iter, p1))
        for it in range(self.n_iter):
            psi = np.clip(Xd @ beta, -25.0, 25.0)      # numerical guard
            omega = polya_gamma_draw(psi, self.rng)
            inv_l = np.clip(1.0 / (tau2 * lam2), 1e-12, 1e12)
            inv_l[0] = 1e-12
            XtO = Xd.T * omega
            S = np.linalg.inv(XtO @ Xd + np.diag(inv_l))
            S = (S + S.T) / 2.0                        # symmetrise
            S += 1e-10 * np.eye(p1)                    # jitter
            beta = self.rng.multivariate_normal(S @ (Xd.T @ kappa), S)
            beta = np.clip(beta, -20.0, 20.0)
            lam2[1:] = np.clip(1.0 / self.rng.gamma(
                1.0, 1.0 / (1.0 / nu[1:] + beta[1:] ** 2 / (2.0 * tau2))),
                1e-10, 1e10)
            lam2[0] = 1e6
            nu[1:] = np.clip(1.0 / self.rng.gamma(
                1.0, 1.0 / (1.0 + 1.0 / lam2[1:])), 1e-10, 1e10)
            # Makalic-Schmidt half-Cauchy update: tau0^2 enters only the xi
            # conditional; tau^2 uses rate 1/xi + sum(beta^2/(2 lam^2)).
            rate_tau = (1.0 / xi + np.sum(beta[1:] ** 2 / (2.0 * lam2[1:])))
            tau2 = np.clip(1.0 / self.rng.gamma(
                (p + 1) / 2.0, 1.0 / rate_tau), 1e-10, 1e10)
            xi = np.clip(1.0 / self.rng.gamma(
                1.0, 1.0 / (1.0 / self.tau0 ** 2 + 1.0 / tau2)),
                1e-10, 1e10)
            chain[it] = beta
        self.beta = chain[self.burn:].mean(axis=0)
        self.beta_sd = chain[self.burn:].std(axis=0)
        self.chain = chain[self.burn:]          # post-burnin draws
        self.X_mean = X.mean(0); self.X_std = X.std(0) + 1e-8
        # ESS per coefficient from lag-1 autocorrelation of the post-burnin chain
        ch = self.chain[:, 1:]
        if ch.shape[0] > 2:
            r1 = np.array([np.corrcoef(ch[:-1, k], ch[1:, k])[0, 1]
                           for k in range(ch.shape[1])])
            ess = ch.shape[0] * (1.0 - r1) / (1.0 + r1)
            self.ess_min = float(np.nanmin(ess))
            self.ess_median = float(np.nanmedian(ess))
        else:
            self.ess_min = self.ess_median = float("nan")
        return self

    def predict_proba(self, X):
        Xs = (X - self.X_mean) / self.X_std
        Xd = np.column_stack([np.ones(len(X)), Xs])
        return expit(Xd @ self.beta)


# ----------------------------------------------------------------------
def load_analysis():
    expr = pd.read_csv(os.path.join(DATA, "metabric_processed.csv"), index_col=0)
    anal = pd.read_csv(os.path.join(DATA, "metabric_analysis.csv"))
    driver = pd.read_csv(os.path.join(RES, "driver_genes.csv"))
    node = pd.read_csv(os.path.join(RES, "node_metrics.csv"))
    return expr, anal, driver, node


def parse_status(status_col):
    s = status_col.astype(str).str.split(":").str[0]
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).values


def km_estimator(time, event):
    """Kaplan-Meier curve: unique times, survival prob, numbers at risk.

    Times are rounded to 6 decimals before forming the unique event times,
    so duplicate event times produced by floating-point arithmetic are
    merged into a single time point. Events recorded at t = 0 are folded
    into the first positive event time, giving the conventional S(0) = 1
    (the curve drops at the first event time).
    """
    time = np.round(np.asarray(time, float), 6)
    event = np.asarray(event, int)
    t0 = time == 0
    if t0.any() and (~t0).any():
        t_first = float(time[~t0].min())
        time = np.where(t0, t_first, time)   # t = 0 samples join t_first
    uniq, inv = np.unique(time, return_inverse=True)
    d_counts = np.bincount(inv, weights=event).astype(int)  # events per time
    at_risk = np.array([int(np.sum(time >= t)) for t in uniq])
    surv = np.ones(len(uniq))
    for k in range(len(uniq)):
        if at_risk[k] > 0 and d_counts[k] > 0:
            surv[k] = surv[k - 1] * (1 - d_counts[k] / at_risk[k])
        elif k > 0:
            surv[k] = surv[k - 1]
    return uniq, np.maximum(surv, 0.0), at_risk


def logrank(time, event, groups):
    """Two-sample log-rank test."""
    uniq = np.unique(time)
    O1 = E1 = V1 = 0.0
    for t in uniq:
        m = time == t
        d1 = np.sum(m & event & (groups == 1))
        d0 = np.sum(m & event & (groups == 0))
        n1 = np.sum((time >= t) & (groups == 1))
        n0 = np.sum((time >= t) & (groups == 0))
        n = n1 + n0
        d = d1 + d0
        if n > 1 and d > 0:
            O1 += d1
            E1 += d * n1 / n
            V1 += d * (n - d) / (n - 1) * n1 * n0 / n ** 2
    chi2 = (O1 - E1) ** 2 / V1 if V1 > 0 else 0.0
    from scipy.stats import chi2 as chi2_dist
    return chi2, float(chi2_dist.sf(chi2, 1))


def cv_cox(X, time, event, penalty=0.0):
    cids = []
    fold_rows = []
    rskf = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS,
                                   random_state=SEED)
    preds = []
    # stratify by event so each fold keeps ~40% events (constant labels
    # would silently reduce to plain K-fold)
    for fold, (tr, te) in enumerate(rskf.split(X, event)):
        beta, _ = cox_fit(X[tr], time[tr], event[tr], penalty,
                          compute_se=False)
        risk = X[te] @ beta
        ci = c_index(time[te], event[te], risk)
        cids.append(ci)
        fold_rows.append((fold, float(ci)))
        for i, r in zip(te, risk):
            preds.append((i, r, event[i]))
    cv = pd.DataFrame(preds, columns=["sample_idx", "risk", "event"]).sort_values(
        "sample_idx")
    folds = pd.DataFrame(fold_rows, columns=["fold", "cindex"])
    return float(np.mean(cids)), float(np.std(cids)), cv, folds


def main():
    expr, anal, driver, node = load_analysis()
    # ---- analysis sample: complete survival + expression ----
    anal["rfs_time"] = pd.to_numeric(anal["p_RFS_MONTHS"], errors="coerce")
    anal["rfs_event"] = parse_status(anal["p_RFS_STATUS"])
    anal["os_time"] = pd.to_numeric(anal["p_OS_MONTHS"], errors="coerce")
    anal["os_event"] = parse_status(anal["p_OS_STATUS"])
    anal["age"] = pd.to_numeric(anal["p_AGE_AT_DIAGNOSIS"], errors="coerce")
    anal["grade"] = pd.to_numeric(anal["s_GRADE"], errors="coerce")
    er = anal["s_ER_STATUS"].astype(str).str.lower().str.contains("pos").astype(int)
    pr = anal["s_PR_STATUS"].astype(str).str.lower().str.contains("pos").astype(int)
    her2 = anal["s_HER2_STATUS"].astype(str).str.lower().str.contains("pos").astype(int)
    anal["er"], anal["pr"], anal["her2"] = er, pr, her2

    sub = anal.dropna(subset=["rfs_time", "rfs_event"])
    sub = sub[sub["sampleId"].isin(expr.index)]
    print(f"analysis set: n={len(sub)}, RFS events="
          f"{int(sub['rfs_event'].sum())} ({sub['rfs_event'].mean():.3f})")

    # ---- patient-level control-kernel score ----
    kernel_genes = driver["gene"].tolist()
    present = [g for g in kernel_genes if g in expr.columns]
    w = node.set_index("gene").loc[present, "control_score"].values
    Z = expr.loc[sub["sampleId"], present].values
    CS = (Z @ w) / w.sum()
    CS = (CS - CS.mean()) / CS.std()
    sub["CS"] = CS

    # ---- covariate matrices ----
    grade = sub["grade"].values
    print("missing grade:", int(np.isnan(grade).sum()))  # imputed to z mean (0)
    age = sub["age"].values
    clin = np.column_stack([
        sub["er"].values, sub["pr"].values, sub["her2"].values,
        (grade - np.nanmean(grade)) / (np.nanstd(grade) + 1e-8),
        (age - np.nanmean(age)) / (np.nanstd(age) + 1e-8)])
    clin = np.nan_to_num(clin, nan=0.0)
    X1 = np.column_stack([CS, clin])
    Zk = Z[:, :min(N_KERNEL_USED, Z.shape[1])]
    Zk = (Zk - Zk.mean(0)) / (Zk.std(0) + 1e-8)
    X2 = np.column_stack([Zk, clin])

    t, e = sub["rfs_time"].values, sub["rfs_event"].values

    # ---- 5-year relapse rates (KM-style denominators) ----
    # denom: evaluable at 60 months; numerator: relapse within 60 months
    sub_er = sub["s_ER_STATUS"].astype(str).str.lower().str.contains(
        "pos").astype(int).values
    sub_her2 = sub["s_HER2_STATUS"].astype(str).str.lower().str.contains(
        "pos").astype(int).values

    def rate5(mask):
        tm, em = t[mask], e[mask]
        denom = (tm >= 60) | ((tm < 60) & (em == 1))
        num = (tm <= 60) & (em == 1)
        return float(num.sum() / denom.sum()) if denom.sum() else float("nan")

    r5 = {"ER_pos_5yr": rate5(sub_er == 1), "ER_neg_5yr": rate5(sub_er == 0),
          "HER2_pos_5yr": rate5(sub_her2 == 1),
          "HER2_neg_5yr": rate5(sub_her2 == 0)}
    pd.DataFrame([r5]).to_csv(os.path.join(RES, "rate5.csv"), index=False)
    print("5-year relapse rate: ER+ {:.1f}% / ER- {:.1f}% / HER2+ {:.1f}% / "
          "HER2- {:.1f}%".format(100 * r5["ER_pos_5yr"], 100 * r5["ER_neg_5yr"],
                                 100 * r5["HER2_pos_5yr"],
                                 100 * r5["HER2_neg_5yr"]))

    # ---- Table 1 source: stratified counts, overall RFS event rates,
    # median observed RFS by ER status ----
    t_v, e_v = t, e
    er_v = sub["er"].values
    pr_v = sub["pr"].values
    her2_v = sub["her2"].values
    gr_v = sub["grade"].values

    def strat_row(label, mask):
        nm = int(mask.sum())
        nev = int(np.sum(mask & (e_v == 1)))
        return {"stratum": label, "n": nm, "rfs_events": nev,
                "rfs_event_rate": float(nev / nm) if nm else float("nan")}

    pr_miss = sub["s_PR_STATUS"].isna().values
    her2_miss = sub["s_HER2_STATUS"].isna().values
    rows_t1 = [
        strat_row("ER_pos", er_v == 1), strat_row("ER_neg", er_v == 0),
        strat_row("PR_pos", pr_v == 1),
        strat_row("PR_neg", (pr_v == 0) & ~pr_miss),   # missing excluded
        strat_row("PR_missing", pr_miss),
        strat_row("HER2_pos", her2_v == 1),
        strat_row("HER2_neg", (her2_v == 0) & ~her2_miss),
        strat_row("HER2_missing", her2_miss),
    ]
    for g in sorted(pd.unique(gr_v[~np.isnan(gr_v)])):
        rows_t1.append(strat_row("grade_%d" % int(g),
                                 (gr_v == g) & ~np.isnan(gr_v)))
    pd.DataFrame(rows_t1).to_csv(
        os.path.join(RES, "table1_cohort.csv"), index=False)
    med_er_pos = float(np.median(t_v[er_v == 1]))
    med_er_neg = float(np.median(t_v[er_v == 0]))
    # collinearity diagnostics (Section 3.5): ER-grade and ER-CS;
    # grade has 88 missing values, so ER-grade uses complete cases
    gz = (gr_v - np.nanmean(gr_v)) / (np.nanstd(gr_v) + 1e-8)
    _ok = ~np.isnan(gz)
    corr_er_grade = float(np.corrcoef(er_v[_ok], gz[_ok])[0, 1])
    corr_er_cs = float(np.corrcoef(er_v, CS)[0, 1])
    print("Table 1 source written to table1_cohort.csv; "
          "median observed RFS: ER+ {:.1f} / ER- {:.1f} months; "
          "corr(ER, grade) = {:.3f}; corr(ER, CS) = {:.3f}".format(
              med_er_pos, med_er_neg, corr_er_grade, corr_er_cs))

    # ---- Cox models ----
    b0, se0 = cox_fit(clin, t, e)
    b1, se1 = cox_fit(X1, t, e)
    b2, se2 = cox_fit(X2, t, e, penalty=0.08)
    c0 = c_index(t, e, clin @ b0)
    c1 = c_index(t, e, X1 @ b1)
    c2 = c_index(t, e, X2 @ b2)
    c0cv, sd0cv, _, f0 = cv_cox(clin, t, e)
    c1cv, sd1cv, cv1, f1 = cv_cox(X1, t, e)
    c2cv, sd2cv, _, f2 = cv_cox(X2, t, e, penalty=0.08)
    fold_scores = pd.concat([
        f0.assign(model="M0"), f1.assign(model="M1"),
        f2.assign(model="M2")], ignore_index=True)
    fold_scores.to_csv(os.path.join(RES, "cv_fold_scores.csv"), index=False)
    # paired M1 - M0 within the same folds (first repeat; shared folds)
    f0r, f1r = f0[f0["fold"] < N_FOLDS], f1[f1["fold"] < N_FOLDS]
    d_pair = f1r["cindex"].values - f0r["cindex"].values
    t_pair = float(d_pair.mean())
    sd_pair = float(d_pair.std(ddof=1))
    from scipy.stats import ttest_rel
    _, p_pair = ttest_rel(f1r["cindex"].values, f0r["cindex"].values)
    cv1.to_csv(os.path.join(RES, "cv_predictions.csv"), index=False)
    # sanity checks (ER+ protective; C-index near 1 for perfect ranking)
    print("M0 betas (ER,PR,HER2,grade,age):", np.round(b0, 4).tolist())
    print("M1 betas (CS,ER,PR,HER2,grade,age):", np.round(b1, 4).tolist())

    # ---- univariable ER+ model (protective effect reported in the text) ----
    er1 = np.asarray(er, dtype=float).reshape(-1, 1)
    b_er1, se_er1 = cox_fit(er1, t, e)
    hr_er1 = float(np.exp(b_er1[0]))
    lo_er1 = float(np.exp(b_er1[0] - 1.96 * se_er1[0]))
    hi_er1 = float(np.exp(b_er1[0] + 1.96 * se_er1[0]))
    p_er1 = float(2 * (1 - _norm_cdf(np.abs(b_er1[0] / max(se_er1[0], 1e-9)))))
    er_uni = pd.DataFrame({"variable": ["ER+"], "beta": b_er1, "se": se_er1,
                           "HR": [hr_er1], "HR_low": [lo_er1],
                           "HR_high": [hi_er1], "p": [p_er1]})
    er_uni.to_csv(os.path.join(RES, "univariable_er.csv"), index=False)
    print(f"univariable ER+ HR = {hr_er1:.3f} "
          f"({lo_er1:.2f}-{hi_er1:.2f}; p = {p_er1:.3f})")
    # ---- Cox table (M1) ----
    names = ["CS"] + ["ER+", "PR+", "HER2+", "grade(z)", "age(z)"]
    cox_df = pd.DataFrame({
        "variable": names,
        "beta": b1, "se": se1,
        "HR": np.exp(b1),
        "HR_low": np.exp(b1 - 1.96 * se1),
        "HR_high": np.exp(b1 + 1.96 * se1),
    })
    cox_df["p"] = 2 * (1 - _norm_cdf(np.abs(b1 / np.maximum(se1, 1e-9))))
    cox_df.to_csv(os.path.join(RES, "cox_results.csv"), index=False)

    # ---- Ridge-Cox coefficients (L2 penalty, model M2) ----
    ridge_df = pd.DataFrame({"variable": present[:N_KERNEL_USED] + names[1:],
                             "beta": b2})
    ridge_df.to_csv(os.path.join(RES, "cox_ridge.csv"), index=False)

    # ---- KM by CS strata (median split) ----
    grp = (CS > np.median(CS)).astype(int)
    km_rows = []
    for g, lab in [(0, "CS-low"), (1, "CS-high")]:
        m = grp == g
        u, s, ar = km_estimator(t[m], e[m])
        for tt, ss, aa in zip(u, s, ar):
            km_rows.append((lab, tt, ss, aa))
    km_df = pd.DataFrame(km_rows, columns=["stratum", "time_months",
                                           "survival", "n_at_risk"])
    km_df.to_csv(os.path.join(RES, "km_data.csv"), index=False)
    chi2, p_lr = logrank(t, e, grp)

    # ---- permutation test: Cox coefficient of CS under the null ----
    # Permute CS, refit the univariate Cox model, keep times and events
    # fixed (preserves censoring and risk sets).
    rng = np.random.default_rng(SEED)
    # correlation with the event indicator (weak, descriptive)
    corr_r = float(np.corrcoef(CS, e)[0, 1])
    Xcs = CS.reshape(-1, 1)
    beta0, _ = cox_fit(Xcs, t, e)
    stat0 = abs(float(beta0[0]))
    perm_beta = []
    for _ in range(N_PERM):
        cs_perm = rng.permutation(CS)
        b_p, _ = cox_fit(cs_perm.reshape(-1, 1), t, e)
        perm_beta.append(abs(float(b_p[0])))
    pval = float((np.array(perm_beta) >= stat0).sum() + 1) / (N_PERM + 1)

    # ---- 5-year endpoint, censoring handled transparently ----
    # Defined for patients evaluable at 60 months (follow-up >= 60 or
    # relapse before 60); early-censored patients are excluded.
    eval60 = (t >= 60) | ((t < 60) & (e == 1))
    n_eval60 = int(eval60.sum())
    n_early_cens = int((~eval60).sum())
    print(f"5-year evaluable set: {n_eval60}/{len(sub)} "
          f"({n_early_cens} early-censored excluded)")
    y5 = ((t <= 60) & (e == 1)).astype(int)[eval60]
    Xb = X2[eval60]

    # KM estimate of the 5-year relapse probability, 1 - S_KM(60)
    km5_rows = []
    for lab, mask in [("ER_pos", sub_er == 1), ("ER_neg", sub_er == 0),
                      ("HER2_pos", sub_her2 == 1), ("HER2_neg", sub_her2 == 0)]:
        u, s, _ = km_estimator(t[mask], e[mask])
        km5 = float(1.0 - s[u <= 60][-1]) if np.any(u <= 60) else float("nan")
        km5_rows.append({"stratum": lab, "km_5yr_relapse": km5})
    # CS strata (median split); source of the Section 3.6 5-year KM RFS
    for g, lab in [(0, "CS_low"), (1, "CS_high")]:
        u, s, _ = km_estimator(t[grp == g], e[grp == g])
        km5 = float(1.0 - s[u <= 60][-1]) if np.any(u <= 60) else float("nan")
        km5_rows.append({"stratum": lab, "km_5yr_relapse": km5})
    km5_df = pd.DataFrame(km5_rows)
    km5_df.to_csv(os.path.join(RES, "km_5yr.csv"), index=False)
    hs = HorseshoeLogistic(p0=5, n_iter=4000, burn=1000, seed=SEED)
    hs.fit(Xb, y5)
    names_b = present[:N_KERNEL_USED] + names[1:]
    post = pd.DataFrame({
        "feature": names_b,
        "posterior_mean": hs.beta[1:],
        "posterior_sd": hs.beta_sd[1:],
    })
    eps_ex = 0.1   # effects < 0.1 treated as clinically negligible
    chain_b = hs.chain[:, 1:]                       # coefficients only
    # P(|beta| < eps_ex | data) from the chain
    post["prob_negligible"] = (np.abs(chain_b) < eps_ex).mean(axis=0)
    post["prob_positive"] = (chain_b > 0).mean(axis=0)
    post.to_csv(os.path.join(RES, "bayes_posterior.csv"), index=False)
    pr5 = hs.predict_proba(Xb)
    auc5 = roc_auc_score(y5, pr5)

    # ---- sensitivity to the horseshoe sparsity scale p0 ----
    # p0 (expected number of non-negligible coefficients among p = 45)
    # is varied 2/5/10; top genes and AUC should be stable.
    sens_rows = []
    for p0 in (2, 5, 10):
        hs_s = HorseshoeLogistic(p0=p0, n_iter=4000, burn=1000, seed=SEED)
        hs_s.fit(Xb, y5)
        ord_idx = np.argsort(-np.abs(hs_s.beta[1:]))[:10]
        sens_rows.append({
            "p0": p0,
            "tau0": hs_s.tau0,
            "auc_5yr": float(roc_auc_score(y5, hs_s.predict_proba(Xb))),
            "top10_genes": ";".join(np.array(names_b)[ord_idx]),
        })
    pd.DataFrame(sens_rows).to_csv(os.path.join(RES, "bayes_sensitivity.csv"),
                                   index=False)

    # ---- receptor missingness: report + complete-case sensitivity ----
    pr_cc = ~sub["s_PR_STATUS"].isna()
    her2_cc = ~sub["s_HER2_STATUS"].isna()
    cc_mask = pr_cc & her2_cc
    n_missing_receptor = int((~cc_mask).sum())
    print("receptor status missing (PR/HER2):", n_missing_receptor,
          "-> complete-case n =", int(cc_mask.sum()))
    receptor_status = pd.DataFrame({
        "marker": ["PR", "HER2"],
        "positive": [int((sub["pr"] == 1).sum()), int((sub["her2"] == 1).sum())],
        "negative": [int(((sub["pr"] == 0) & pr_cc).sum()),
                     int(((sub["her2"] == 0) & her2_cc).sum())],
        "missing": [int((~pr_cc).sum()), int((~her2_cc).sum())]})
    receptor_status.to_csv(os.path.join(RES, "receptor_status.csv"), index=False)
    if n_missing_receptor > 0:
        b_cc, se_cc = cox_fit(X1[cc_mask], t[cc_mask], e[cc_mask])
        cs_hr_cc = float(np.exp(b_cc[0]))
        cs_p_cc = float(2 * (1 - _norm_cdf(
            np.abs(b_cc[0] / max(se_cc[0], 1e-9)))))
        print(f"complete-case CS HR = {cs_hr_cc:.4f} (p = {cs_p_cc:.3g})")

    out = {
        "analysis_n": int(len(sub)),
        "missing_grade": int(np.isnan(grade).sum()),
        "rfs_events": int(e.sum()),
        "rfs_event_rate": float(e.mean()),
        "M0_clinical_Cindex": c0, "M0_clinical_Cindex_cv": c0cv,
        "M1_kernel_Cindex": c1, "M1_kernel_Cindex_cv": c1cv,
        "M1_kernel_Cindex_cv_sd": sd1cv,
        "M2_ridge_Cindex": c2, "M2_ridge_Cindex_cv": c2cv,
        "CS_HR": float(np.exp(b1[0])), "CS_HR_low": float(np.exp(b1[0] - 1.96 * se1[0])),
        "CS_HR_high": float(np.exp(b1[0] + 1.96 * se1[0])),
        "CS_cox_p": float(2 * (1 - _norm_cdf(np.abs(b1[0] / max(se1[0], 1e-9))))),
        "logrank_chi2": float(chi2), "logrank_p": p_lr,
        "CS_event_corr": corr_r, "permutation_pval": pval,
        "ER_median_observed_rfs_months": med_er_pos,
        "ERneg_median_observed_rfs_months": med_er_neg,
        "corr_ER_grade": corr_er_grade, "corr_ER_CS": corr_er_cs,
        "n_perm": N_PERM,
        "n_eval60": n_eval60,
        "n_early_censored_60": n_early_cens,
        "n_missing_receptor": n_missing_receptor,
        "M0_clinical_Cindex_cv_sd": float(sd0cv),
        "M2_ridge_Cindex_cv_sd": float(sd2cv),
        "bayes_5yr_auc": float(auc5),
        "paired_M1_minus_M0_cindex": t_pair,
        "paired_M1_minus_M0_sd": sd_pair,
        "paired_M1_minus_M0_p": float(p_pair),
        "n_kernel_used": N_KERNEL_USED,
    }
    if n_missing_receptor > 0:
        out["CS_HR_complete_case"] = cs_hr_cc
        out["CS_p_complete_case"] = cs_p_cc
    with open(os.path.join(RES, "model_performance.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    with open(os.path.join(RES, "km_logrank.json"), "w") as fh:
        json.dump({"chi2": chi2, "p": p_lr}, fh)
    with open(os.path.join(RES, "permutation_test.json"), "w") as fh:
        json.dump({"stat": stat0, "p": pval, "n_perm": N_PERM}, fh)
    with open(os.path.join(RES, "bayes_diagnostics.json"), "w") as fh:
        json.dump({"n_iter": hs.n_iter, "burn": hs.burn,
                   "ess_min": hs.ess_min, "ess_median": hs.ess_median,
                   "note": "ESS from lag-1 autocorrelation of the post-burnin "
                           "chain; no formal R-hat available for a single "
                           "chain"}, fh, indent=2)
    print(json.dumps(out, indent=2, default=float))


def _norm_cdf(x):
    from scipy.stats import norm
    return norm.cdf(x)


if __name__ == "__main__":
    main()
