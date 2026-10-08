#!/usr/bin/env python3
"""
10_controls.py - control analyses for the control-kernel score (CS).

The CS is a centrality-weighted average of 60 kernel-gene expressions. To
separate the contribution of the network-based gene selection from the mere
fact that some genes are prognostically informative, we compare the true CS
against two null distributions of random 60-gene scores drawn from the
SAME 99-gene network panel (fixed seed; the whole-transcriptome null used in
an earlier draft was dropped because it draws from a different candidate
universe and is not an apples-to-apples comparison):

  null B  : 2,000 random 60-gene sets from the 99-gene panel, unweighted
            mean expression (selection effect only)
  null C  : 2,000 random 60-gene sets from the 99-gene panel, weighted by
            random rank-based weights (selection and weighting varied
            jointly, matched to the CS construction)

In addition the equal-weight mean of the SAME 60 kernel genes is reported,
so that the weighting effect can be read off directly (CS vs equal-weight
on fixed genes) and the selection effect from nulls B/C (random genes from
the same candidate panel vs the network-selected kernel).

Outputs:
  results/controls_summary.csv      : per-score and per-null statistics
  results/controls_random_null.csv  : null distributions (null_type column)
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
logrank = _statmod.logrank

SEED = 42
N_RANDOM_SETS = 2000
KERNEL_SIZE = 60


def load():
    expr = pd.read_csv(os.path.join(DATA, "metabric_processed.csv"), index_col=0)
    anal = pd.read_csv(os.path.join(DATA, "metabric_analysis.csv"))
    driver = pd.read_csv(os.path.join(RES, "driver_genes.csv"))
    node = pd.read_csv(os.path.join(RES, "node_metrics.csv"))
    anal = anal.copy()
    anal["rfs_time"] = pd.to_numeric(anal["p_RFS_MONTHS"], errors="coerce")
    anal["rfs_event"] = pd.to_numeric(
        anal["p_RFS_STATUS"].astype(str).str.split(":").str[0],
        errors="coerce").fillna(0).astype(int)
    sub = anal.dropna(subset=["rfs_time", "rfs_event"])
    sub = sub[sub["sampleId"].isin(expr.index)]
    t = sub["rfs_time"].values
    e = sub["rfs_event"].values
    genes = driver["gene"].tolist()          # 60-gene control kernel
    present = [g for g in genes if g in expr.columns]
    w = node.set_index("gene").loc[present, "control_score"].values
    Z = expr.loc[sub["sampleId"], present].values
    panel_all = [g for g in node["gene"] if g in expr.columns]   # 99-gene panel
    Zp = expr.loc[sub["sampleId"], panel_all].values
    sub_idxs = sub["sampleId"].values
    return present, w, Z, Zp, panel_all, t, e, list(expr.columns), sub_idxs


def score_stats(score, t, e):
    score = (score - score.mean()) / (score.std() + 1e-8)
    b, se = cox_fit(score.reshape(-1, 1), t, e)
    from scipy.stats import norm
    p_wald = float(2 * (1 - norm.cdf(abs(b[0] / max(se[0], 1e-9)))))
    chi2, p_lr = logrank(t, e, (score > np.median(score)).astype(int))
    return {"coef": float(b[0]), "p_wald": p_wald, "logrank_p": p_lr}


def main():
    present, w, Z, Zp, panel_all, t, e, all_genes, sub_idxs = load()
    rng = np.random.default_rng(SEED)
    panel = np.asarray(panel_all)            # 99-gene panel (kernel candidates)
    P = len(panel)
    assert P == 99, f"expected 99-gene panel, got {P}"
    assert KERNEL_SIZE <= P

    # true CS and equal-weight mean of the same kernel genes
    cs = (Z @ w) / w.sum()
    ew = Z.mean(axis=1)
    rows = [{"score": "CS (centrality-weighted)", **score_stats(cs, t, e)},
            {"score": "Equal-weight (60 kernel genes)", **score_stats(ew, t, e)}]

    w_template = np.linspace(1.0 / KERNEL_SIZE, 1.0, KERNEL_SIZE)

    def random_rank_weights():
        return rng.permutation(w_template)

    null_rows = []

    def run_null(tag, n, gene_pool, Zpool, weight_mode):
        for k in range(n):
            idx = rng.choice(len(gene_pool), size=KERNEL_SIZE, replace=False)
            Zk = Zpool[:, idx]
            if weight_mode == "equal":
                sk = Zk.mean(axis=1)
            else:
                rk = random_rank_weights()
                sk = (Zk @ rk) / rk.sum()
            st = score_stats(sk, t, e)
            null_rows.append({"null_type": tag, "random_set": k, **st})

    # null B: same 99-gene panel, random genes, equal weight
    run_null("panel99_equal", N_RANDOM_SETS, panel, Zp, "equal")
    # null C: same 99-gene panel, random genes, random rank weights
    run_null("panel99_rank", N_RANDOM_SETS, panel, Zp, "rank")

    null_df = pd.DataFrame(null_rows)
    cs_p = rows[0]["logrank_p"]

    summary = {"cs_logrank_p": cs_p, "n_random_sets": N_RANDOM_SETS}
    for tag in sorted(null_df["null_type"].unique()):
        sub_p = null_df.loc[null_df["null_type"] == tag, "logrank_p"].values
        nb = int((sub_p < cs_p).sum())
        summary[f"{tag}_n_better"] = nb
        summary[f"{tag}_frac_better"] = float(nb / len(sub_p))
    # equal-weight kernel (weighting effect) is already in rows

    pd.DataFrame(rows).to_csv(os.path.join(RES, "controls_summary.csv"),
                              index=False)
    null_df.to_csv(os.path.join(RES, "controls_random_null.csv"), index=False)
    with open(os.path.join(RES, "controls_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(pd.DataFrame(rows).to_string(index=False))
    for tag in sorted(null_df["null_type"].unique()):
        nb = summary[f"{tag}_n_better"]
        print(f"{tag}: log-rank p < CS p in {nb} / "
              f"{N_RANDOM_SETS} random sets "
              f"(fraction = {summary[f'{tag}_frac_better']:.3f})")


if __name__ == "__main__":
    main()
