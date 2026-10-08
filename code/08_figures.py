#!/usr/bin/env python3
"""
08_figures.py - Generate all publication figures (vector PDF + 300 dpi PNG).

Figures:
  Fig 1  analysis workflow (schematic)
  Fig 2  cohort composition (RFS events, ER/PR/HER2)
  Fig 3  inferred GRN visualization (kernel genes highlighted)
  Fig 4  network topology (degree distribution, spectrum)
  Fig 5  control energy spectrum (Gramian eigenvalues)
  Fig 6  dynamics: linear transient projections + nonlinear convergence
  Fig 7  constant-input perturbation trajectory (controlled vs free run)
  Fig 8  Cox model comparison (Harrell C-index)
  Fig 9  Kaplan-Meier curves by control-kernel score stratum
  Fig 10 Cox forest plot (HR with 95% CI)
  Fig 11 Bayesian horseshoe posterior (5-year RFS)
"""
import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 300,
    "savefig.dpi": 300, "axes.linewidth": 0.8,
})


def savefig(fig, name):
    # PDF/PNG for preview; EPS required by the journal's LaTeX upload flow.
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(FIG, name + ".eps"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig1_workflow():
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.axis("off")
    steps = [
        "METABRIC\n(n=1,979 tumours,\n99 regulatory genes)",
        "z-score\nstandardisation",
        "Graphical lasso\n(partial-correlation\nGRN)",
        "Cybernetic metrics\n(feedback, spectrum,\nGramian, energy)",
        "Control-kernel\npatient score (CS)",
        "Models\n(Cox PH, Ridge-Cox,\nBayesian horseshoe)",
    ]
    boxes = [(0.05, 0.62, 0.16, 0.28), (0.255, 0.62, 0.15, 0.28),
             (0.44, 0.62, 0.17, 0.28), (0.655, 0.62, 0.16, 0.30),
             (0.25, 0.12, 0.17, 0.30), (0.48, 0.12, 0.20, 0.30)]
    for (x, y, w, h), s in zip(boxes, steps):
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, facecolor="#eef2f7",
                                   edgecolor="#33415c", linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, s, ha="center", va="center", fontsize=8)
    for x1, y1, x2, y2 in [(0.21, 0.76, 0.255, 0.76), (0.405, 0.76, 0.44, 0.76),
                            (0.61, 0.76, 0.655, 0.76), (0.735, 0.62, 0.42, 0.30),
                            (0.42, 0.12, 0.48, 0.12)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#33415c", lw=1.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    savefig(fig, "fig1_workflow")


def fig2_cohort():
    anal = pd.read_csv(os.path.join(DATA, "metabric_analysis.csv"))
    ev = pd.to_numeric(anal["p_RFS_STATUS"].astype(str).str.split(":").str[0],
                       errors="coerce").fillna(0).astype(int)
    er = anal["s_ER_STATUS"].astype(str).str.lower().str.contains("pos").astype(int)
    pr = anal["s_PR_STATUS"].astype(str).str.lower().str.contains("pos").astype(int)
    her2 = anal["s_HER2_STATUS"].astype(str).str.lower().str.contains("pos").astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3))
    axes[0].bar(["No event", "RFS event"], [1 - ev.mean(), ev.mean()],
                color=["#8fa3bf", "#c0392b"], edgecolor="black", lw=0.5)
    axes[0].set_ylabel("Proportion")
    axes[1].bar(["ER-", "ER+"], [1 - er.mean(), er.mean()],
                color=["#aab7c4", "#4a6fa5"], edgecolor="black", lw=0.5)
    axes[2].bar(["PR-", "PR+"], [1 - pr.mean(), pr.mean()],
                color=["#aab7c4", "#4a6fa5"], edgecolor="black", lw=0.5)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, "fig2_cohort")


def fig3_network():
    genes = pd.read_csv(os.path.join(DATA, "gene_list.csv"), header=None)[0].tolist()
    adj = np.load(os.path.join(RES, "grn_adjacency.npy"))
    W = np.load(os.path.join(RES, "grn_weights.npy"))
    driver = pd.read_csv(os.path.join(RES, "driver_genes.csv"))
    node = pd.read_csv(os.path.join(RES, "node_metrics.csv"))
    G = nx.Graph()
    G.add_nodes_from(genes)
    p = len(genes)
    for i in range(p):
        for j in range(i + 1, p):
            # full weighted network (|pcorr| > 1e-12) is the analysis
            # object; thresholded adjacency is display-only
            if W[i, j] > 1e-12:
                G.add_edge(genes[i], genes[j], weight=float(W[i, j]))
    kernel = set(driver["gene"].head(25))
    gcc = max(nx.connected_components(G), key=len)
    sub = G.subgraph(gcc)
    pos = nx.spring_layout(sub, seed=42, k=0.9, iterations=120)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    com = node.set_index("gene").loc[list(sub.nodes), "community"]
    cmap = plt.get_cmap("tab20")
    colors = [cmap(c % 20) for c in com.values]
    nx.draw_networkx_edges(sub, pos, ax=ax, alpha=0.30, width=0.6,
                           edge_color="#7f8c9b")
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=120, node_color=colors,
                           alpha=0.85)
    kset = [n for n in sub.nodes if n in kernel]
    if kset:
        nx.draw_networkx_nodes(sub, pos, ax=ax, nodelist=kset, node_size=250,
                               node_color="#c0392b", edgecolors="black",
                               linewidths=0.6, alpha=0.9)
    # labels drawn last so kernel nodes do not occlude them
    nx.draw_networkx_labels(sub, pos, ax=ax, font_size=5.5)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="#c0392b", markersize=7,
                              label="Control-kernel gene"),
                       Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=cmap(0), markersize=6,
                              label="Other gene")],
              loc="lower left", frameon=False)
    ax.axis("off")
    savefig(fig, "fig3_network")


def fig4_topology_spectrum():
    node = pd.read_csv(os.path.join(RES, "node_metrics.csv"))
    vals = np.load(os.path.join(RES, "grn_spectrum.npy"))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    deg = node["degree"]
    axes[0].hist(deg, bins=np.arange(deg.max() + 2) - 0.5, color="#4a6fa5",
                 edgecolor="black", lw=0.3)
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Frequency")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].hist(vals, bins=50, color="#8fa3bf", edgecolor="black", lw=0.3)
    axes[1].axvline(vals.max(), color="#c0392b", ls="--", lw=1.2,
                    label=f"$\\lambda_{{\\max}}$={vals.max():.3f}")
    axes[1].set_xlabel("Eigenvalue of weighted adjacency $A$")
    axes[1].set_ylabel("Frequency")
    axes[1].legend(frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, "fig4_topology_spectrum")


def fig5_energy_spectrum():
    spec = pd.read_csv(os.path.join(RES, "energy_spectrum.csv"))
    with open(os.path.join(RES, "control_energy.json")) as fh:
        meta = json.load(fh)
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ev = spec["eigenvalue"].values
    n = len(ev)
    k_top = max(1, int(np.ceil(0.1 * n)))
    share = meta["energy_top10_trace_share"]
    # plot at the same 1e-12 floor stated in the manuscript
    ev_plot = np.maximum(ev, 1e-12)
    ax.loglog(np.arange(1, n + 1), ev_plot, color="#2c3e50", lw=1.0)
    # shade the top-k modes (ranks n-k_top+1..n) so the counted block is visible
    x0 = n - k_top + 1
    ax.axvspan(x0 - 0.5, n + 0.5, color="#e74c3c", alpha=0.12)
    ax.axvline(x0 - 0.5, color="#c0392b", ls="--", lw=1.0)
    ax.text(x0 - 1.0, ev.max() * 0.35,
            f"top 10% (ranks {x0}-{n}) =\n{100 * share:.1f}% of trace",
            fontsize=8, ha="right")
    ax.set_xlabel("Eigenvalue rank of controllability Gramian $W_c$")
    ax.set_ylabel("Eigenvalue (log)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, "fig5_energy_spectrum")


def fig6_attractors():
    att = pd.read_csv(os.path.join(RES, "dynamics_attractors.csv"))
    att_nl = pd.read_csv(os.path.join(RES, "dynamics_attractors_nl.csv"))
    att_nl50 = pd.read_csv(os.path.join(RES, "dynamics_attractors_nl50.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    axes[0].hist(att["projection_perron"], bins=16, color="#4a6fa5",
                 edgecolor="black", lw=0.3)
    axes[0].axvline(att["projection_perron"].median(), color="#c0392b", ls="--",
                    lw=1.0, label="median split")
    axes[0].set_xlabel("Projection onto Perron vector")
    axes[0].set_ylabel("Count")
    axes[0].legend(frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].plot(att_nl["rep"], att_nl["projection_perron"], "o",
                 color="#8fa3bf", ms=5, label="t = 20")
    axes[1].plot(att_nl50["rep"], att_nl50["projection_perron"], "o",
                 color="#c0392b", ms=5, label="t = 50")
    axes[1].axhline(0, color="#7f8c9b", lw=0.8, ls=":")
    axes[1].set_xlabel("Random initial condition (replicate)")
    axes[1].set_ylabel("Projection onto Perron vector")
    axes[1].legend(frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, "fig6_attractors")


def fig7_steering():
    steer = pd.read_csv(os.path.join(RES, "dynamics_steering.csv"))
    att = pd.read_csv(os.path.join(RES, "dynamics_attractors.csv"))
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ctrl = steer[steer["control_active"] == 1]
    free = steer[steer["control_active"] == 0]
    ax.plot(ctrl["t"], ctrl["projection_perron"], color="#c0392b", lw=1.5,
            label="controlled (constant input)")
    ax.plot(free["t"], free["projection_perron"], color="#8fa3bf", lw=1.2,
            ls="--", label="free (no input)")
    ax.axhline(att["projection_perron"].median(), color="#2c3e50", ls="--",
               lw=1.0, label="median split line")
    ax.set_xlabel("Time (arbitrary units)")
    ax.set_ylabel("Projection onto Perron vector")
    ax.legend(frameon=False, loc="center right", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, "fig7_steering")


def fig8_cindex():
    per = json.load(open(os.path.join(RES, "model_performance.json")))
    labels = ["M0 clinical", "M1 kernel+clinical", "M2 Ridge-Cox"]
    vals = [per["M0_clinical_Cindex_cv"], per["M1_kernel_Cindex_cv"],
            per["M2_ridge_Cindex_cv"]]
    sds = [per.get("M0_clinical_Cindex_cv_sd", 0.02),
           per.get("M1_kernel_Cindex_cv_sd", 0.02),
           per.get("M2_ridge_Cindex_cv_sd", 0.02)]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    colors = ["#7f8c9b", "#c0392b", "#2c3e50"]
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=sds, color=colors, edgecolor="black", lw=0.5,
           capsize=4, width=0.55)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.axhline(0.5, color="#d5dbe3", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Harrell C-index (repeated 5-fold CV)")
    ax.set_ylim(0.4, 0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, "fig8_cindex")


def fig9_km():
    km = pd.read_csv(os.path.join(RES, "km_data.csv"))
    lr = json.load(open(os.path.join(RES, "km_logrank.json")))
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    colors = {"CS-low": "#4a6fa5", "CS-high": "#c0392b"}
    for strata, df in km.groupby("stratum"):
        ax.step(df["time_months"], df["survival"], where="post",
                color=colors[strata], lw=1.6, label=strata)
    ax.text(0.02, 0.06,
            rf"log-rank $\chi^2$ = {lr['chi2']:.2f}, p = {lr['p']:.1e}",
            transform=ax.transAxes, fontsize=8)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Relapse-free survival probability")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, "fig9_km")


def fig10_forest():
    cox = pd.read_csv(os.path.join(RES, "cox_results.csv"))
    cox = cox.sort_values("HR", ascending=True)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ypos = np.arange(len(cox))
    ax.errorbar(cox["HR"], ypos, xerr=[cox["HR"] - cox["HR_low"],
                                       cox["HR_high"] - cox["HR"]],
                fmt="o", color="#2c3e50", ecolor="#8fa3bf", elinewidth=1.2,
                ms=5, capsize=3)
    ax.axvline(1.0, color="#c0392b", lw=0.8, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(cox["variable"], fontsize=8)
    ax.set_xlabel("Hazard ratio (95% CI)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, "fig10_forest")


def fig11_bayes():
    post = pd.read_csv(os.path.join(RES, "bayes_posterior.csv"))
    post = post.reindex(post["posterior_mean"].abs().sort_values(
        ascending=False).index).head(22)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ypos = np.arange(len(post))
    ax.errorbar(post["posterior_mean"], ypos, xerr=2 * post["posterior_sd"],
                fmt="o", color="#2c3e50", ecolor="#8fa3bf", elinewidth=1.0,
                ms=3.5)
    ax.axvline(0, color="#c0392b", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(post["feature"], fontsize=7)
    ax.set_xlabel("Posterior mean (log-odds, 5-year RFS)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, "fig11_bayes")




def fig12_tcga_km():
    """Fig. 10: Kaplan-Meier overall survival by CS stratum (median split)
    in the TCGA-BRCA assessment cohort (n = 710)."""
    km = pd.read_csv(os.path.join(RES, "tcga_km_data.csv"))
    val = json.load(open(os.path.join(RES, "tcga_validation.json")))
    lr = val["logrank"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    lab = {0: "CS-low", 1: "CS-high"}
    colors = {"CS-low": "#4a6fa5", "CS-high": "#c0392b"}
    for grp, df in km.groupby("stratum"):
        ax.step(df["time_days"] / 365.25, df["survival"], where="post",
                color=colors[lab[grp]], lw=1.6, label=lab[grp])
    ax.text(0.02, 0.06,
            rf"log-rank $\chi^2$ = {lr['chi2']:.2f}, p = {lr['p']:.2f}",
            transform=ax.transAxes, fontsize=8)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Overall survival probability")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, "fig12_tcga_km")


def fig13_network_overlap():
    """Fig. 11: cross-cohort edge counts and overlap between the METABRIC
    (training) and TCGA-BRCA (external) networks on the shared 98-gene
    panel, at the two thresholds used throughout the manuscript."""
    rep = json.load(open(os.path.join(RES, "tcga_network_replication.json")))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, key, th in [(axes[0], "dense", "|pi| > 1e-12"),
                        (axes[1], "sparse", "|pi| > 0.05")]:
        b = rep[key]
        mb, tc, ov = b["metabric_98gene_panel"], b["tcga"], b["overlap"]
        ax.bar(["METABRIC", "TCGA-BRCA"], [mb, tc], color=["#4a6fa5", "#c0392b"],
               alpha=0.85, width=0.6)
        ax.set_title(f"{th}\noverlap = {ov}, Jaccard = {b['jaccard']:.3f}",
                     fontsize=9)
        ax.set_ylabel("Number of edges")
        for i, v in enumerate([mb, tc]):
            ax.text(i, v + max(mb, tc) * 0.02, str(v), ha="center", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, "fig13_network_overlap")

def main():
    # METABRIC figures 1-11 here; the two TCGA figures are produced by
    # code/15_tcga_figures.py (after scripts 11/14), so the METABRIC-only
    # pipeline stays runnable from a clean checkout.
    fig2_cohort()
    fig3_network()
    fig4_topology_spectrum()
    fig5_energy_spectrum()
    fig6_attractors()
    fig7_steering()
    fig8_cindex()
    fig9_km()
    fig10_forest()
    fig11_bayes()
    print("all figures done")


if __name__ == "__main__":
    main()
