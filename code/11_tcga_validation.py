"""
11_tcga_validation.py - External validation of the control-kernel score in
TCGA-BRCA (overall survival).

The control-kernel score (CS) is computed exactly as in the METABRIC main
analysis: CS_s = sum_k c_k z_sk / sum_k c_k, where c_k are the structure-
based, outcome-blind control scores of the 60 control-kernel genes inferred
on METABRIC (results/driver_genes.csv), and z_sk are per-cohort z-scored
expression values in TCGA. The kernel genes and weights are therefore fixed
from the training cohort and never touch TCGA survival data during
construction.

Endpoints: TCGA-BRCA has no curated relapse-free-survival endpoint; we
validate against overall survival (days) constructed from vital status and
follow-up days. This is stated explicitly in the manuscript as a
prognostic-association validation, not an endpoint-identical replication.

This script prepares the analysis table and delegates all survival
statistics to code/11_tcga_validation.R (R survival::coxph, Efron ties;
concordance; survdiff; survfit), writing results/tcga_validation.json.

Models:
  M0  : ER + PR + HER2(IHC) + age
  M1  : M0 + CS
  CSu : univariable CS
"""

import os
import json
import gzip
import shutil
import subprocess
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "TCGA-BRCA")
RESULTS = os.path.join(ROOT, "results")


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_genes():
    genes = pd.read_csv(os.path.join(ROOT, "supplementary", "Table_S1_genes.csv"))
    return genes["gene_symbol"].tolist()


def load_kernel():
    drv = pd.read_csv(os.path.join(RESULTS, "driver_genes.csv"))
    return drv["gene"].tolist(), drv["control_score"].astype(float).values


def load_tcga_expression(genes):
    """Read HiSeqV2.gz (rows = genes, columns = samples), keep requested genes."""
    wanted = set(genes)
    expr = {}
    header = None
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
        raise RuntimeError("no requested genes found in expression matrix")
    df = pd.DataFrame(expr).T
    df.columns = header[1:]
    return df


def load_tcga_clinical():
    clin = pd.read_csv(os.path.join(DATA, "BRCA_clinicalMatrix.tsv"),
                       sep="\t", index_col=0)
    return clin


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    genes = load_genes()
    kernel, ck = load_kernel()
    print(f"99-gene list: {len(genes)}; kernel: {len(kernel)}")

    clin_path = os.path.join(DATA, "BRCA_clinicalMatrix.tsv")
    if not os.path.exists(clin_path):
        raise RuntimeError(
            "data/TCGA-BRCA/BRCA_clinicalMatrix.tsv not found; the TCGA "
            "clinical matrix must be placed in the repository (see README)")

    expr = load_tcga_expression(genes)
    clin = load_tcga_clinical()
    print(f"expression: {expr.shape[0]} genes x {expr.shape[1]} samples; "
          f"clinical rows: {len(clin)}")

    # tumour samples only (Xena parses sample_type_id as integer 1)
    tum = clin[clin["sample_type_id"] == 1].copy()
    print(f"tumour samples: {len(tum)}")
    common = sorted(set(expr.columns) & set(tum.index))
    print(f"samples with expression and clinical: {len(common)}")
    E = expr[common].T  # samples x genes
    tum = tum.loc[common]

    # clinical covariates with STRICT three-state receptor coding: a receptor
    # is coded 1 (positive) / 0 (negative) only when the IHC field explicitly
    # reads "Positive"/"Negative"; Indeterminate, Equivocal and missing
    # entries are treated as unknown and excluded (they are NOT collapsed
    # into the negative group). The counts are reported in
    # results/tcga_receptor_summary.csv.
    def state3(s):
        s = s.astype(str).str.strip()
        pos = (s == "Positive").astype(float)
        neg = (s == "Negative").astype(float)
        return pos.where((s == "Positive") | (s == "Negative"))

    er = state3(tum["breast_carcinoma_estrogen_receptor_status"])
    pr = state3(tum["breast_carcinoma_progesterone_receptor_status"])
    her2 = state3(tum["lab_proc_her2_neu_immunohistochemistry_receptor_status"])
    age = pd.to_numeric(tum["age_at_initial_pathologic_diagnosis"],
                        errors="coerce")

    # receptor-state disclosure table (see Section 3.7)
    def state_counts(s):
        s = s.astype(str).str.strip()
        return {"positive": int((s == "Positive").sum()),
                "negative": int((s == "Negative").sum()),
                "unknown": int((~s.isin(["Positive", "Negative"])).sum())}
    pd.DataFrame({
        "marker": ["ER", "PR", "HER2"],
        **{k: [state_counts(tum[c])[k] for c in [
            "breast_carcinoma_estrogen_receptor_status",
            "breast_carcinoma_progesterone_receptor_status",
            "lab_proc_her2_neu_immunohistochemistry_receptor_status"]]
           for k in ["positive", "negative", "unknown"]}
    }).to_csv(os.path.join(RESULTS, "tcga_receptor_summary.csv"), index=False)

    # overall survival constructed from vital status and follow-up days
    # (the Xena clinical matrix has no OS_Time/OS_event columns for BRCA)
    vital = tum["vital_status"].astype(str).str.strip().str.upper()
    d2d = pd.to_numeric(tum["days_to_death"], errors="coerce")
    d2f = pd.to_numeric(tum["days_to_last_followup"], errors="coerce")
    os_event = (vital == "DECEASED").astype(int)
    os_event = os_event.where(vital.isin(["DECEASED", "LIVING"]))
    os_time = pd.Series(np.where(os_event == 1, d2d, d2f), index=tum.index)
    os_time = os_time.where(os_event.notna())
    print(f"vital status present: {vital.isin(['DECEASED','LIVING']).sum()} "
          f"(deceased {int((vital=='DECEASED').sum())})")

    # complete cases with positive follow-up time: receptor unknowns are
    # excluded here (strict coding), not encoded as negative
    cc_pre = pd.DataFrame({"ER": er, "PR": pr, "HER2": her2, "age": age,
                           "t": os_time, "e": os_event})
    cc = cc_pre.dropna().copy()
    cc = cc[cc["t"] > 0]
    print(f"complete cases (strict ER+PR+HER2+age+OS, t>0): {len(cc)}")

    # z-score per gene WITHIN the retained cohort (after receptor/OS
    # filtering), so that the z-transformation does not borrow information
    # from samples that are later excluded from the analysis
    Ecc = E.loc[cc.index]
    Ez = (Ecc - Ecc.mean(axis=0)) / Ecc.std(axis=0)

    # kernel genes present in the retained cohort
    k_present = [g for g in kernel if g in Ez.columns]
    ck_present = ck[[i for i, g in enumerate(kernel) if g in Ez.columns]]
    missing_k = [g for g in kernel if g not in Ez.columns]
    print(f"kernel genes present in TCGA: {len(k_present)}/{len(kernel)}"
          + (f" (missing: {missing_k})" if missing_k else ""))

    # CS (METABRIC weights, outcome-blind)
    CS = (Ez[k_present].values @ ck_present) / ck_present.sum()
    CS = (CS - CS.mean()) / CS.std()

    cc["CS"] = CS
    er, pr, her2 = cc["ER"], cc["PR"], cc["HER2"]
    age = cc["age"]
    print(f"events: {int(cc['e'].sum())} ({100*cc['e'].mean():.1f}%)")
    print(f"OS time range: {cc['t'].min():.0f}-{cc['t'].max():.0f} days "
          f"(median {cc['t'].median():.0f})")
    print(f"CS: median {cc['CS'].median():.3f}, IQR "
          f"[{cc['CS'].quantile(0.25):.3f}, {cc['CS'].quantile(0.75):.3f}]")

    inp = os.path.join(RESULTS, "tcga_input.csv")
    cc.to_csv(inp, index=False)

    # delegate survival statistics to R (survival::coxph, Efron ties)
    rscript = os.path.join(ROOT, "code", "11_tcga_validation.R")
    subprocess.run(["Rscript", rscript, inp], check=True)
    with open(os.path.join(RESULTS, "tcga_validation.json")) as f:
        out = json.load(f)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
