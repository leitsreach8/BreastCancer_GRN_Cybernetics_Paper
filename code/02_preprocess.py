#!/usr/bin/env python3
"""
02_preprocess.py - Build the METABRIC analysis set.

Merges sample-level clinical data, patient-level survival data and the mRNA
expression matrix (99 a-priori regulatory genes, 1980 samples; 1,979 after
RFS completeness filtering), applies
per-gene z-score standardisation, and saves the analysis-ready tables.

Outputs:
  data/metabric_processed.csv   : z-scored expression (samples x genes)
  data/metabric_analysis.csv    : merged clinical + survival table
  data/gene_list.csv            : gene symbols (order matches columns)
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")


def main():
    expr = pd.read_csv(os.path.join(DATA, "metabric_expr.csv"), index_col=0)
    clin_s = pd.read_csv(os.path.join(DATA, "metabric_clinical.csv"))
    clin_p = pd.read_csv(os.path.join(DATA, "metabric_patient_clinical.csv"))

    # ---- restrict to samples with complete mRNA data ----
    complete = expr.notna().all(axis=1)
    expr = expr.loc[complete]
    print(f"samples with full mRNA: {expr.shape[0]}")

    # ---- z-score per gene ----
    mu = expr.mean(axis=0)
    sd = expr.std(axis=0, ddof=0).replace(0, 1e-9)
    expr_z = (expr - mu) / sd
    expr_z.to_csv(os.path.join(DATA, "metabric_processed.csv"))
    expr_z.columns.to_series().to_csv(os.path.join(DATA, "gene_list.csv"),
                                      index=False, header=False)
    print("z-scored expression:", expr_z.shape)

    # ---- merge clinical ----
    clin_s = clin_s.set_index("sampleId")
    clin_p = clin_p.set_index("patientId")
    merged = pd.DataFrame(index=expr_z.index)
    for df, prefix in [(clin_s, "s_"), (clin_p, "p_")]:
        for col in df.columns:
            if col in ("sampleId", "patientId"):
                continue
            merged[f"{prefix}{col}"] = df[col]
    merged = merged.reset_index().rename(columns={"index": "sampleId"})
    merged["patientId"] = merged["sampleId"]  # METABRIC: sample == patient id
    # survival analysis requires complete RFS status and follow-up time;
    # samples with a missing RFS status are excluded here rather than
    # silently treated as censored downstream
    before = len(merged)
    merged = merged.dropna(subset=["p_RFS_STATUS", "p_RFS_MONTHS"])
    if len(merged) < before:
        print(f"excluded {before - len(merged)} sample(s) with incomplete RFS "
              f"status/time; analysis sample n = {len(merged)}")
    merged.to_csv(os.path.join(DATA, "metabric_analysis.csv"), index=False)
    # verification of the patient-level alignment: METABRIC sample IDs equal
    # patient IDs, so patient-level columns must match the sample index
    # without loss; any mismatch would silently shrink the analysis sample
    n_rfs = int(merged["p_RFS_MONTHS"].notna().sum())
    n_rfs_stat = int(merged["p_RFS_STATUS"].notna().sum())
    print("patient-level match: p_RFS_MONTHS", n_rfs, "/", len(merged),
          "; p_RFS_STATUS", n_rfs_stat, "/", len(merged))
    assert n_rfs == len(merged), "patient-level RFS months must fully align"
    assert n_rfs_stat == len(merged), "patient-level RFS status must fully align"
    print("merged analysis table:", merged.shape)
    print("cols:", list(merged.columns))


if __name__ == "__main__":
    main()
