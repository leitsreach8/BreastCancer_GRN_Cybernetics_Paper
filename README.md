# Breast Cancer GRN: Controllability, Feedback Structure and Prognosis

Reproducible pipeline for the manuscript

> "The Breast Cancer Gene Network as a Complex Adaptive System:
> Controllability, Feedback Structure and Relapse-Free Survival in METABRIC,
> with Cross-Cohort Transportability Assessment in TCGA-BRCA"

## Overview

The pipeline performs the following analyses:

1. Load and preprocess the METABRIC cohort (expression + clinical).
2. Infer a sparse 99-gene conditional-dependence network with the graphical lasso.
3. Compute network-topology and control-theoretic descriptors
   (cycle space, spectral radius, exact controllability, controllability-Gramian
   spectrum, control kernel).
4. Simulate linear and saturating dynamics (stability, fixed-point analysis,
   constant-input perturbation).
5. Construct a patient-level control-kernel score (CS) with outcome-blind,
   structure-based weights.
6. Fit survival models (Cox, Ridge-Cox, Bayesian horseshoe logistic), run
   permutation and log-rank tests, repeated and full-pipeline cross-validation,
   a Schoenfeld proportional-hazards check, time-dependent AUC,
   Kaplan-Meier strata, and generate all figures.
7. Cross-cohort transportability assessment in TCGA-BRCA (overall survival),
   including cross-cohort network-structure replication.

An explicit epistemic distinction is maintained throughout: the inferred
network is an observational conditional-dependence structure; its descriptors
are structural properties of that estimated object; the score is a
control-energy-inspired summary of expression; and any association with
survival is an empirical prognostic association whose generalisability is
tested, not assumed.

## Data sources

- **METABRIC**: gene expression (Illumina HT-12 v3) and clinical annotations,
  downloaded from cBioPortal
  (`https://www.cbioportal.org/study/summary?id=brca_metabric`).
- **TCGA-BRCA** (cross-cohort assessment): RNA-seq (HiSeqV2) and clinical matrix
  from the UCSC Xena platform
  (`https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz`,
  `.../BRCA_clinicalMatrix`).

## Requirements

- Python 3.11+ with numpy, pandas, scipy, scikit-learn, networkx, matplotlib
  (see `requirements.txt`).
- R with the `survival` and `jsonlite` packages (used for the TCGA-BRCA Cox,
  concordance and log-rank analyses, for the TCGA Kaplan-Meier figure, and for
  the Schoenfeld proportional-hazards check).
## Pipeline

Run the scripts in order from the repository root (paths are relative to the
root). `run_all.sh` reproduces the full METABRIC pipeline; the TCGA external
validation is run afterwards.

```
bash code/run_all.sh        # resolve genes, fetch METABRIC, preprocess,
                            # network inference, metrics, dynamics,
                            # statistical models, figures, CV, controls,
                            # supplementary checks, PH check
python code/11_tcga_validation.py   # prepares TCGA data and calls the R script
Rscript code/11_tcga_validation.R    # survival statistics for TCGA-BRCA
```

Scripts 02--14 (driven by `run_all.sh`) reproduce the METABRIC analyses;
script 11 performs the TCGA-BRCA cross-cohort assessment (survival
statistics in R) and script 14 the cross-cohort network-structure
replication. All numerical results are written to `results/` as JSON/CSV
files; all figures are written to `figures/`.

## Key results files

| File | Content |
|---|---|
| `results/grn_meta.json` | network inference meta-data (genes, edges, alpha, bootstrap stability under two edge criteria) |
| `results/network_metrics.json` | topology and control descriptors (incl. zero-eigenvalue multiplicity) |
| `results/control_energy.json` | Gramian spectrum, reachable-subspace dynamic range, effective rank, control kernel |
| `results/dynamics_meta.json` | stability, fixed points, steering energies, displacement fraction |
| `results/model_performance.json` | Cox models, C-indices (full-pipeline CV), CS correlations, paired CV |
| `results/permutation_test.json` | permutation p-value for the CS association |
| `results/km_logrank.json` | Kaplan-Meier strata and log-rank test |
| `results/bayes_posterior.csv` | Bayesian horseshoe posterior summaries (chain diagnostics in `bayes_diagnostics.json`) |
| `results/ph_check.json` | Schoenfeld PH check, CS x log(time) interaction, period-specific HRs |
| `results/kernel_size_sensitivity.csv` | kernel-size sensitivity (K = 20/40/60) |
| `results/cs_clinical_correlations.json` | CS correlations with ER/PR/HER2/grade/age |
| `results/time_auc.csv` | Uno IPCW time-dependent AUC (1/3/5/10 years; in-sample note in JSON) |
| `results/controls_summary.json` | equal-weight and two random-gene null comparisons (same 99-gene panel) |
| `results/tcga_validation.json` | TCGA-BRCA cross-cohort assessment (strict three-state receptor cohort) |
| `results/tcga_receptor_summary.csv` | TCGA receptor-status disclosure (excluded/unknown counts) |
| `results/tcga_network_replication.json` | cross-cohort network overlap |

## Notes

- METABRIC patient-level RFS fields are aligned to samples with an assertion
  in script 02; the analysis set has n = 1,979 after RFS completeness
  filtering.
- The TCGA-BRCA assessment applies a strict three-state definition of
  receptor status (only explicit Positive/Negative calls; missing or
  indeterminate calls are disclosed in `results/tcga_receptor_summary.csv`
  and excluded), leaving n = 710 tumours with 78 OS events for analysis;
  CS shows no association with OS (HR = 1.00, 95% CI 0.80--1.25, p = 0.99).
- Survival statistics for the TCGA assessment and the proportional-hazards
  check are computed with the R `survival` package because the dense
  event-time ties in TCGA make the Python implementation unreliable; the
  METABRIC Python outputs were checked against independent R fits and agree.
- The manuscript reports the TCGA assessment as a negative result by design;
  the code and results files reproduce exactly the numbers reported in the
  manuscript.
