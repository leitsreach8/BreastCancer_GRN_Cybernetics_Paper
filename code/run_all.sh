#!/bin/bash
# run_all.sh - reproduce the full METABRIC analysis pipeline
set -e
cd "$(dirname "$0")/.."
PY=./.venv/bin/python

if [ ! -x "$PY" ]; then
    echo "ERROR: $PY not found; create the environment first:" >&2
    echo "  python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

echo "[1/16] resolve authoritative gene IDs"
$PY code/resolve_genes.py

echo "[2/16] fetch METABRIC clinical + mRNA from cBioPortal"
$PY code/fetch_metabric.py

echo "[3/16] preprocess (z-score, merge survival)"
$PY code/02_preprocess.py

echo "[4/16] network inference (graphical lasso)"
$PY code/03_network_inference.py

echo "[5/16] cybernetic network metrics"
$PY code/04_network_metrics.py

echo "[6/16] control energy"
$PY code/05_control_energy.py

echo "[7/16] dynamical simulation"
$PY code/06_dynamics_simulation.py

echo "[8/16] statistical models (Cox, horseshoe, permutation)"
$PY code/07_statistical_models.py

echo "[9/16] figures (METABRIC; TCGA figures are generated at the end)"
$PY code/08_figures.py

echo "[10/16] full-pipeline cross-validation"
$PY code/09_full_pipeline_cv.py

echo "[11/16] control analyses (equal-weight and random gene sets)"
$PY code/10_controls.py

echo "[12/16] supplementary checks (kernel-size sensitivity, CS-clinical correlations, time-dependent AUC)"
$PY code/12_supplementary_checks.py

echo "[13/16] proportional-hazards check (R survival)"
if command -v Rscript >/dev/null 2>&1; then
    Rscript code/13_ph_check.R
else
    echo "WARNING: Rscript not found; skipping the Schoenfeld proportional-hazards check" >&2
fi

echo "[14/16] TCGA-BRCA cross-cohort transportability assessment (R survival)"
$PY code/11_tcga_validation.py

echo "[15/16] TCGA-BRCA network replication (98-gene panel)"
$PY code/14_tcga_network_replication.py

echo "[16/16] TCGA-BRCA figures (KM curves, network-overlap bars)"
$PY code/15_tcga_figures.py

echo "ALL DONE"
