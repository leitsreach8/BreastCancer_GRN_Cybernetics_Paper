#!/usr/bin/env python3
"""
15_tcga_figures.py - TCGA-BRCA figures for the cross-cohort assessment.

Generates:
  Fig 12  TCGA-BRCA Kaplan-Meier curves (CS-low vs CS-high, OS)
  Fig 13  cross-cohort network-overlap bars (dense vs sparse threshold)

Run AFTER code/11_tcga_validation.py + code/11_tcga_validation.R and
code/14_tcga_network_replication.py have produced their result files.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from importlib import import_module

fig08 = import_module("08_figures")


def main():
    fig08.fig12_tcga_km()
    fig08.fig13_network_overlap()
    print("TCGA figures done")


if __name__ == "__main__":
    main()
