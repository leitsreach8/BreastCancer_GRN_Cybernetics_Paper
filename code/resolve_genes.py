#!/usr/bin/env python3
"""
resolve_genes.py - Resolve the a-priori breast-cancer regulatory gene set to
authoritative cBioPortal entrez IDs (using the downloaded full gene table),
so that every ID used in downstream analysis is verified against the portal.

Writes data/gene_map.csv (symbol, entrez_gene_id).
"""
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

# A-priori regulatory gene set (symbols only; IDs are resolved authoritatively)
SYMBOLS = [
    # endocrine / ER signalling
    "ESR1", "ESR2", "PGR", "AR", "TFF1",
    # growth factor / ERBB signalling
    "ERBB2", "ERBB3", "EGFR", "EGF", "HBEGF", "IGF1R", "GRB2", "SHC1",
    "NRAS", "KRAS", "BRAF", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3",
    # PI3K / AKT / mTOR
    "PIK3CA", "PIK3R1", "PIK3R2", "AKT1", "AKT2", "AKT3", "PTEN", "MTOR",
    "RPS6KB1", "RPS6KA1", "RPS6KA2", "RPS6KA3",
    # cell cycle / proliferation
    "MKI67", "TOP2A", "CCNA2", "CCNB1", "CCNB2", "CCND1", "CCND2", "CCNE1",
    "CDK1", "CDC20", "CDC25A", "CDC25B", "CDC25C", "BUB1", "AURKA", "PLK1",
    "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "MYBL2", "CENPI",
    # TP53 / apoptosis
    "TP53", "MDM2", "CDKN1A", "BAX", "BAD", "BCL2", "CASP3", "CASP8",
    "CASP9", "ATM",
    # EMT / adhesion / invasion
    "CDH1", "CDH2", "CDH3", "FN1", "L1CAM", "TUBB3",
    # angiogenesis / hypoxia
    "VEGFA", "VEGFB", "VEGFC", "HIF1A", "HIF1AN", "CXCR4", "CXCL8",
    # JAK-STAT / immune
    "JAK1", "JAK2", "JAK3", "STAT1", "STAT3", "STAT5A", "STAT5B",
    # FGFR / retinoic acid
    "FGFR1", "FGFR2", "FGFR3", "FGFR4", "RARA", "RARB", "RXRA",
    # misc growth regulators
    "MYC", "IGFBP2", "IGFBP3", "ERRFI1", "RASA1", "CXCR2",
]


def main():
    with open(os.path.join(DATA, "cbioportal_genes.json")) as fh:
        genes = json.load(fh)
    sym2id = {}
    for g in genes:
        sym = g["hugoGeneSymbol"]
        if sym not in sym2id:
            sym2id[sym] = g["entrezGeneId"]

    resolved, missing = [], []
    for sym in SYMBOLS:
        if sym in sym2id:
            resolved.append((sym, sym2id[sym]))
        else:
            missing.append(sym)

    assert len(resolved) == len(SYMBOLS), (len(resolved), len(SYMBOLS))
    assert len(resolved) == 99, len(resolved)   # the 99 a-priori panel genes
    print(f"resolved {len(resolved)} / {len(SYMBOLS)}")
    if missing:
        print("missing:", missing)
    with open(os.path.join(DATA, "gene_map.csv"), "w") as fh:
        fh.write("symbol,entrez_gene_id\n")
        for sym, gid in resolved:
            fh.write(f"{sym},{gid}\n")
    print("wrote data/gene_map.csv")


if __name__ == "__main__":
    main()
