#!/usr/bin/env python3
"""
fetch_metabric.py - Pull METABRIC (brca_metabric) clinical data and mRNA
expression for the resolved a-priori regulatory gene set from the cBioPortal
REST API (gene-level GET requests, robust retries).

Outputs (data/):
  metabric_clinical.csv          : sample-level clinical table (wide)
  metabric_patient_clinical.csv  : patient-level clinical table (wide),
                                   incl. age, RFS/OS time and status
  metabric_expr.csv              : mRNA expression matrix (samples x genes)

If metabric_expr.csv already exists, expression is not re-fetched and only
the two clinical tables are (re)written.
"""
import json
import os
import time
import urllib.request
import urllib.error
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
API = "https://api.cbioportal.org/api"
STUDY = "brca_metabric"
PROFILE = "brca_metabric_mrna"
SAMPLE_LIST = "brca_metabric_mrna"
RETRIES = 6
SLEEP = 3.0


def http_get_json(url: str, retries: int = RETRIES):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url),
                                        timeout=90) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            print(f"  retry {attempt + 1}/{retries} {url[:80]}: {exc}",
                  flush=True)
            time.sleep(SLEEP)
    raise RuntimeError(f"failed after {retries} retries: {url[:80]}")


def fetch_clinical():
    rows = []
    page, size = 0, 1000
    while True:
        url = (f"{API}/studies/{STUDY}/clinical-data"
               f"?clinicalDataType=SAMPLE&pageSize={size}&pageNumber={page}")
        batch = http_get_json(url)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < size:
            break
        page += 1
        if page > 60:
            break
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index=["sampleId", "patientId"],
                          columns="clinicalAttributeId", values="value",
                          aggfunc="first").reset_index()
    return wide


def fetch_patient_clinical():
    rows = []
    page, size = 0, 1000
    while True:
        url = (f"{API}/studies/{STUDY}/clinical-data"
               f"?clinicalDataType=PATIENT&pageSize={size}&pageNumber={page}")
        batch = http_get_json(url)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < size:
            break
        page += 1
        if page > 60:
            break
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index=["patientId"],
                          columns="clinicalAttributeId", values="value",
                          aggfunc="first").reset_index()
    return wide


def fetch_gene(gene_id: int):
    url = (f"{API}/molecular-profiles/{PROFILE}/molecular-data"
           f"?sampleListId={SAMPLE_LIST}&entrezGeneId={gene_id}")
    data = http_get_json(url)
    if not data:
        return {}
    return {d["sampleId"]: d.get("value") for d in data}


def main():
    os.makedirs(DATA, exist_ok=True)
    gene_map = pd.read_csv(os.path.join(DATA, "gene_map.csv"))

    clin_path = os.path.join(DATA, "metabric_clinical.csv")
    pclin_path = os.path.join(DATA, "metabric_patient_clinical.csv")
    expr_path = os.path.join(DATA, "metabric_expr.csv")
    if all(os.path.exists(p) for p in (clin_path, pclin_path, expr_path)):
        print("all data files already present; nothing to fetch", flush=True)
        return

    if os.path.exists(clin_path):
        clin = pd.read_csv(clin_path)
        print(f"sample clinical already present: {len(clin)} rows",
              flush=True)
    else:
        clin = fetch_clinical()
        print(f"sample clinical: {len(clin)} rows", flush=True)
        clin.to_csv(clin_path, index=False)

    if os.path.exists(pclin_path):
        pclin = pd.read_csv(pclin_path)
        print(f"patient clinical already present: {len(pclin)} rows",
              flush=True)
    else:
        pclin = fetch_patient_clinical()
        print(f"patient clinical: {len(pclin)} rows", flush=True)
        pclin.to_csv(pclin_path, index=False)

    if os.path.exists(expr_path):
        print("expression matrix already present; skipping re-fetch",
              flush=True)
        return
    samples = list(clin["sampleId"])
    expr = {}
    for row in gene_map.itertuples():
        vals = fetch_gene(row.entrez_gene_id)
        if not vals:
            print(f"  no data: {row.symbol}", flush=True)
            continue
        expr[row.symbol] = [vals.get(s, np.nan) for s in samples]
        print(f"  ok: {row.symbol} (n={len(vals)})", flush=True)
        time.sleep(0.3)

    mat = pd.DataFrame(expr, index=samples)
    mat.index.name = "sampleId"
    mat.to_csv(os.path.join(DATA, "metabric_expr.csv"))
    print(f"expression matrix: {mat.shape} (samples x genes)", flush=True)


if __name__ == "__main__":
    main()
