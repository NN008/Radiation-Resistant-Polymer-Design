#!/usr/bin/env python3
"""
Build tidy CSVs for plotting from your existing run outputs.
Matches your layout under Runs/ and 4Validation/Outputs/*.
"""

import pandas as pd, numpy as np, glob, json
from pathlib import Path

TG_THRESH  = 215.0
MAC_THRESH = 0.0569

ROOT = Path(".")
RUNS = sorted((ROOT/"Runs").glob("loop_*"))
VALS = sorted((ROOT/"4Validation/Outputs").glob("Out*"))
LOGS = ROOT/"logs"
LOGS.mkdir(exist_ok=True)

# ----------------- helpers: de-duplicate / sanitize -----------------

def dedup_cols(cols):
    """Make column names unique: ['a','a','b'] -> ['a','a.1','b']"""
    seen, out = {}, []
    for c in cols:
        c = str(c)
        if c in seen:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out

def sanitize_df(df):
    """Ensure unique columns and clean index."""
    if df is None or df.empty:
        return df
    df.columns = dedup_cols([str(c) for c in df.columns])
    # drop exact duplicate columns if any
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    return df.reset_index(drop=True)

# ----------------- io + normalizers -----------------

def norm_cols(df):
    """Rename common columns to: pred_Tg, pred_MAC, novelty, cluster_id, control_Tg, control_MAC, score, pred_int_width."""
    df = sanitize_df(df)
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if "tg" in lc and ("pred" in lc or "hat" in lc): ren[c] = "pred_Tg"
        if "mac" in lc and ("pred" in lc or "hat" in lc): ren[c] = "pred_MAC"
        if "novel" in lc: ren[c] = "novelty"
        if "cluster" in lc or "butina" in lc: ren[c] = "cluster_id"
        if "control" in lc and "tg" in lc: ren[c] = "control_Tg"
        if "control" in lc and "mac" in lc: ren[c] = "control_MAC"
        if "score" in lc: ren[c] = "score"
        if ("pred" in lc and "int" in lc and "width" in lc) or ("pi" in lc and "width" in lc) or "uncert" in lc:
            ren[c] = "pred_int_width"
    df = df.rename(columns=ren)
    # if we only have similarity, convert to novelty = 1 - sim
    if "novelty" not in df.columns:
        # try to find a likely similarity col
        sim_col = None
        for c in df.columns:
            lc = c.lower()
            if "tanimoto" in lc or ("sim" in lc and "nn" in lc):
                sim_col = c
                break
        if sim_col is not None:
            df["novelty"] = 1.0 - df[sim_col].clip(lower=0, upper=1)
    return sanitize_df(df)

def safe_csv(path):
    try:
        df = pd.read_csv(path)
        return sanitize_df(df)
    except Exception as e:
        print(f"[skip] {path}: {e}")
        return None

def is_hit_row(row, tg_thr=TG_THRESH, mac_thr=MAC_THRESH):
    try:
        return int((row.get("pred_Tg", -1e9) >= tg_thr) and (row.get("pred_MAC", -1e9) >= mac_thr))
    except Exception:
        return 0

# ----------------- aggregate: generated candidates -----------------

rows = []
for loop in RUNS:
    # Prefer predicted.csv; fall back to polymer_candidates*.csv / valid_with_desc.csv
    cand_paths = [loop/"predicted.csv"] + list(loop.glob("polymer_candidates*.csv")) + [loop/"valid_with_desc.csv"]
    for p in cand_paths:
        if not p.exists(): 
            continue
        df = safe_csv(p)
        if df is None or df.empty: 
            continue
        df = norm_cols(df)
        keep = [c for c in ["pred_Tg","pred_MAC","novelty","cluster_id","score","pred_int_width"] if c in df.columns]
        if not keep: 
            continue
        tmp = df[keep].copy()
        tmp["loop_id"] = loop.name
        tmp["is_hit"] = [is_hit_row(r) for _, r in tmp.iterrows()]
        rows.append(sanitize_df(tmp))

# Also include 4Validation Outputs as extra rows
for out in VALS:
    for p in out.glob("polymer_candidates*.csv"):
        df = safe_csv(p)
        if df is None or df.empty: 
            continue
        df = norm_cols(df)
        keep = [c for c in ["pred_Tg","pred_MAC","novelty","cluster_id","score","pred_int_width"] if c in df.columns]
        if not keep:
            continue
        tmp = df[keep].copy()
        tmp["loop_id"] = f"4Validation/{out.name}"
        tmp["is_hit"] = [is_hit_row(r) for _, r in tmp.iterrows()]
        rows.append(sanitize_df(tmp))

if rows:
    clean_rows = [sanitize_df(r) for r in rows]
    gen = pd.concat(clean_rows, ignore_index=True, sort=False)
    gen = sanitize_df(gen)
    gen.to_csv(LOGS/"generated_candidates.csv", index=False)
    print("[OK] logs/generated_candidates.csv", gen.shape)
else:
    print("[WARN] No candidate rows found. Check Runs/*/predicted.csv etc.")

# ----------------- aggregate: conditional sweeps (optional) -----------------

sw_rows = []
for loop in RUNS:
    p = loop/"out_next"/"selected_next_batch.csv"
    if not p.exists(): 
        continue
    df = safe_csv(p)
    if df is None or df.empty: 
        continue
    df = norm_cols(df)
    keep = [c for c in ["control_Tg","control_MAC","pred_Tg","pred_MAC"] if c in df.columns]
    if keep:
        sw_rows.append(sanitize_df(df[keep].copy()))

if sw_rows:
    sw = pd.concat(sw_rows, ignore_index=True, sort=False)
    sw = sanitize_df(sw)
    sw.to_csv(LOGS/"conditional_sweeps.csv", index=False)
    print("[OK] logs/conditional_sweeps.csv", sw.shape)
else:
    print("[info] No conditional sweeps found (ok to skip).")

# ----------------- aggregate: round metrics (lightweight) -----------------

rm_rows = []
for loop in RUNS:
    pred = loop/"predicted.csv"
    if not pred.exists():
        continue
    d = safe_csv(pred)
    if d is None or d.empty:
        continue
    d = norm_cols(d)
    n = len(d)
    hits = int(((d.get("pred_Tg", pd.Series([-1e9]*n)) >= TG_THRESH) &
                (d.get("pred_MAC", pd.Series([-1e9]*n)) >= MAC_THRESH)).sum())
    rm_rows.append({"run": loop.name, "n_samples": n, "hit_count": hits, "valid_frac": 1.0})

if rm_rows:
    rmdf = pd.DataFrame(rm_rows)
    rmdf = sanitize_df(rmdf)
    rmdf.to_csv(LOGS/"round_metrics.csv", index=False)
    print("[OK] logs/round_metrics.csv", len(rmdf))
else:
    print("[info] No round metrics (predicted.csv) found; skip rounds figure.")
