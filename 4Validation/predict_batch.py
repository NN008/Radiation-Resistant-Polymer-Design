#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighted predict_batch: no model loading.

Reads a featurized CSV that already contains Tg/MAC (e.g., Tg_pred/MAC_pred or Tg/MAC),
computes a Tg-weighted composite score after normalization, writes:
  • --out  : all candidates with Tg_norm/MAC_norm/weighted_score
  • --hits : selected hits (ranked by weighted_score)
  • non_hits.csv (or --nonhits): non-hits with a 'reason' column

Default normalization: threshold-relative
    Tg_norm  = Tg_value  / TG_CUTOFF
    MAC_norm = MAC_value / MAC_CUTOFF
Default weights: w_tg=0.7, w_mac=0.3
Hit rule (default):
    weighted_score >= 1.0 AND Tg_norm >= min_tg_frac AND MAC_norm >= min_mac_frac

Usage:
  python predict_batch.py \
    --in   Runs/.../valid_with_desc.csv \
    --out  Runs/.../predicted.csv \
    --hits Runs/.../hits.csv \
    --tg 215.0 --mac 0.0569 \
    --w_tg 0.7 --w_mac 0.3 \
    --min_tg_frac 0.85 --min_mac_frac 0.85 \
    [--eps 1e-5] [--norm threshold|minmax|zscore] [--nonhits Runs/.../non_hits.csv]
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _norm_key(s: str) -> str:
    return s.lower().replace("_", "").strip()


def _find_pair(df: pd.DataFrame, pairs):
    lookup = { _norm_key(c): c for c in df.columns }
    for a, b in pairs:
        ca = lookup.get(_norm_key(a))
        cb = lookup.get(_norm_key(b))
        if ca and cb:
            return ca, cb
    return None, None


def _threshold_norm(series: pd.Series, cutoff: float) -> pd.Series:
    return series.astype(float) / float(cutoff)


def _minmax_norm(series: pd.Series) -> pd.Series:
    lo, hi = np.nanpercentile(series.astype(float), [5, 95])
    if hi <= lo:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series.astype(float) - lo) / (hi - lo)


def _zscore(series: pd.Series) -> pd.Series:
    mu = float(series.astype(float).mean())
    sd = float(series.astype(float).std(ddof=0))
    if sd == 0:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series.astype(float) - mu) / sd


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in",   dest="inp",   required=True, help="Featurized CSV (already has Tg/MAC predictions)")
    ap.add_argument("--out",  dest="outp",  required=True, help="Output CSV for all candidates")
    ap.add_argument("--hits", dest="hits",  required=True, help="Output CSV for selected hits (ranked)")
    ap.add_argument("--nonhits", dest="nonhits", default=None, help="Optional CSV for non-hits with reasons")
    ap.add_argument("--tg", type=float, default=203.7, help="Tg cutoff for normalization/expectation (°C)")
    ap.add_argument("--mac", type=float, default=0.0569, help="MAC cutoff for normalization/expectation")
    ap.add_argument("--w_tg", type=float, default=0.7, help="Weight for Tg in weighted score")
    ap.add_argument("--w_mac", type=float, default=0.3, help="Weight for MAC in weighted score")
    ap.add_argument("--min_tg_frac", type=float, default=0.85, help="Minimum Tg_norm required (soft floor)")
    ap.add_argument("--min_mac_frac", type=float, default=0.85, help="Minimum MAC_norm required (soft floor)")
    ap.add_argument("--eps", type=float, default=0.0, help="Tolerance for threshold compares")
    ap.add_argument("--norm", choices=["threshold","minmax","zscore"], default="threshold",
                    help="Normalization method for Tg/MAC before weighting")
    # kept for CLI compatibility; ignored
    ap.add_argument("--models", dest="models", default=None, help="Ignored (no model loading)")
    args = ap.parse_args()

    # normalize weights to sum to 1.0
    if args.w_tg < 0 or args.w_mac < 0:
        raise ValueError("w_tg and w_mac must be non-negative.")
    total_w = args.w_tg + args.w_mac
    if total_w <= 0:
        raise ValueError("w_tg + w_mac must be > 0.")
    if abs(total_w - 1.0) > 1e-6:
        args.w_tg /= total_w
        args.w_mac /= total_w
        print(f"[warn] normalized weights → w_tg={args.w_tg:.3f}, w_mac={args.w_mac:.3f}")

    df = pd.read_csv(args.inp)
    if "SMILES" not in df.columns:
        raise ValueError("Input must contain a 'SMILES' column.")

    tg_col, mac_col = _find_pair(df, [
        ("Tg_pred","MAC_pred"),
        ("tg_pred","mac_pred"),
        ("Tg","MAC"),
        ("tg","mac"),
    ])
    if not tg_col or not mac_col:
        raise ValueError("Could not find Tg/MAC columns (looked for Tg_pred/MAC_pred or Tg/MAC).")

    # Unify Score/score if needed (avoid duplicates)
    if "score" in df.columns:
        pass
    elif "Score" in df.columns:
        df = df.rename(columns={"Score": "score"})
    else:
        df["score"] = pd.to_numeric(df[tg_col], errors="coerce").astype(float) * \
                      pd.to_numeric(df[mac_col], errors="coerce").astype(float)

    # Numeric Tg/MAC and finiteness
    tg_vals  = pd.to_numeric(df[tg_col],  errors="coerce").astype(float)
    mac_vals = pd.to_numeric(df[mac_col], errors="coerce").astype(float)
    finite_mask = np.isfinite(tg_vals.values) & np.isfinite(mac_vals.values)

    eps = max(0.0, float(args.eps))

    # --- Normalization
    if args.norm == "threshold":
        Tg_norm  = _threshold_norm(tg_vals,  max(args.tg, eps))
        MAC_norm = _threshold_norm(mac_vals, max(args.mac, eps))
    elif args.norm == "minmax":
        Tg_norm  = _minmax_norm(tg_vals)
        MAC_norm = _minmax_norm(mac_vals)
    else:
        Tg_z  = _zscore(tg_vals)
        MAC_z = _zscore(mac_vals)
        Tg_norm  = (Tg_z  - Tg_z.median())  / (Tg_z.std(ddof=0)  + 1e-9) + 1.0
        MAC_norm = (MAC_z - MAC_z.median()) / (MAC_z.std(ddof=0) + 1e-9) + 1.0

    df["Tg_norm"]  = Tg_norm
    df["MAC_norm"] = MAC_norm
    df["weighted_score"] = args.w_tg * df["Tg_norm"] + args.w_mac * df["MAC_norm"]

    # Floors and weighted threshold
    tg_floor_fail  = df["Tg_norm"]  < (args.min_tg_frac  - eps)
    mac_floor_fail = df["MAC_norm"] < (args.min_mac_frac - eps)
    weighted_fail  = df["weighted_score"] < (1.0 - eps)

    # Hits must be finite AND pass floors AND weighted threshold
    hits_mask = finite_mask & (~tg_floor_fail.values) & (~mac_floor_fail.values) & (~weighted_fail.values)

    # --- Reasons for non-hits
    reason = np.full(len(df), "", dtype=object)
    reason[~finite_mask] = "non_finite_prediction"

    both_floors = finite_mask & tg_floor_fail.values & mac_floor_fail.values
    only_tg     = finite_mask & tg_floor_fail.values & (~mac_floor_fail.values)
    only_mac    = finite_mask & (~tg_floor_fail.values) & mac_floor_fail.values
    below_w     = finite_mask & (~tg_floor_fail.values) & (~mac_floor_fail.values) & weighted_fail.values

    reason[both_floors] = "below_both_floors"
    reason[only_tg]     = "below_tg_floor"
    reason[only_mac]    = "below_mac_floor"
    reason[below_w]     = "below_weighted_threshold"

    # Attach helper columns
    df["_tg_col_used"]  = tg_col
    df["_mac_col_used"] = mac_col
    df["_tg_val"]  = tg_vals
    df["_mac_val"] = mac_vals
    df["_reason"]  = reason  # empty string => hit

    # Save all predictions
    Path(args.outp).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.outp, index=False)

    # Save hits (ranked)
    hits = df[hits_mask].copy().sort_values("weighted_score", ascending=False)
    Path(args.hits).parent.mkdir(parents=True, exist_ok=True)
    hits.to_csv(args.hits, index=False)

    # Save non-hits with explicit reason
    nonhits_path = args.nonhits or str(Path(args.hits).with_name("non_hits.csv"))
    non_hits = df[~hits_mask].copy()
    non_hits["reason"] = np.where(non_hits["_reason"] == "", "unknown", non_hits["_reason"])
    Path(nonhits_path).parent.mkdir(parents=True, exist_ok=True)
    non_hits.to_csv(nonhits_path, index=False)

    # Console summary
    total = len(df)
    n_hits = hits.shape[0]
    n_non  = non_hits.shape[0]
    counts = non_hits["reason"].value_counts(dropna=False).to_dict()

    print(f"[predict] wrote {args.outp}  (n={total}) using Tg='{tg_col}', MAC='{mac_col}'")
    print(f"[hits]    wrote {args.hits} (n_hits={n_hits}, hit_rate={n_hits/total if total else 0:.3f})")
    print(f"[nonhits] wrote {nonhits_path} (n_non_hits={n_non})  breakdown: {counts}")
    print(
        f"[weighted_score]   norm={args.norm}  w_tg={args.w_tg:.2f}  w_mac={args.w_mac:.2f}  "
        f"floors: Tg_norm≥{args.min_tg_frac:.2f}, MAC_norm≥{args.min_mac_frac:.2f}  eps={eps:g}"
    )


if __name__ == "__main__":
    main()