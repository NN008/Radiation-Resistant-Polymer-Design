#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrain Tg/MAC predictor ensembles using the promoted labels merged with descriptors.

Reads (by default):
  - out_next/promotion_labels.csv
  - a 17-RDKit-descriptor CSV for candidates (auto-detected unless --cands is passed)

Auto-detect order for the descriptors CSV (first that exists):
  1) ./polymer_candidates_with_rdkit17.csv                (legacy root)
  2) ./4Validation/polymer_candidates_with_rdkit17.csv    (preferred)
  3) ./4Validation/valid_with_desc.csv                    (from validation)
  4) latest ./Runs/**/valid_with_desc.csv                 (from last loop run)

Writes/updates:
  - data/pool.csv        (growing labeled pool, dedup by SMILES)
  - models/desc_scaler.pkl
  - models/tg_rf_ens.pkl
  - models/mac_rf_ens.pkl
  - models/metadata.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# ---- Paths relative to project root ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../PolymerDesign
LABELS = PROJECT_ROOT / "out_next" / "promotion_labels.csv"
POOL   = PROJECT_ROOT / "data" / "pool.csv"
MODELS = PROJECT_ROOT / "2Predictors" / "Retrained" / "models"

ENSEMBLE_K = 5
N_EST      = 500
BASE_SEED  = 1000  # reproducible ensembles


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--cands",
        type=str,
        default=None,
        help="Path to the candidates CSV that includes the 17 RDKit descriptors."
    )
    return ap.parse_args()


def autodetect_cols(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    def norm(s: str) -> str:
        return s.strip().lower().replace("_", "").replace("-", "")
    lookup = {norm(c): c for c in df.columns}
    for n in names:
        c = lookup.get(norm(n))
        if c:
            return c
    return None


def pick_descriptor_cols(df: pd.DataFrame, known: List[str]) -> List[str]:
    cols = [c for c in df.columns if c not in known and pd.api.types.is_numeric_dtype(df[c])]
    # Prefer exactly 17 if present (your RDKit-17 set)
    return cols[:17] if len(cols) >= 17 else cols


def fit_ensemble(X: np.ndarray, y: np.ndarray, k: int, n_estimators: int, seed: int) -> List[RandomForestRegressor]:
    models = []
    for i in range(k):
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            n_jobs=-1,
            random_state=seed + i,
        )
        rf.fit(X, y)
        models.append(rf)
    return models


def find_latest_valid_with_desc(runs_root: Path) -> Optional[Path]:
    """
    Return the newest Runs/*/valid_with_desc.csv if present.
    """
    if not runs_root.exists():
        return None
    candidates = sorted(runs_root.glob("loop_*/valid_with_desc.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def resolve_cands_path(cli_path: Optional[str]) -> Path:
    """
    Resolve the descriptors CSV path via CLI or auto-detection.
    """
    if cli_path:
        p = Path(cli_path)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"--cands file not found: {p}")
        return p

    # Auto-detect in priority order
    candidates = [
        PROJECT_ROOT / "polymer_candidates_with_rdkit17.csv",
        PROJECT_ROOT / "4Validation" / "polymer_candidates_with_rdkit17.csv",
        PROJECT_ROOT / "4Validation" / "valid_with_desc.csv",
    ]
    latest_run = find_latest_valid_with_desc(PROJECT_ROOT / "Runs")
    if latest_run:
        candidates.append(latest_run)

    for p in candidates:
        if p.exists():
            print(f"[auto] using descriptors file: {p}")
            return p

    raise FileNotFoundError(
        "Could not find a descriptors CSV.\n"
        "Tried:\n  - ./polymer_candidates_with_rdkit17.csv\n"
        "  - ./4Validation/polymer_candidates_with_rdkit17.csv\n"
        "  - ./4Validation/valid_with_desc.csv\n"
        "  - latest ./Runs/*/valid_with_desc.csv\n"
        "Or pass an explicit path with --cands."
    )


def main():
    args = parse_args()

    if not LABELS.exists():
        raise FileNotFoundError(f"Missing {LABELS}. Run promotion_labels.py first.")

    cands_path = resolve_cands_path(args.cands)

    labs = pd.read_csv(LABELS).drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    cands = pd.read_csv(cands_path)

    col_smiles = autodetect_cols(cands, ["SMILES", "smiles"])
    if col_smiles is None:
        raise ValueError("Candidates CSV must have a 'SMILES' column (case-insensitive).")

    merged = labs.merge(cands, how="left", left_on="SMILES", right_on=col_smiles).drop_duplicates(subset=["SMILES"])
    if merged.empty:
        raise RuntimeError(
            f"Promotion labels did not match any SMILES in {cands_path.name} "
            f"({len(labs)} labels, {len(cands)} candidates)."
        )

    POOL.parent.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(POOL) if POOL.exists() else pd.DataFrame()
    pool = pd.concat([pool, merged], ignore_index=True).drop_duplicates(subset=["SMILES"])
    pool.to_csv(POOL, index=False)
    print(f"[pool] updated {POOL} → {len(pool)} rows")

    # Targets: prefer true (proxy) labels if present
    col_tg  = "Tg_true"  if "Tg_true"  in pool.columns else autodetect_cols(pool, ["Tg", "tg", "Tg_pred", "tg_pred"])
    col_mac = "MAC_true" if "MAC_true" in pool.columns else autodetect_cols(pool, ["MAC", "mac", "MAC_pred", "mac_pred"])
    if not col_tg or not col_mac:
        raise ValueError("Pool must contain Tg_true/MAC_true or Tg/MAC columns (case-insensitive).")

    known = [
        "SMILES", "score", "Tg_pred", "MAC_pred", "unc_Tg", "unc_MAC",
        "cluster_id", "rank_in_cluster", "flags", "notes", col_tg, col_mac
    ]
    desc_cols = pick_descriptor_cols(pool, known)
    if len(desc_cols) == 0:
        raise ValueError("No numeric descriptor columns found for training.")
    if len(desc_cols) < 17:
        print(f"[warn] training with {len(desc_cols)} descriptor columns (expected 17).")

    X  = pool[desc_cols].to_numpy(dtype=float)
    yT = pool[col_tg].to_numpy(dtype=float)
    yM = pool[col_mac].to_numpy(dtype=float)

    if len(pool) < 8:
        print(f"[warn] pool has only {len(pool)} rows; uncertainty estimates may be weak.")

    scaler = StandardScaler().fit(X)  # trees don't need it strictly; standardize for portability
    Xs = scaler.transform(X)

    tg_models  = fit_ensemble(Xs, yT, k=ENSEMBLE_K, n_estimators=N_EST, seed=BASE_SEED)
    mac_models = fit_ensemble(Xs, yM, k=ENSEMBLE_K, n_estimators=N_EST, seed=BASE_SEED + 777)

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler,     MODELS / "desc_scaler.pkl")
    joblib.dump(tg_models,  MODELS / "tg_rf_ens.pkl")
    joblib.dump(mac_models, MODELS / "mac_rf_ens.pkl")

    meta = {
        "desc_cols": desc_cols,
        "ensemble_k": ENSEMBLE_K,
        "n_estimators": N_EST,
        "base_seed": BASE_SEED,
        "n_pool_rows": int(len(pool)),
        "targets": {"tg": col_tg, "mac": col_mac},
        "cands_path": str(cands_path),
        "labels_path": str(LABELS),
    }
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[models] saved scaler + RF ensembles to {MODELS}")
    print(f"[meta]   wrote {MODELS / 'metadata.json'}")


if __name__ == "__main__":
    main()
