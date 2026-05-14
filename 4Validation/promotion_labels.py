#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create proxy "true" labels for the promoted set (4 rows by default).
- Reads: out_next/selected_next_batch.csv
- Optional: uses independent Tg/MAC models if available for cross-check
- Writes:
    out_next/promotion_labels.csv
    out_next/promotion_labels_meta.json
"""

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# -------------------- Config --------------------
IN_SEL  = Path("out_next/selected_next_batch.csv")
OUT_LAB = Path("out_next/promotion_labels.csv")
OUT_LOG = Path("out_next/promotion_labels_meta.json")

# Optional independent models (if you trained separate RFs earlier)
TG_MODEL_PATH  = Path("tg_predictor_rf_final.pkl")
MAC_MODEL_PATH = Path("mac_predictor_rf_final.pkl")

# If you want to force specific descriptor columns, list 17 names here; else autodetect numerics.
DESC_COLS_HINT: List[str] | None = None

# -------------------- RDKit (optional) --------------------
RDKit_OK = True
try:
    from rdkit import Chem
except Exception:
    RDKit_OK = False

def autodetect_cols(df: pd.DataFrame, names: list[str]) -> str | None:
    def norm(s: str) -> str: return s.strip().lower().replace("_","").replace("-","")
    lookup = {norm(c): c for c in df.columns}
    for n in names:
        c = lookup.get(norm(n))
        if c: return c
    return None

def pick_17_desc_cols(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return cols[:17] if len(cols) >= 17 else cols

def rdkit_flag(smiles: str) -> str:
    if not RDKit_OK:
        return "no_rdkit"
    try:
        m = Chem.MolFromSmiles(smiles)
        return "ok" if m is not None else "bad_smiles"
    except Exception:
        return "bad_smiles"

def try_load_model(path: Path):
    try:
        import joblib
        if path.exists():
            return joblib.load(path)
    except Exception:
        return None
    return None

def try_predict(model, X: np.ndarray) -> np.ndarray | None:
    if model is None: return None
    try:
        return model.predict(X)
    except Exception:
        # Accept a (scaler, model) tuple/pipeline
        try:
            scaler, inner = model
            return inner.predict(scaler.transform(X))
        except Exception:
            return None

def main():
    if not IN_SEL.exists():
        raise FileNotFoundError(f"Missing {IN_SEL}. Run the batch selector first.")

    sel = pd.read_csv(IN_SEL).drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

    col_smiles = autodetect_cols(sel, ["SMILES","smiles"])
    col_tg  = autodetect_cols(sel, ["Tg","tg","tg_pred","Tg_pred","Tg_predicted"])
    col_mac = autodetect_cols(sel, ["MAC","mac","mac_pred","MAC_pred","MAC_predicted"])
    if not col_smiles or not col_tg or not col_mac:
        raise ValueError("Need SMILES + Tg(_pred) + MAC(_pred) columns in selected_next_batch.csv")

    sel["flags"] = sel[col_smiles].map(rdkit_flag)

    # Descriptor matrix (if present) to feed optional independent models
    if DESC_COLS_HINT is not None:
        desc_cols = DESC_COLS_HINT
    else:
        desc_cols = pick_17_desc_cols(sel, exclude=[col_smiles, col_tg, col_mac, "score", "cluster_id", "rank_in_cluster"])
    X = sel[desc_cols].to_numpy(dtype=float) if len(desc_cols) else None

    tg_model  = try_load_model(TG_MODEL_PATH)
    mac_model = try_load_model(MAC_MODEL_PATH)
    tg_proxy  = try_predict(tg_model,  X) if X is not None else None
    mac_proxy = try_predict(mac_model, X) if X is not None else None

    if tg_proxy is not None and mac_proxy is not None and len(tg_proxy)==len(sel)==len(mac_proxy):
        # Conservative proxy = average with current predictions
        sel["Tg_true"]  = (sel[col_tg].astype(float)  + tg_proxy.astype(float)) / 2.0
        sel["MAC_true"] = (sel[col_mac].astype(float) + mac_proxy.astype(float)) / 2.0
        sel["notes"]    = "proxy=avg(current_pred,independent_model)"
        used_independent = True
    else:
        sel["Tg_true"]  = sel[col_tg].astype(float)
        sel["MAC_true"] = sel[col_mac].astype(float)
        sel["notes"]    = "proxy=copy(current_pred)"
        used_independent = False

    out = sel[[col_smiles, "Tg_true", "MAC_true", "flags", "notes"]].copy()
    out.columns = ["SMILES","Tg_true","MAC_true","flags","notes"]
    OUT_LAB.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_LAB, index=False)

    meta = {
        "n_rows": int(len(out)),
        "desc_cols_used": desc_cols,
        "used_independent_models": used_independent,
        "inputs": str(IN_SEL),
        "outputs": str(OUT_LAB)
    }
    OUT_LOG.write_text(json.dumps(meta, indent=2))
    print(f"[labels] wrote {OUT_LAB} ({len(out)} rows)")
    print(f"[meta]   wrote {OUT_LOG}")

if __name__ == "__main__":
    main()
