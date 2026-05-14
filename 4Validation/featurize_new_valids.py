#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Featurize a valid SMILES CSV with 17 RDKit descriptors (cached) using YOUR training names.
Usage:
  python featurize_new_valids.py --in loops/valid.csv --out loops/valid_with_desc.csv --cache data/desc_cache.csv
"""

import argparse
from pathlib import Path
import pandas as pd

RDKit_OK = True
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
except Exception:
    RDKit_OK = False

# EXACT descriptor names (and order) from your training dataset
DESC_NAMES = [
    "MolWt", "ExactMolWt", "HeavyAtomMolWt", "MolLogP", "TPSA",
    "NumHAcceptors", "NumHDonors", "NumRings", "FractionCSP3", "LabuteASA",
    "BalabanJ", "BertzCT", "Chi0v", "Chi1n", "Kappa1", "Kappa2", "NumHeavyAtoms"
]

def calc17(smiles: str):
    """Return descriptor vector in the exact DESC_NAMES order."""
    if not RDKit_OK:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        return [
            float(Descriptors.MolWt(m)),                     # MolWt
            float(Descriptors.ExactMolWt(m)),                # ExactMolWt
            float(Descriptors.HeavyAtomMolWt(m)),            # HeavyAtomMolWt
            float(Descriptors.MolLogP(m)),                   # MolLogP
            float(Descriptors.TPSA(m)),                      # TPSA
            float(Descriptors.NumHAcceptors(m)),             # NumHAcceptors
            float(Descriptors.NumHDonors(m)),                # NumHDonors
            float(rdMolDescriptors.CalcNumRings(m)),         # NumRings
            float(Descriptors.FractionCSP3(m)),              # FractionCSP3
            float(rdMolDescriptors.CalcLabuteASA(m)),        # LabuteASA
            float(Descriptors.BalabanJ(m)),                  # BalabanJ
            float(Descriptors.BertzCT(m)),                   # BertzCT
            float(Descriptors.Chi0v(m)),                     # Chi0v
            float(Descriptors.Chi1n(m)),                     # Chi1n
            float(Descriptors.Kappa1(m)),                    # Kappa1
            float(Descriptors.Kappa2(m)),                    # Kappa2
            float(m.GetNumHeavyAtoms()),                     # NumHeavyAtoms
        ]
    except Exception:
        # robust to occasional RDKit edge-cases
        return None

def _load_cache(cache_path: Path) -> pd.DataFrame:
    """Load cache; if legacy RD17_* headers exist, rename to your DESC_NAMES in order."""
    if not cache_path.exists():
        return pd.DataFrame(columns=["SMILES"] + DESC_NAMES)

    cache = pd.read_csv(cache_path)
    # Legacy support: RD17_1..RD17_17 -> DESC_NAMES
    legacy_cols = [f"RD17_{i+1}" for i in range(17)]
    if all(c in cache.columns for c in legacy_cols):
        cache = cache.rename(columns={legacy_cols[i]: DESC_NAMES[i] for i in range(17)})

    # Ensure columns exist in correct order
    for name in ["SMILES"] + DESC_NAMES:
        if name not in cache.columns:
            cache[name] = pd.Series(dtype="float64" if name != "SMILES" else "object")
    return cache[["SMILES"] + DESC_NAMES]

def ensure_cache(smiles_list, cache_path: Path) -> pd.DataFrame:
    cache = _load_cache(cache_path)
    have = set(cache["SMILES"].astype(str))
    need = [s for s in map(str, smiles_list) if s not in have]

    if need:
        rows = []
        for s in need:
            vals = calc17(s)
            if vals is None:
                continue
            rows.append([s] + vals)
        if rows:
            add = pd.DataFrame(rows, columns=["SMILES"] + DESC_NAMES)
            cache = pd.concat([cache, add], ignore_index=True).drop_duplicates(subset=["SMILES"])
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache.to_csv(cache_path, index=False)

    return cache

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="outp", required=True)
    ap.add_argument("--cache", dest="cache", default="data/desc_cache.csv")
    args = ap.parse_args()

    if not RDKit_OK:
        raise RuntimeError("RDKit is required to compute descriptors.")

    df = pd.read_csv(args.inp)
    if "SMILES" not in df.columns:
        raise ValueError("Input CSV must contain a 'SMILES' column.")

    cache = ensure_cache(df["SMILES"].astype(str).tolist(), Path(args.cache))
    feat = df.merge(cache, how="left", on="SMILES")
    Path(args.outp).parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(args.outp, index=False)
    print(f"[featurize] wrote {args.outp} with columns: {', '.join(DESC_NAMES)}")
    print(f"[cache]     using {args.cache}")

if __name__ == "__main__":
    main()
