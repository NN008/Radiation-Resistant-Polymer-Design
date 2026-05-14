#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validation / Generation pipeline for the fixed conditional Transformer.

- Loads the training bundle (config+state+vocab+scaler) to avoid tokenizer drift.
- Scales (Tg, MAC) conditions with the saved scaler.
- Generates many SMILES near your target condition.
- RDKit validity + canonicalization.
- Descriptors -> RF predictors (Tg, MAC) -> filter and score.
- Keeps running top-200 heap, checkpointing, and optional diversity (Butina).

Edit the CONFIG section paths as needed, then:
    python validate_and_rank.py
"""

import os, json, gc, inspect, random, heapq
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from contextlib import nullcontext

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.ML.Cluster import Butina
import joblib

import sys
from pathlib import Path

# Add the folder containing sample2.py to sys.path
sys.path.append(str(Path.home() / "cluster/PolymerDesign/3Smile_Generation/Transformer/Scripts"))

# ------------------------
# CONFIG (edit to your paths)
# ------------------------
# Preferred: use the training bundle created by export_bundle.py
BUNDLE_DIR       = "/csl/users/2026nnandaku/cluster/PolymerDesign/3Smile_Generation/Transformer/bundle"
# Optional fallback: monolithic file containing dict {state_dict, config, tokens, scaler}
MONOLITHIC_PATH  = "Transformer/transformer_polymer_gen_FULL.pt"

# RF predictors
TG_MODEL_PATH    = "/csl/users/2026nnandaku/cluster/PolymerDesign/2Predictors/OgPredictors/tg_predictor_rf_final.pkl"
MAC_MODEL_PATH   = "/csl/users/2026nnandaku/cluster/PolymerDesign/2Predictors/OgPredictors/mac_predictor_rf_final.pkl"

# Target / thresholds
TARGET_TG_C      = -1.0   # will be set after calibration
TARGET_MAC       = -1.0   # will be set after calibration
TG_THRESHOLD     = -1.0   # force auto-calibration
MAC_THRESHOLD    = -1.0   # force auto-calibration

# Sampling volume / behavior
N_SAMPLES        = 10000           # total valid molecules to attempt to keep
BATCH_SIZE       = 500
MAX_TRIES_PER_KEEP = 30            # tries per kept sample (safety)
SAMPLING_KW      = dict(temperature=0.80, top_k=64, top_p=0.95, max_len=128)

# Diversity
DIVERSIFY        = True
DIVERSITY_THRESH = 0.40            # Butina distance threshold

# Output / resume
OUTDIR           = "."
STRICT_TOP200_CSV    = os.path.join(OUTDIR, "polymer_candidates.csv")
PASSED_ALL_CSV       = os.path.join(OUTDIR, "polymer_passed_all.csv")
UNFILTERED_TOP_CSV   = os.path.join(OUTDIR, "polymer_top200_unfiltered.csv")
NEAR_MISS_CSV        = os.path.join(OUTDIR, "polymer_near_misses.csv")
RAW_LOG_PATH         = os.path.join(OUTDIR, "raw_smiles_log.csv")
CKPT_PATH            = os.path.join(OUTDIR, "pipeline_topk.ckpt")
SEEN_PATH            = os.path.join(OUTDIR, "seen_smiles.txt")
PASSED_SET_PATH      = os.path.join(OUTDIR, "passed_smiles.txt")

# Descriptor set (must match RF training)
FEATURES = [
    "MolWt","ExactMolWt","HeavyAtomMolWt","MolLogP","TPSA",
    "NumHAcceptors","NumHDonors","NumRings","FractionCSP3","LabuteASA",
    "BalabanJ","BertzCT","Chi0v","Chi1n","Kappa1","Kappa2","NumHeavyAtoms"
]

# ------------------------
# Device / AMP / Seeds
# ------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
AMP = torch.amp.autocast("cuda") if use_cuda else nullcontext
print(f"[setup] device={device}", flush=True)

SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ------------------------
# Imports that depend on your repo layout
# ------------------------
# Use the fixed model (pad-masked, condition-only encoder)
try:
    from sample2 import CondTransformer  # recommended
except Exception:
    # fallback to whatever name you used for the fixed transformer
    from sample import CondTransformer

# Tokenizer helpers from the training code
try:
    from tokenizer import SMILESTokenizer, load_tokens
except Exception:
    # Last-resort fallback if tokenizer module isn't available
    class SMILESTokenizer:
        def __init__(self, tokens):
            self.tokens = tokens
            self.stoi = {ch: i for i, ch in enumerate(tokens)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
        def decode(self, ids):
            out = []
            for i in ids:
                ch = self.itos.get(int(i), "<unk>")
                if ch == "<eos>": break
                if ch not in ("<pad>", "<bos>", "<unk>"):
                    out.append(ch)
            return "".join(out)
    def load_tokens(path):
        with open(path) as f:
            return json.load(f)

# ------------------------
# Bundle/Monolithic loader
# ------------------------
def load_model_and_tokenizer():
    """
    Try to load from bundle (preferred).
    Fallback to the monolithic .pt that contains dict {state_dict, config, tokens, scaler}.
    Returns: (model.eval().to(device), tokenizer, scaler)
    """
    if os.path.isdir(BUNDLE_DIR) and \
       all(os.path.exists(os.path.join(BUNDLE_DIR, f)) for f in ("model_state.pt", "config.json", "vocab.json", "scaler.pkl")):
        print("[load] using bundle directory", BUNDLE_DIR, flush=True)
        state = torch.load(os.path.join(BUNDLE_DIR, "model_state.pt"), map_location="cpu")
        cfg   = json.load(open(os.path.join(BUNDLE_DIR, "config.json")))
        tokens = load_tokens(os.path.join(BUNDLE_DIR, "vocab.json"))
        scaler = joblib.load(os.path.join(BUNDLE_DIR, "scaler.pkl"))
        tokenizer = SMILESTokenizer(tokens)
        cfg.setdefault("pad_id", tokenizer.stoi.get("<pad>", 0))
        model = CondTransformer(**cfg)
        model.load_state_dict(state)
        return model.to(device).eval(), tokenizer, scaler

    # Fallback: monolithic file
    print("[load] bundle not found; trying monolithic file", MONOLITHIC_PATH, flush=True)
    pkg = torch.load(MONOLITHIC_PATH, map_location="cpu")
    if isinstance(pkg, dict) and all(k in pkg for k in ("state_dict", "config", "tokens", "scaler")):
        tokens = pkg["tokens"]
        cfg    = pkg["config"]
        scaler = pkg["scaler"]
        tokenizer = SMILESTokenizer(tokens)
        cfg.setdefault("pad_id", tokenizer.stoi.get("<pad>", 0))
        model = CondTransformer(**cfg)
        model.load_state_dict(pkg["state_dict"])
        return model.to(device).eval(), tokenizer, scaler

    # Last resort: assume pickled nn.Module
    model = pkg
    # Need a tokenizer; try to recover tokens from a nearby vocab.json
    vocab_guess = os.path.join(os.path.dirname(MONOLITHIC_PATH), "vocab.json")
    if os.path.exists(vocab_guess):
        tokens = load_tokens(vocab_guess)
        tokenizer = SMILESTokenizer(tokens)
    else:
        raise RuntimeError("Could not load tokenizer tokens; provide vocab.json or use bundle.")
    # Scaler guess
    scaler_guess = os.path.join(os.path.dirname(MONOLITHIC_PATH), "scaler.pkl")
    scaler = joblib.load(scaler_guess) if os.path.exists(scaler_guess) else None
    return model.to(device).eval(), tokenizer, scaler

transformer, tokenizer, scaler = load_model_and_tokenizer()

# RF predictors
tg_model  = joblib.load(TG_MODEL_PATH)
mac_model = joblib.load(MAC_MODEL_PATH)
TG_FEATURES  = list(getattr(tg_model,  "feature_names_in_", []))
MAC_FEATURES = list(getattr(mac_model, "feature_names_in_", []))
if not TG_FEATURES or not MAC_FEATURES:
    raise RuntimeError("Saved RF models must have feature_names_in_. Refit/save with sklearn>=1.0.")

# ------------------------
# Conditioning via saved scaler
# ------------------------
def to_cond_scaled(tg_real: float, mac_real: float, device):
    if scaler is None:
        raise RuntimeError("No scaler available; please use the bundle or provide scaler.pkl.")
    arr = np.array([[tg_real, mac_real]], dtype=np.float32)
    scaled = scaler.transform(arr)[0]
    return torch.tensor(scaled, dtype=torch.float32, device=device)

#TARGET_SCALED = to_cond_scaled(TARGET_TG_C, TARGET_MAC, device)
#print(f"[cond] TARGET scaled → Tg={float(TARGET_SCALED[0]):.3f}, MAC={float(TARGET_SCALED[1]):.3f}", flush=True)

TARGET_SCALED = None  # set later inside main()

def _cond_high(device):
    if TARGET_SCALED is None:
        raise RuntimeError("TARGET_SCALED not set; set it in main() after _init_thresholds().")
    return TARGET_SCALED

# ------------------------
# Checkpoint / resume helpers
# ------------------------
def _load_seen(path=SEEN_PATH):
    seen = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s: seen.add(s)
    return seen

def _append_seen(iter_smiles, path=SEEN_PATH):
    with open(path, "a", encoding="utf-8") as f:
        for s in iter_smiles: f.write(s + "\n")

def _save_ckpt(top_heap, processed):
    state = {"processed": int(processed),
             "topk": [(-neg, smi, tg, mac) for (neg, smi, tg, mac) in top_heap]}
    with open(CKPT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f)

def _load_ckpt():
    if not os.path.exists(CKPT_PATH): return [], 0
    with open(CKPT_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)
    heap = [(-score, smi, tg, mac) for (score, smi, tg, mac) in state.get("topk", [])]
    heapq.heapify(heap)
    return heap, int(state.get("processed", 0))

def _push_topk(heap, k, score, smi, tg, mac):
    if any(h[1] == smi for h in heap): return
    item = (-float(score), smi, float(tg), float(mac))
    if len(heap) < k: heapq.heappush(heap, item)
    else:
        if item < heap[0]: heapq.heapreplace(heap, item)

def _heap_to_df(heap):
    rows = [{"SMILES": smi, "Tg_pred": tg, "MAC_pred": mac, "Score": -neg}
            for (neg, smi, tg, mac) in heap]
    if not rows: return pd.DataFrame(columns=["SMILES","Tg_pred","MAC_pred","Score"])
    rows.sort(key=lambda r: r["Score"], reverse=True)
    return pd.DataFrame(rows)

def _load_passed_set():
    s = set()
    if os.path.exists(PASSED_SET_PATH):
        with open(PASSED_SET_PATH) as f:
            for ln in f:
                ln = ln.strip()
                if ln: s.add(ln)
    return s

def _append_passed(rows_df, passed_seen):
    if rows_df.empty: return passed_seen
    out = rows_df[["SMILES","Tg_pred","MAC_pred","Score"]].copy()
    mask = ~out["SMILES"].isin(passed_seen)
    out = out[mask]
    if out.empty: return passed_seen
    write_header = not os.path.exists(PASSED_ALL_CSV)
    out.to_csv(PASSED_ALL_CSV, mode="a", index=False, header=write_header)
    with open(PASSED_SET_PATH, "a") as f:
        for s in out["SMILES"]: f.write(s + "\n")
    passed_seen.update(out["SMILES"].tolist())
    return passed_seen

# ------------------------
# Sampler
# ------------------------
def _looks_smilesy(s):
    if not s or " " in s: return False
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789[]=()#@+-\\/*,.:;*")
    if any(ch not in allowed for ch in s): return False
    # basic parentheses balance
    bal = 0
    for ch in s:
        if ch == "(": bal += 1
        elif ch == ")":
            bal -= 1
            if bal < 0: return False
    return bal == 0

def _sample_once(model, tok, cond, device):
    if hasattr(model, "generate"):
        return model.generate(tok, cond, device=device, **SAMPLING_KW)
    # fallback for older pickled modules
    try:
        sig = inspect.signature(model.sample)
        kwargs = {k: v for k, v in SAMPLING_KW.items() if k in sig.parameters}
    except (TypeError, ValueError):
        kwargs = {}
    return model.sample(tok, cond, device=device, **kwargs)

def generate_valid_smiles(model, n_keep):
    kept, tries = [], 0
    with torch.inference_mode():
        with AMP:
            pbar = tqdm(total=n_keep, desc="Sampling(valid)", mininterval=0.5)
            while len(kept) < n_keep and (tries < n_keep * MAX_TRIES_PER_KEEP):
                tries += 1
                cond = _cond_high(device)
                try:
                    s = _sample_once(model, tokenizer, cond, device)
                    if not s or not _looks_smilesy(s): continue
                    mol = Chem.MolFromSmiles(s)
                    if mol is None: continue
                    can = Chem.MolToSmiles(mol, canonical=True)
                    kept.append(can)
                    pbar.update(1)
                except Exception:
                    continue
            pbar.close()
    return kept

# ------------------------
# Chemistry + descriptors
# ------------------------
def clean_and_unique(smiles_list, seen):
    unique, new_seen = [], []
    for s in smiles_list:
        if not s: 
            continue
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            Chem.SanitizeMol(mol)
            can = Chem.MolToSmiles(mol, canonical=True)
            if can in seen:
                continue
            unique.append(can)
            new_seen.append(can)
        except Exception:
            continue
    return unique, new_seen

def compute_descriptors(smiles_list):
    rows = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        try:
            rows.append({
                "SMILES": s,
                "MolWt": Descriptors.MolWt(mol),
                "ExactMolWt": Descriptors.ExactMolWt(mol),
                "HeavyAtomMolWt": Descriptors.HeavyAtomMolWt(mol),
                "MolLogP": Descriptors.MolLogP(mol),
                "TPSA": Descriptors.TPSA(mol),
                "NumHAcceptors": Descriptors.NumHAcceptors(mol),
                "NumHDonors": Descriptors.NumHDonors(mol),
                "NumRings": rdMolDescriptors.CalcNumRings(mol),
                "FractionCSP3": Descriptors.FractionCSP3(mol),
                "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
                "BalabanJ": Descriptors.BalabanJ(mol),
                "BertzCT": Descriptors.BertzCT(mol),
                "Chi0v": Descriptors.Chi0v(mol),
                "Chi1n": Descriptors.Chi1n(mol),
                "Kappa1": Descriptors.Kappa1(mol),
                "Kappa2": Descriptors.Kappa2(mol),
                "NumHeavyAtoms": mol.GetNumHeavyAtoms(),
            })
        except Exception:
            continue
    if not rows: return pd.DataFrame(columns=["SMILES"] + FEATURES)
    df = pd.DataFrame(rows)
    return df[["SMILES"] + FEATURES]

def _predict_with_model(model, feats_df, needed_cols, label):
    missing = [c for c in needed_cols if c not in feats_df.columns]
    if missing:
        print(f"[ERROR] Missing features for {label}: {missing}", flush=True)
        return None
    return model.predict(feats_df[needed_cols])

def predict_and_score(df):
    if df.empty:
        return pd.DataFrame(columns=["SMILES","Tg_pred","MAC_pred","Score"])

    feats = df.drop(columns=["SMILES"], errors="ignore")

    tg_pred  = _predict_with_model(tg_model,  feats, TG_FEATURES,  "Tg")
    mac_pred = _predict_with_model(mac_model, feats, MAC_FEATURES, "MAC")
    if tg_pred is None or mac_pred is None:
        return pd.DataFrame(columns=["SMILES","Tg_pred","MAC_pred","Score"])

    out = pd.DataFrame({
        "SMILES": df["SMILES"].values,
        "Tg_pred": tg_pred,
        "MAC_pred": mac_pred
    })
    out["Score"] = out["Tg_pred"] * out["MAC_pred"]

    print("Tg_pred (°C)  p50/p90/p95/max:",
          float(np.median(tg_pred)),
          float(np.percentile(tg_pred, 90)),
          float(np.percentile(tg_pred, 95)),
          float(np.max(tg_pred)), flush=True)
    print("MAC_pred      p50/p90/p95/max:",
          float(np.median(mac_pred)),
          float(np.percentile(mac_pred, 90)),
          float(np.percentile(mac_pred, 95)),
          float(np.max(mac_pred)), flush=True)
    return out

# ------------------------
# Threshold auto-recalibration using RF preds on training CSV (optional)
# ------------------------
TRAIN_CSV = "/csl/users/2026nnandaku/cluster/PolymerDesign/1Dataset/PI1M_Tg_MAC.csv"

def _suggest_thresholds_from_predictions(csv_path, sample_n=20000):
    df = pd.read_csv(csv_path)
    df = df[df["SMILES"].notna()]
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=SEED)
    ddf = compute_descriptors(df["SMILES"].tolist())
    if ddf.empty:
        return TG_THRESHOLD, MAC_THRESHOLD
    tg_pred  = tg_model.predict(ddf[TG_FEATURES])
    mac_pred = mac_model.predict(ddf[MAC_FEATURES])
    tg95  = float(np.quantile(tg_pred, 0.95))
    mac95 = float(np.quantile(mac_pred, 0.95))
    return tg95, mac95

def _init_thresholds():
    # Always calibrate to the 95th percentile of your RF predictions on the training CSV
    try:
        tg95p, mac95p = _suggest_thresholds_from_predictions(TRAIN_CSV)
        final_tg, final_mac = tg95p, mac95p
    except Exception:
        # If training CSV access fails, fall back to the config constants
        final_tg, final_mac = TG_THRESHOLD, MAC_THRESHOLD
    print(f"[thresh] using Tg>{final_tg:.2f} °C, MAC>{final_mac:.4f}", flush=True)
    return final_tg, final_mac

# ------------------------
# Diversity (Butina)
# ------------------------
def _morgan_fp(smi, radius=2, nbits=1024):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    try: return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    except Exception: return None

def diversify_topk(df, k=200, thresh=0.4):
    if df.empty: return df
    smiles = df["SMILES"].tolist()
    fps = [_morgan_fp(s) for s in smiles]
    dists = []
    for i in range(len(fps)):
        row = []
        for j in range(i):
            if fps[i] is None or fps[j] is None:
                d = 1.0
            else:
                d = 1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
            row.append(d)
        dists.extend(row)
    clusters = Butina.ClusterData(dists, len(fps), thresh, isDistData=True)
    selected = []
    for clus in clusters:
        best_i, best_score = None, -1.0
        for idx in clus:
            sc = float(df.iloc[idx]["Score"])
            if sc > best_score: best_score, best_i = sc, idx
        if best_i is not None: selected.append(best_i)
        if len(selected) >= k: break
    if len(selected) < k:
        taken = set(selected)
        for idx in df.sort_values("Score", ascending=False).index:
            if idx not in taken:
                selected.append(idx)
                if len(selected) >= k: break
    selected = list(dict.fromkeys(selected))
    return df.iloc[selected].sort_values("Score", ascending=False).head(k).reset_index(drop=True)

# ------------------------
# Main
# ------------------------
def main():
    # Prepare logs
    if not os.path.exists(RAW_LOG_PATH):
        with open(RAW_LOG_PATH, "w", encoding="utf-8") as f: f.write("SMILES\n")

    topk_heap, processed = _load_ckpt()
    seen = _load_seen()
    passed_seen = _load_passed_set()
    print(f"[resume] processed={processed}, seen={len(seen)}, heap={len(topk_heap)}, passed_seen={len(passed_seen)}", flush=True)

    final_tg, final_mac = _init_thresholds()
    final_mac = 0.056896
    # Use the calibrated thresholds as the sampling condition target
    global TARGET_SCALED
    TARGET_SCALED = to_cond_scaled(final_tg, final_mac, device)
    print(f"[cond] TARGET scaled → Tg={float(TARGET_SCALED[0]):.3f}, MAC={float(TARGET_SCALED[1]):.3f}", flush=True)
    remain, batch_idx = max(0, N_SAMPLES - processed), 0
        
    while remain > 0:
        n_now = min(BATCH_SIZE, remain); batch_idx += 1
        print(f"[batch {batch_idx}] sampling {n_now} (tries up to {n_now*MAX_TRIES_PER_KEEP})…", flush=True)

        raw_valid = generate_valid_smiles(transformer, n_now)
        with open(RAW_LOG_PATH, "a", encoding="utf-8") as f:
            for s in raw_valid: f.write(s + "\n")

        uniq, new_seen = clean_and_unique(raw_valid, seen)
        print(f"[batch {batch_idx}] unique new valid = {len(uniq)}", flush=True)
        if new_seen: _append_seen(new_seen); seen.update(new_seen)

        if len(uniq) > 0:
            desc_df = compute_descriptors(uniq)
            scored  = predict_and_score(desc_df)
            filt = scored[(scored["Tg_pred"] > final_tg) & (scored["MAC_pred"] > final_mac)].copy()
            passed_seen = _append_passed(filt, passed_seen)
            for _, row in filt.iterrows():
                _push_topk(topk_heap, 200, row["Score"], row["SMILES"], row["Tg_pred"], row["MAC_pred"])
        else:
            print(f"[batch {batch_idx}] nothing to score.", flush=True)

        processed += n_now
        remain = max(0, N_SAMPLES - processed)

        _save_ckpt(topk_heap, processed)
        print(f"[checkpoint] processed={processed}, heap={len(topk_heap)}", flush=True)

        # free
        del raw_valid
        if 'uniq' in locals(): del uniq
        if 'new_seen' in locals(): del new_seen
        if 'desc_df' in locals(): del desc_df
        if 'scored' in locals(): del scored
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Strict top-200 from heap
    top_strict = _heap_to_df(topk_heap).head(200).reset_index(drop=True)

    # Unfiltered top-200 by rescoring all seen
    print("[final] rescoring all seen SMILES to produce unfiltered top-200…", flush=True)
    seen_all = []
    if os.path.exists(SEEN_PATH):
        with open(SEEN_PATH, "r", encoding="utf-8") as f:
            seen_all = [ln.strip() for ln in f if ln.strip()]

    CH = 5000
    rows = []
    for i in range(0, len(seen_all), CH):
        chunk = seen_all[i:i+CH]
        ddf = compute_descriptors(chunk)
        sc  = predict_and_score(ddf)
        rows.append(sc)
    all_scored = (pd.concat(rows, ignore_index=True)
                  if rows else pd.DataFrame(columns=["SMILES","Tg_pred","MAC_pred","Score"]))
    all_top200 = all_scored.sort_values("Score", ascending=False).head(200).reset_index(drop=True)

    near = all_scored[(all_scored["Tg_pred"] > final_tg*0.9) &
                      (all_scored["MAC_pred"] > final_mac*0.9)].sort_values("Score", ascending=False)

    top_div = diversify_topk(top_strict, k=200, thresh=DIVERSITY_THRESH) if (DIVERSIFY and not top_strict.empty) else top_strict

    top_div.to_csv(STRICT_TOP200_CSV, index=False)
    all_top200.to_csv(UNFILTERED_TOP_CSV, index=False)
    near.to_csv(NEAR_MISS_CSV, index=False)
    if not os.path.exists(PASSED_ALL_CSV):
        pd.DataFrame(columns=["SMILES","Tg_pred","MAC_pred","Score"]).to_csv(PASSED_ALL_CSV, index=False)
    print(f"[done] strict_top200 → {STRICT_TOP200_CSV} (rows={len(top_div)})", flush=True)
    print(f"[done] passed_all (cumulative) → {PASSED_ALL_CSV}", flush=True)
    print(f"[done] unfiltered_top200 → {UNFILTERED_TOP_CSV} (rows={len(all_top200)})", flush=True)
    print(f"[done] near_misses → {NEAR_MISS_CSV} (rows={len(near)})", flush=True)

if __name__ == "__main__":
    main()
