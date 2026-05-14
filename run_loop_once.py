#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_loop_once.py — screening loop (paths resolved relative to project root).

Pipeline:
  0) 4Validation/run_pipeline.py  → writes polymer_candidates.csv (we ignore its old Tg/MAC/score)
  1) 4Validation/featurize_new_valids.py → add 17 RDKit descriptors (cache: data/desc_cache.csv)
  2) 4Validation/predict_batch.py (frozen models/) → predicted.csv, hits.csv, non_hits.csv
     • Weighted composite score drives hits
     • if hits=0 and --fallback_k>0 → take top-K by weighted_score as hits
  3) 4Validation/batch.py (diverse selection) → selected_next_batch.csv (under this run)
  4) 4Validation/promotion_labels.py (absolute path) → promotion_labels.csv
     • called from project root; we mirror selected into project-root out_next/ for compatibility
  (No retraining step — models remain frozen)

All artifacts for the run go into Runs/loop_YYYYMMDD_HHMMSS/
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Weighted-score config (forwarded to predict_batch.py)
    ap.add_argument("--tg_threshold", type=float, default=203.7, help="Tg cutoff used for normalization")
    ap.add_argument("--mac_threshold", type=float, default=0.0569, help="MAC cutoff used for normalization")
    ap.add_argument("--w_tg", type=float, default=0.7, help="Weight for Tg in weighted score")
    ap.add_argument("--w_mac", type=float, default=0.3, help="Weight for MAC in weighted score")
    ap.add_argument("--min_tg_frac", type=float, default=0.85, help="Minimum Tg_norm required (soft floor)")
    ap.add_argument("--min_mac_frac", type=float, default=0.85, help="Minimum MAC_norm required (soft floor)")
    ap.add_argument("--norm", choices=["threshold","minmax","zscore"], default="threshold",
                    help="Normalization method in predict_batch")
    ap.add_argument("--eps", type=float, default=0.0, help="Tolerance/epsilon in predict_batch")

    # selection knobs
    ap.add_argument("--fallback_k", type=int, default=4, help="Top-K fallback if no hits (0 to disable)")
    ap.add_argument("--select_k", type=int, default=4, help="K to select; pass -1 to auto-size inside batch.py")
    ap.add_argument("--min_similarity", type=float, default=0.75)
    ap.add_argument("--fp_bits", type=int, default=2048)
    ap.add_argument("--fp_radius", type=int, default=2)

    # where per-run artifacts go
    ap.add_argument("--runs_root", default="Runs", help="Root folder for per-run artifacts (relative to project root)")
    # descriptor cache (reused across runs)
    ap.add_argument("--cache", default="data/desc_cache.csv", help="Descriptor cache (relative to project root)")

    # (Optional) pass-through knobs for auto-K if you added them to batch.py
    ap.add_argument("--k_min", type=int, default=None, help="(optional) forwarded to batch.py when provided")
    ap.add_argument("--k_max", type=int, default=None, help="(optional) forwarded to batch.py when provided")
    ap.add_argument("--k_frac", type=float, default=None, help="(optional) forwarded to batch.py when provided")

    return ap.parse_args()


def run(cmd, cwd=None):
    print("[cmd]", " ".join(cmd), f"(cwd={cwd or Path.cwd()})", flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def main():
    args = parse_args()

    # -------- Resolve project layout (relative to this file) --------
    PROJECT_ROOT = Path(__file__).resolve().parent

    VALIDATION_DIR = PROJECT_ROOT / "4Validation"
    PREDICTORS_DIR = PROJECT_ROOT / "2Predictors" / "OgPredictors"  # frozen/OG models dir (ignored by predict_batch)
    MODELS_DIR     = PREDICTORS_DIR

    GEN_SCRIPT        = VALIDATION_DIR / "run_pipeline.py"
    FEATURIZE_SCRIPT  = VALIDATION_DIR / "featurize_new_valids.py"
    PREDICT_SCRIPT    = VALIDATION_DIR / "predict_batch.py"
    SELECTOR_SCRIPT   = VALIDATION_DIR / "batch.py"
    LABELS_SCRIPT     = VALIDATION_DIR / "promotion_labels.py"

    # Preferred generator output locations (newest existing wins)
    GEN_OUT_PREFS = [
        VALIDATION_DIR / "polymer_candidates.csv",                 # current
        VALIDATION_DIR / "Outputs" / "polymer_candidates.csv",     # historical
        PROJECT_ROOT     / "polymer_candidates.csv",               # legacy
    ]

    CACHE_PATH = PROJECT_ROOT / args.cache
    PROJECT_OUTNEXT = PROJECT_ROOT / "out_next"

    # -------- Create per-run directory under Runs/ --------
    ts = time.strftime("%Y%m%d_%H%M%S")
    runs_root = PROJECT_ROOT / args.runs_root
    run_dir = runs_root / f"loop_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # paths inside this run
    gen_copy_csv = run_dir / "polymer_candidates.csv"
    feat_csv     = run_dir / "valid_with_desc.csv"
    pred_csv     = run_dir / "predicted.csv"
    hits_csv     = run_dir / "hits.csv"
    nonhits_csv  = run_dir / "non_hits.csv"
    out_next     = run_dir / "out_next"; out_next.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"

    # ensure shared folders exist
    (PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    PROJECT_OUTNEXT.mkdir(parents=True, exist_ok=True)

    # -------- 0) Generate (ignore built-in scoring) --------
    print(f"[generate] running {GEN_SCRIPT} in {VALIDATION_DIR}")
    run([sys.executable, str(GEN_SCRIPT)], cwd=str(VALIDATION_DIR))

    # find newest generator CSV among preferred locations
    candidates = [p for p in GEN_OUT_PREFS if p.exists()]
    if not candidates:
        raise FileNotFoundError(
            "Expected generator output not found at any of:\n  " +
            "\n  ".join(str(p) for p in GEN_OUT_PREFS) +
            "\nCheck where run_pipeline.py writes polymer_candidates.csv."
        )
    gen_out = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"[generate] found generator output → {gen_out}")

    # load & sanity check
    cand = pd.read_csv(gen_out)
    if "SMILES" not in cand.columns:
        raise ValueError(f"{gen_out} must contain a 'SMILES' column.")
    cand.to_csv(gen_copy_csv, index=False)
    print(f"[generate] copied → {gen_copy_csv} (n={len(cand)})")

    # -------- 1) Featurize (creates/updates cache) --------
    run([
        sys.executable, str(FEATURIZE_SCRIPT),
        "--in", str(gen_copy_csv),
        "--out", str(feat_csv),
        "--cache", str(CACHE_PATH),
    ], cwd=str(PROJECT_ROOT))

    # Mirror into Validation with a conventional name for compatibility:
    VALIDATION_DESC = VALIDATION_DIR / "polymer_candidates_with_rdkit17.csv"
    shutil.copy2(feat_csv, VALIDATION_DESC)
    print(f"[featurize] mirrored descriptors → {VALIDATION_DESC}")

    # -------- 2) Predict & filter (weighted-scoring; models ignored) --------
    predict_cmd = [
        sys.executable, str(PREDICT_SCRIPT),
        "--in", str(feat_csv),
        "--out", str(pred_csv),
        "--hits", str(hits_csv),
        "--nonhits", str(nonhits_csv),
        "--models", str(MODELS_DIR),  # accepted but ignored by the lightweight script
        "--tg", str(args.tg_threshold),
        "--mac", str(args.mac_threshold),
        "--w_tg", str(args.w_tg),
        "--w_mac", str(args.w_mac),
        "--min_tg_frac", str(args.min_tg_frac),
        "--min_mac_frac", str(args.min_mac_frac),
        "--norm", str(args.norm),
        "--eps", str(args.eps),
    ]
    run(predict_cmd, cwd=str(PROJECT_ROOT))

    # Handle empty hits with top-K fallback (by weighted_score)
    pred_df = pd.read_csv(pred_csv) if pred_csv.exists() else pd.DataFrame()
    n_scored = len(pred_df) if not pred_df.empty else 0
    best_weighted_score = float(pred_df["weighted_score"].max()) if n_scored and "weighted_score" in pred_df.columns else 0.0

    hits_df = pd.read_csv(hits_csv) if hits_csv.exists() else pd.DataFrame()
    if hits_df.empty and args.fallback_k > 0 and not pred_df.empty:
        print(f"[fallback] no hits after weighted criteria. Using top-{args.fallback_k} by weighted_score.", flush=True)
        hits_df = pred_df.sort_values("weighted_score", ascending=False).head(args.fallback_k).copy()
        hits_df.to_csv(hits_csv, index=False)

    n_hits = len(hits_df)
    if n_hits == 0:
        summary = {
            "run_dir": str(run_dir),
            "n_candidates": int(len(cand)),
            "n_scored": int(n_scored),
            "n_hits": 0,
            "best_weighted_score": float(best_weighted_score),
            "weighted_config": {
                "tg_cutoff": args.tg_threshold,
                "mac_cutoff": args.mac_threshold,
                "w_tg": args.w_tg, "w_mac": args.w_mac,
                "min_tg_frac": args.min_tg_frac, "min_mac_frac": args.min_mac_frac,
                "norm": args.norm, "eps": args.eps,
            },
            "selected_k": 0,
            "selected_csv": None,
            "note": "No hits; nothing selected.",
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[summary] → {summary_path}")
        print("✔ One-loop pass complete (no hits; stopped before selection).")
        return

    # -------- 3) Select diverse K into this run's out_next --------
    selector_cmd = [
        sys.executable, str(SELECTOR_SCRIPT),
        "--csv", str(hits_csv),
        "--outdir", str(out_next),
        "--k", str(args.select_k),
        "--min_similarity", str(args.min_similarity),
        "--fp_bits", str(args.fp_bits),
        "--fp_radius", str(args.fp_radius),
    ]
    if args.k_min is not None:
        selector_cmd += ["--k_min", str(args.k_min)]
    if args.k_max is not None:
        selector_cmd += ["--k_max", str(args.k_max)]
    if args.k_frac is not None:
        selector_cmd += ["--k_frac", str(args.k_frac)]

    run(selector_cmd, cwd=str(PROJECT_ROOT))

    selected_csv = out_next / "selected_next_batch.csv"
    if not selected_csv.exists():
        raise FileNotFoundError("Selection step did not produce selected_next_batch.csv in the run's out_next.")

    # Mirror selected into project-root out_next for promotion_labels.py compatibility
    PROJECT_OUTNEXT.mkdir(parents=True, exist_ok=True)
    proj_selected = PROJECT_OUTNEXT / "selected_next_batch.csv"
    shutil.copy2(selected_csv, proj_selected)
    print(f"[select] mirrored selected_next_batch.csv → {proj_selected}")

    # -------- 4) Labels --------
    if not LABELS_SCRIPT.exists():
        raise FileNotFoundError(f"Labeling script not found: {LABELS_SCRIPT}")

    run([
        sys.executable, str(LABELS_SCRIPT),
        "--selected", str(proj_selected),
        "--tg", str(args.tg_threshold),
        "--mac", str(args.mac_threshold)
    ], cwd=str(PROJECT_ROOT))

    proj_labels = PROJECT_OUTNEXT / "promotion_labels.csv"
    if not proj_labels.exists():
        raise FileNotFoundError("promotion_labels.py did not write out_next/promotion_labels.csv in project root.")
    run_labels = out_next / "promotion_labels.csv"
    shutil.copy2(proj_labels, run_labels)
    print(f"[labels]  copied promotion_labels.csv → {run_labels}")

    # -------- Summary JSON --------
    summary = {
        "run_dir": str(run_dir),
        "n_candidates": int(len(cand)),
        "n_scored": int(n_scored),
        "n_hits": int(n_hits),
        "best_weighted_score": float(best_weighted_score),
        "weighted_config": {
            "tg_cutoff": args.tg_threshold,
            "mac_cutoff": args.mac_threshold,
            "w_tg": args.w_tg, "w_mac": args.w_mac,
            "min_tg_frac": args.min_tg_frac, "min_mac_frac": args.min_mac_frac,
            "norm": args.norm, "eps": args.eps,
        },
        "selected_k": int(args.select_k),
        "selected_csv": str(selected_csv),
        "labels_csv": str(run_labels),
        "note": "Models kept frozen; weighted_score drives selection.",
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[summary] → {summary_path}")
    print("⟡ Screening phase complete — weighted candidates curated and archived at:", run_dir)


if __name__ == "__main__":
    main()