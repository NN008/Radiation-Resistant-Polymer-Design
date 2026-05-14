#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Select the next candidate batch from validated polymers using score + diversity.

Inputs:
  - CSV with columns:
      SMILES, Tg, MAC, (optional) weighted_score or score, (optional) unc_Tg, unc_MAC,
      and your 17 RDKit descriptors (any column names; auto-detected).

Outputs (in --outdir):
  - selected_next_batch.csv
  - clustered_candidates.csv
  - selection_report.txt

Policy (when --k < 0): best-per-cluster (layered)
  • Take the best (rank-1) from each cluster (layer 0).
  • If you want more, set --layers > 1 to also take rank-2 per cluster (layer 1), etc.,
    in priority order of clusters (priority = quality of the cluster leader).
  • If --k >= 0 is given, we select up to K using the same layered order.

PLUS: After selection, prune pairs with Tanimoto similarity ≥ --min_similarity,
keeping the higher-scoring one (uses the same fingerprint settings).

Run example:
  python batch.py \
    --csv polymer_candidates_with_rdkit17.csv \
    --outdir out_next \
    --k -1 \
    --layers 1 \
    --min_similarity 0.70 --fp_bits 2048 --fp_radius 2
"""

import argparse
import textwrap
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

# ------------------------- RDKit (optional) -------------------------
RDKit_OK = True
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina
except Exception:
    RDKit_OK = False
    warnings.warn("RDKit not available. Falling back to descriptor-only diversity (no post-selection pruning).", RuntimeWarning)


def _norm_name(s: str) -> str:
    return s.strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def _find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    """Case/format-insensitive column finder."""
    lookup = {_norm_name(c): c for c in df.columns}
    for n in names:
        nn = _norm_name(n)
        if nn in lookup:
            return lookup[nn]
    return None


def _assert_has(df: pd.DataFrame, needed: List[str]) -> None:
    missing = [c for c in needed if _find_col(df, [c]) is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _compute_score_if_needed(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    # Prefer a precomputed weighted score; else fall back to Tg*MAC "score".
    col_score = _find_col(df, ["weighted_score"])
    col_tg   = _find_col(df, ["Tg", "tg", "tg_pred", "Tg_pred", "Tg_predicted"])
    col_mac  = _find_col(df, ["MAC", "mac", "mac_pred", "MAC_pred", "MAC_predicted"])

    if col_score is None:
        col_score = _find_col(df, ["score"])
        if col_score is None:
            if col_tg is None or col_mac is None:
                raise ValueError("Cannot compute score: need Tg and MAC columns or an existing score column.")
            df["score"] = pd.to_numeric(df[col_tg], errors="coerce").astype(float) * \
                          pd.to_numeric(df[col_mac], errors="coerce").astype(float)
            col_score = "score"
    return df, col_score


def _get_unc_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    return _find_col(df, ["unc_Tg", "uncTg", "uncertainty_Tg"]), _find_col(df, ["unc_MAC", "uncMAC", "uncertainty_MAC"])


def _pick_descriptor_columns(df: pd.DataFrame, exclude_cols: List[str]) -> List[str]:
    exclude_norm = set(_norm_name(c) for c in exclude_cols)
    numeric_cols = []
    for c in df.columns:
        if _norm_name(c) in exclude_norm:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    # Keep exactly 17 if clearly present; otherwise keep all numeric (fallback is robust).
    if len(numeric_cols) >= 17:
        pref_subs = ["desc", "descriptor", "rdkit", "calc", "prop"]
        preferred = [c for c in numeric_cols if any(s in _norm_name(c) for s in pref_subs)]
        if len(preferred) >= 17:
            return preferred[:17]
    return numeric_cols


# --------------------- RDKit utilities ---------------------
def _smiles_to_mol(smiles: str):
    return Chem.MolFromSmiles(smiles) if RDKit_OK else None


def _mol_to_fp(mol, radius=2, nBits=2048):
    if not RDKit_OK or mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    except Exception:
        return None


def _butina_cluster_fps(fps: List, dist_cutoff: float) -> List[Tuple[int]]:
    """Return list of clusters (tuples of indices) from fingerprints."""
    dists = []
    n = len(fps)
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])  # distance = 1 - similarity
    clusters = Butina.ClusterData(dists, nPts=n, distThresh=dist_cutoff, isDistData=True)
    return list(clusters)


# -------------- Descriptor-only diversity (fallback) -------------------
def _safe_minmax_scale(X: np.ndarray) -> np.ndarray:
    X = X.astype(float)
    mins = np.nanmin(X, axis=0)
    maxs = np.nanmax(X, axis=0)
    rng = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
    X = (X - mins) / rng
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    return X


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _greedy_maxmin_diverse_indices(X: np.ndarray, scores: np.ndarray, k: int) -> List[int]:
    """
    Pick k items with max-min diversity seeded by highest score.
    X: (n,d) normalized descriptor matrix
    scores: (n,) numeric, higher is better
    """
    n = X.shape[0]
    if n <= k:
        return list(range(n))

    order = np.argsort(-scores)  # seed with best score
    selected = [int(order[0])]
    remaining = set(range(n))
    remaining.remove(selected[0])

    while len(selected) < k and remaining:
        best_i = None
        best_val = -1.0
        for i in list(remaining):
            min_dist = min(1.0 - _cosine_sim(X[i], X[j]) for j in selected)
            val = (min_dist, scores[i])  # tie-break by score
            if best_i is None or val > best_val:
                best_val = val
                best_i = i
        selected.append(best_i)
        remaining.remove(best_i)

    return selected


# --------------------------- Selection logic ---------------------------
def _rank_key(row, col_score, col_unc_tg, col_unc_mac, col_mac, col_tg):
    # Lower uncertainties are better; if missing, treat as 0 penalty.
    unc_sum = 0.0
    if col_unc_tg is not None and not pd.isna(row[col_unc_tg]):
        unc_sum += float(row[col_unc_tg])
    if col_unc_mac is not None and not pd.isna(row[col_unc_mac]):
        unc_sum += float(row[col_unc_mac])
    # Sort by: score desc, uncertainty asc, MAC desc, Tg desc
    return (-float(row[col_score]),
            float(unc_sum),
            -float(row[col_mac]) if col_mac else 0.0,
            -float(row[col_tg]) if col_tg else 0.0)


def _post_prune_similar(selected_df: pd.DataFrame,
                        col_smiles: str,
                        col_score: str,
                        min_similarity: float,
                        fp_bits: int,
                        fp_radius: int) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Greedy keep-by-descending-score. If a candidate is ≥ min_similarity to any kept one,
    drop it and record the reason keyed by __rowid__.
    Returns pruned_df, reasons_map.
    """
    reasons_map: Dict[int, str] = {}

    if (not RDKit_OK) or selected_df.empty or col_smiles not in selected_df.columns:
        return selected_df.copy().reset_index(drop=True), reasons_map

    # Build FPs
    smiles = selected_df[col_smiles].astype(str).tolist()
    mols   = [ _smiles_to_mol(s) for s in smiles ]
    fps    = [ _mol_to_fp(m, radius=fp_radius, nBits=fp_bits) if m is not None else None for m in mols ]

    scores = pd.to_numeric(selected_df[col_score], errors="coerce").fillna(-np.inf).to_numpy(dtype=float)
    order  = list(np.argsort(-scores))  # high score first

    kept_idx: List[int] = []
    kept_fps: List = []
    kept_smiles: List[str] = []

    # We'll need __rowid__ for reasons
    rid_col = "__rowid__"
    if rid_col not in selected_df.columns:
        selected_df = selected_df.reset_index(drop=True)
        selected_df[rid_col] = np.arange(len(selected_df))

    for i in order:
        if fps[i] is None:
            # No fingerprint; keep conservatively
            kept_idx.append(i)
            kept_fps.append(None)
            kept_smiles.append(smiles[i])
            continue

        drop = False
        for k_idx, k_fp in zip(kept_idx, kept_fps):
            if k_fp is None:
                continue
            sim = DataStructs.TanimotoSimilarity(fps[i], k_fp)
            if sim >= float(min_similarity):
                # Drop i; winner is k_idx (higher score due to sort order)
                rid_loser = int(selected_df.iloc[i][rid_col])
                winner_smiles = kept_smiles[kept_idx.index(k_idx)]
                reasons_map[rid_loser] = (
                    f"dropped by cross-cluster similarity ≥ {min_similarity:.3f} vs {winner_smiles} "
                    f"(Tanimoto≈{sim:.3f}; lower {col_score})"
                )
                drop = True
                break

        if not drop:
            kept_idx.append(i)
            kept_fps.append(fps[i])
            kept_smiles.append(smiles[i])

    pruned = selected_df.iloc[kept_idx].copy().reset_index(drop=True)
    return pruned, reasons_map


def select_next_batch(df: pd.DataFrame,
                      k: int = -1,
                      layers: int = 1,
                      min_similarity: float = 0.70,
                      fp_bits: int = 2048,
                      fp_radius: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Returns:
      selected_df, clustered_df, report_text
    """
    # Core columns
    col_smiles = _find_col(df, ["SMILES", "smiles"])
    _assert_has(df, ["SMILES"])
    df, col_score = _compute_score_if_needed(df)
    col_tg  = _find_col(df, ["Tg", "tg", "tg_pred", "Tg_pred", "Tg_predicted"])
    col_mac = _find_col(df, ["MAC", "mac", "mac_pred", "MAC_pred", "MAC_predicted"])
    col_unc_tg, col_unc_mac = _get_unc_cols(df)

    # Sanitize and copy
    work = df.copy().reset_index(drop=True)
    work["__rowid__"] = np.arange(len(work))

    # Try RDKit path first
    clusters = []
    cluster_ids = np.full(len(work), -1, dtype=int)
    fp_ok_count = 0
    invalid_smiles_idx = []
    RDKit_OK_fallback = False

    if RDKit_OK:
        mols = []
        fps  = []
        for i, s in enumerate(work[col_smiles]):
            mol = None
            try:
                mol = _smiles_to_mol(str(s))
            except Exception:
                mol = None
            mols.append(mol)
            if mol is None:
                invalid_smiles_idx.append(i)
                fps.append(None)
                continue
            fp = _mol_to_fp(mol, radius=fp_radius, nBits=fp_bits)
            if fp is None:
                invalid_smiles_idx.append(i)
            else:
                fp_ok_count += 1
            fps.append(fp)

        rdkit_valid_mask = np.array([f is not None for f in fps])
        valid_idx = np.where(rdkit_valid_mask)[0]
        if len(valid_idx) >= 2:
            cutoff = max(0.0, min(1.0, 1.0 - float(min_similarity)))
            fps_valid = [fps[i] for i in valid_idx]
            clusters_idx = _butina_cluster_fps(fps_valid, dist_cutoff=cutoff)
            for cid, tup in enumerate(clusters_idx):
                for rel_i in tup:
                    abs_i = int(valid_idx[rel_i])
                    cluster_ids[abs_i] = cid
            cur = int(len(clusters_idx))
            for i in range(len(work)):
                if cluster_ids[i] < 0:
                    cluster_ids[i] = cur
                    cur += 1
        else:
            RDKit_OK_fallback = True
        max_cid = int(cluster_ids.max())
        clusters = [tuple(np.where(cluster_ids == c)[0].tolist()) for c in range(max_cid + 1)]
    else:
        RDKit_OK_fallback = True

    # If RDKit path failed or insufficient, use descriptor-only greedy diversity
    if (not RDKit_OK) or RDKit_OK_fallback or len(clusters) == 0:
        exclude = [col for col in [col_smiles, col_score, col_tg, col_mac, col_unc_tg, col_unc_mac] if col]
        desc_cols = _pick_descriptor_columns(work, exclude_cols=exclude)
        if len(desc_cols) == 0:
            raise ValueError("No numeric descriptor columns found for fallback diversity selection.")
        X = work[desc_cols].to_numpy()
        X = _safe_minmax_scale(X)
        # single cluster; layered policy will handle ordering
        cluster_ids = np.full(len(work), 0, dtype=int)
        clusters = [tuple(range(len(work)))]

    # Rank each cluster
    ranked_lists = []
    for cid, members in enumerate(clusters):
        rows = work.iloc[list(members)].copy()
        rows["__cluster_id__"] = cid

        # Stable multi-key ranking: score↓, (unc_Tg+unc_MAC)↑, MAC↓, Tg↓
        rows["__rk0"] = -pd.to_numeric(rows[col_score], errors="coerce").fillna(-np.inf)
        rk1 = 0.0
        if col_unc_tg:  rk1 += pd.to_numeric(rows[col_unc_tg], errors="coerce").fillna(0.0)
        if col_unc_mac: rk1 += pd.to_numeric(rows[col_unc_mac], errors="coerce").fillna(0.0)
        rows["__rk1"] = rk1
        rows["__rk2"] = -pd.to_numeric(rows[col_mac], errors="coerce").fillna(0.0) if col_mac else 0.0
        rows["__rk3"] = -pd.to_numeric(rows[col_tg],  errors="coerce").fillna(0.0) if col_tg  else 0.0

        rows = rows.sort_values(by=["__rk0", "__rk1", "__rk2", "__rk3"],
                                ascending=[True, True, True, True]) \
                   .drop(columns=["__rk0", "__rk1", "__rk2", "__rk3"])
        rows["__rank_in_cluster__"] = np.arange(1, len(rows) + 1)
        ranked_lists.append(rows)

    ranked_all = pd.concat(ranked_lists, ignore_index=True)
    ranked_all["cluster_id"] = ranked_all["__cluster_id__"]
    ranked_all["rank_in_cluster"] = ranked_all["__rank_in_cluster__"]
    ranked_all = ranked_all.drop(columns=["__cluster_id__", "__rank_in_cluster__"], errors="ignore")

    # --- Cluster priority by leader quality (best row per cluster) ---
    def _leader_key_for_cluster(cid: int):
        rows_c = ranked_lists[cid]
        if rows_c.empty:
            return (float("inf"), float("inf"), float("inf"), float("inf"))
        best = rows_c.iloc[0]
        return _rank_key(best, col_score, col_unc_tg, col_unc_mac, col_mac, col_tg)

    cluster_order = sorted(range(len(clusters)), key=_leader_key_for_cluster)

    # Best-per-cluster layered selection
    if k is None or k < 0:
        K_target = min(len(ranked_all), len(clusters) * max(1, int(layers)))
    else:
        K_target = min(k, len(ranked_all))

    selected_rows = []
    per_cluster_rows = {cid: list(ranked_lists[cid].itertuples(index=False, name=None))
                        for cid in range(len(clusters))}
    layer = 0
    while len(selected_rows) < K_target:
        progressed = False
        for cid in cluster_order:
            rows_c = per_cluster_rows[cid]
            if layer < len(rows_c):
                selected_rows.append(rows_c[layer])
                progressed = True
                if len(selected_rows) == K_target:
                    break
        if not progressed:
            break
        layer += 1

    selected_df = pd.DataFrame(selected_rows, columns=ranked_all.columns)

    # ---------------- Post-selection pruning for cross-cluster similarity ----------------
    # Keep higher-scoring one when Tanimoto ≥ min_similarity.
    pruned_reasons: Dict[int, str] = {}
    if not selected_df.empty:
        selected_df, pruned_reasons = _post_prune_similar(
            selected_df,
            col_smiles=_find_col(selected_df, ["SMILES", "smiles"]),
            col_score=col_score,
            min_similarity=float(min_similarity),
            fp_bits=int(fp_bits),
            fp_radius=int(fp_radius),
        )

    # ---------------- Why-not-selected analysis (with detailed reasons) ----------------
    sel_by_cluster = {}
    if len(selected_df) > 0:
        smi_col_sel  = _find_col(selected_df, ["SMILES", "smiles"])
        for _, r in selected_df.iterrows():
            sel_by_cluster[int(r["cluster_id"])] = r

    not_selected = ranked_all[~ranked_all["__rowid__"].isin(selected_df["__rowid__"] if len(selected_df) else [])].copy()
    reasons = []
    smi_col_all   = _find_col(ranked_all, ["SMILES", "smiles"])
    rdkit_invalid_set = set(invalid_smiles_idx)  # indices in 'work', same as '__rowid__'

    def _unc_sum(row):
        s = 0.0
        if col_unc_tg and not pd.isna(row[col_unc_tg]):  s += float(row[col_unc_tg])
        if col_unc_mac and not pd.isna(row[col_unc_mac]): s += float(row[col_unc_mac])
        return s

    def _explain_outranking(r, w):
        eps = 1e-12
        sc_r, sc_w = float(r[col_score]), float(w[col_score])
        if sc_r + eps < sc_w: return "lower score"
        ur, uw = _unc_sum(r), _unc_sum(w)
        if ur > uw + eps:     return "higher uncertainty (unc_Tg+unc_MAC)"
        if col_mac and not (pd.isna(r[col_mac]) or pd.isna(w[col_mac])):
            mr, mw = float(r[col_mac]), float(w[col_mac])
            if mr + eps < mw: return "lower MAC"
        if col_tg and not (pd.isna(r[col_tg]) or pd.isna(w[col_tg])):
            tr, tw = float(r[col_tg]), float(w[col_tg])
            if tr + eps < tw: return "lower Tg"
        return "tie-break"

    cluster_priority_rank = {cid: i + 1 for i, cid in enumerate(cluster_order)}

    for _, r in not_selected.iterrows():
        rid = int(r["__rowid__"])
        smi = r[smi_col_all]
        # If dropped by pruning, explain that first.
        if rid in pruned_reasons:
            reasons.append(f"  {smi} — DROPPED after selection: {pruned_reasons[rid]}")
            continue

        cid = int(r["cluster_id"])
        ric = int(r.get("rank_in_cluster", 999999))
        rdkit_tag = " [RDKit FP invalid]" if rid in rdkit_invalid_set else ""

        if cid in sel_by_cluster:
            w = sel_by_cluster[cid]
            smi_winner = w[smi_col_sel] if smi_col_sel else "(selected)"
            ric_winner = int(w.get("rank_in_cluster", 1))
            why = _explain_outranking(r, w)
            reasons.append(
                f"  {smi} — DROPPED: lower layer within the same cluster (rank {ric} vs {ric_winner}); "
                f"policy selects the cluster representative first (best-per-cluster). Reason: {why}.{rdkit_tag}"
            )
        else:
            pr = cluster_priority_rank.get(cid, None)
            rank_str = f" (cluster priority {pr}/{len(clusters)})" if pr is not None else ""
            reasons.append(
                f"  {smi} — NOT SELECTED: budget filled before this cluster’s layer 0 pick "
                f"(best-per-cluster policy){rank_str}.{rdkit_tag}"
            )

    # ---------------- Build report ----------------
    lines = []
    lines.append("=== Selection Report ===\n")
    lines.append(f"Total candidates: {len(work)}")
    lines.append(f"RDKit available: {RDKit_OK}")
    if RDKit_OK:
        lines.append(f"Valid RDKit fingerprints: {fp_ok_count}/{len(work)}")
    lines.append(f"Clusters formed: {len(clusters)}")
    lines.append("Policy: best-per-cluster (layered) + post-selection pruning of pairs with "
                 f"Tanimoto ≥ {float(min_similarity):.3f}")
    lines.append("Cluster priority (best-first): " + ", ".join(map(str, cluster_order)))
    lines.append(f"Requested/Effective K: {'auto' if (k is None or k < 0) else k}, Selected: {len(selected_df)}")
    lines.append(f"Min similarity for Butina/pruning: {float(min_similarity):.3f}")
    if len(invalid_smiles_idx) > 0:
        bad_idx_str = ", ".join(map(str, invalid_smiles_idx))
        lines.append(f"Invalid SMILES rows (excluded from RDKit path): {bad_idx_str}")

    # Show selected with the exact score column used
    score_display_col = col_score
    sel_cols_show = [c for c in [
        _find_col(selected_df, ['SMILES','smiles']),
        _find_col(selected_df, ['Tg','tg','Tg_pred','tg_pred','Tg_predicted']),
        _find_col(selected_df, ['MAC','mac','MAC_pred','mac_pred','MAC_predicted']),
        score_display_col,
        "cluster_id", "rank_in_cluster"
    ] if c]
    topview = selected_df[sel_cols_show].copy()
    topview = topview.sort_values(by=score_display_col, ascending=False)
    lines.append(f"\nSelected (sorted by {score_display_col}):")
    lines.append(topview.to_string(index=False))

    lines.append("\nNot selected and reasons:")
    if reasons:
        lines.extend(reasons)
    else:
        lines.append("  (none)")

    lines.append("\nCluster sizes:")
    for cid, members in enumerate(clusters):
        lines.append(f"  - Cluster {cid}: {len(members)}")
    report_text = "\n".join(lines)

    # For convenience: ensure nice front columns for outputs
    col_smiles_disp = _find_col(selected_df, ["SMILES","smiles"])
    col_tg_disp  = _find_col(selected_df, ["Tg","tg","Tg_pred","tg_pred","Tg_predicted"])
    col_mac_disp = _find_col(selected_df, ["MAC","mac","MAC_pred","mac_pred","MAC_predicted"])

    front_cols = [c for c in [col_smiles_disp, col_tg_disp, col_mac_disp, score_display_col,
                              "cluster_id", "rank_in_cluster"] if c in selected_df.columns]
    back_cols_sel = [c for c in selected_df.columns if c not in front_cols]
    selected_df = selected_df[front_cols + back_cols_sel]

    front_cols_all = [c for c in [col_smiles_disp, col_tg_disp, col_mac_disp, score_display_col,
                                  "cluster_id", "rank_in_cluster"] if c in ranked_all.columns]
    back_cols_all  = [c for c in ranked_all.columns if c not in front_cols_all]
    ranked_all = ranked_all[front_cols_all + back_cols_all]

    return selected_df, ranked_all, report_text


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--csv", required=True, help="Path to validated candidates CSV (with SMILES, Tg, MAC, etc.)")
    ap.add_argument("--outdir", default="out_next", help="Directory for outputs")
    ap.add_argument("--k", type=int, default=-1, help="If <0, auto: best-per-cluster layered; else fixed-K cap")
    ap.add_argument("--layers", type=int, default=1,
                    help="When --k < 0 (auto), select best-per-cluster for this many layers (1=leaders only).")
    ap.add_argument("--k_min", type=int, default=2, help="(unused in layered policy; kept for CLI compatibility)")
    ap.add_argument("--k_max", type=int, default=9, help="(unused in layered policy; kept for CLI compatibility)")
    ap.add_argument("--k_frac", type=float, default=0.33, help="(unused in layered policy; kept for CLI compatibility)")
    ap.add_argument("--min_similarity", type=float, default=0.70, help="Butina + pruning min Tanimoto similarity")
    ap.add_argument("--fp_bits", type=int, default=2048, help="Morgan fingerprint bit length")
    ap.add_argument("--fp_radius", type=int, default=2, help="Morgan fingerprint radius")
    args = ap.parse_args()

    in_csv = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    df = pd.read_csv(in_csv)

    # --- Decide K (auto layered vs fixed) ---
    if args.k < 0:
        k_use = -1
        print(f"[auto] best-per-cluster layered policy: layers={args.layers}")
    else:
        k_use = min(args.k, len(df))

    selected_df, clustered_df, report_text = select_next_batch(
        df,
        k=k_use,
        layers=args.layers,
        min_similarity=args.min_similarity,
        fp_bits=args.fp_bits,
        fp_radius=args.fp_radius
    )

    sel_path = outdir / "selected_next_batch.csv"
    clu_path = outdir / "clustered_candidates.csv"
    rep_path = outdir / "selection_report.txt"

    selected_df.to_csv(sel_path, index=False)
    clustered_df.to_csv(clu_path, index=False)
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    k_effective = len(selected_df)
    print(textwrap.dedent(f"""
    ✔ Done.

    Outputs:
      - {sel_path}
      - {clu_path}
      - {rep_path}

    Notes:
      • Effective K used: {k_effective}
      • Policy: {'best-per-cluster layered' if args.k < 0 else 'fixed-K'} with
        post-selection pruning at Tanimoto ≥ {float(args.min_similarity):.3f}
    """).strip())


if __name__ == "__main__":
    main()
