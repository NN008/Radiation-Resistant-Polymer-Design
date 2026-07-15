#!/usr/bin/env python3
"""
Make Methods/Results figures (matplotlib only).
Reads tidy CSVs (build_figure_data.py) AND directly reads 1Dataset/PI1M_with_Tg_Final.csv
to build a 4x4 grid of per-descriptor threshold counts.
"""

import os
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from pathlib import Path
from math import sqrt

# -------------------- GLOBALS --------------------
TG_THRESH  = 215.0
MAC_THRESH = 0.0569

FIGSIZE_STD = (7, 4.5)
DPI = 300

# -------------------- IO UTILS -------------------
def _read(path):
    p = Path(path)
    if not p.exists():
        print(f"[skip] {p}")
        return None
    return pd.read_csv(p)

def _save(fig, name):
    fig.tight_layout()
    fig.savefig(name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {name}")

# ================== METHODS FIGS =================

def fig_data_funnel(out="Data_Funnel.png"):
    """(Simple placeholder) If you already have a funnel image, keep this or skip."""
    stages = ["Raw PI1M+Bicerano+JCIM","Standardized SMILES",
              "De-salted / neutral","Pass SMARTS filters",
              "≤120 heavy atoms","Has '*' endpoints"]
    counts = [995803, 988775, 988775, 988775, 988775, 988775]  # use your real counts if desired
    y = np.arange(len(stages))[::-1]
    fig = plt.figure(figsize=(10, 6))
    plt.barh(y, counts)
    plt.yticks(y, stages)
    plt.xlabel("Entries")
    plt.title("Data curation & standardization pipeline (actual counts)")
    total = max(1, counts[0])
    for i, v in enumerate(counts[::-1]):
        pct = 100.0 * v / total
        plt.text(v * 1.01, i, f"{v:,} ({pct:.1f}%)", va="center", fontsize=11)
    _save(fig, out)

def fig_tokenizer_pretty(out="Tokenizer_Conditioning.png"):
    from matplotlib.patches import FancyBboxPatch
    input_str = "<Tg:+0.5><MAC:+0.4>  [BOS]  c1ccc(cc1)N*  [EOS]"
    tokens = ["<Tg:+0.5>", "<MAC:+0.4>", "[BOS]",
              "c","1","c","c","c","(","c","c","1",")","N","*","[EOS]"]
    fig = plt.figure(figsize=(12, 3.6))
    ax = plt.gca(); ax.set_axis_off()
    ax.text(0.03, 0.80, "Input string:", ha="left", va="center", fontsize=18)
    ax.text(0.23, 0.80, input_str, ha="left", va="center",
            fontsize=18, fontfamily="monospace")
    ax.text(0.03, 0.46, "Tokenized sequence:", ha="left", va="center", fontsize=18)
    x = 0.23; y = 0.46; w = 0.055; h = 0.10; pad = 0.008
    for tok in tokens:
        ax.add_patch(FancyBboxPatch((x, y-h/2), w, h,
                    boxstyle="round,pad=0.01,rounding_size=0.02",
                    fc="white", ec="black", lw=1.3))
        ax.text(x + w/2, y, tok, ha="center", va="center",
                fontsize=16, fontfamily="monospace")
        x += w + pad
    ax.text(0.03, 0.16,
            "Prefix tokens encode property targets; the model learns a mapping "
            "(properties → structure). Wildcard '*' marks polymerizable endpoints.",
            ha="left", va="center", fontsize=14)
    _save(fig, out)

# -------- NEW: 16-panel descriptor thresholds (no RDKit needed) --------

def fig_descriptor_thresholds_from_final(
    out="assets/Descriptor_Thresholds2.png",
    src_path="1Dataset/PI1M_Tg_MAC.csv",
    save_individual=False,
    individual_dir="DescriptorPanels"
):
    """
    Build a 4×4 panel: each subplot is ONE RDKit descriptor with counts at fixed thresholds.
    Uses existing columns in PI1M_with_Tg_Final.csv (no recompute).

    Columns expected (16):
      ExactMolWt, HeavyAtomMolWt, MolLogP, TPSA, NumHAcceptors, NumHDonors, NumRings,
      FractionCSP3, LabuteASA, BalabanJ, BertzCT, Chi0v, Chi1n, Kappa1, Kappa2, NumHeavyAtoms
    """
    REQUIRED = [
        "ExactMolWt","HeavyAtomMolWt","MolLogP","TPSA","NumHAcceptors","NumHDonors",
        "NumRings","FractionCSP3","LabuteASA","BalabanJ","BertzCT","Chi0v","Chi1n",
        "Kappa1","Kappa2","NumHeavyAtoms"
    ]

    df = pd.read_csv(src_path, low_memory=False)
    for c in REQUIRED:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns in {src_path}: {missing}")
        return

    total = len(df)
    TH = {
        "ExactMolWt":     [("≥200",200),("≥400",400),("≥600",600)],
        "HeavyAtomMolWt": [("≥200",200),("≥400",400),("≥600",600)],
        "MolLogP":        [("≥0",0),("≥2",2),("≥4",4)],
        "TPSA":           [("≥20",20),("≥60",60),("≥100",100)],
        "NumHAcceptors":  [("≥1",1),("≥2",2),("≥4",4)],
        "NumHDonors":     [("≥1",1),("≥2",2)],
        "NumRings":       [("≥1",1),("≥2",2),("≥3",3)],
        "FractionCSP3":   [("≥0.2",0.2),("≥0.4",0.4),("≥0.6",0.6)],
        "LabuteASA":      [("≥200",200),("≥300",300),("≥400",400)],
        "BalabanJ":       [("≥1.0",1.0),("≥2.0",2.0)],
        "BertzCT":        [("≥300",300),("≥600",600),("≥900",900)],
        "Chi0v":          [("≥2",2),("≥4",4),("≥6",6)],
        "Chi1n":          [("≥2",2),("≥4",4),("≥6",6)],
        "Kappa1":         [("≥1.0",1.0),("≥2.0",2.0)],
        "Kappa2":         [("≥1.0",1.0),("≥2.0",2.0)],
        "NumHeavyAtoms":  [("≥20",20),("≥40",40),("≥60",60)],
    }

    order = REQUIRED[:]
    ncols, nrows = 4, 4
    fig = plt.figure(figsize=(ncols*5.2, nrows*3.5))

    if save_individual:
        Path(individual_dir).mkdir(parents=True, exist_ok=True)

    # Unified million-scale axis settings
    XMAX = 1_000_000
    TICKS = np.linspace(0, XMAX, 6)  # 0, 0.2M, ..., 1.0M
    tick_formatter = FuncFormatter(lambda x, pos: ("0" if x == 0 else f"{x/1_000_000:.1f}".rstrip("0").rstrip(".")))

    for i, name in enumerate(order, start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        labs, counts = [], []
        thr_list = TH[name]
        col = df[name].dropna()
        for label, thr in thr_list:
            cnt = int((col >= thr).sum())
            labs.append(label)
            counts.append(cnt)

        y = np.arange(len(labs))[::-1]
        ax.barh(y, counts)
        ax.set_yticks(y)
        ax.set_yticklabels(labs)
        ax.set_title(name)

        # ---- unified axis: 0..1 with ×10⁶ label ----
        ax.set_xlim(0, XMAX)
        ax.xaxis.set_major_locator(FixedLocator(TICKS))
        ax.xaxis.set_major_formatter(tick_formatter)
        ax.set_xlabel("Count (×10⁶)")

        # annotations, clamped so they don't cross the right edge
        pad = 0.01 * XMAX
        right_guard = 0.985 * XMAX
        for j, v in enumerate(counts[::-1]):
            pct = (v/total*100.0) if total else 0.0
            x_annot = min(v + pad, right_guard)
            ax.text(x_annot, j, f"{v:,} ({pct:.1f}%)", va="center", fontsize=9)

        # optional per-panel images (inherits same axis style)
        if save_individual:
            f2 = plt.figure(figsize=(5.5, 3.5))
            ax2 = f2.add_subplot(1,1,1)
            ax2.barh(y, counts)
            ax2.set_yticks(y); ax2.set_yticklabels(labs)
            ax2.set_title(name)
            ax2.set_xlim(0, XMAX)
            ax2.xaxis.set_major_locator(FixedLocator(TICKS))
            ax2.xaxis.set_major_formatter(tick_formatter)
            ax2.set_xlabel("Count (×10⁶)")
            for j, v in enumerate(counts[::-1]):
                pct = (v/total*100.0) if total else 0.0
                ax2.text(min(v + pad, right_guard), j, f"{v:,} ({pct:.1f}%)", va="center", fontsize=9)
            f2.tight_layout()
            out_single = Path(individual_dir) / f"{name}_thresholds.png"
            f2.savefig(out_single, dpi=DPI, bbox_inches="tight"); plt.close(f2)
            print(f"[OK] {out_single}")

    plt.suptitle("Descriptor thresholds on Final Dataset\nNote: MolWt omitted; ExactMolWt used.",
                 y=0.995, fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out} (n={total:,})")


# ================== RESULTS FIGS =================

def fig_pareto(in_csv="logs/generated_candidates.csv", out="assets/Pareto_Tg_MAC.png"):
    df = _read(in_csv)
    if df is None: return
    fig = plt.figure(figsize=(7.2, 5.4))
    ax = plt.gca()
    ax.scatter(df["pred_Tg"], df["pred_MAC"],
               s=16, alpha=0.35, edgecolors="none",
               label="Candidates", zorder=2)
    ax.axvline(TG_THRESH, lw=1.2, zorder=3)
    ax.axhline(MAC_THRESH, lw=1.2, zorder=3)
    if "is_hit" in df.columns and (df["is_hit"] == 1).any():
        h = df[df["is_hit"] == 1]
        ax.scatter(h["pred_Tg"], h["pred_MAC"], s=120, marker="*",
                   label="Dual-objective hits", zorder=4)
    # shade feasible quadrant using current limits
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    xmin = max(TG_THRESH, x0)
    ymin = (MAC_THRESH - y0) / (y1 - y0)
    ax.axvspan(xmin, x1, ymin=max(0.0, ymin), ymax=1.0, alpha=0.08, zorder=1)
    ax.set_xlabel("Predicted Tg (°C)")
    ax.set_ylabel(r"Predicted MAC (cm$^2$/g)")
    ax.set_title("Dual-objective map (predicted Tg vs MAC)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    _save(fig, out)

def _wilson_interval(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * sqrt((p*(1-p) + z*z/(4*n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def fig_rounds(in_csv="logs/round_metrics.csv", out="assets/Round_HitRate.png"):
    df = _read(in_csv)
    if df is None or not {"n_samples", "hit_count"}.issubset(df.columns):
        print("[info] no round_metrics; skipping.")
        return
    if "run" in df.columns:
        df = df.sort_values("run").reset_index(drop=True)
    n = df["n_samples"].to_numpy()
    k = df["hit_count"].to_numpy()
    p = np.divide(k, n, out=np.zeros_like(k, dtype=float), where=n>0)
    lo, hi = [], []
    for ki, ni in zip(k, n):
        a, b = _wilson_interval(int(ki), int(ni))
        lo.append(a*100.0); hi.append(b*100.0)
    fig = plt.figure(figsize=FIGSIZE_STD)
    idx = np.arange(len(df))
    plt.plot(idx, p*100.0)
    plt.fill_between(idx, lo, hi, alpha=0.15, label="95% CI")
    for i, (pi, ni) in enumerate(zip(p, n)):
        plt.text(i, pi*100.0 + 1.0, f"n={int(ni)}", ha="center", fontsize=9)
    plt.xlabel("Run index (sorted)")
    plt.ylabel("Hit rate (%)")
    plt.title("Closed-loop efficiency by run")
    plt.legend(loc="upper left", frameon=True)
    _save(fig, out)

# ===================== MAIN ======================

if __name__ == "__main__":
    # Methods
    fig_descriptor_thresholds_from_final(
        out="assets/Descriptor_Thresholds2.png",
        src_path="1Dataset/PI1M_with_Tg_Final.csv",
        save_individual=False   # set True to also dump 16 separate PNGs
    )
    fig_pareto("logs/generated_candidates.csv", "assets/Pareto_Tg_MAC.png")
    print("Pareto figure saved as assets/Pareto_Tg_MAC.png")
    plt.show()
