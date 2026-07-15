# Closed-Loop Deep Generative Framework for Radiation-Resistant Polymer Inverse Design

A closed-loop generative AI pipeline for the inverse design of radiation-resistant polymers, combining surrogate property prediction, a property-conditional Transformer generator, and diversity-aware candidate selection.

## Overview
Radiation-resistant polymers are critical for aerospace, medical, and energy applications, yet traditional discovery cycles span decades. Inspired by AlphaFold's impact on structural biology, this framework couples ML property predictors with a generative Transformer in an autonomous loop — enabling rapid, targeted discovery of novel polymer candidates with high glass transition temperature (Tg) and mass attenuation coefficient (MAC).

## Framework

### 1. Dataset Enrichment
- Merged three polymer datasets: **PI1M**, **Bicerano**, and **JCIM BigSMILES**
- Standardized all entries to canonical SMILES with wildcard atoms (`*`) marking polymerization points
- Computed 17 RDKit molecular descriptors per repeat unit (MolWt, MolLogP, TPSA, FractionCSP3, BertzCT, NumRings, etc.)

### 2. Surrogate Models (Random Forest)
- Trained separate RF regressors for Tg and MAC using scaffold-aware splits
- Used to impute missing labels across the full dataset
- **MAC: R² > 0.99**, **Tg: R² > 0.90**
- Quantile regression forests provide uncertainty estimates (ΔPI) used to penalize low-confidence candidates

### 3. Property-Conditional Transformer
- Decoder-only GPT-style architecture conditioned on standardized Tg and MAC control tokens
- Generates chemically valid polymer SMILES sequences
- **>98% SMILES validity**, **>95% uniqueness**, mean novelty ~0.79 (Tanimoto distance)
- SMARTS-based safety filters remove unstable substructures

### 4. Closed-Loop Optimization
Each iteration: Generate → Featurize → Predict → Score → Cluster & Select → Calibrate

Composite scoring function:
```
S(x) = α·sTg(x) + β·sMAC(x) - γ·(1 - novelty(x)) - δ(x)
```
Butina clustering (ECFP2, cutoff=0.72) enforces chemical diversity among selected candidates.

## Results
- Generated 10,000 valid polymer candidates in a single run
- **4 dual-objective hits** meeting Tg ≥ 215°C and MAC ≥ 0.0569 cm²/g with high novelty
- **3.7× higher hit rate** vs. brute-force generation
- **+42% novelty** among top candidates
- Hit rate converged to 40–50% after 10 closed-loop iterations

### Top Candidates
| Lead | Type | Predicted Tg | Predicted MAC |
|---|---|---|---|
| Lead 1 | Pyridine–phenol amide (benzoxazole) | 225–235°C | 0.0569–0.0571 cm²/g |
| Lead 2 | Amide + ring carboxyl (polyimide route) | 220–230°C | 0.0569–0.0570 cm²/g |
| Lead 3 | Catechol amide (benzoxazole/polyimide) | 225–235°C | 0.0569–0.0571 cm²/g |

## Tech Stack
- Python, PyTorch, scikit-learn, RDKit
- pandas, NumPy, Matplotlib
- Jupyter Notebook

## Data & Reproducibility
Datasets and pseudocode are too large for GitHub and are hosted on Zenodo:

**[Zenodo Repository](https://zenodo.org/records/17033425)** — includes enriched datasets, descriptor caches, and pseudocode

### Data Sources
- **PI1M** — polymer SMILES dataset: [PI1M](https://github.com/Ramprasad-Group/PI1M)
- **Bicerano** — polymer property compilation: Bicerano, J. (2002). *Prediction of Polymer Properties* (3rd ed.). CRC Press.
- **JCIM BigSMILES** — Lin et al. (2019). [DOI: 10.1021/acs.jcim.9b00701](https://doi.org/10.1021/acs.jcim.9b00701)

## Structure
```
Radiation-Resistant-Polymer-Design/
├── 1Dataset/            # Dataset merging, SMILES standardization, descriptor computation
├── 2Predictors/         # RF regressors for Tg and MAC (surrogate models)
├── 3Smile_Generation/   # Property-conditional Transformer generator
├── 4Validation/         # Candidate scoring, validation, and output batches
├── data/                # Descriptor cache, pooled data
├── Runs/                # Timestamped closed-loop run artifacts
├── out_next/            # Staging dir the closed loop reads its next batch from
├── scripts/             # Standalone figure/report generation scripts (run from repo root)
├── logs/                # Aggregated CSV outputs used by the figure scripts (round_metrics, generated_candidates, etc.)
├── assets/              # Generated figures for the poster/paper (Figures/, Figures2/, MethodFigs/, ...)
├── run_loop_once.py     # Main closed-loop driver (run from repo root)
├── README.md
└── Pub_readme.md        # Companion README for the Zenodo data release
```

## Usage
```bash
pip install torch scikit-learn rdkit pandas numpy matplotlib
jupyter notebook
```
> Model weights (.pt, .ckpt, .pkl) are not included. Re-run training notebooks to regenerate. Full datasets available on [Zenodo](https://zenodo.org/records/17033425).
