# STEP 3: Set up
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm

# === CONFIG ===
CHUNK_SIZE = 50000
SAVE_DIR = "/content/drive/MyDrive/Polymer_DesignProj"
os.makedirs(SAVE_DIR, exist_ok=True)

# STEP 4: Upload SMILES file manually in Colab UI
from google.colab import files
uploaded = files.upload()
file_path = next(iter(uploaded))
df = pd.read_csv(file_path)
df.columns = ['SMILES']
df["Name"] = [f"PI1M_{i:05d}" for i in range(len(df))]

# STEP 5: Descriptor function
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * 17
    return [
        Descriptors.MolWt(mol),
        Descriptors.ExactMolWt(mol),
        Descriptors.HeavyAtomMolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        rdMolDescriptors.CalcNumRings(mol),
        Descriptors.FractionCSP3(mol),
        rdMolDescriptors.CalcLabuteASA(mol),
        Descriptors.BalabanJ(mol),
        Descriptors.BertzCT(mol),
        Descriptors.Chi0v(mol),
        Descriptors.Chi1n(mol),
        Descriptors.Kappa1(mol),
        Descriptors.Kappa2(mol),
        mol.GetNumHeavyAtoms()
    ]

descriptor_names = [
    'MolWt', 'ExactMolWt', 'HeavyAtomMolWt', 'MolLogP', 'TPSA',
    'NumHAcceptors', 'NumHDonors', 'NumRings', 'FractionCSP3', 'LabuteASA',
    'BalabanJ', 'BertzCT', 'Chi0v', 'Chi1n', 'Kappa1', 'Kappa2', 'NumHeavyAtoms'
]

# STEP 6: Process in chunks and save each one
tqdm.pandas()
for start in range(0, len(df), CHUNK_SIZE):
    end = min(start + CHUNK_SIZE, len(df))
    chunk = df.iloc[start:end].copy().reset_index(drop=True)
    desc_values = chunk['SMILES'].progress_apply(compute_descriptors)
    desc_df = pd.DataFrame(desc_values.tolist(), columns=descriptor_names)
    combined = pd.concat([chunk, desc_df], axis=1)
    combined.to_csv(f"{SAVE_DIR}/PI1M_descriptors_{start}_{end}.csv", index=False)
    print(f"Saved: PI1M_descriptors_{start}_{end}.csv")
    
# STEP 7: Merge all chunks into final CSV
import glob
chunk_files = sorted(glob.glob(f"{SAVE_DIR}/PI1M_descriptors_*.csv"))
final_df = pd.concat([pd.read_csv(f) for f in chunk_files])
final_df.to_csv(f"{SAVE_DIR}/PI1M_all_descriptors.csv", index=False)
print("All done! Final dataset saved to:")
print(f"{SAVE_DIR}/PI1M_all_descriptors.csv")

import pandas as pd

# === Step 1: Load your PI1M dataset ===
df_pi1m = pd.read_csv("PI1M_with_Tg.csv")

# If there's already a Tg column, rename it to avoid conflict during merge
if "Tg" in df_pi1m.columns:
    df_pi1m.rename(columns={"Tg": "Tg_existing"}, inplace=True)

# === Step 2: Load and clean external Tg datasets ===
tg2 = pd.read_csv("Bicerano_bigsmiles.csv", encoding="ISO-8859-1")
tg3 = pd.read_csv("JCIM_sup_bigsmiles.csv", encoding="ISO-8859-1")

# Convert Bicerano (K) to °C
tg2["Tg (K) exp"] = tg2["Tg (K) exp"] - 273.15

# Keep only SMILES + Tg columns and rename consistently
tg2_clean = tg2[["SMILES", "Tg (K) exp"]].rename(columns={"Tg (K) exp": "Tg"})
tg3_clean = tg3[["SMILES", "Tg (C)"]].rename(columns={"Tg (C)": "Tg"})

# Drop NaNs and combine
all_tg = pd.concat([tg2_clean, tg3_clean], ignore_index=True)
all_tg = all_tg.dropna(subset=["SMILES", "Tg"]).drop_duplicates(subset="SMILES")

# === Step 3: Merge with PI1M ===
df_merged = df_pi1m.merge(all_tg, on="SMILES", how="left")

# === Step 4: Fill in Tg where missing from existing
if "Tg_existing" in df_merged.columns:
    df_merged["Tg"] = df_merged["Tg_existing"].combine_first(df_merged["Tg"])
    df_merged.drop(columns=["Tg_existing"], inplace=True)

# === Step 5: Save final result ===
df_merged.to_csv("PI1M_with_Tg.csv", index=False)
print("Done! Final dataset saved as PI1M_with_Tg_FULL.csv")
num_with_tg = df_merged["Tg"].notnull().sum()
total_polymers = len(df_merged)

print(f"{num_with_tg} polymers have Tg values out of {total_polymers} total")
print(f"That's {100 * num_with_tg / total_polymers:.2f}% coverage")
# Load the merged dataset
df = pd.read_csv("PI1M_with_Tg_FULL.csv")

# Count non-null Tg values
filled_count = df["Tg"].notnull().sum()
total_count = len(df)

print(f"Tg values filled: {filled_count} out of {total_count} polymers")
df_main = pd.read_csv('PI1M_with_Tg_merged.csv')
df_lieconv = pd.read_csv('PI1M_MF_3.csv')
# Step 2: Standardize SMILES column names and strip whitespace
df_main['SMILES'] = df_main['SMILES'].astype(str).str.strip()
df_lieconv['Smiles'] = df_lieconv['Smiles'].astype(str).str.strip()
df_lieconv = df_lieconv.rename(columns={'Smiles': 'SMILES', 'Tg_pred': 'Tg_lieconv'})
# Step 3: Merge on SMILES
df_merged = df_main.merge(df_lieconv[['SMILES', 'Tg_lieconv']], on='SMILES', how='left')
# Step 4: Track how many missing Tg values existed before
num_missing_before = df_merged['Tg'].isna().sum()
# Step 5: Fill missing Tg using LieConv predictions
df_merged['Tg'] = df_merged['Tg'].combine_first(df_merged['Tg_lieconv'])
# Step 6: Track how many were still missing after filling
num_missing_after = df_merged['Tg'].isna().sum()
num_filled = num_missing_before - num_missing_after
# Step 7: Drop temp column
df_merged.drop(columns=['Tg_lieconv'], inplace=True)
# Step 8: Save and report
df_merged.to_csv('PI1M_with_Tg_merged.csv', index=False)
print(f"Tg merge complete. {num_filled} Tg values were filled using LieConv predictions.")
print(f"Final Tg missing count: {num_missing_after}")
# --- Step 1: Setup ---
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
# --- Step 2: Load your partially-filled dataset ---
df = pd.read_csv('PI1M_with_Tg.csv')  # contains SMILES, 17 descriptors, Tg (partially filled)
# --- Step 3: Define the 17 descriptors you want to use ---
descriptor_cols = ['MolWt', 'ExactMolWt', 'HeavyAtomMolWt', 'MolLogP', 'TPSA',
                   'NumHAcceptors', 'NumHDonors', 'NumRings', 'FractionCSP3',
                   'LabuteASA', 'BalabanJ', 'BertzCT', 'Chi0v', 'Chi1n',
                   'Kappa1', 'Kappa2', 'NumHeavyAtoms']
# --- Step 4: Build a low-fidelity Tg estimate using a subset of features ---
proxy_features = [
    'MolWt',
    'TPSA',
    'NumRings',
    'FractionCSP3',
    'Chi0v',
    'Kappa1',
    'NumHAcceptors',
    'NumHeavyAtoms',
    'Chi1n',
    'LabuteASA'
]
proxy_df = df.dropna(subset=proxy_features + ['Tg'])

ridge_proxy = Ridge()
ridge_proxy.fit(proxy_df[proxy_features], proxy_df['Tg'])

df['Tg_LF'] = ridge_proxy.predict(df[proxy_features].fillna(0))
# --- Step 5: Train---
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error
# --- Prepare training data ---
features = descriptor_cols + ['Tg_LF']
known_mask = df['Tg'].notna()
X_train = df.loc[known_mask, features]
y_train = df.loc[known_mask, 'Tg']
# --- Scale for Ridge ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# --- Train RidgeCV (cross-validated Ridge) ---
ridge = RidgeCV(alphas=np.logspace(-2, 2, 10))
ridge.fit(X_train_scaled, y_train)
# --- Grid search for Random Forest ---
rf = RandomForestRegressor(random_state=42)
param_grid_rf = {
    'n_estimators': [300],
    'max_depth': [15, 20],
    'min_samples_leaf': [3, 5],
    'max_features': ['sqrt']
}
rf_cv = GridSearchCV(rf, param_grid_rf, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
rf_cv.fit(X_train, y_train)
rf_best = rf_cv.best_estimator_
# --- Grid search for XGBoost ---
xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
param_grid_xgb = {
    'n_estimators': [300, 500],
    'max_depth': [4, 5],
    'learning_rate': [0.03, 0.05],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'reg_lambda': [2.0],
    'reg_alpha': [0.1]
}
xgb_cv = GridSearchCV(xgb, param_grid_xgb, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
xgb_cv.fit(X_train, y_train)
xgb_best = xgb_cv.best_estimator_
# --- Optional: Evaluate performance on a held-out validation set ---
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

ridge_val = ridge.predict(scaler.transform(X_val))
rf_val = rf_best.predict(X_val)
xgb_val = xgb_best.predict(X_val)

mae_ridge = mean_absolute_error(y_val, ridge_val)
mae_rf = mean_absolute_error(y_val, rf_val)
mae_xgb = mean_absolute_error(y_val, xgb_val)

print("MAE scores on validation set:")
print(f"RidgeCV:      {mae_ridge:.4f}")
print(f"RandomForest: {mae_rf:.4f}")
print(f"XGBoost:      {mae_xgb:.4f}")
# --- Step 6: Predict missing Tg values using weighted ensemble ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# --- Step 6-A: Split known Tg data for validation-based weighting ---
X_val_sub, X_val_hold, y_val_sub, y_val_hold = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

# Retrain models on sub-training set
xgb.fit(X_val_sub, y_val_sub)
rf.fit(X_val_sub, y_val_sub)
ridge.fit(scaler.transform(X_val_sub), y_val_sub)

# Evaluate performance on holdout set
ridge_val = ridge.predict(scaler.transform(X_val_hold))
rf_val = rf.predict(X_val_hold)
xgb_val = xgb.predict(X_val_hold)

mae_ridge = mean_absolute_error(y_val_hold, ridge_val)
mae_rf = mean_absolute_error(y_val_hold, rf_val)
mae_xgb = mean_absolute_error(y_val_hold, xgb_val)
# --- Step 6-B: Compute inverse MAE-based weights ---
inv_mae = np.array([1 / mae_xgb, 1 / mae_rf, 1 / mae_ridge])
weights = inv_mae / inv_mae.sum()
# Optional: Retrain models on full known data for final predictions
xgb.fit(X_train, y_train)
rf.fit(X_train, y_train)
ridge.fit(scaler.transform(X_train), y_train)
# --- Step 6-C: Predict missing Tg values ---
missing_mask = df['Tg'].isna()
X_missing = df.loc[missing_mask, features]

# Fill missing descriptor values — choose zero-fill or mean-fill
X_missing_filled = X_missing.fillna(0)

# Ridge needs scaled inputs
X_missing_scaled = scaler.transform(X_missing_filled)

pred_xgb = xgb.predict(X_missing)
pred_rf = rf.predict(X_missing)
pred_ridge = ridge.predict(X_missing_scaled)
# --- Step 6-D: Report ---
print("Step 6 complete: Tg imputed using MAE-weighted ensemble")
print(f"Ridge MAE:      {mae_ridge:.2f}")
print(f"Random Forest: {mae_rf:.2f}")
print(f"XGBoost:       {mae_xgb:.2f}")
print(f"Weights used:  Ridge={weights[2]:.2f}, RF={weights[1]:.2f}, XGB={weights[0]:.2f}")
print(f"Tg values predicted: {missing_mask.sum()}")
print("Still missing Tg values:", df['Tg'].isna().sum())
df = pd.read_csv("PI1M_with_Tg.csv")
print("Any missing Tg values left? ", df['Tg'].isna().sum())
# --- Step 6: Predict missing Tg values using MAE-weighted ensemble ---
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import numpy as np

# Recreate Tg_LF in case it was dropped
proxy_features = [
    'MolWt', 'TPSA', 'NumRings', 'FractionCSP3',
    'Chi0v', 'Kappa1', 'NumHAcceptors', 'NumHeavyAtoms',
    'Chi1n', 'LabuteASA'
]
proxy_df = df.dropna(subset=proxy_features + ['Tg'])
ridge_proxy = Ridge()
ridge_proxy.fit(proxy_df[proxy_features], proxy_df['Tg'])
df['Tg_LF'] = ridge_proxy.predict(df[proxy_features].fillna(0))

# --- Sanity filter: exclude fragments and non-polymers ---
valid_candidates = (
    (df['MolWt'] > 250) &
    (df['NumHeavyAtoms'] > 20) &
    (df['SMILES'].str.len() >= 5)
)

# --- Identify missing Tg values that are valid for prediction ---
missing_mask = df['Tg'].isna() & valid_candidates

# Save skipped molecules for inspection
skipped = df[df['Tg'].isna() & ~valid_candidates]
skipped.to_csv('skipped_unrealistic_structures.csv', index=False)

# Prepare data for prediction
X_missing = df.loc[missing_mask, descriptor_cols + ['Tg_LF']]

# Impute any missing values
X_missing_imputed = X_missing.copy()
for col in X_missing.columns:
    if X_missing[col].isna().any():
        X_missing_imputed[col] = X_missing[col].fillna(X_train[col].mean())

# Scale for Ridge
X_missing_scaled = scaler.transform(X_missing_imputed)

# Predict with trained models
pred_ridge = ridge.predict(X_missing_scaled)
pred_rf    = rf.predict(X_missing_imputed)
pred_xgb   = xgb.predict(X_missing_imputed)

# --- Weighted ensemble using inverse MAE ---
mae_ridge = 30.56
mae_rf    = 24.80
mae_xgb   = 25.09
w_ridge   = 1 / mae_ridge
w_rf      = 1 / mae_rf
w_xgb     = 1 / mae_xgb
w_total   = w_ridge + w_rf + w_xgb

ensemble_preds = (
    (w_ridge * pred_ridge + w_rf * pred_rf + w_xgb * pred_xgb) / w_total
)

# --- Assign predicted values ---
df.loc[missing_mask, 'Tg'] = ensemble_preds
df['Tg_predicted'] = 0
df.loc[missing_mask, 'Tg_predicted'] = 1

# --- Optional: Remove nonsensical predictions (e.g., Tg > 1000 C) ---
df.loc[df['Tg'] > 1000, ['Tg', 'Tg_predicted']] = [np.nan, 0]

# --- Final printout ---
print(f"Step 6 complete: {missing_mask.sum()} Tg values imputed using MAE-weighted ensemble")
print(f"Skipped {len(skipped)} unrealistic structures — saved as skipped_unrealistic_structures.csv")
# --- Save the final dataset ---
df.to_csv('PI1M_with_Tg_filled_FINAL.csv', index=False)
print("Final dataset saved as PI1M_with_Tg_filled_FINAL.csv")
print(f"Ridge MAE:      {mae_ridge:.2f}")
print(f"Random Forest:  {mae_rf:.2f}")
print(f"XGBoost:        {mae_xgb:.2f}")
print(f"Weights used:   Ridge={weights[0]:.2f}, RF={weights[1]:.2f}, XGB={weights[2]:.2f}")
print(f"Tg values predicted: {missing_mask.sum()}")
outliers = df[df['Tg'] < -273.15]  # Tg above 800 K is rare
print(outliers[['SMILES', 'Tg', 'Tg_predicted']])
import numpy as np

# Replace unphysical Tg values with NaN
df.loc[df['Tg'] < -273.15, ['Tg', 'Tg_predicted']] = [np.nan, 0]
df.to_csv('PI1M_with_Tg_filled_FINAL.csv', index=False)