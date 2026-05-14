import warnings
warnings.filterwarnings("ignore")  

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import joblib
import sys

print("Script started")
sys.stdout.flush()

# -------------------------
# 1. Load dataset
# -------------------------
df = pd.read_csv('/csl/users/2026nnandaku/cluster/PolymerDesign/PI1M_with_Tg_Final.csv')
df = df[df['Tg'].notna()].copy()
print(f"Dataset loaded: {df.shape}")
sys.stdout.flush()

# -------------------------
# 2. Subsample for tuning
# -------------------------
df_sample = df.sample(n=75000, random_state=42)
print("Sampled 75,000 rows for GridSearch")
sys.stdout.flush()

# -------------------------
# 3. Split features and target
# -------------------------
X_sample = df_sample.drop(columns=['SMILES', 'Name', 'Tg', 'Tg_LF', 'Tg_predicted'])
y_sample = df_sample['Tg']

X_train, X_val, y_train, y_val = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42
)
print("Data split into train/val")
sys.stdout.flush()

# -------------------------
# 4. GridSearchCV on sample
# -------------------------
param_grid = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [15, 20, 25, 30],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}

rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    scoring=rmse_scorer,
    cv=5,
    n_jobs=-1,
    verbose=3
)

print("Starting GridSearchCV")
sys.stdout.flush()
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

print("\nGrid Search Complete. Best Params:")
print(best_params)
sys.stdout.flush()

# -------------------------
# 5. Train final model on full data using best params
# -------------------------
X_full = df.drop(columns=['SMILES', 'Name', 'Tg', 'Tg_LF', 'Tg_predicted'])
y_full = df['Tg']

print("Training final model on full dataset...")
sys.stdout.flush()

final_model = RandomForestRegressor(
    **best_params,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_full, y_full)
print("Final model training complete")
sys.stdout.flush()

# -------------------------
# 6. Evaluate on holdout test set
# -------------------------
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)
y_pred = final_model.predict(X_test_full)

mae = mean_absolute_error(y_test_full, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_full, y_pred))
r2 = r2_score(y_test_full, y_pred)

print(f"\nFinal Model MAE:  {mae:.3f}")
print(f"Final Model RMSE: {rmse:.3f}")
print(f"Final Model R²:   {r2:.3f}")
sys.stdout.flush()

# -------------------------
# 7. Save final model
# -------------------------
model_path = '/csl/users/2026nnandaku/cluster/PolymerDesign/tg_predictor_rf_final.pkl'
joblib.dump(final_model, model_path)
print(f"Model saved to {model_path}")
sys.stdout.flush()