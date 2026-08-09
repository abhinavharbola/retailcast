# Kaggle Setup:

## 1. Create the notebook
1. Go to the competition dataset page: **Store Sales - Time Series Forecasting** (Kaggle).
2. Click "New Notebook" from that dataset page, this auto-attaches the dataset as input at
   `/kaggle/input/competetions/store-sales-time-series-forecasting/`.
3. Settings (right panel): Accelerator = **None/CPU** (no GPU needed anywhere in this project).
4. Settings → Internet = **ON** (needed to `pip install` prophet/lightgbm/xgboost/mlflow/dagshub).

## 2. Install packages
Each notebook's first cell already runs its own `!pip install -q ...` line, scoped to just
what that notebook needs (e.g. notebook 04 installs `lightgbm xgboost mlflow dagshub
pyarrow`). You don't need to add anything manually - just run that first cell.
(`numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn` are preinstalled on Kaggle, no need to
reinstall.) `kaggle/requirements-ipynb.txt` is the reference list of pinned versions if you
need to debug an install issue.

## 3. DagsHub + MLflow remote tracking (needed from Notebook 4 onward)
1. You already have a DagsHub account from your MLOps project, create a **new repo** there for RetailCast (or reuse the same one with a different experiment name, your call).
2. Get your DagsHub token: DagsHub → Settings → Tokens.
3. In Kaggle: Add-ons → Secrets → add two secrets:
   - `DAGSHUB_TOKEN` = your token
   - `DAGSHUB_REPO` = `yourusername/retailcast` (or whatever repo name you use)
4. Each notebook that logs to MLflow will read these secrets, code included in Notebook 4.

## 4. LLM API keys (needed later, in local dashboard stage, not needed for Notebooks 1-5)
Not required yet. We'll set these up when we get to the GenAI narrative layer.

## 5. Order of execution
Run these notebooks **in order**, each one saves its output to `/kaggle/working/` as the input for the next:
1. `01_eda.ipynb` → saves `retailcast_subset.parquet`, `stationarity_results.csv`, `series_activation_diagnostics.csv`, `subset_config.json`
2. `02_feature_engineering.ipynb` → saves `demand_pattern_classification.csv`, `retailcast_features.parquet`
3. `03_statistical_models.ipynb` → saves `prophet_results.csv`, `sarima_results.csv`
4. `04_ml_models.ipynb` → saves `ml_results.csv`, `final_holdout_predictions.parquet`, `lightgbm_feature_importance.csv`, `cost_of_error.json`, logs to MLflow/DagsHub
5. `05_anomaly_detection.ipynb` → saves `anomaly_results.parquet`, `anomaly_eval_metrics.csv`

**Important**: Kaggle notebook sessions are ephemeral, at the end of each notebook, go to
"Save Version" → "Save & Run All" so outputs persist, then download the output files
(top-right "Output" tab) to keep locally, since the next notebook needs them as input
(upload the previous notebook's output as a new Kaggle Dataset, or attach the previous
notebook itself as an input source via "Add Data" → "Notebook Output").