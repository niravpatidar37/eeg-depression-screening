# EEG Depression Screening (Hybrid CNN–LSTM) + MLflow (SQLite)

End-to-end pipeline for binary classification of EEG-derived spectral features:
- **Classes**: `Depressive disorder` vs `Healthy control`
- **Model**: Hybrid **CNN + LSTM** (CNN over 32×32 “feature image”, LSTM over CNN rows)
- **Tracking**: **MLflow** with **SQLite backend** (`mlflow.db`) + artifact store (`mlartifacts/`)
- **Imbalance handling**: **Class-weighted CrossEntropyLoss** (no SMOTE)
- **Training stability**: Weight decay + LR scheduler + early stopping

---

## Project Structure

Recommended:

```
MLOps/
mlflow.db                 # created by MLflow UI/backend
mlartifacts/              # created by MLflow (artifact root)
models/                   # local checkpoint copies (optional)
artifacts/                # local artifacts (optional)
dataset/
EEG.csv
train.py                  # your training script
README.md
```

> If you don’t want local `models/` and `artifacts/`, you can remove them and rely on MLflow artifacts only.

---

## Dataset

Place your dataset at:

```
dataset/EEG.csv
```

The pipeline:
- filters `specific.disorder` to:
  - `Depressive disorder`
  - `Healthy control`
- drops metadata columns:
  - `no.`, `sex`, `age`, `eeg.date`, `education`, `IQ`, `main.disorder`, `specific.disorder`
- drops junk column (present in the dataset):
  - `Unnamed: 122`
- trims/pads features to **1024** and reshapes to **(1, 32, 32)**

---

## Environment Setup

### 1) Create and activate a virtual environment (Windows)

```powershell
cd F:\MLOps
python -m venv venv
.\venv\Scripts\activate
```

### 2) Install dependencies

```powershell
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib mlflow
```

> If your CUDA version differs, install the matching PyTorch wheel for your system.

---

## Run MLflow UI (SQLite backend)

Start MLflow UI from the **project root** (same folder where you want `mlflow.db`):

```powershell
cd F:\MLOps
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root .\mlartifacts --host 127.0.0.1 --port 5000
```

Open:

* [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Why SQLite?

MLflow has a deprecation warning for the filesystem tracking backend (`./mlruns`) in Feb 2026.
Using SQLite avoids that and gives better feature compatibility.

---

## Train the Model

Run your training script:

```powershell
python train.py
```

The script logs to:

* Tracking DB: `sqlite:///mlflow.db`
* Artifacts: `./mlartifacts`

### What gets logged to MLflow

* **Params**: config values, feature-importance stats, best epoch
* **Metrics**: `train_loss`, `train_acc`, `val_loss`, `val_acc`, learning rate by epoch
* **Artifacts**:

  * `models/best_model_state_dict.pth` (checkpoint)
  * `artifacts/scaler.joblib`
  * `artifacts/feature_importance.csv`
  * `artifacts/classification_report.json`
  * `artifacts/confusion_matrix.npy`
  * model export via `mlflow.pytorch.log_model(...)` (depending on MLflow version)

---

## Viewing Runs in MLflow UI

1. Open MLflow UI: [http://127.0.0.1:5000](http://127.0.0.1:5000)
2. Go to **Experiments**
3. Select **EEG_Depression_Screening**
4. Click a run to view:

   * **Metrics charts** over epochs
   * **Parameters**
   * **Artifacts** (downloadable from UI)

> If you don’t see runs, you likely started MLflow UI from a different folder (so it’s reading a different `mlflow.db`), or your code didn’t set the same tracking URI.

---

## Common Issues

### 1) “No GPU detected”

The script exits if CUDA is not available. Fix by:

* installing a CUDA-enabled PyTorch build
* ensuring NVIDIA drivers are installed
* verifying `nvidia-smi` works

### 2) MLflow UI 404 on `/traces/search`

This usually indicates an MLflow version mismatch (UI expecting traces API). Fix:

```powershell
pip install -U mlflow
```

### 3) VS Code shows “Authentication required” / “Antigravity”

That’s a VS Code webview/extension overlay. Use a normal browser:

* [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Notes on Model & Accuracy

* You may see **high train accuracy** with fluctuating validation accuracy due to:

  * small dataset size (≈294 samples after filtering)
  * reshaping tabular features into a 2D grid (CNN inductive bias may not match data)
* Improvements already included:

  * class-weighted loss (no SMOTE)
  * early stopping
  * weight decay
  * ReduceLROnPlateau scheduler

### Recommended next steps (optional)

* Increase `TEST_SIZE` to 0.2–0.3 for more stable validation
* Use **StratifiedKFold (5-fold)** CV and log fold metrics to MLflow
* Compare against tabular baselines (LogReg / RandomForest / XGBoost/LightGBM)

---

