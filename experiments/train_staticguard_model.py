"""
Module: train_staticguard_model.py
Purpose:
    Train a baseline SVM (RBF kernel) model on FSR-based features to classify activities,
    collapsing sitting/standing into a single "static" class and excluding hand-only classes
    (where FSR is not informative). Saves the trained pipeline and a JSON metadata file.

Workflow:
    1) Load training features CSV.
    2) Exclude hand-only classes.
    3) Collapse {sitting, standing} -> static.
    4) Clean features (handle inf/NaN).
    5) Remove classes with < 2 samples for basic stability.
    6) Fit StandardScaler + SVC(RBF).
    7) Persist model (.joblib) and metadata (.json).

Inputs:
    - outputs/features_fsr_train.csv

Outputs:
    - outputs/models_test1-12/svm_rbf_BASE_NOGUARD_UNIFIED.joblib
    - outputs/models_test1-12/svm_rbf_BASE_NOGUARD_UNIFIED_meta.json

Dependencies:
    - numpy, pandas
    - scikit-learn
    - joblib
"""

from pathlib import Path
import sys
import json
from collections import Counter

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline  # No imblearn, no "guard" wrapper

# =========================
# Paths and configuration
# =========================
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RANDOM_STATE = 42
DATA_PATH = ROOT / "outputs" / "features_fsr_train.csv"
OUT_DIR = ROOT / "outputs" / "models_test1-12"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SVM_C = 5
SVM_GAMMA = 0.1

print(f"Loading features: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Exclude hand-only classes (FSR not informative)
HAND = {"hands_up", "hands_forward", "hand_up_back"}
df = df[~df["label"].isin(HAND)].reset_index(drop=True)

TARGET = "label" if "label" in df.columns else None
if TARGET is None:
    raise RuntimeError("Missing 'label' column.")

# Collapse sitting/standing -> static
y = df[TARGET].astype(str).replace({"sitting": "static", "standing": "static"})
X = df.drop(columns=[TARGET], errors="ignore").select_dtypes(include=[np.number]).copy()

# Basic cleaning
X.replace([np.inf, -np.inf], np.nan, inplace=True)
if X.isna().any().any():
    X = X.fillna(X.mean())

# Remove classes with < 2 samples (stability)
counts = y.value_counts()
keep = y.map(counts) >= 2
if keep.sum() != len(y):
    print("Removing classes with < 2 samples:", sorted(set(y[~keep])))
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)

print("Shape X:", X.shape, "| classes:", Counter(y))

# Baseline pipeline: scaler + SVM RBF (no ROS, no guard)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA, probability=False, random_state=RANDOM_STATE)),
])

print("Fitting BASE model (no ROS, no guard)…")
pipe.fit(X, y)

model_name = "svm_rbf_BASE_NOGUARD_UNIFIED"
model_path = OUT_DIR / f"{model_name}.joblib"
dump(pipe, model_path)
print("Saved model:", model_path)

meta = {
    "data_path": str(DATA_PATH),
    "feature_columns": X.columns.tolist(),
    "classes_": sorted(set(y)),
    "params": {"C": SVM_C, "gamma": SVM_GAMMA},
    "use_guard": False,
    "use_ros": False,
}
meta_path = OUT_DIR / f"{model_name}_meta.json"
meta_path.write_text(json.dumps(meta, indent=2))
print("Saved metadata:", meta_path)
