# experiments/train_staticguard_model.py
from pathlib import Path
import sys, json
from collections import Counter
import numpy as np
import pandas as pd
from joblib import dump

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline  # NO imblearn, NO guard

RANDOM_STATE = 42
DATA_PATH = ROOT / "outputs" / "features_fsr_train.csv"
OUT_DIR = ROOT / "outputs" / "models_test1-12"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SVM_C = 5
SVM_GAMMA = 0.1

print(f"📥 Carico features: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Escludi classi solo-braccia (FSR non informativo)
HAND = {"hands_up", "hands_forward", "hand_up_back"}
df = df[~df["label"].isin(HAND)].reset_index(drop=True)

TARGET = "label" if "label" in df.columns else None
if TARGET is None:
    raise RuntimeError("Manca la colonna 'label'.")

# Collassa sitting/standing -> static
y = df[TARGET].astype(str).replace({"sitting": "static", "standing": "static"})
X = df.drop(columns=[TARGET], errors="ignore").select_dtypes(include=[np.number]).copy()

# Pulizia
X.replace([np.inf, -np.inf], np.nan, inplace=True)
if X.isna().any().any():
    X = X.fillna(X.mean())

# Rimuovi classi con <2 esempi (stabilità)
counts = y.value_counts()
keep = y.map(counts) >= 2
if keep.sum() != len(y):
    print("⚠️  Rimuovo classi con <2 esempi:", sorted(set(y[~keep])))
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)

print("📊 X:", X.shape, "| classi:", Counter(y))

# Pipeline base: scaler + SVM RBF (NO ROS, NO GUARD)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA, probability=False, random_state=RANDOM_STATE)),
])

print("🧠 Fit modello BASE (NO ROS, NO GUARD)…")
pipe.fit(X, y)

model_name = "svm_rbf_BASE_NOGUARD_UNIFIED"
model_path = OUT_DIR / f"{model_name}.joblib"
dump(pipe, model_path)
print("✅ Salvato modello:", model_path)

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
print("✅ Salvati metadati:", meta_path)
