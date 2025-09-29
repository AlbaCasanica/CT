"""
Module: eval_unified_per_test.py
Purpose:
    Evaluate a unified SVM model on multiple test sets, excluding hand-only classes.
    Generates per-test classification reports, confusion matrices, and a summary CSV.

Main functionalities:
    - Load trained model and feature metadata
    - Prepare feature matrices and labels from CSV files
    - Evaluate model performance (accuracy, F1-scores) per test set
    - Save classification report and confusion matrix for each test set
    - Generate summary table across all test sets

Input:
    - Pre-computed feature CSV files: outputs/features_test{i}.csv (i = 1..12)
    - Trained SVM model (joblib) and metadata JSON

Output:
    - Per-test folder: report.txt, confusion_matrix.csv
    - Global summary: summary.csv

Dependencies:
    - numpy, pandas
    - scikit-learn
    - joblib
"""

from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# =========================
# Paths and configuration
# =========================
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "outputs" / "models_test1-12" / "svm_rbf_BASE_NOGUARD_UNIFIED.joblib"
META_PATH  = ROOT / "outputs" / "models_test1-12" / "svm_rbf_BASE_NOGUARD_UNIFIED_meta.json"

OUT_DIR = ROOT / "outputs" / "eval_unified"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hand-only classes to exclude
HAND_CLASSES = {"hands_up", "hands_forward", "hand_up_back"}
# Map sitting/standing to a common "static" label
STATIC_MAP   = {"sitting": "static", "standing": "static"}


# =========================
# Model loading
# =========================
def load_model_and_cols():
    """
    Load the trained SVM model and its feature metadata.

    Returns
    -------
    model : sklearn estimator
        Pre-trained SVM model loaded from disk.
    feat_cols : list of str
        List of feature column names used during training.
    """
    model = load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    feat_cols = meta.get("feature_columns", [])
    return model, feat_cols


# =========================
# Data preparation
# =========================
def prepare_Xy(feat_path: Path, feat_cols):
    """
    Prepare feature matrix X and label vector y from a feature CSV file.

    - Drops hand-only classes
    - Maps sitting/standing to "static"
    - Aligns feature columns to training order
    - Cleans NaNs and infinities

    Parameters
    ----------
    feat_path : Path
        Path to feature CSV file.
    feat_cols : list of str
        Expected feature column names from training.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix aligned to training features.
    y : pandas.Series
        Labels mapped and cleaned.
    rep_id : pandas.Series
        Repetition IDs (if available).
    """
    df = pd.read_csv(feat_path, dtype={"rep_id": str})
    if "label" not in df.columns:
        raise RuntimeError(f"{feat_path.name}: missing 'label' column.")

    # Exclude hand-only classes
    df = df[~df["label"].isin(HAND_CLASSES)].reset_index(drop=True)

    # Map static classes
    y = df["label"].astype(str).replace(STATIC_MAP)

    # Select numeric feature columns
    X = df.drop(columns=["label"], errors="ignore").select_dtypes(include=[np.number]).copy()

    # Align with training feature order
    cols = [c for c in feat_cols if c in X.columns]
    X = X.reindex(columns=cols, fill_value=0.0)

    # Clean NaNs/Infs
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X.isna().any().any():
        X = X.fillna(X.mean())

    return X, y, df.get("rep_id", pd.Series(range(len(df))))


# =========================
# Evaluation per test set
# =========================
def eval_one(model, feat_cols, test_id: int):
    """
    Evaluate model on a single test set and save results.

    Parameters
    ----------
    model : sklearn estimator
        Pre-trained model.
    feat_cols : list of str
        Feature column names used in training.
    test_id : int
        Test set identifier.

    Returns
    -------
    dict or None
        Summary dictionary with test_id, n, acc, f1_macro, f1_weighted,
        or None if file is missing.
    """
    feat_path = ROOT / "outputs" / f"features_test{test_id}.csv"
    if not feat_path.exists():
        return None

    X, y_true, rep_id = prepare_Xy(feat_path, feat_cols)
    if len(y_true) == 0:
        return {"test_id": test_id, "n": 0, "acc": np.nan, "f1_macro": np.nan, "f1_weighted": np.nan}

    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1w = f1_score(y_true, y_pred, average="weighted")
    labels = sorted(set(y_true) | set(y_pred))

    rep_dir = OUT_DIR / f"test_{test_id}"
    rep_dir.mkdir(parents=True, exist_ok=True)

    # Classification report
    rpt = classification_report(y_true, y_pred, labels=labels, digits=3)
    with open(rep_dir / "report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.3%}\nF1-macro: {f1m:.3f}\nF1-weighted: {f1w:.3f}\n\n{rpt}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    cm_df.to_csv(rep_dir / "confusion_matrix.csv")

    print(f"[T{test_id:02}] acc={acc:.3%}  f1m={f1m:.3f}  n={len(y_true)}")
    return {"test_id": test_id, "n": int(len(y_true)), "acc": acc, "f1_macro": f1m, "f1_weighted": f1w}


# =========================
# Main entry point
# =========================
def main():
    """
    Main routine:
    - Load model and metadata
    - Evaluate all test sets (1..12)
    - Save global summary
    """
    print(f"📦 Model: {MODEL_PATH.name}")
    model, feat_cols = load_model_and_cols()

    rows = []
    for tid in range(1, 13):
        res = eval_one(model, feat_cols, tid)
        if res is not None:
            rows.append(res)

    if rows:
        df = pd.DataFrame(rows).sort_values("test_id")
        df.to_csv(OUT_DIR / "summary.csv", index=False)
        print("\n📊 Summary saved to:", (OUT_DIR / "summary.csv").relative_to(ROOT))
        print(df.to_string(index=False))
    else:
        print("⚠️ No features_test*.csv found in outputs/.")


if __name__ == "__main__":
    main()
