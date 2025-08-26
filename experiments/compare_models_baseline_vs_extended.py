"""
Module: compare_models_baseline_vs_extended.py
Purpose:
    Compare baseline vs extended feature sets on the same dataset using multiple classifiers.
    Uses Leave-One-Group-Out (LOGO) cross-validation by test_id to simulate per-test generalization.

Feature sets:
    - Baseline: classic summary statistics.
    - Extended: baseline + newly added features (activation %, CoP stats, ratios, asymmetries).

Workflow:
    1) Load the training features CSV (expects columns label, rep_id, test_id and numeric features).
    2) Build matrices X, y, groups for baseline and extended feature patterns.
    3) Evaluate SVM/MLP/RF with LOGO CV (scaler + PCA inside each fold).
    4) Print results and save a CSV summary.

Inputs:
    - outputs/features_fsr_train.csv

Outputs:
    - outputs/model_compare_outputs.csv

Dependencies:
    - numpy, pandas
    - scikit-learn
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# =========================
# Config
# =========================
PATH_CSV = Path("outputs/features_fsr_train.csv")  # run from CT/experiments
LABEL_COL = "label"
REP_ID_COL = "rep_id"
TEST_ID_COL = "test_id"

# Baseline feature patterns (classic statistics)
BASELINE_PATTERNS = [
    r"^avg_mean_",
    r"^avg_std_",
    r"^avg_var_",
    r"^spatial_std_mean_",
    r"^spatial_var_mean_",
    r"^total_mean_",
    r"^samples_",
    r"^load_asym$",
]

# Extended feature patterns (newly added features)
EXTRA_PATTERNS = [
    r"^activation_pct_",
    r"^cop_",
    r"^foreheel_ratio",
    r"^medlat_ratio",
    r"_diff$",
    r"_absdiff$",
]

# Classifiers to compare
CLASSIFIERS = {
    "SVM": SVC(kernel="rbf", gamma="scale", C=1),
    "NN": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=0),
    "RF": RandomForestClassifier(n_estimators=100, random_state=0),
}


# =========================
# Helpers
# =========================
def load_data(path: Path) -> pd.DataFrame:
    """
    Load the dataset and perform minimal validation.

    Parameters
    ----------
    path : Path
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded data.

    Raises
    ------
    SystemExit
        If the file does not exist or required columns are missing.
    """
    if not path.exists():
        sys.exit(f"ERROR: File not found: {path}")
    df = pd.read_csv(path)
    for col in (LABEL_COL, TEST_ID_COL):
        if col not in df.columns:
            sys.exit(f"ERROR: Missing required column '{col}' in {path}")
    return df


def select_columns(df: pd.DataFrame, patterns) -> list[str]:
    """
    Select columns whose names match any of the provided regex patterns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    patterns : list[str]
        List of regex strings.

    Returns
    -------
    list of str
        Matching column names.
    """
    regex = re.compile("|".join(patterns))
    cols = [c for c in df.columns if regex.search(c)]
    return cols


def build_matrix(df: pd.DataFrame, patterns):
    """
    Build X, y, groups, and the list of selected columns given regex patterns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset with LABEL_COL and TEST_ID_COL.
    patterns : list[str]
        Regex patterns to select feature columns.

    Returns
    -------
    X : np.ndarray
        Feature matrix (float).
    y : np.ndarray
        Label vector.
    groups : np.ndarray
        Group vector (test_id) for LOGO CV.
    cols : list[str]
        Selected feature names.

    Raises
    ------
    SystemExit
        If no columns match the provided patterns.
    """
    cols = select_columns(df, patterns)
    if not cols:
        sys.exit(f"ERROR: no columns found for patterns {patterns}")

    X = df[cols].values.astype(float)
    y = df[LABEL_COL].values
    groups = df[TEST_ID_COL].values
    return X, y, groups, cols


def evaluate_model(clf, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """
    Evaluate a classifier with Leave-One-Group-Out (LOGO) CV.

    Inside each fold:
        - Fit StandardScaler on the training split.
        - Fit PCA(n_components=0.95) on the scaled training split.
        - Transform both train and test splits with the fitted scaler and PCA.
        - Fit classifier and predict.
        - Accumulate predictions and compute metrics at the end.

    Parameters
    ----------
    clf : sklearn estimator
        Classifier to evaluate.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    groups : np.ndarray
        Group labels (test_id) for LOGO CV.

    Returns
    -------
    acc : float
        Overall accuracy across all folds.
    f1 : float
        Macro-F1 across all folds.
    """
    logo = LeaveOneGroupOut()
    y_true, y_pred = [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        scaler = StandardScaler()
        pca = PCA(n_components=0.95)

        X_train = scaler.fit_transform(X[train_idx])
        X_train = pca.fit_transform(X_train)

        X_test = scaler.transform(X[test_idx])
        X_test = pca.transform(X_test)

        clf.fit(X_train, y[train_idx])
        preds = clf.predict(X_test)

        y_true.extend(y[test_idx])
        y_pred.extend(preds)

    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    return acc, f1


# =========================
# Main
# =========================
def main():
    df = load_data(PATH_CSV)
    print("[INFO] Sample of available columns:", df.columns.tolist()[:30])

    # Baseline
    Xb, y, groups, base_cols = build_matrix(df, BASELINE_PATTERNS)

    # Extended = baseline + extra
    Xe, _, _, ext_cols = build_matrix(df, BASELINE_PATTERNS + EXTRA_PATTERNS)

    results = []
    for name, clf in CLASSIFIERS.items():
        acc_b, f1_b = evaluate_model(clf, Xb, y, groups)
        acc_e, f1_e = evaluate_model(clf, Xe, y, groups)
        results.append([name, "baseline", acc_b, f1_b, len(base_cols)])
        results.append([name, "extended", acc_e, f1_e, len(ext_cols)])

    res_df = pd.DataFrame(results, columns=["Classifier", "FeatureSet", "Acc", "F1", "nFeatures"])
    print("\n=== RESULTS ===")
    print(res_df)

    out_path = Path("outputs/model_compare_outputs.csv")
    res_df.to_csv(out_path, index=False)
    print(f"\n[OK] Results saved to {out_path}")


if __name__ == "__main__":
    main()
