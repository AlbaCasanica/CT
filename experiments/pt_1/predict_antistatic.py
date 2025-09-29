"""
Module: predict_antistatic.py
Purpose:
    Post-process model predictions to reduce false "static" classifications.
    It uses a simple dynamicity score (based on activation percentage and CoP path speed)
    to reassign high-dynamic samples that were predicted as "static" to the model's
    second-best class.

Workflow:
    1) Load model and input features CSV.
    2) Align features to the training schema (from meta JSON, if provided).
    3) Predict with the full pipeline.
    4) Compute a dynamicity score = zscore(activation_pct_both) + zscore(cop_path_per_sec_both).
    5) For samples predicted "static" with dynamicity above a quantile threshold q,
       replace "static" with the second-best class (via decision_function or predict_proba).
    6) Save CSV with columns: pred, rep_id.

Inputs:
    - --input : path to features CSV (must include numeric features; rep_id optional)
    - --model : path to fitted pipeline/joblib
    - --meta  : path to JSON with {"feature_columns": [...]}, optional
    - --q     : quantile for dynamicity threshold (1.00 = no correction)

Outputs:
    - CSV with columns: pred, rep_id

Dependencies:
    - numpy, pandas
    - scikit-learn
    - joblib
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from joblib import load
from sklearn.exceptions import NotFittedError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# =========================
# Small helpers
# =========================
def robust_z(x: np.ndarray) -> np.ndarray:
    """
    Robust z-score using median and std (with epsilon fallback).

    Parameters
    ----------
    x : np.ndarray
        Input vector.

    Returns
    -------
    np.ndarray
        Standardized vector.
    """
    m = np.nanmedian(x)
    s = np.nanstd(x)
    if s == 0 or np.isnan(s):
        s = 1.0
    return (x - m) / s


def get_inner_model(model):
    """
    Return the classifier exposing decision_function/predict_proba.

    Notes
    -----
    - If the model is a wrapped estimator (e.g., StaticGuardedEstimator),
      it should expose `base_estimator`.
    - Otherwise return the model itself (pipeline or estimator).
    """
    if hasattr(model, "base_estimator"):
        return model.base_estimator
    return model


def second_best_labels(inner, X: pd.DataFrame, classes_: np.ndarray):
    """
    Compute the second-best class for each row using decision_function or predict_proba.

    Parameters
    ----------
    inner : sklearn estimator
        Classifier exposing decision_function or predict_proba.
    X : pandas.DataFrame
        Feature matrix aligned to the classifier's expected columns.
    classes_ : np.ndarray
        Class labels in the estimator.

    Returns
    -------
    np.ndarray or None
        Array of second-best labels, or None if neither API is available.
    """
    if hasattr(inner, "decision_function"):
        S = inner.decision_function(X)
        if S.ndim == 1:  # binary case
            S = np.vstack([-S, S]).T
        idx1 = np.argmax(S, axis=1)
        S[np.arange(len(S)), idx1] = -1e12
        idx2 = np.argmax(S, axis=1)
        return classes_[idx2]

    if hasattr(inner, "predict_proba"):
        P = inner.predict_proba(X)
        idx1 = np.argmax(P, axis=1)
        P[np.arange(len(P)), idx1] = -1e9
        idx2 = np.argmax(P, axis=1)
        return classes_[idx2]

    return None


# =========================
# Main logic
# =========================
def main(inp, outp, model_path, meta_path, q=0.60):
    """
    Apply anti-static correction and save predictions.

    Parameters
    ----------
    inp : str or Path
        Path to input features CSV.
    outp : str or Path
        Path to output CSV (pred, rep_id).
    model_path : str or Path
        Path to fitted joblib model/pipeline.
    meta_path : str or Path
        Path to JSON meta with "feature_columns" (optional).
    q : float
        Quantile threshold for dynamicity (higher -> more conservative).
        Example: 0.60 is moderately aggressive; 1.00 disables corrections.
    """
    print(f"Model: {model_path}")
    model = load(model_path)

    print(f"Input features: {inp}")
    df = pd.read_csv(inp, dtype={"rep_id": str})

    # Feature schema from meta (if available)
    feat_cols = None
    if meta_path and Path(meta_path).exists():
        try:
            meta = json.loads(Path(meta_path).read_text())
            feat_cols = meta.get("feature_columns", None)
        except Exception:
            feat_cols = None

    # Build feature matrix
    X = df.select_dtypes(include=[np.number]).copy()
    X.drop(columns=["label"], inplace=True, errors="ignore")
    if feat_cols:
        X = X.reindex(columns=[c for c in feat_cols if c in X.columns], fill_value=0.0)

    # Base prediction from the full pipeline
    try:
        base_pred = model.predict(X)
    except NotFittedError:
        raise SystemExit("Model is not fitted: retrain and pass a valid --model path.")

    # Count "static" before correction
    before_static = int(np.sum(base_pred == "static"))

    # If dynamicity fields are missing, save base predictions
    if "activation_pct_both" not in df.columns and "cop_path_per_sec_both" not in df.columns:
        print("Missing 'activation_pct_both'/'cop_path_per_sec_both' → saving base predictions.")
        pd.DataFrame(
            {"pred": base_pred, "rep_id": df.get("rep_id", pd.Series(range(len(df))))}
        ).to_csv(outp, index=False)
        print(f"Saved: {outp} (static before={before_static})")
        return

    # Dynamicity score
    act = df.get("activation_pct_both", pd.Series(np.zeros(len(df))))
    path = df.get("cop_path_per_sec_both", pd.Series(np.zeros(len(df))))
    dyn = robust_z(act.values) + robust_z(path.values)
    thr = np.nanquantile(dyn, q)
    print(f"Anti-static: dynamicity threshold q{int(q * 100)} = {thr:.3f}")

    # Find inner classifier and its classes
    inner = get_inner_model(model)
    classes_ = getattr(inner, "classes_", None)
    if classes_ is None and hasattr(model, "named_steps"):
        try:
            classes_ = model.named_steps["clf"].classes_
        except Exception:
            classes_ = None

    y = base_pred.astype(str).copy()
    changed = 0
    if classes_ is not None and ("static" in classes_.tolist()):
        sec = second_best_labels(inner, X, classes_)
        if sec is not None:
            idx_static = classes_.tolist().index("static")
            for i in range(len(y)):
                # Only change those predicted as "static" that are highly dynamic
                if y[i] == "static" and dyn[i] > thr:
                    y[i] = sec[i]
                    changed += 1
    else:
        print("No classes_ or 'static' not found among classes → no correction applied.")

    after_static = int(np.sum(np.array(y) == "static"))
    print(f"Anti-static: static before={before_static} → after={after_static} | corrections={changed}")

    pd.DataFrame({"pred": y, "rep_id": df.get("rep_id", pd.Series(range(len(df))))}).to_csv(outp, index=False)
    print(f"Saved: {outp}")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Post-process predictions to reduce false 'static' labels using a dynamicity threshold."
    )
    ap.add_argument("--input", required=True, help="Path to input features CSV.")
    ap.add_argument("--output", required=True, help="Path to output CSV (pred, rep_id).")
    ap.add_argument(
        "--model",
        default=str(ROOT / "outputs" / "models_test1-12" / "svm_rbf_BASE_NOGUARD_UNIFIED.joblib"),
        help="Path to fitted joblib model/pipeline.",
    )
    ap.add_argument(
        "--meta",
        default=str(ROOT / "outputs" / "models_test1-12" / "svm_rbf_BASE_NOGUARD_UNIFIED_meta.json"),
        help="Path to JSON with {'feature_columns': [...]}.",
    )
    ap.add_argument(
        "--q",
        type=float,
        default=1.00,
        help="Dynamicity quantile threshold (1.00 disables corrections).",
    )
    args = ap.parse_args()
    main(args.input, args.output, args.model, args.meta, args.q)
