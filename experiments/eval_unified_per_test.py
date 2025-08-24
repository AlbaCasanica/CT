# experiments/eval_unified_per_test.py
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "outputs" / "models_test1-12" / "svm_rbf_BASE_NOGUARD_UNIFIED.joblib"
META_PATH  = ROOT / "outputs" / "models_test1-12" / "svm_rbf_BASE_NOGUARD_UNIFIED_meta.json"

OUT_DIR = ROOT / "outputs" / "eval_unified"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HAND_CLASSES = {"hands_up", "hands_forward", "hand_up_back"}  # FSR-only
STATIC_MAP   = {"sitting": "static", "standing": "static"}

def load_model_and_cols():
    model = load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    feat_cols = meta.get("feature_columns", [])
    return model, feat_cols

def prepare_Xy(feat_path: Path, feat_cols):
    df = pd.read_csv(feat_path, dtype={"rep_id": str})
    if "label" not in df.columns:
        raise RuntimeError(f"{feat_path.name}: manca la colonna 'label'.")

    # filtro FSR-only
    df = df[~df["label"].isin(HAND_CLASSES)].reset_index(drop=True)

    y = df["label"].astype(str).replace(STATIC_MAP)

    X = df.drop(columns=["label"], errors="ignore").select_dtypes(include=[np.number]).copy()
    # ordina e allinea alle colonne usate in training
    cols = [c for c in feat_cols if c in X.columns]
    X = X.reindex(columns=cols, fill_value=0.0)

    # pulizia
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X.isna().any().any():
        X = X.fillna(X.mean())

    return X, y, df.get("rep_id", pd.Series(range(len(df))))

def eval_one(model, feat_cols, test_id: int):
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

    # Report txt
    rpt = classification_report(y_true, y_pred, labels=labels, digits=3)
    with open(rep_dir / "report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.3%}\nF1-macro: {f1m:.3f}\nF1-weighted: {f1w:.3f}\n\n{rpt}")

    # Confusion matrix csv
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    cm_df.to_csv(rep_dir / "confusion_matrix.csv")

    print(f"[T{test_id:02}] acc={acc:.3%}  f1m={f1m:.3f}  n={len(y_true)}")
    return {"test_id": test_id, "n": int(len(y_true)), "acc": acc, "f1_macro": f1m, "f1_weighted": f1w}

def main():
    print(f"📦 Modello: {MODEL_PATH.name}")
    model, feat_cols = load_model_and_cols()

    rows = []
    for tid in range(1, 13):
        res = eval_one(model, feat_cols, tid)
        if res is not None:
            rows.append(res)

    if rows:
        df = pd.DataFrame(rows).sort_values("test_id")
        df.to_csv(OUT_DIR / "summary.csv", index=False)
        print("\n📊 Riepilogo salvato:", (OUT_DIR / "summary.csv").relative_to(ROOT))
        print(df.to_string(index=False))
    else:
        print("⚠️ Nessun features_test*.csv trovato in outputs/.")

if __name__ == "__main__":
    main()
