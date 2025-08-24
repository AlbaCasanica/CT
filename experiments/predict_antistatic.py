# experiments/predict_antistatic.py
import argparse, json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from joblib import load
from sklearn.exceptions import NotFittedError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def robust_z(x):
    m = np.nanmedian(x)
    s = np.nanstd(x)
    if s == 0 or np.isnan(s): s = 1.0
    return (x - m) / s

def get_inner_model(model):
    """Ritorna il classificatore che espone decision_function/predict_proba."""
    # StaticGuardedEstimator → ha base_estimator
    if hasattr(model, "base_estimator"):
        return model.base_estimator
    return model  # Pipeline o stimatore liscio

def second_best_labels(inner, X, classes_):
    # Prova decision_function, poi predict_proba
    if hasattr(inner, "decision_function"):
        S = inner.decision_function(X)
        if S.ndim == 1:  # binario
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

def main(inp, outp, model_path, meta_path, q=0.60):  # q un po' più aggressivo
    print(f"📥 Modello: {model_path}")
    model = load(model_path)

    print(f"📥 Input features: {inp}")
    df = pd.read_csv(inp, dtype={"rep_id": str})

    # Colonne feature come in training (dal meta, se c'è)
    feat_cols = None
    if meta_path and Path(meta_path).exists():
        try:
            meta = json.loads(Path(meta_path).read_text())
            feat_cols = meta.get("feature_columns", None)
        except Exception:
            pass

    X = df.select_dtypes(include=[np.number]).copy()
    X.drop(columns=["label"], inplace=True, errors="ignore")
    if feat_cols:
        X = X.reindex(columns=[c for c in feat_cols if c in X.columns], fill_value=0.0)

    # Predizione base (dalla pipeline completa)
    try:
        base_pred = model.predict(X)
    except NotFittedError:
        raise SystemExit("❌ Modello non fittato: riallena e passa il path corretto con --model.")

    # Statistiche di “static” prima
    before_static = int(np.sum(base_pred == "static"))

    # Punteggio dinamica piedi
    if "activation_pct_both" not in df.columns and "cop_path_per_sec_both" not in df.columns:
        print("⚠️  Mancano 'activation_pct_both'/'cop_path_per_sec_both' → salvo predizioni base.")
        pd.DataFrame({"pred": base_pred, "rep_id": df.get("rep_id", pd.Series(range(len(df))))}).to_csv(outp, index=False)
        print(f"✅ Salvato: {outp} (static before={before_static})")
        return

    act  = df.get("activation_pct_both", pd.Series(np.zeros(len(df))))
    path = df.get("cop_path_per_sec_both", pd.Series(np.zeros(len(df))))
    dyn = robust_z(act.values) + robust_z(path.values)
    thr = np.nanquantile(dyn, q)
    print(f"🔎 Anti-static: soglia dinamica q{int(q*100)} = {thr:.3f}")

    # Prendi il classificatore interno per la seconda scelta
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
                if y[i] == "static" and dyn[i] > thr:
                    # seconda scelta già calcolata esclude la migliore (che era static?)
                    # in alcuni casi la migliore del classificatore interno potrebbe non essere 'static'
                    # ma noi corriamo solo se la pred finale è 'static' e dyn alto
                    y[i] = sec[i]
                    changed += 1
    else:
        print("ℹ️  Non trovo classes_ o 'static' tra le classi → nessuna correzione.")

    after_static = int(np.sum(np.array(y) == "static"))
    print(f"🛡️  Anti-static: static prima={before_static} → dopo={after_static} | correzioni={changed}")

    pd.DataFrame({"pred": y, "rep_id": df.get("rep_id", pd.Series(range(len(df))))}).to_csv(outp, index=False)
    print(f"✅ Salvato: {outp}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=str(ROOT / "outputs" / "models_test1-12" / "svm_rbf_BASE_NOGUARD_UNIFIED.joblib"))
    ap.add_argument("--meta",  default=str(ROOT / "outputs" / "models_test1-12" / "svm_rbf_BASE_NOGUARD_UNIFIED_meta.json"))
    ap.add_argument("--q", type=float, default=1.00, help="Quantile soglia dinamica (1.00 = nessuna correzione)")
    args = ap.parse_args()
    main(args.input, args.output, args.model, args.meta, args.q)
