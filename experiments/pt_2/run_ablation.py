"""
Ablation study on features:
- baseline (only "Maria's" features: e.g., force_*)
- extended (baseline + all new features)
- leave-one-out: extended minus one group at a time (cop_y_*, activation_pct, symmetry, etc.)

Expected input: CSV with one row per (rep_id, phase) and numeric feature columns.
Minimum columns: rep_id, label  (+ optional phase, test_id). If test_id is missing, it is inferred from rep_id "test::...".

Example:
python -m experiments.run_ablation \
  --in outputs/features_master_phased.csv \
  --outdir outputs/ablation \
  --cv folds \
  --n-splits 5 \
  --standardize
"""

from __future__ import annotations
import argparse, os, re, json
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score

# ----------------------- utilities -----------------------

def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _extract_test_id(rep_id: str) -> str:
    # handles formats like "test::rep" or filename::rep
    return rep_id.split("::", 1)[0] if "::" in rep_id else rep_id.split("_")[0]

def _feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Define feature groups for leave-one-out."""
    cols = df.columns

    groups = {
        # Maria's baseline (adjust if needed): all statistics starting with 'force_'
        "BASELINE_force": [c for c in cols if c.startswith("force_") and _is_numeric_series(df[c])],

        # new features/groups
        "NEW_cop_y":      [c for c in cols if c.startswith("cop_y_") and _is_numeric_series(df[c])],
        "NEW_activation": [c for c in cols if c == "activation_pct" and _is_numeric_series(df[c])],
        "NEW_symmetry":   [c for c in cols if ("symmetry_index" in c) and _is_numeric_series(df[c])],
        "NEW_force_LR":   [c for c in cols if (c.startswith("force_L_") or c.startswith("force_R_")) and _is_numeric_series(df[c])],
    }
    # drop empty groups
    groups = {k:v for k,v in groups.items() if len(v) > 0}
    return groups

def _build_feature_sets(groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Create the set dictionary: baseline, extended, and leave-one-out."""
    baseline = groups.get("BASELINE_force", [])
    new_parts = []
    for k,v in groups.items():
        if k.startswith("NEW_"):
            new_parts.extend(v)
    extended = sorted(set(baseline + new_parts))

    sets = {"baseline": sorted(set(baseline)),
            "extended": extended}

    # leave-one-out for each NEW_ group present
    for k,v in groups.items():
        if not k.startswith("NEW_") or not v: 
            continue
        name = f"ext_minus__{k.replace('NEW_','').lower()}"
        sets[name] = sorted(set(extended) - set(v))

    return sets

def _make_model(standardize: bool, n_estimators: int = 300, max_depth: int | None = None, random_state: int = 42):
    base = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    if standardize:
        # RF does not need it, but standardizing keeps things uniform if you switch models later
        return make_pipeline(StandardScaler(with_mean=True, with_std=True), base)
    return base

def _cv_split(y, groups, mode: str, n_splits: int, seed: int = 42):
    if mode == "folds":
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(np.zeros(len(y)), y):
            yield tr, te
    elif mode == "group":
        # group by test_id: prevents leakage between train/test from the same test
        gkf = GroupKFold(n_splits=n_splits)
        for tr, te in gkf.split(np.zeros(len(y)), y, groups):
            yield tr, te
    else:
        raise ValueError("cv mode non valido. Usa 'folds' oppure 'group'.")

def _eval_set(X: pd.DataFrame, y: pd.Series, groups: pd.Series, cv_mode: str, n_splits: int, standardize: bool) -> Dict[str, float]:
    accs, f1s = [], []
    for tr, te in _cv_split(y, groups, cv_mode, n_splits):
        model = _make_model(standardize=standardize)
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict(X.iloc[te])
        accs.append(accuracy_score(y.iloc[te], p))
        f1s.append(f1_score(y.iloc[te], p, average="macro"))
    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs, ddof=1) if len(accs)>1 else 0.0),
            "f1_mean": float(np.mean(f1s)),  "f1_std": float(np.std(f1s, ddof=1) if len(f1s)>1 else 0.0)}

# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser(description="Ablation study: baseline vs extended + leave-one-out")
    ap.add_argument("--in", dest="inp", required=True, help="CSV con feature per (rep_id, phase).")
    ap.add_argument("--outdir", required=True, help="Cartella output per risultati e plot.")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--rep-col", default="rep_id")
    ap.add_argument("--cv", choices=["folds","group"], default="folds", help="CV standard (folds) o per gruppo test_id (group).")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--standardize", action="store_true", help="Applica StandardScaler prima del modello.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    plots_dir = os.path.join(args.outdir, "plots"); os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(args.inp)
    if not {args.label_col, args.rep_col}.issubset(df.columns):
        raise SystemExit("Mancano colonne minime: rep_id e label.")

    # y and groups (for CV on test_id)
    y = df[args.label_col].astype(str)
    if "test_id" in df.columns:
        groups = df["test_id"].astype(str)
    else:
        groups = df[args.rep_col].astype(str).map(_extract_test_id)

    # select numeric features
    meta = {args.label_col, args.rep_col, "phase", "test_id"}
    cand = [c for c in df.columns if c not in meta and _is_numeric_series(df[c])]
    feats_df = df[cand].copy()

    # groups and sets
    groups_map = _feature_groups(feats_df)
    sets = _build_feature_sets(groups_map)
    with open(os.path.join(args.outdir, "feature_sets.json"), "w") as f:
        json.dump({k: sorted(v) for k,v in sets.items()}, f, indent=2)

    # evaluate each set
    rows = []
    for name, feats in sets.items():
        if len(feats) == 0:
            print(f"[WARN] set '{name}' vuoto, salto.")
            continue
        X = feats_df[feats].fillna(0.0)
        res = _eval_set(X, y, groups, args.cv, args.n_splits, args.standardize)
        rows.append({"set": name, "n_features": len(feats), **res})
        print(f"[OK] {name}: n={len(feats)} | acc={res['acc_mean']:.3f}±{res['acc_std']:.3f} | f1={res['f1_mean']:.3f}±{res['f1_std']:.3f}")

    out_csv = os.path.join(args.outdir, "ablation_results.csv")
    out = pd.DataFrame(rows).sort_values("set")
    out.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")

    # bar plot of Δ relative to baseline
    try:
        import matplotlib.pyplot as plt
        base_row = out[out["set"]=="baseline"].iloc[0] if (out["set"]=="baseline").any() else None
        if base_row is not None:
            out["delta_f1_vs_baseline"] = out["f1_mean"] - float(base_row["f1_mean"])
            out["delta_acc_vs_baseline"] = out["acc_mean"] - float(base_row["acc_mean"])
            fig = plt.figure(figsize=(8,5))
            order = [r for r in out["set"] if r!="baseline"]
            vals = out.set_index("set").loc[order, "delta_f1_vs_baseline"]
            plt.bar(range(len(order)), vals.values)
            plt.xticks(range(len(order)), order, rotation=30, ha="right")
            plt.ylabel("Δ F1 (vs baseline)")
            plt.title("Ablation — impatto sul F1 (extended / leave-one-out)")
            plt.tight_layout()
            fig.savefig(os.path.join(plots_dir, "ablation_delta_f1.png"), dpi=160)
            plt.close(fig)
    except Exception as e:
        print("[WARN] plot non generato:", e)

if __name__ == "__main__":
    main()
