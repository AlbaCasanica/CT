"""
Segment dynamic repetitions into two sub-phases:
  - phase 1: down (descent)  = start -> minimum point
  - phase 2: up   (ascent)   = minimum point -> end

Pivot used to find the minimum: total force (preferred), otherwise CoP_y, otherwise the sum of FSR sensors.
Robustness:
- smoothing via rolling median (window estimated from the time step);
- discard cases with minimum at the edges (can be disabled);
- if the time column is not useful (constant/non-monotonic), use a per-repetition index;
- automatically detects dynamic labels (excludes {static, idle, idling, rest}).

Output:
- Adds: phase ("down"/"up"), phase_id (1/2), idxmin_rel (position of the minimum).
- If segmentation fails for a dynamic repetition: adds 'segmentation_error' and leaves the repetition unchanged.
"""

import argparse
import pandas as pd
import numpy as np
from experiments.pt_2.phase_split import segment_all_reps

# --------------------------- time helpers ---------------------------

def _safe_parse_time(series: pd.Series) -> pd.Series:
    """Try converting a series to datetime. If it fails, return it unchanged."""
    try:
        return pd.to_datetime(series)
    except Exception:
        return series

def _choose_time_column(df: pd.DataFrame, rep_col: str, time_col: str) -> str:
    """
    Choose the time column to use:
      1) try using 'time_col' (parse as datetime when possible);
      2) if it's not informative (constant or non-monotonic within reps), create '_tidx' as a per-repetition counter.
    Returns the name of the column to use.
    """
    col = time_col
    if col not in df.columns:
        df["_tidx"] = df.groupby(rep_col).cumcount().astype(float)
        print(f"[INFO] Colonna tempo '{time_col}' assente → uso indice per rep: _tidx")
        return "_tidx"

    df[col] = _safe_parse_time(df[col])

    nunique = df[col].nunique(dropna=True)
    if nunique <= 1:
        df["_tidx"] = df.groupby(rep_col).cumcount().astype(float)
        print(f"[INFO] '{time_col}' non informativa (nunique={nunique}) → uso indice per rep: _tidx")
        return "_tidx"

    # monotonicity per repetition (accept if ≥70% of reps are monotonic)
    monotone_ratio = df.groupby(rep_col, sort=False)[col].apply(lambda s: s.is_monotonic_increasing).mean()
    if monotone_ratio < 0.7:
        df["_tidx"] = df.groupby(rep_col).cumcount().astype(float)
        print(f"[INFO] '{time_col}' non monotona in molte rep (ok≈{monotone_ratio:.2f}) → uso indice per rep: _tidx")
        return "_tidx"

    return col

# --------------------------- dynamic label helper ---------------------------

def _infer_dynamic_values(df: pd.DataFrame, label_col: str) -> tuple:
    """
    Treat as dynamic all labels except {static, idle, idling, rest}.
    Returns a tuple of dynamic values (strings exactly as present in the dataframe).
    """
    if label_col not in df.columns:
        print(f"[WARN] Colonna label '{label_col}' assente: userò ('dynamic','dyn',1).")
        return ("dynamic", "dyn", 1)

    exclude_static = {"static", "idle", "idling", "rest"}
    uniq = df[label_col].dropna().unique().tolist()
    dyn_vals = [v for v in uniq if str(v).lower() not in exclude_static]

    if not dyn_vals:
        # conservative fallback
        print("[WARN] Nessuna etichetta dinamica inferita → fallback ('dynamic','dyn',1).")
        return ("dynamic", "dyn", 1)

    dyn_vals_sorted = tuple(sorted(map(str, dyn_vals)))
    print(f"[INFO] Valori dinamici per la segmentazione: {dyn_vals_sorted}")
    return dyn_vals_sorted

# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Segmenta ripetizioni dinamiche in fasi down/up.")
    ap.add_argument("--in",  dest="inp", required=True,
                    help="CSV per-sample (con rep_id, label, colonna tempo e sensori FSR/force/CoP).")
    ap.add_argument("--out", dest="out", required=True,
                    help="CSV di output con colonne phase/phase_id.")
    ap.add_argument("--rep-col", default="rep_id", help="Nome colonna identificativa della ripetizione.")
    ap.add_argument("--label-col", default="label", help="Nome colonna etichetta (dynamic/static/altre posture).")
    ap.add_argument("--time-col", default="ReconstructedTime", help="Nome colonna temporale di input.")
    ap.add_argument("--prefer", choices=["force", "cop"], default="force",
                    help="Pivot preferita: 'force' (default) o 'cop'.")
    ap.add_argument("--win-sec", type=float, default=0.20,
                    help="Finestra smoothing in secondi (per la rolling median).")
    ap.add_argument("--min-samples-per-phase", type=int, default=5,
                    help="Minimo numero di campioni per ciascuna fase.")
    ap.add_argument("--allow-min-at-edges", action="store_true",
                    help="Permette il minimo ai bordi (di default è considerato errore).")
    # CSV read options
    ap.add_argument("--sep", default=",", help="Separatore CSV (default ',').")
    ap.add_argument("--decimal", default=".", help="Separatore decimale (default '.').")
    ap.add_argument("--encoding", default=None, help="Encoding file (default: autodetect).")
    args = ap.parse_args()

    # 1) Read CSV
    df = pd.read_csv(args.inp, sep=args.sep, decimal=args.decimal, encoding=args.encoding)

    # 2) Choose a usable time column (or create a per-repetition index)
    time_col_used = _choose_time_column(df, args.rep_col, args.time_col)
    if time_col_used != args.time_col:
        print(f"[INFO] Uso '{time_col_used}' come asse temporale per la segmentazione.")

    # 3) Automatically infer dynamic labels
    dyn_vals = _infer_dynamic_values(df, args.label_col)

    # 4) Segment all repetitions
    out = segment_all_reps(
        df,
        rep_col=args.rep_col,
        label_col=args.label_col,
        time_col=time_col_used,
        prefer=args.prefer,
        win_sec=args.win_sec,
        min_samples_per_phase=args.min_samples_per_phase,
        allow_min_at_edges=args.allow_min_at_edges,
        dynamic_values=dyn_vals
    )

    # 5) Write output
    out.to_csv(args.out, index=False)
    n_rows = len(out)
    n_phased = out["phase"].notna().sum() if "phase" in out.columns else 0
    n_err = out["segmentation_error"].notna().sum() if "segmentation_error" in out.columns else 0
    print(f"[OK] Salvato: {args.out} | righe={n_rows} | righe con phase={n_phased} | rep fallite={n_err}")

if __name__ == "__main__":
    main()
