"""
Extract per-phase (down/up) features from a per-sample CSV that is already segmented.
Works with a single foot as well as (optionally) with L/R columns if present.

Expected input: per-sample CSV with at least:
  - rep_id (configurable)
  - label  (configurable)
  - phase  (down / up)  <-- produced by run_phase_split.py
  - FSR columns: e.g., Fsr.01, Fsr.02, ... (their sum = force_total)
  - (optional) force_total, cop_y, force_total_L/R, cop_y_L/R

Output: a CSV with ONE row per (rep_id, phase) (and label), containing feature statistics.

Example:
python -m experiments.fsr_feature_extraction_phased \
  --in outputs/mitch_B0308_right_phased.csv \
  --out outputs/features_test1_phased_right.csv \
  --rep-col rep_id --label-col label --phase-col phase --time-col ReconstructedTime
"""

from __future__ import annotations
import argparse
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# --------------------------- utilities ---------------------------

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _find_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Try to detect standard columns if they already exist.
    Otherwise, they will be computed from raw FSR signals.
    """
    def pick(*patterns):
        for p in patterns:
            hits = [c for c in df.columns if re.fullmatch(p, c, flags=re.I)]
            if hits:
                return hits[0]
        return None

    cols = dict(
        force_total = pick(r"force_total", r"total_force", r"F_total"),
        force_total_L = pick(r"force_total_?L", r"L_force_total"),
        force_total_R = pick(r"force_total_?R", r"R_force_total"),
        cop_y = pick(r"cop_y", r"copY"),
        cop_y_L = pick(r"cop_y_?L", r"L_cop_y"),
        cop_y_R = pick(r"cop_y_?R", r"R_cop_y"),
    )
    return cols

def _fsr_like_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if re.search(r"(fsr|sensor|F\d+)", c, flags=re.I)]
    # keep only numeric (or coercible) columns
    keep = []
    for c in cols:
        s = _to_num(df[c])
        if s.notna().any():
            keep.append(c)
    return keep

def _ensure_force_cols(df: pd.DataFrame, meta_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """
    Ensure that at least 'force_total' is available.
    If it doesn't exist, create it by summing FSR-like columns.
    """
    cols = _find_cols(df)
    if cols["force_total"] is None:
        fsr_cols = [c for c in _fsr_like_cols(df) if c not in meta_cols]
        if not fsr_cols:
            raise ValueError("Nessuna colonna force_total trovata e nessuna FSR-like disponibile per calcolarla.")
        df = df.copy()
        df["force_total"] = pd.concat([_to_num(df[c]) for c in fsr_cols], axis=1).sum(axis=1, min_count=1)
        cols["force_total"] = "force_total"
    return df, cols

def _duration_sec(g: pd.DataFrame, time_col: str) -> float:
    """Compute duration in seconds if the time column is datetime or a monotonic numeric; otherwise use n_samples."""
    if time_col not in g.columns:
        return float(len(g))
    s = g[time_col]
    if np.issubdtype(s.dtype, np.datetime64):
        dt = (s.iloc[-1] - s.iloc[0]).total_seconds()
        return float(dt) if np.isfinite(dt) and dt >= 0 else float(len(g))
    else:
        s_num = _to_num(s)
        if s_num.notna().all():
            dt = float(s_num.iloc[-1] - s_num.iloc[0])
            return dt if np.isfinite(dt) and dt >= 0 else float(len(g))
        return float(len(g))

# --------------------------- base features ---------------------------

def _basic_stats(x: pd.Series, prefix: str) -> Dict[str, float]:
    x = _to_num(x).dropna()
    if x.empty:
        return {
            f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan, f"{prefix}_min": np.nan,
            f"{prefix}_p25": np.nan,  f"{prefix}_p50": np.nan, f"{prefix}_p75": np.nan,
            f"{prefix}_max": np.nan,  f"{prefix}_iqr": np.nan,
        }
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_p25": float(q25),
        f"{prefix}_p50": float(q50),
        f"{prefix}_p75": float(q75),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_iqr": float(q75 - q25),
    }

def _activation_pct(df: pd.DataFrame, threshold: float, fsr_cols: List[str]) -> float:
    """Percentage of active sensors (value > threshold) averaged over time."""
    if not fsr_cols:
        return np.nan
    arr = pd.concat([_to_num(df[c]) for c in fsr_cols], axis=1)
    active = (arr > threshold).mean(axis=1)  # active share per row
    return float(active.mean()) * 100.0

def _symmetry_index(a: pd.Series, b: pd.Series) -> float:
    """Symmetry index: (A - B) / (A + B). Mean over time. Range [-1, 1]."""
    A = _to_num(a); B = _to_num(b)
    den = A + B
    si = (A - B) / den.replace(0, np.nan)
    return float(si.mean()) if si.notna().any() else np.nan

# --------------------------- pipeline ---------------------------

def compute_features_per_phase(
    df: pd.DataFrame,
    rep_col: str = "rep_id",
    label_col: str = "label",
    phase_col: str = "phase",
    time_col: str = "ReconstructedTime",
    activation_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Return a DataFrame with ONE row per (rep_id, phase) including the label.
    Features computed:
      - stats on force_total (mean, std, min, p25, p50, p75, max, iqr)
      - duration (sec or n_samples)
      - n_samples
      - activation_pct (if raw FSR are present)
      - (if present) stats on cop_y; symmetry index for force_total_L/R and cop_y_L/R
    """
    # meta checks
    for c in [rep_col, label_col, phase_col]:
        if c not in df.columns:
            raise ValueError(f"Colonna richiesta assente: {c}")

    meta_cols = [rep_col, label_col, phase_col, time_col]
    df, cols = _ensure_force_cols(df, meta_cols=meta_cols)

    # raw FSR for activation%
    fsr_cols = [c for c in _fsr_like_cols(df) if c not in meta_cols]

    # optional columns
    f_total = cols["force_total"]
    fL, fR = cols["force_total_L"], cols["force_total_R"]
    cop, copL, copR = cols["cop_y"], cols["cop_y_L"], cols["cop_y_R"]

    # Fallback: if 'cop_y' is missing, derive it from L/R if available
    if (not cop) or (cop not in df.columns):
        if copL and (copL in df.columns) and copR and (copR in df.columns):
            df = df.copy()
            df["cop_y"] = (pd.to_numeric(df[copL], errors="coerce")
                       + pd.to_numeric(df[copR], errors="coerce")) / 2.0
            cop = "cop_y"
        elif copL and (copL in df.columns):
            df = df.copy()
            df["cop_y"] = pd.to_numeric(df[copL], errors="coerce")
            cop = "cop_y"
        elif copR and (copR in df.columns):
            df = df.copy()
            df["cop_y"] = pd.to_numeric(df[copR], errors="coerce")
            cop = "cop_y"


    rows = []
    group_keys = [rep_col, phase_col]
    # carry the label along (we'll take the first as the aggregate value)
    grouped = df.groupby(group_keys, sort=False)

    for (rep, ph), g in grouped:
        # base fields
        feat = {
            rep_col: rep,
            phase_col: ph,
            label_col: g[label_col].iloc[0] if label_col in g.columns else None,
            "n_samples": int(len(g)),
            "duration_sec": _duration_sec(g, time_col),
        }

        # force_total stats
        feat.update(_basic_stats(g[f_total], "force"))

        # cop_y stats (if available)
        if cop and cop in g.columns:
            feat.update(_basic_stats(g[cop], "cop_y"))
        else:
            # placeholder if not present
            feat.update({k: np.nan for k in [
                "cop_y_mean","cop_y_std","cop_y_min","cop_y_p25","cop_y_p50","cop_y_p75","cop_y_max","cop_y_iqr"
            ]})

        # activation %
        feat["activation_pct"] = _activation_pct(g, activation_threshold, fsr_cols)

        # if L/R are available → symmetry
        if fL and fR and fL in g.columns and fR in g.columns:
            feat["force_symmetry_index"] = _symmetry_index(g[fL], g[fR])
            # separate L/R stats useful for descriptive reporting
            feat.update({**_basic_stats(g[fL], "force_L"), **_basic_stats(g[fR], "force_R")})
        else:
            feat["force_symmetry_index"] = np.nan

        if copL and copR and copL in g.columns and copR in g.columns:
            feat["cop_y_symmetry_index"] = _symmetry_index(g[copL], g[copR])
        else:
            feat["cop_y_symmetry_index"] = np.nan

        rows.append(feat)

    out = pd.DataFrame(rows)
    # neat column ordering
    main = [rep_col, label_col, phase_col, "n_samples", "duration_sec"]
    rest = [c for c in out.columns if c not in main]
    return out[main + sorted(rest)]

# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Estrazione feature per (rep_id, phase) da CSV per-sample phased.")
    ap.add_argument("--in", dest="inp", required=True, help="CSV per-sample con colonna 'phase'.")
    ap.add_argument("--out", dest="out", required=True, help="CSV di output per-(rep,phase).")
    ap.add_argument("--rep-col", default="rep_id")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--phase-col", default="phase")
    ap.add_argument("--time-col", default="ReconstructedTime")
    ap.add_argument("--activation-th", type=float, default=0.0,
                    help="Soglia per considerare un sensore FSR 'attivo' (default=0).")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--decimal", default=".")
    ap.add_argument("--encoding", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.inp, sep=args.sep, decimal=args.decimal, encoding=args.encoding)

    out = compute_features_per_phase(
        df,
        rep_col=args.rep_col,
        label_col=args.label_col,
        phase_col=args.phase_col,
        time_col=args.time_col,
        activation_threshold=args.activation_th,
    )

    out.to_csv(args.out, index=False)
    print(f"[OK] Salvato: {args.out} | righe={len(out)} | rep*phase uniche={out[['%s'%args.rep_col, '%s'%args.phase_col]].drop_duplicates().shape[0]}")

if __name__ == "__main__":
    main()
