from __future__ import annotations
import re
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict

PHASE_DOWN = "down"   # phase 1: descent (start -> minimum)
PHASE_UP   = "up"     # phase 2: ascent  (minimum -> end)

def _median_dt(t: pd.Series) -> float:
    """
    Estimate the average time step in seconds (works with datetime or float).
    Patch: avoid .view(); use .astype('int64') for compatibility with recent pandas.
    """
    if np.issubdtype(t.dtype, np.datetime64):
        # nanoseconds -> seconds
        td = t.astype("int64").astype("float64") / 1e9
        diffs = np.diff(td)
    else:
        diffs = np.diff(t.to_numpy(dtype="float64"))
    diffs = diffs[np.isfinite(diffs)]
    return np.median(diffs) if diffs.size else 0.01

def _roll_median(x: pd.Series, win_samples: int) -> pd.Series:
    win = max(3, int(win_samples) | 1)  # enforce odd window size
    return x.rolling(win, center=True, min_periods=max(1, win//3)).median()

def _infer_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Try to find useful columns:
    - global or side-specific CoP_y (cop_y, cop_y_L/R, etc.)
    - global or side-specific force_total
    Returns a dict with keys: cop_y, cop_y_L, cop_y_R, force_total, force_total_L, force_total_R.
    """
    cols = {k: None for k in ["cop_y","cop_y_L","cop_y_R","force_total","force_total_L","force_total_R"]}
    def pick(*patterns):
        for p in patterns:
            c = [c for c in df.columns if re.fullmatch(p, c, flags=re.I)]
            if c: return c[0]
        return None
    cols["cop_y"]         = pick(r"cop_y", r"cop_y_y", r"copY")
    cols["cop_y_L"]       = pick(r"cop_y_?L", r"L_cop_y")
    cols["cop_y_R"]       = pick(r"cop_y_?R", r"R_cop_y")
    cols["force_total"]   = pick(r"force_total", r"total_force", r"F_total")
    cols["force_total_L"] = pick(r"force_total_?L", r"L_force_total")
    cols["force_total_R"] = pick(r"force_total_?R", r"R_force_total")
    return cols

def _to_numeric_safe(s: pd.Series) -> pd.Series:
    """Coerce to numeric with errors='coerce' (strings -> NaN, index preserved)."""
    return pd.to_numeric(s, errors="coerce")

def _build_pivot(df: pd.DataFrame,
                 prefer: str = "force",
                 inferred: Optional[Dict[str, Optional[str]]] = None) -> pd.Series:
    """
    Build the pivot series used to locate the repetition's minimum.
    prefer: 'force' (default) or 'cop'.

    Strategy:
      1) If prefer='force': use force_total or L+R; fallback to cop_y (global or L/R mean).
      2) If prefer='cop': vice versa.
      3) Final fallback: sum of all FSR-like columns (coerced to numeric).
    """
    if inferred is None:
        inferred = _infer_cols(df)

    # Helper accessors
    f_tot  = inferred["force_total"]
    f_L    = inferred["force_total_L"]
    f_R    = inferred["force_total_R"]
    cop    = inferred["cop_y"]
    cop_L  = inferred["cop_y_L"]
    cop_R  = inferred["cop_y_R"]

    def has(col): return (col is not None) and (col in df.columns)

    def force_series():
        if has(f_tot):
            s = _to_numeric_safe(df[f_tot])
            if s.notna().any(): return s
        parts = []
        if has(f_L): parts.append(_to_numeric_safe(df[f_L]))
        if has(f_R): parts.append(_to_numeric_safe(df[f_R]))
        if parts:
            s = pd.concat(parts, axis=1).sum(axis=1, min_count=1)
            if s.notna().any(): return s
        return None

    def cop_series():
        if has(cop):
            s = _to_numeric_safe(df[cop])
            if s.notna().any(): return s
        parts = []
        if has(cop_L): parts.append(_to_numeric_safe(df[cop_L]))
        if has(cop_R): parts.append(_to_numeric_safe(df[cop_R]))
        if parts:
            s = pd.concat(parts, axis=1).mean(axis=1)
            if s.notna().any(): return s
        return None

    # Build by preference
    if prefer == "force":
        s = force_series()
        if s is None:
            s = cop_series()
    else:
        s = cop_series()
        if s is None:
            s = force_series()

    if s is not None and s.notna().any():
        return s

    # Final fallback: sum of FSR-like columns (numeric coercion)
    fsr_like = [c for c in df.columns if re.search(r"(fsr|sensor|F\d+)", c, flags=re.I)]
    fsr_like = [c for c in fsr_like if c in df.columns]
    if fsr_like:
        arr = [_to_numeric_safe(df[c]) for c in fsr_like]
        s = pd.concat(arr, axis=1).sum(axis=1, min_count=1)
        if s.notna().any():
            return s

    raise ValueError("Cannot build pivot: no valid numeric column for force/CoP/FSR.")

def split_rep_phases(rep_df: pd.DataFrame,
                     time_col: str = "ReconstructedTime",
                     prefer: str = "force",
                     win_sec: float = 0.20,
                     min_samples_per_phase: int = 5,
                     allow_min_at_edges: bool = False,
                     add_phase_column: bool = True
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Split a single repetition into two phases:
      - Phase 1 (down): start -> index of the pivot series minimum
      - Phase 2 (up):   (minimum) -> end
    Returns (df_phase1, df_phase2, idxmin_global)

    Robustness features:
      - rolling-median smoothing (window from win_sec and average dt)
      - pivot coerced to numeric; if all-NaN after smoothing, try raw series
      - robust idxmin (fill NaNs with +inf)
      - guard against minima at the edges (can be disabled)
      - enforce a minimum number of samples per phase
    """
    if rep_df.empty:
        raise ValueError("rep_df is empty.")

    rep_df = rep_df.sort_values(time_col).reset_index(drop=True)
    t = rep_df[time_col]
    dt = _median_dt(t)
    win_samples = max(3, int(round(win_sec / max(dt, 1e-6))))

    pivot_raw = _build_pivot(rep_df, prefer=prefer)
    pivot_smooth = _roll_median(_to_numeric_safe(pivot_raw), win_samples)

    # If smoothing produced all NaNs, fallback to the raw series
    if pivot_smooth.isna().all():
        pivot_smooth = _to_numeric_safe(pivot_raw)

    # Still all NaN -> explicit error
    if pivot_smooth.isna().all():
        raise ValueError("Pivot is all NaN (FSR/force/CoP non-numeric or empty).")

    # Minimum index robust to NaNs
    idxmin = int(pivot_smooth.fillna(np.inf).idxmin())

    # Minimum at the edges?
    if (idxmin <= 0 or idxmin >= len(rep_df)-1) and not allow_min_at_edges:
        # Try a larger window
        pivot_smooth2 = _roll_median(_to_numeric_safe(pivot_raw), int(win_samples*1.5)+1)
        if pivot_smooth2.isna().all():
            pivot_smooth2 = _to_numeric_safe(pivot_raw)
        idxmin2 = int(pivot_smooth2.fillna(np.inf).idxmin())
        if 0 < idxmin2 < len(rep_df)-1:
            idxmin = idxmin2
        else:
            # Last attempt: direct raw series
            idxmin_raw = int(_to_numeric_safe(pivot_raw).fillna(np.inf).idxmin())
            if 0 < idxmin_raw < len(rep_df)-1:
                idxmin = idxmin_raw
            else:
                raise ValueError("Minimum at the edge: cannot reliably segment.")

    # Slices: phase1 = [0 .. idxmin], phase2 = [idxmin .. end]
    df1 = rep_df.iloc[:idxmin+1].copy()
    df2 = rep_df.iloc[idxmin:].copy()

    if len(df1) < min_samples_per_phase or len(df2) < min_samples_per_phase:
        raise ValueError(f"Insufficient segmentation: "
                         f"phase1={len(df1)}, phase2={len(df2)} (<{min_samples_per_phase}).")

    if add_phase_column:
        df1["phase"] = PHASE_DOWN
        df2["phase"] = PHASE_UP
        df1["phase_id"] = 1
        df2["phase_id"] = 2

    return df1, df2, idxmin

def segment_all_reps(df: pd.DataFrame,
                     rep_col: str = "rep_id",
                     label_col: str = "label",
                     dynamic_values = ("dynamic","dyn",1),
                     time_col: str = "ReconstructedTime",
                     prefer: str = "force",
                     win_sec: float = 0.20,
                     min_samples_per_phase: int = 5,
                     allow_min_at_edges: bool = False) -> pd.DataFrame:
    """
    Apply the split to ALL dynamic repetitions and return a single DataFrame
    with additional columns: phase, phase_id, idxmin_rel (position of the minimum within the rep).
    Non-dynamic repetitions are kept unchanged (without a 'phase' column).
    """
    out = []
    dyn_set = {str(x).lower() for x in dynamic_values}
    for (rep, lab), grp in df.groupby([rep_col, label_col], sort=False):
        is_dynamic = str(lab).lower() in dyn_set
        if is_dynamic:
            try:
                d1, d2, idxmin = split_rep_phases(
                    grp, time_col=time_col, prefer=prefer, win_sec=win_sec,
                    min_samples_per_phase=min_samples_per_phase,
                    allow_min_at_edges=allow_min_at_edges, add_phase_column=True
                )
                d1["idxmin_rel"] = idxmin
                d2["idxmin_rel"] = idxmin
                out.extend([d1, d2])
            except Exception as e:
                g = grp.copy()
                g["phase"] = None
                g["phase_id"] = None
                g["segmentation_error"] = str(e)
                out.append(g)
        else:
            out.append(grp.copy())

    res = pd.concat(out, axis=0).sort_values([rep_col, time_col]).reset_index(drop=True)
    return res
