"""
Module: fsr_feature_extraction.py
Purpose:
    End-to-end pipeline to compute FSR-based features per sample, aggregate them per repetition,
    and merge left/right feet into a single feature table per (rep_id, label). Designed to be
    reused across tests via environment variable overrides.

Main steps:
    1) Load left/right Mitch FSR CSVs (with minimal cleaning).
    2) (Optional) Normalize repetition IDs over time within each label and drop mixed-label reps.
    3) Compute per-sample features (left/right).
    4) Aggregate per-repetition features (left/right).
    5) Merge left/right (inner join on rep_id, label), compute symmetric and asymmetry metrics.
    6) Save to outputs/features_test{ID}.csv (ID set via env or default).

Input expectations (per-side CSV):
    Required columns: ReconstructedTime, label, rep_id, Fsr.01..Fsr.16
    If time column differs, an automatic fallback renaming is attempted.

Outputs:
    - CSV with columns: [test_id], rep_id, label, and numeric features.

Environment overrides:
    LEFT_CSV, RIGHT_CSV, OUT_PATH, TEST_ID

Dependencies:
    - numpy, pandas
    - pre_processing.fsr_features: per_sample_features, aggregate_per_rep, combine_left_right
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# Project root for local imports
# =========================
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from pre_processing.fsr_features import (  # noqa: E402
    per_sample_features,
    aggregate_per_rep,
    combine_left_right,
)

# =========================
# Configuration (parametric via ENV) — baseline style, no calibration
# =========================
DEFAULT_LEFT  = ROOT / "data" / "test_4" / "mitch_B0308-old_left_big_del_end_dupli_new_time_fsr_pressure_labeled_no_idle_median_filter_segmented_ppsorted.csv"
DEFAULT_RIGHT = ROOT / "data" / "test_4" / "mitch_B0510-new_right_big_del_end_dupli_new_time_fsr_pressure_labeled_no_idle_median_filter_segmented_ppsorted.csv"

OUT_DIR  = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OUT = OUT_DIR / "features_test4.csv"

# ENV overrides (so the same script can be reused for any test)
LEFT_CSV  = Path(os.getenv("LEFT_CSV",  str(DEFAULT_LEFT)))
RIGHT_CSV = Path(os.getenv("RIGHT_CSV", str(DEFAULT_RIGHT)))
OUT_PATH  = Path(os.getenv("OUT_PATH",  str(DEFAULT_OUT)))

ADD_TEST_ID    = True
TEST_ID_VALUE  = int(os.getenv("TEST_ID", "4"))

# Pipeline flags
NORMALIZE_REP_IDS  = True   # align rep_id by start time within each label
DROP_MIXED         = True   # drop repetitions containing >1 label
CALIBRATE_PER_FOOT = False  # deliberately disabled: no REF_* required

# Minimal required metadata (as in Test 1)
REQUIRED_META = {"ReconstructedTime", "label", "rep_id"}


# =========================
# Helpers
# =========================
def _read_csv_safe(path: Path) -> pd.DataFrame:
    """
    Read CSV and drop common stray index columns, if present.
    """
    df = pd.read_csv(path)
    for junk in ("Unnamed: 0", "index"):
        if junk in df.columns:
            df = df.drop(columns=junk)
    return df


def _check_required(df: pd.DataFrame, side_name: str):
    """
    Validate required metadata and a minimum number of FSR channels.
    """
    missing = REQUIRED_META - set(df.columns)
    if missing:
        raise KeyError(
            f"[{side_name}] missing columns {missing}. "
            f"Required: ReconstructedTime, label, rep_id + Fsr.01..Fsr.16"
        )
    fsr_cols = [c for c in df.columns if c.startswith("Fsr.")]
    if len(fsr_cols) < 8:
        raise KeyError(
            f"[{side_name}] too few FSR columns (found {len(fsr_cols)}). Expected Fsr.01..Fsr.16"
        )


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce critical columns to consistent dtypes.
    """
    df = df.copy()
    if "rep_id" in df.columns:
        df["rep_id"] = df["rep_id"].astype(str)
    if "label" in df.columns:
        df["label"] = df["label"].astype(str)
    if "ReconstructedTime" in df.columns:
        df["ReconstructedTime"] = pd.to_numeric(df["ReconstructedTime"], errors="coerce")
    return df


def _normalize_rep_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize repetition IDs to <label>_<idx> where idx is the order by start time within each label.
    """
    need = {"label", "rep_id", "ReconstructedTime"}
    if not need.issubset(df.columns):
        raise ValueError("Need label, rep_id, ReconstructedTime to normalize repetitions.")
    t0 = (
        df.groupby(["label", "rep_id"])["ReconstructedTime"]
          .min().reset_index().rename(columns={"ReconstructedTime": "t0"})
    )
    t0 = t0.sort_values(["label", "t0"]).copy()
    t0["rep_idx"] = t0.groupby("label").cumcount() + 1
    t0["rep_id_std"] = t0["label"].astype(str) + "_" + t0["rep_idx"].astype(int).astype(str)
    key = t0.set_index(["label", "rep_id"])["rep_id_std"].to_dict()

    out = df.copy()
    out["rep_id"] = out.set_index(["label", "rep_id"]).index.map(key)
    out.reset_index(drop=True, inplace=True)
    return out


def _drop_mixed_reps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove repetitions that contain more than one label.
    """
    nunique = df.groupby("rep_id")["label"].nunique()
    keep_ids = nunique[nunique == 1].index
    dropped = nunique[nunique > 1]
    if len(dropped):
        print(f"⚠️  DROP_MIXED: removed {len(dropped)} mixed-label repetitions:", list(dropped.index)[:10], "…")
    return df[df["rep_id"].isin(keep_ids)].copy()


# =========================
# Load
# =========================
print(f"📥 Reading L: {LEFT_CSV.name}")
print(f"📥 Reading R: {RIGHT_CSV.name}")
df_L = _read_csv_safe(LEFT_CSV)
df_R = _read_csv_safe(RIGHT_CSV)

# Fallback: rename time column if needed
for d in (df_L, df_R):
    if "ReconstructedTime" not in d.columns:
        for cand in ("reconstructed_time", "Time", "time", "timestamp"):
            if cand in d.columns:
                d.rename(columns={cand: "ReconstructedTime"}, inplace=True)
                break

_check_required(df_L, "LEFT")
_check_required(df_R, "RIGHT")

df_L = _coerce_types(df_L)
df_R = _coerce_types(df_R)

# (Per-foot calibration deliberately disabled)
if CALIBRATE_PER_FOOT:
    print("ℹ️ CALIBRATE_PER_FOOT=True but the block is not implemented in this simplified version.")

# Normalize rep_id and remove mixed-label reps if requested
if NORMALIZE_REP_IDS:
    df_L = _normalize_rep_ids(df_L)
    df_R = _normalize_rep_ids(df_R)

if DROP_MIXED:
    before_L = df_L["rep_id"].nunique()
    before_R = df_R["rep_id"].nunique()
    df_L = _drop_mixed_reps(df_L)
    df_R = _drop_mixed_reps(df_R)
    after_L = df_L["rep_id"].nunique()
    after_R = df_R["rep_id"].nunique()
    print(f"🔎 DROP_MIXED: L {before_L}→{after_L} reps | R {before_R}→{after_R} reps")

print("DEBUG #reps per label (L):", df_L.groupby("label")["rep_id"].nunique().to_dict())
print("DEBUG #reps per label (R):", df_R.groupby("label")["rep_id"].nunique().to_dict())

# =========================
# Per-sample → per-rep → merge L/R
# =========================
print("⚙️  Computing per-sample features (L/R)…")
per_L = per_sample_features(df_L, side="L")
per_R = per_sample_features(df_R, side="R")

print("📦 Aggregating per repetition…")
rep_L = aggregate_per_rep(per_L, side="L")
rep_R = aggregate_per_rep(per_R, side="R")

# Overlap check before merge
L_set = set(zip(rep_L["label"], rep_L["rep_id"]))
R_set = set(zip(rep_R["label"], rep_R["rep_id"]))
common = L_set & R_set
print(f"🔗 Overlap L/R: common={len(common)} | only L={len(L_set - R_set)} | only R={len(R_set - L_set)}")

print("🔗 Merging left/right on (rep_id, label) [inner join]…")
features = combine_left_right(rep_L, rep_R)
features = features.drop_duplicates(subset=["rep_id", "label"]).reset_index(drop=True)

# Add test_id if requested
id_cols = ["rep_id", "label"]
if ADD_TEST_ID and "test_id" not in features.columns:
    features.insert(0, "test_id", TEST_ID_VALUE)
    id_cols = ["test_id"] + id_cols

# Order columns: identifiers first, then numeric features
num_cols = features.select_dtypes(include=np.number).columns.tolist()
save_cols = [c for c in id_cols if c in features.columns] + [c for c in num_cols if c not in id_cols]
features_out = features[save_cols].copy()

# =========================
# Save
# =========================
features_out.to_csv(OUT_PATH, index=False)
print(f"\n✅ Saved: {OUT_PATH}")
print("Shape:", features_out.shape)
print("\nPreview (first 5 rows):")
print(features_out.head())
