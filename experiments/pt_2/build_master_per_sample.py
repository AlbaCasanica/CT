"""
Create a per-sample "master" CSV by merging multiple raw files (not yet segmented by phase),
optionally adding CoP (cop_y, optional L/R) and force_total.

Example:
python -m experiments.build_master_per_sample \
  --in-glob "data/test_1/*.csv" \
  --out data/master_per_sample.csv \
  --side auto
"""

from __future__ import annotations
import argparse, glob, os, re
import pandas as pd
import numpy as np

# Try to import your function that computes per-sample CoP/forces
# (adjust these imports to your paths if needed)
try:
    # e.g., if you keep it in pre_processing/
    from pre_processing.fsr_features import per_sample_features
except Exception:
    try:
        # alternative version if you copied it under experiments/
        from experiments.fsr_features import per_sample_features
    except Exception:
        per_sample_features = None  # fallback: we can only sum FSRs for force_total

FSR_COL_PAT = re.compile(r"(?:^|_)(fsr\.?\d+|sensor\d+|F\d+)$", re.I)

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _fsr_cols(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        if FSR_COL_PAT.search(c):
            # keep only columns that contain at least one numeric value
            if _to_num(df[c]).notna().any():
                cols.append(c)
    return cols

def _infer_side_from_filename(fname: str) -> str | None:
    name = os.path.basename(fname).lower()
    if "left" in name or "_l" in name:
        return "L"
    if "right" in name or "_r" in name:
        return "R"
    return None

def _add_test_and_unique_rep_id(df: pd.DataFrame, filename: str,
                                rep_col="rep_id", label_col="label"):
    test_id = os.path.splitext(os.path.basename(filename))[0]
    df = df.copy()
    df["test_id"] = test_id
    # unique rep_id: <test_id>::<rep_id>
    if rep_col in df.columns:
        df["rep_id"] = df[rep_col].astype(str)
        df["rep_id"] = df["test_id"] + "::" + df["rep_id"]
    else:
        # if rep_id is missing, create a rough incremental one within each label block
        df["__row_idx"] = df.groupby(label_col).cumcount()
        df["rep_id"] = df["test_id"] + "::" + df[label_col].astype(str) + "::" + df["__row_idx"].astype(str)
        df.drop(columns=["__row_idx"], inplace=True)
    return df

def _augment_with_features(df: pd.DataFrame, side: str | None):
    """If possible, compute cop_y and force_total using your per_sample_features()."""
    if per_sample_features is None:
        # fallback: only force_total from FSR sum
        fsr = _fsr_cols(df)
        if fsr:
            df = df.copy()
            df["force_total"] = pd.concat([_to_num(df[c]) for c in fsr], axis=1).sum(axis=1, min_count=1)
        return df, []
    # use your function
    extra = per_sample_features(df, side=side, coords=None)
    # keep cop_y* and force_total* if present
    keep = [c for c in extra.columns if c.startswith(("cop_y", "force_total"))]
    if not keep:
        # fallback: at least force_total from FSR
        fsr = _fsr_cols(df)
        if fsr:
            df = df.copy()
            df["force_total"] = pd.concat([_to_num(df[c]) for c in fsr], axis=1).sum(axis=1, min_count=1)
        return df, []
    out = df.join(extra[keep])

    # ---- NORMALIZE CoP: ensure a global 'cop_y' column ----
    if "cop_y" not in out.columns:
        has_L = "cop_y_L" in out.columns
        has_R = "cop_y_R" in out.columns
        if has_L and has_R:
            out["cop_y"] = (out["cop_y_L"] + out["cop_y_R"]) / 2.0
        elif has_L:
            out["cop_y"] = out["cop_y_L"]
        elif has_R:
            out["cop_y"] = out["cop_y_R"]
    # -------------------------------------------------------

    # ---- COMPUTE CoP PATH LENGTH (cop_path_len_both) ----
    if "cop_y" in out.columns:
        # sort by time
        if "ReconstructedTime" in out.columns:
            out = out.sort_values(["rep_id", "ReconstructedTime"])
        else:
            out = out.sort_values(["rep_id"]).reset_index(drop=True)

        # frame-to-frame absolute differences for each rep
        out["__cop_diff"] = out.groupby("rep_id")["cop_y"].diff().abs()

        # compute path length per repetition
        cop_path = out.groupby("rep_id")["__cop_diff"].sum().rename("cop_path_len_both")
        out = out.join(cop_path, on="rep_id")

        out.drop(columns=["__cop_diff"], inplace=True)
    # -----------------------------------------------------

    return out, keep


def main():
    ap = argparse.ArgumentParser(description="Build master per-sample CSV (unisce test, aggiunge CoP/forze).")
    ap.add_argument("--in-glob", required=True, help="Pattern dei CSV raw (non segmentati per fase). Es: 'data/test_1/*.csv'")
    ap.add_argument("--out", required=True, help="CSV di output master.")
    ap.add_argument("--rep-col", default="rep_id")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--time-col", default="ReconstructedTime")
    ap.add_argument("--side", default="auto", choices=["auto","L","R","none"],
                    help="L/R per calcolare cop_y_L/R se supportato dal tuo modulo; 'auto' prova dal filename; 'none' cop_y globale.")
    ap.add_argument("--sep", default=","); ap.add_argument("--decimal", default="."); ap.add_argument("--encoding", default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.in_glob))
    if not paths:
        raise SystemExit(f"Nessun file trovato per pattern: {args.in_glob}")

    chunks = []
    all_kept_cols = set()

    for p in paths:
        df = pd.read_csv(p, sep=args.sep, decimal=args.decimal, encoding=args.encoding)
        # normalize required columns
        for need in [args.label_col, args.time_col]:
            if need not in df.columns:
                raise ValueError(f"{p}: manca la colonna obbligatoria '{need}'")
        # side
        if args.side == "auto":
            side = _infer_side_from_filename(p)
        elif args.side == "none":
            side = None
        else:
            side = args.side

        # add test_id and unique rep_id
        df = _add_test_and_unique_rep_id(df, p, rep_col=args.rep_col, label_col=args.label_col)

        # enrich with cop/forces if available
        df_aug, kept = _augment_with_features(df, side)
        all_kept_cols.update(kept)

        chunks.append(df_aug)

    master = pd.concat(chunks, axis=0, ignore_index=True)

    # Order columns for convenience
    preferred = ["test_id", "rep_id", args.label_col, args.time_col]
    # place FSR columns right after
    fsr_cols = _fsr_cols(master)
    # then derived columns (if any)
    derived = sorted([c for c in master.columns if c not in preferred + fsr_cols])
    ordered = preferred + fsr_cols + derived
    master = master[ordered]

    master.to_csv(args.out, index=False)
    print(f"[OK] Salvato master: {args.out} | righe={len(master)} | file_uniti={len(paths)} | colonne_CoP/force aggiunte={sorted(all_kept_cols)}")

if __name__ == "__main__":
    main()
