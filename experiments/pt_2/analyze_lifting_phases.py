"""
Analyze lifting phases (down vs up) using phase-aggregated features.

What this script does:
- Automatically detects feature pairs named as <base>_phase_down and <base>_phase_up.
- For each detected pair, it produces and saves a boxplot (down vs up).
- Computes descriptive statistics (mean, std, median, delta) for each pair.
- Runs a paired Welch t-test (ttest_rel) between down and up values.
- Exports a CSV summary with statistics and p-values, sorted by significance.

Assumptions/requirements:
- The input dataset already contains phase-aggregated features with the exact suffixes
  `_phase_down` and `_phase_up`.
- The dataset contains a 'label' column and includes rows with label 'lifting'.
- Non-numeric columns are ignored when building phase pairs.
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

# NOTE: `Path` is used below; ensure it is imported elsewhere if you integrate this script.
# from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

PLOTS_DIR = "outputs/plots"
OUT_CSV   = "outputs/lifting_phase_stats.csv"
DATA_CSV  = "outputs/features_fsr_train.csvlp"  # Update this path if you use a different input file.

os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------- Load & filter ----------
# Read the full dataset and keep only rows labeled as "lifting".
df = pd.read_csv(DATA_CSV)
if "label" not in df.columns:
    raise ValueError("Column 'label' not found in dataset.")
df = df[df["label"].str.lower() == "lifting"].copy()
print(f"Lifting rows: {len(df)}")

if len(df) == 0:
    # Stop early if there are no lifting rows; the rest of the analysis requires them.
    raise SystemExit("No lifting rows found. Re-run extraction with lifting present.")

# ---------- Detect phase feature pairs ----------
# Pattern expected: <base>_phase_down and <base>_phase_up, where <base> is the common feature name.
down_cols = [c for c in df.columns if c.endswith("_phase_down")]
pairs = []
for c_down in down_cols:
    base = re.sub(r"_phase_down$", "", c_down)
    c_up = f"{base}_phase_up"
    if c_up in df.columns:
        # Keep only if both columns are numeric; otherwise skip.
        if np.issubdtype(df[c_down].dtype, np.number) and np.issubdtype(df[c_up].dtype, np.number):
            pairs.append((base, c_down, c_up))

pairs = sorted(set(pairs))
print(f"Detected {len(pairs)} phase feature pairs.")
if not pairs:
    # If no pairs are found, ensure that the upstream feature extraction produced the expected columns.
    raise SystemExit("No phase feature pairs found. Check Step B extraction produced *_phase_down/_phase_up.")

# ---------- Helper to save current fig ----------
def save_fig(name: str):
    """Save the current Matplotlib figure to the plots output directory."""
    path = os.path.join(PLOTS_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[saved] {path}")

# ---------- Per-feature analysis ----------
# Accumulate per-feature statistics and test results for a final CSV export.
rows = []  # list of dicts; each dict represents one feature pair summary

for base, c_down, c_up in pairs:
    # Ensure paired comparison: drop any rows with NaNs in either phase.
    sub = df[[c_down, c_up]].dropna()
    n = len(sub)
    if n < 3:
        # Too few valid paired samples for a robust test; skip stats but still attempt plotting.
        print(f"Skipping stats for {base}: only {n} valid paired rows.")
        continue

    x = sub[c_down].to_numpy()  # values for the "down" phase
    y = sub[c_up].to_numpy()    # values for the "up" phase
    delta = y - x               # simple difference (up - down)

    # Descriptive statistics per phase and for the delta.
    desc = {
        "feature_base": base,
        "n": int(n),
        "down_mean": float(np.mean(x)),
        "down_std": float(np.std(x, ddof=1)),
        "up_mean": float(np.mean(y)),
        "up_std": float(np.std(y, ddof=1)),
        "delta_mean": float(np.mean(delta)),
        "delta_std": float(np.std(delta, ddof=1)),
        "down_median": float(np.median(x)),
        "up_median": float(np.median(y)),
    }

    # Paired t-test (within-sample comparison of down vs up).
    try:
        tval, pval = ttest_rel(x, y, nan_policy="omit")
        desc["t_stat"] = float(tval)
        desc["p_value"] = float(pval)
    except Exception as e:
        # If the test fails (e.g., due to degenerate variance), record NaNs and continue.
        desc["t_stat"] = np.nan
        desc["p_value"] = np.nan
        print(f"ttest_rel failed for {base}: {e}")

    rows.append(desc)

    # Visualization: boxplot comparing distributions for down vs up.
    plt.figure(figsize=(5.5, 4))
    plt.boxplot([x, y], labels=["down", "up"])
    plt.title(f"{base} (lifting: down vs up)")
    plt.ylabel(base)
    save_fig(f"{base}_lifting_down_vs_up.png")
    plt.show()

# ---------- Export summary ----------
if rows:
    out_df = pd.DataFrame(rows)
    # Order by p-value ascending (most significant differences first), then by feature name.
    out_df = out_df.sort_values(by=["p_value", "feature_base"], na_position="last").reset_index(drop=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[saved] {OUT_CSV}")

    # Quick text preview of the most significant features.
    print("\nTop 10 features by significance (smallest p-values):")
    print(out_df.loc[:, ["feature_base", "n", "down_mean", "up_mean", "delta_mean", "t_stat", "p_value"]].head(10))
else:
    # No valid summaries generated; verify that phase features exist and contain numeric values.
    print("No valid rows to summarize. Check that phase features were generated correctly.")
