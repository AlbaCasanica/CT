"""
Analyze CoP features for STATIC vs DYNAMIC trials.
- Builds a macro_label (static/dynamic)
- Shows & saves boxplots and summary stats
- Runs a Welch t-test on CoP path length
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

ROOT = Path(__file__).resolve().parents[2]

# ---------- Argument parsing ----------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--in",
    dest="input",
    type=Path,
    required=True,
    help="Input CSV file (e.g., data/master_per_sample.csv)"
)
parser.add_argument(
    "--outdir",
    type=Path,
    default=ROOT / "outputs/cop_static_vs_dynamic",
    help="Output directory for plots"
)
args = parser.parse_args()

PLOTS_DIR = args.outdir

# ---------- Helper functions ----------
def ensure_dir(path: Path) -> None:
    """Ensure that a directory exists (create if missing)."""
    path.mkdir(parents=True, exist_ok=True)

def save_current_fig(filename: str) -> None:
    """Save the current matplotlib figure into the output directory."""
    ensure_dir(PLOTS_DIR)
    out_path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[saved] {out_path}")

# ---------- Load dataset ----------
df = pd.read_csv(args.input)

print("Dataset shape:", df.shape)
print("Fine-grained class distribution:\n", df["label"].value_counts(), "\n")

# ---------- Map fine-grained labels -> macro_label (static / dynamic) ----------
STATIC_CLASSES = {"sitting", "standing", "hand_up_back", "hands_forward", "hands_up"}
DYNAMIC_CLASSES = {"lifting", "pull", "push", "squatting", "walking"}

def to_macro(lbl: str) -> str:
    """Convert detailed class label into 'static' or 'dynamic' macro label."""
    if lbl in STATIC_CLASSES:
        return "static"
    elif lbl in DYNAMIC_CLASSES:
        return "dynamic"
    else:
        return "other"

df["macro_label"] = df["label"].apply(to_macro)
print("Macro-label distribution:\n", df["macro_label"].value_counts(), "\n")

unknown = df.loc[df["macro_label"] == "other", "label"].unique()
if len(unknown) > 0:
    print("WARNING: Found unmapped labels ->", list(unknown))

# ---------- Boxplots grouped by macro_label ----------
plt.figure(figsize=(6, 4))
df.boxplot(column="cop_path_len_both", by="macro_label")
plt.title("CoP Path Length (static vs dynamic)")
plt.suptitle("")
plt.ylabel("Path length (units depend on coords)")
plt.xlabel("")
save_current_fig("cop_path_len_static_vs_dynamic.png")
plt.show()

if "cop_y_std" in df.columns:
    plt.figure(figsize=(6, 4))
    df.boxplot(column="cop_y_std", by="macro_label")
    plt.title("CoP_y Std (static vs dynamic)")
    plt.suptitle("")
    plt.ylabel("cop_y_std")
    plt.xlabel("")
    save_current_fig("cop_y_std_static_vs_dynamic.png")
    plt.show()

# ---------- Descriptive statistics ----------
print("\nSummary statistics for CoP path length by macro_label:")
print(df.groupby("macro_label")["cop_path_len_both"].describe(), "\n")

if "cop_y_std" in df.columns:
    print("Summary statistics for cop_y_std by macro_label:")
    print(df.groupby("macro_label")["cop_y_std"].describe(), "\n")

# ---------- Welch t-test: static vs dynamic ----------
static_vals = df.loc[df["macro_label"] == "static", "cop_path_len_both"].dropna()
dynamic_vals = df.loc[df["macro_label"] == "dynamic", "cop_path_len_both"].dropna()

if len(static_vals) > 1 and len(dynamic_vals) > 1:
    t, p = ttest_ind(static_vals, dynamic_vals, equal_var=False)
    print(f"Welch t-test on cop_path_len_both: t = {t:.2f}, p = {p:.3e}")
else:
    print("Not enough samples in one of the groups to run the t-test.")
