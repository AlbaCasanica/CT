"""
Descriptive analysis of phase differences (down/up) starting from
outputs/features_master_phased.csv (one row per (rep_id, phase)).

What it produces:
1) Summary CSVs:
   - outputs/phased_summary_by_label.csv           -> mean, std, median by label x phase
   - outputs/phased_pct_down_gt_up_by_label.csv    -> % of (rep_id) with down > up for each metric
   - outputs/phased_deltas_by_label.csv            -> paired differences (up - down) summarized by label
2) Plots (.png) in outputs/plots/:
   - boxplot_down_vs_up_force_mean_[overall|perlabel].png
   - boxplot_down_vs_up_cop_y_mean_[overall|perlabel].png (if cop_y is available)
   - violin_delta_up_minus_down_[force_mean|cop_y_mean].png (distribution of paired differences)

Example:
python -m experiments.analyze_phased_features \
  --in outputs/features_master_phased.csv \
  --outdir outputs
"""

from __future__ import annotations
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


METRICS_DEFAULT = ["force_mean", "force_iqr", "force_max", "n_samples", "force_std"]
COP_METRICS = ["cop_y_mean", "cop_y_iqr", "cop_y_std"]

def _ensure_outdirs(outdir: str):
    """Create the output directory and its 'plots' subfolder if they don't exist; return the plots dir path."""
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def _has_cop(df: pd.DataFrame) -> bool:
    """Check whether any CoP_y metrics are present in the dataframe columns."""
    return any(c.startswith("cop_y_") for c in df.columns)

def _summary_by_label(df: pd.DataFrame, out_csv: str, metrics: list[str]):
    """Compute mean, std, and median by (label, phase) for the selected metrics and save to CSV."""
    key = ["label", "phase"]
    use_cols = [c for c in metrics if c in df.columns]
    summ = df.groupby(key)[use_cols].agg(["mean", "std", "median"]).round(3)
    summ.to_csv(out_csv)
    print(f"[OK] summary per label salvato -> {out_csv}")
    return summ

def _pct_down_gt_up(df: pd.DataFrame, out_csv: str, metrics: list[str]):
    """
    For each label and metric, compute the percentage of repetitions with down > up.
    The calculation is done on a wide table indexed by rep_id with one column per phase.
    """
    rows = []
    for label, g in df.groupby("label", sort=False):
        # Build a wide table: rows = rep_id, columns = phase for each metric
        rep_ids = g["rep_id"].unique()
        for metric in metrics:
            if metric not in g.columns: 
                continue
            wide = g.pivot(index="rep_id", columns="phase", values=metric)
            wide = wide.dropna()
            if wide.empty:
                rows.append({"label": label, "metric": metric, "n_reps": 0, "pct_down_gt_up": np.nan})
                continue
            pct = float((wide.get("down") > wide.get("up")).mean() * 100.0)
            rows.append({"label": label, "metric": metric, "n_reps": int(len(wide)), "pct_down_gt_up": round(pct, 1)})
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"[OK] % down>up per label salvato -> {out_csv}")
    return out

def _paired_deltas(df: pd.DataFrame, out_csv: str, metrics: list[str]):
    """
    Compute paired differences (up - down) for each rep_id and label.
    Summarize mean, std, median, and IQR by label and metric.
    """
    rows = []
    for label, g in df.groupby("label", sort=False):
        wide = {}
        repids = sorted(g["rep_id"].unique())
        for metric in metrics:
            if metric not in g.columns: 
                continue
            w = g.pivot(index="rep_id", columns="phase", values=metric)
            w = w.dropna()
            if w.empty: 
                continue
            delta = (w.get("up") - w.get("down")).dropna()
            if delta.empty: 
                continue
            d = delta.to_numpy()
            q25, q50, q75 = np.percentile(d, [25, 50, 75])
            rows.append({
                "label": label,
                "metric": metric,
                "n_reps": int(len(d)),
                "delta_mean": float(np.mean(d)),
                "delta_std": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
                "delta_median": float(q50),
                "delta_iqr": float(q75 - q25),
                "delta_min": float(np.min(d)),
                "delta_max": float(np.max(d)),
            })
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"[OK] deltas (up-down) per label salvato -> {out_csv}")
    return out

def _boxplot_overall(df: pd.DataFrame, metric: str, title: str, out_png: str):
    """Create a single overall boxplot comparing down vs up for the given metric and save it."""
    plt.figure(figsize=(6, 5))
    data = [df.loc[df["phase"]=="down", metric].dropna(),
            df.loc[df["phase"]=="up",   metric].dropna()]
    plt.boxplot(data, labels=["down", "up"], showfliers=False)
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[OK] plot -> {out_png}")

def _boxplot_per_label(df: pd.DataFrame, metric: str, out_png: str, max_cols: int = 3):
    """
    Create a grid of boxplots (down vs up) per label for the given metric.
    Handles missing data by printing a placeholder message in the corresponding subplot.
    """
    labels = df["label"].dropna().unique().tolist()
    if not labels:
        return
    n = len(labels)
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=(5*ncols, 4*nrows))
    for i, lab in enumerate(labels, 1):
        sub = df[df["label"] == lab]
        ax = plt.subplot(nrows, ncols, i)
        data = [sub.loc[sub["phase"]=="down", metric].dropna(),
                sub.loc[sub["phase"]=="up",   metric].dropna()]
        if any(len(d)==0 for d in data):
            ax.text(0.5, 0.5, "dati insufficienti", ha="center", va="center")
            ax.set_axis_off()
            continue
        ax.boxplot(data, labels=["down","up"], showfliers=False)
        ax.set_title(str(lab))
        ax.set_ylabel(metric)
    plt.suptitle(f"Boxplot down vs up per label — {metric}", y=0.995)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[OK] plot -> {out_png}")

def _violin_deltas(df: pd.DataFrame, metric: str, out_png: str):
    """
    Build a violin plot of paired differences (up - down) for the given metric (overall).
    The differences are computed by pivoting to (rep_id x phase) and subtracting.
    """
    # Compute up - down for each rep_id (overall)
    w = df.pivot_table(index="rep_id", columns="phase", values=metric, aggfunc="first")
    w = w.dropna()
    if w.empty:
        return
    d = (w["up"] - w["down"]).dropna()
    plt.figure(figsize=(6,5))
    plt.violinplot(d.to_numpy(), showmeans=True, showextrema=True)
    plt.axhline(0, linestyle="--")
    plt.xticks([1], [f"Δ = up - down"])
    plt.ylabel(metric)
    plt.title(f"Differenza paired (up - down) — {metric}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[OK] plot -> {out_png}")

def main():
    ap = argparse.ArgumentParser(description="Analisi descrittiva down vs up da features per-fase.")
    ap.add_argument("--in", dest="inp", required=True, help="CSV features per (rep_id, phase). Es: outputs/features_master_phased.csv")
    ap.add_argument("--outdir", default="outputs", help="Cartella output (CSV + plots/).")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    if not {"rep_id","label","phase"}.issubset(df.columns):
        raise SystemExit("Il file deve contenere almeno: rep_id, label, phase.")

    plots_dir = _ensure_outdirs(args.outdir)

    # Available metrics
    metrics = [m for m in METRICS_DEFAULT if m in df.columns]
    if _has_cop(df):
        metrics_cop = [m for m in COP_METRICS if m in df.columns]
        metrics_all = metrics + metrics_cop
    else:
        metrics_cop = []
        metrics_all = metrics

    # 1) Summary by label x phase
    _summary_by_label(df, os.path.join(args.outdir, "phased_summary_by_label.csv"), metrics_all)

    # 2) % of reps with down > up (per metric)
    _pct_down_gt_up(df, os.path.join(args.outdir, "phased_pct_down_gt_up_by_label.csv"), metrics_all)

    # 3) Paired differences (up - down), summarized by label
    _paired_deltas(df, os.path.join(args.outdir, "phased_deltas_by_label.csv"), metrics_all)

    # 4) Synthetic plots
    if "force_mean" in df.columns:
        _boxplot_overall(df, "force_mean", "Forza media — down vs up (overall)",
                         os.path.join(plots_dir, "boxplot_down_vs_up_force_mean_overall.png"))
        _boxplot_per_label(df, "force_mean",
                           os.path.join(plots_dir, "boxplot_down_vs_up_force_mean_perlabel.png"))
        _violin_deltas(df, "force_mean",
                       os.path.join(plots_dir, "violin_delta_up_minus_down_force_mean.png"))

    if "cop_y_mean" in df.columns:
        _boxplot_overall(df, "cop_y_mean", "CoP_y medio — down vs up (overall)",
                         os.path.join(plots_dir, "boxplot_down_vs_up_cop_y_mean_overall.png"))
        _boxplot_per_label(df, "cop_y_mean",
                           os.path.join(plots_dir, "boxplot_down_vs_up_cop_y_mean_perlabel.png"))
        _violin_deltas(df, "cop_y_mean",
                       os.path.join(plots_dir, "violin_delta_up_minus_down_cop_y_mean.png"))

    print("[DONE] Analisi completata.")

if __name__ == "__main__":
    main()
