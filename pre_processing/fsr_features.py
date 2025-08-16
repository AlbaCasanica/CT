import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def fsr_columns(df: pd.DataFrame) -> List[str]:
    """Restituisce i nomi delle colonne FSR in ordine (Fsr.01..Fsr.16)."""
    cols = [c for c in df.columns if c.startswith("Fsr.")]
    return sorted(cols, key=lambda c: int(c.split(".")[1]))

def default_grid_coords() -> Dict[str, Tuple[float,float]]:
    """Crea coordinate normalizzate 4x4 per i sensori FSR (0..1)."""
    coords = {}
    grid_size = 4
    for i in range(1, 17):
        idx = i - 1
        x = (idx % grid_size) / (grid_size - 1)
        y = (idx // grid_size) / (grid_size - 1)
        coords[f"Fsr.{i:02d}"] = (x, y)
    return coords

def per_sample_features(df: pd.DataFrame, side: str, coords: Dict[str, Tuple[float,float]] | None = None) -> pd.DataFrame:
    """Calcola feature per ogni campione (timestamp)."""
    if coords is None:
        coords = default_grid_coords()
    cols = fsr_columns(df)
    X = df[cols].astype(float).values

    # Media pressionesource .venv/bin/activate
    avg = X.mean(axis=1)

    # Centro di pressione (CoP)
    xy = np.array([coords[c] for c in cols])  # (16,2)
    wsum = X.sum(axis=1, keepdims=True) + 1e-9
    cop = (X @ xy) / wsum
    cop_x, cop_y = cop[:, 0], cop[:, 1]

    # Attivazione % (sensori sopra soglia)
    thr = np.percentile(X, 95) * 0.10
    active_fraction = (X >= thr).sum(axis=1) / X.shape[1]

    # Dispersione spaziale
    spatial_std = X.std(axis=1, ddof=0)
    spatial_var = spatial_std**2

    out = pd.DataFrame({
        "ReconstructedTime": df["ReconstructedTime"].values,
        "label": df["label"].values,
        "rep_id": df["rep_id"].values,
        f"avg_press_{side}": avg,
        f"cop_x_{side}": cop_x,
        f"cop_y_{side}": cop_y,
        f"active_fraction_{side}": active_fraction,
        f"spatial_std_{side}": spatial_std,
        f"spatial_var_{side}": spatial_var,
    })
    return out

def _cop_path_len(x: np.ndarray, y: np.ndarray) -> float:
    dx = np.diff(x)
    dy = np.diff(y)
    return float(np.sum(np.sqrt(dx*dx + dy*dy)))

def aggregate_per_rep(per_sample: pd.DataFrame, side: str) -> pd.DataFrame:
    """Aggrega le feature per ripetizione (rep_id)."""
    rows = []
    for (rep, label), g in per_sample.groupby(["rep_id", "label"]):
        avg_mean = float(g[f"avg_press_{side}"].mean())
        avg_std = float(g[f"avg_press_{side}"].std(ddof=0))
        avg_var = float(avg_std**2)
        spatial_std_mean = float(g[f"spatial_std_{side}"].mean())
        spatial_var_mean = float(g[f"spatial_var_{side}"].mean())
        activation_pct = float(g[f"active_fraction_{side}"].mean() * 100.0)
        cop_x_mean = float(g[f"cop_x_{side}"].mean())
        cop_y_mean = float(g[f"cop_y_{side}"].mean())
        cop_x_std = float(g[f"cop_x_{side}"].std(ddof=0))
        cop_y_std = float(g[f"cop_y_{side}"].std(ddof=0))
        path_len = _cop_path_len(g[f"cop_x_{side}"].to_numpy(), g[f"cop_y_{side}"].to_numpy())
        rows.append({
            "rep_id": rep,
            "label": label,
            f"avg_mean_{side}": avg_mean,
            f"avg_std_{side}": avg_std,
            f"avg_var_{side}": avg_var,
            f"spatial_std_mean_{side}": spatial_std_mean,
            f"spatial_var_mean_{side}": spatial_var_mean,
            f"activation_pct_{side}": activation_pct,
            f"cop_x_mean_{side}": cop_x_mean,
            f"cop_y_mean_{side}": cop_y_mean,
            f"cop_x_std_{side}": cop_x_std,
            f"cop_y_std_{side}": cop_y_std,
            f"cop_path_len_{side}": path_len,
            f"samples_{side}": int(len(g))
        })
    return pd.DataFrame(rows)

def combine_left_right(rep_L: pd.DataFrame, rep_R: pd.DataFrame) -> pd.DataFrame:
    """Unisce piede sinistro e destro e calcola medie 'both'."""
    df = pd.merge(rep_L, rep_R, on=["rep_id","label"], how="inner")
    out = pd.DataFrame({"rep_id": df["rep_id"], "label": df["label"]})
    def pair_mean(base: str) -> np.ndarray:
        return df[[f"{base}_L", f"{base}_R"]].mean(axis=1).to_numpy()
    # Baseline
    out["avg_mean_both"] = pair_mean("avg_mean")
    # Extended
    for base in ["avg_std","avg_var","spatial_std_mean","spatial_var_mean",
                 "activation_pct","cop_x_mean","cop_y_mean","cop_x_std","cop_y_std","cop_path_len","samples"]:
        out[f"{base}_both"] = pair_mean(base)
    return out
