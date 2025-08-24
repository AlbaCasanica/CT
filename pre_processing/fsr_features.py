import re
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple

# -----------------------------
# Utility su colonne e griglia
# -----------------------------
def fsr_columns(df: pd.DataFrame) -> List[str]:
    """
    Restituisce i nomi delle colonne FSR.
    Attesi nomi tipo 'Fsr.01'..'Fsr.16'. L'ordine finale è crescente per indice numerico.
    """
    cols = [c for c in df.columns if c.startswith("Fsr.")]
    def _num(c: str) -> int:
        m = re.search(r"(\d+)", c)
        return int(m.group(1)) if m else 10**9
    return sorted(cols, key=_num)


def default_grid_coords() -> Dict[str, Tuple[float, float]]:
    """
    Coordinate normalizzate (0..1) per la griglia 4x4, in row-major.
    Nota: inverti righe/colonne se la tua mappatura fisica è diversa.
    """
    coords = {}
    grid_size = 4
    for i in range(1, 17):
        idx = i - 1
        x = (idx % grid_size) / (grid_size - 1)
        y = (idx // grid_size) / (grid_size - 1)
        coords[f"Fsr.{i:02d}"] = (x, y)
    return coords


def _region_masks(cols: List[str]) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Crea i mask (indici) per heel/fore e medial/lateral in modo robusto:
    - ordina i 16 sensori per numero 1..16
    - costruisce la griglia 4x4 in row-major su QUESTO ordine
    - restituisce indici rispetto all'ordine ORIGINALE di `cols`
    """
    n = len(cols)
    assert n == 16, f"Attesi 16 canali FSR, trovati {n} (cols={list(cols)})"

    def num(c: str) -> int:
        m = re.search(r"(\d+)", c)
        if not m:
            raise ValueError(f"Impossibile estrarre numero da nome colonna: {c}")
        return int(m.group(1))

    nums = np.array([num(c) for c in cols])
    order = np.argsort(nums)  # indici originali ordinati 1..16

    heel   = [int(order[k]) for k in range(16) if (k // 4) in (0, 1)]  # due righe "posteriori"
    fore   = [int(order[k]) for k in range(16) if (k // 4) in (2, 3)]  # due righe "anteriori"
    medial = [int(order[k]) for k in range(16) if (k % 4)  in (0, 1)]  # colonne mediali
    lateral= [int(order[k]) for k in range(16) if (k % 4)  in (2, 3)]  # colonne laterali

    # Se la tua convenzione è invertita, scambia heel<->fore qui sopra.
    return heel, fore, medial, lateral


# -----------------------------
# Helpers per aggregazione
# -----------------------------
def _iqr(arr: np.ndarray) -> float:
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))

def _cv(arr: np.ndarray, eps: float = 1e-9) -> float:
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return float(s / (m + eps))


# -----------------------------
# Feature per campione (timestamp)
# -----------------------------
def per_sample_features(
    df: pd.DataFrame,
    side: str,
    coords: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Calcola feature per timestamp: somme/ratio regionali, CoP, attivazione, dispersione.
    Richiede in df: colonne FSR (Fsr.01..Fsr.16) + ReconstructedTime, label, rep_id.
    """
    if coords is None:
        coords = default_grid_coords()

    cols = fsr_columns(df)                      # 'Fsr.01'..'Fsr.16'
    X = df[cols].astype(float).to_numpy()       # (T,16)
    eps = 1e-9

    # Maschere regionali (una volta sola)
    H, F, M, L = _region_masks(cols)

    # Somme per regione (ASSOLUTE)
    heel_sum    = X[:, H].sum(axis=1)
    fore_sum    = X[:, F].sum(axis=1)
    medial_sum  = X[:, M].sum(axis=1)
    lateral_sum = X[:, L].sum(axis=1)
    total_sum   = X.sum(axis=1)

    # Frazioni di carico per regione (rispetto al totale)
    fore_frac    = fore_sum   / (total_sum + eps)
    heel_frac    = heel_sum   / (total_sum + eps)
    medial_frac  = medial_sum / (total_sum + eps)
    lateral_frac = lateral_sum/ (total_sum + eps)

    # Ratio per-campione (robuste con eps)
    foreheel_ratio = fore_sum   / (heel_sum + eps)
    medlat_ratio   = medial_sum / (lateral_sum + eps)

    # Centro di pressione 2D (peso = valore FSR)
    xy   = np.array([coords[c] for c in cols], dtype=float)  # (16,2)
    cop  = (X @ xy) / (total_sum.reshape(-1, 1) + eps)
    cop_x, cop_y = cop[:, 0], cop[:, 1]

    # Attivazione % (soglia robusta su tutto X)
    thr = np.percentile(X, 95) * 0.10
    active_fraction = (X >= thr).sum(axis=1) / X.shape[1]

    # Dispersione spaziale istantanea
    spatial_std = X.std(axis=1, ddof=0)
    spatial_var = spatial_std ** 2

    out = pd.DataFrame({
        "ReconstructedTime": df["ReconstructedTime"].values,
        "label": df["label"].values,
        "rep_id": df["rep_id"].values,

        # carico/attivazione
        f"total_sum_{side}": total_sum,
        f"avg_press_{side}": X.mean(axis=1),
        f"active_fraction_{side}": active_fraction,

        # CoP
        f"cop_x_{side}": cop_x,
        f"cop_y_{side}": cop_y,

        # dispersione
        f"spatial_std_{side}": spatial_std,
        f"spatial_var_{side}": spatial_var,

        # frazioni regionali
        f"fore_frac_{side}":    fore_frac,
        f"heel_frac_{side}":    heel_frac,
        f"medial_frac_{side}":  medial_frac,
        f"lateral_frac_{side}": lateral_frac,

        # ratio per-campione
        f"foreheel_ratio_{side}": foreheel_ratio,
        f"medlat_ratio_{side}":   medlat_ratio,
    })

    # DEBUG leggero (prime righe)
    if len(out) > 0:
        print(f"[DEBUG per_sample {side}] "
              f"fore~{out[f'fore_frac_{side}'].head(3).round(3).tolist()} "
              f"heel~{out[f'heel_frac_{side}'].head(3).round(3).tolist()} "
              f"med~{out[f'medial_frac_{side}'].head(3).round(3).tolist()} "
              f"lat~{out[f'lateral_frac_{side}'].head(3).round(3).tolist()}")
    return out


# -----------------------------
# Aggregazione per ripetizione
# -----------------------------
def _cop_path_len(x: np.ndarray, y: np.ndarray) -> float:
    dx = np.diff(x)
    dy = np.diff(y)
    return float(np.sum(np.sqrt(dx * dx + dy * dy)))


def aggregate_per_rep(per_sample: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    Aggrega le feature per ripetizione (rep_id).
    - Mantiene i campi legacy (avg_* etc.)
    - Aggiunge statistiche robuste per CoP e ratio: mean / std / IQR / CV
    """
    rows = []
    for (rep, label), g in per_sample.groupby(["rep_id", "label"]):
        # carico
        total_mean = float(g[f"total_sum_{side}"].mean())

        avg_mean = float(g[f"avg_press_{side}"].mean())
        avg_std  = float(g[f"avg_press_{side}"].std(ddof=0))
        avg_var  = float(avg_std ** 2)

        spatial_std_mean = float(g[f"spatial_std_{side}"].mean())
        spatial_var_mean = float(g[f"spatial_var_{side}"].mean())

        activation_pct = float(g[f"active_fraction_{side}"].mean() * 100.0)

        # CoP
        cop_x = g[f"cop_x_{side}"].to_numpy()
        cop_y = g[f"cop_y_{side}"].to_numpy()
        cop_x_mean = float(cop_x.mean())
        cop_y_mean = float(cop_y.mean())
        cop_x_std  = float(cop_x.std(ddof=0))
        cop_y_std  = float(cop_y.std(ddof=0))
        cop_x_iqr  = _iqr(cop_x)
        cop_y_iqr  = _iqr(cop_y)
        cop_x_cv   = _cv(cop_x)
        cop_y_cv   = _cv(cop_y)
        path_len   = _cop_path_len(cop_x, cop_y)

        # medie temporali delle frazioni
        fore_mean    = float(g[f"fore_frac_{side}"].mean())
        heel_mean    = float(g[f"heel_frac_{side}"].mean())
        medial_mean  = float(g[f"medial_frac_{side}"].mean())
        lateral_mean = float(g[f"lateral_frac_{side}"].mean())

        # Ratio "da medie" (come in versione precedente, utile per compatibilità)
        EPS = 1e-9
        foreheel_ratio_from_means = float(fore_mean / (heel_mean + EPS))
        medlat_ratio_from_means   = float(medial_mean / (lateral_mean + EPS))

        # Ratio statistiche direttamente dai per-campione
        fr_series = g[f"foreheel_ratio_{side}"].to_numpy()
        ml_series = g[f"medlat_ratio_{side}"].to_numpy()
        fr_mean = float(fr_series.mean())
        fr_std  = float(fr_series.std(ddof=0))
        fr_iqr  = _iqr(fr_series)
        fr_cv   = _cv(fr_series)

        ml_mean = float(ml_series.mean())
        ml_std  = float(ml_series.std(ddof=0))
        ml_iqr  = _iqr(ml_series)
        ml_cv   = _cv(ml_series)

        # timing
        t = g["ReconstructedTime"].to_numpy()
        duration = float(t[-1] - t[0]) if len(t) > 1 else 1.0
        path_per_sec = float(path_len / max(duration, 1e-9))

        rows.append({
            "rep_id": rep,
            "label": label,

            # carico
            f"total_mean_{side}": total_mean,

            # legacy
            f"avg_mean_{side}": avg_mean,
            f"avg_std_{side}": avg_std,
            f"avg_var_{side}": avg_var,
            f"spatial_std_mean_{side}": spatial_std_mean,
            f"spatial_var_mean_{side}": spatial_var_mean,
            f"activation_pct_{side}": activation_pct,

            # CoP (stat)
            f"cop_x_mean_{side}": cop_x_mean,
            f"cop_y_mean_{side}": cop_y_mean,
            f"cop_x_std_{side}":  cop_x_std,
            f"cop_y_std_{side}":  cop_y_std,
            f"cop_x_iqr_{side}":  cop_x_iqr,
            f"cop_y_iqr_{side}":  cop_y_iqr,
            f"cop_x_cv_{side}":   cop_x_cv,
            f"cop_y_cv_{side}":   cop_y_cv,

            f"cop_path_len_{side}": path_len,
            f"cop_path_per_sec_{side}": path_per_sec,
            f"samples_{side}": int(len(g)),

            # Ratio da medie (compat)
            f"foreheel_ratio_{side}": foreheel_ratio_from_means,
            f"medlat_ratio_{side}":   medlat_ratio_from_means,

            # Ratio statistiche dai per-sample
            f"foreheel_ratio__mean_{side}": fr_mean,
            f"foreheel_ratio__std_{side}":  fr_std,
            f"foreheel_ratio__iqr_{side}":  fr_iqr,
            f"foreheel_ratio__cv_{side}":   fr_cv,

            f"medlat_ratio__mean_{side}": ml_mean,
            f"medlat_ratio__std_{side}":  ml_std,
            f"medlat_ratio__iqr_{side}":  ml_iqr,
            f"medlat_ratio__cv_{side}":   ml_cv,
        })
    return pd.DataFrame(rows)


# -----------------------------
# Unione piede sx/dx
# -----------------------------
def combine_left_right(rep_L: pd.DataFrame, rep_R: pd.DataFrame) -> pd.DataFrame:
    """
    Unisce piede sinistro e destro su (rep_id,label), calcola medie "both"
    e feature di asimmetria (diff/absdiff) per vari indicatori.
    """
    df = pd.merge(rep_L, rep_R, on=["rep_id", "label"], how="inner")
    out = pd.DataFrame({"rep_id": df["rep_id"], "label": df["label"]})

    def pair_mean(base: str) -> np.ndarray:
        return df[[f"{base}_L", f"{base}_R"]].mean(axis=1).to_numpy()

    def pair_diff(base: str) -> np.ndarray:
        return (df[f"{base}_R"] - df[f"{base}_L"]).to_numpy()

    # --- Medie "both" (conserviamo quelli più utili) ---
    for base in [
        "avg_mean", "avg_std", "avg_var",
        "spatial_std_mean", "spatial_var_mean",
        "activation_pct",
        "cop_x_mean", "cop_y_mean",
        "cop_x_std", "cop_y_std",
        "cop_x_iqr", "cop_y_iqr",
        "cop_x_cv",  "cop_y_cv",
        "cop_path_len", "cop_path_per_sec",
        "samples",
        "foreheel_ratio", "medlat_ratio",
        "foreheel_ratio__mean", "foreheel_ratio__std", "foreheel_ratio__iqr", "foreheel_ratio__cv",
        "medlat_ratio__mean",   "medlat_ratio__std",   "medlat_ratio__iqr",   "medlat_ratio__cv",
        "total_mean",
    ]:
        cols = [f"{base}_L", f"{base}_R"]
        if all(c in df.columns for c in cols):
            out[f"{base}_both"] = pair_mean(base)

    # --- Asimmetrie L/R (diff e absdiff) ---
    eps = 1e-9
    # asimmetria di carico (usa total_mean come proxy)
    if {"total_mean_L", "total_mean_R"}.issubset(df.columns):
        load_asym = (df["total_mean_L"] - df["total_mean_R"]) / (df["total_mean_L"] + df["total_mean_R"] + eps)
        out["load_asym"] = load_asym.to_numpy()

    # diff/absdiff per ratio e CoP
    for base in ["foreheel_ratio", "medlat_ratio", "cop_x_mean", "cop_y_mean"]:
        cols = [f"{base}_L", f"{base}_R"]
        if all(c in df.columns for c in cols):
            d = pair_diff(base)
            out[f"{base}_diff"]    = d
            out[f"{base}_absdiff"] = np.abs(d)

    return out
