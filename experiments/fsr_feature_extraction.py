import os, sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pre_processing.fsr_features import (
    per_sample_features, aggregate_per_rep, combine_left_right
)

# cartella dati
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

df_L = pd.read_csv(os.path.join(data_dir, "mitch_B0510-new_left_big_pressure_labeled_no_idle_median_filter_k9_segmented_fsr_num_ppsorted.csv"))
df_R = pd.read_csv(os.path.join(data_dir, "mitch_B0308-old_right_big_pressure_labeled_no_idle_median_filter_k9_segmented_fsr_num_ppsorted.csv"))

per_L = per_sample_features(df_L, side="L")
per_R = per_sample_features(df_R, side="R")

rep_L = aggregate_per_rep(per_L, "L")
rep_R = aggregate_per_rep(per_R, "R")

features = combine_left_right(rep_L, rep_R)

# outputs
out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "fsr_features_first_test.csv")
features.to_csv(out_path, index=False)

print(f"Salvato: {out_path}")
print(features.head())
