import clean_mitch_timestamps
import labeling


mitch_left = ""
mitch_right = ""

df_dupli_left, outpath_dupli_left = clean_mitch_timestamps.remove_end_duplicates(mitch_left)
df_dupli_right, outpath_dupli_right = clean_mitch_timestamps.remove_end_duplicates(mitch_right)

df_time_left, otupath_time_left = clean_mitch_timestamps.add_ReconstructedTime(outpath_dupli_left, df_dupli_left)
df_time_right, otupath_time_right = clean_mitch_timestamps.add_ReconstructedTime(outpath_dupli_right, df_dupli_right)

filepath_start_stop_labels = ""
df_label_left, outpath_label_left = labeling.apply_corrected_labels(otupath_time_left, filepath_start_stop_labels, df_time_left)
df_label_right, outpath_label_right = labeling.apply_corrected_labels(otupath_time_right, filepath_start_stop_labels, df_time_right)

df_fsr_left, outpath_fsr_left = labeling.extract_pressure_data(outpath_label_left, df_label_left)
df_fsr_right, outpath_fsr_right = labeling.extract_pressure_data(outpath_label_right, df_label_right)

df_fsr_no_idle_left, outpath_fsr_no_idle_left = labeling.remove_idle(outpath_fsr_left, df_fsr_left)
df_fsr_no_idle_right, outpath_fsr_no_idle_right = labeling.remove_idle(outpath_fsr_right, df_fsr_right)
