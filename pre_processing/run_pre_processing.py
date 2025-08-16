from labeling import apply_corrected_labels, extract_pressure_data, remove_idle, sort_push_pull
from clean_mitch_timestamps import run_fix_timestamp_mitch
import rotating_imu_data
from filter_sensor_data import median_filter_medfilt
from segment_repetitions import make_start_stop_segmentation_file, apply_repetition_label


raw_mitch_left      = ""
raw_mitch_right     = ""
start_stop_labels   = ""

"""
"-- Fix time, delet end duplicates and fix fsr numbers mitch --"
df_fix_left, path_fix_left = run_fix_timestamp_mitch(raw_mitch_left)
df_fix_right, path_fix_right = run_fix_timestamp_mitch(raw_mitch_right)


"-- Add labes to mitch --"
df_label_left, path_label_left = apply_corrected_labels(path_fix_left, start_stop_labels, df_fix_left)
df_label_right, path_label_right = apply_corrected_labels(path_fix_right, start_stop_labels, df_fix_right)
"""
# Take a brake, check the the plot with all labels and update label file if nessesary

# Filepaths nessessary from here on:
label_file_arm      = ""
label_file_back     = ""
label_file_left     = ""
label_file_right    = ""
filepath_foler      = "" 

"""
"-- Update labels if nessessary --"
df_label_arm,   path_label_arm      = apply_corrected_labels(label_file_arm, start_stop_labels)
df_label_back,  path_label_back     = apply_corrected_labels(label_file_back, start_stop_labels)
df_label_left,  path_label_left     = apply_corrected_labels(label_file_left, start_stop_labels)
df_label_right, path_label_right    = apply_corrected_labels(label_file_right, start_stop_labels)


"-- Rotate Data --"
'''
df_standing_arm = labeling.extract_one_activity(labeled_file_imu_arm, "standing")
df_standing_back = labeling.extract_one_activity(labeled_file_imu_back, "standing")

g_arm = rotating_imu_data.compute_mean_gravity(df_standing_arm)
g_back = rotating_imu_data.compute_mean_gravity(df_standing_back)

R_arm = rotating_imu_data.compute_rotation(g_arm)
R_back = rotating_imu_data.compute_rotation(g_back)

df_arm_rotated, out_path_rotate_arm = rotating_imu_data.rotate_sensor_data(labeled_file_imu_arm, R_arm)
df_back_rotated, out_path_rotate_back = rotating_imu_data.rotate_sensor_data(labeled_file_imu_back, R_back)

# Optional: plot z-axis of rotated data:
rotating_imu_data.plot_z_axis_before_and_after(labeled_file_imu_arm, labeled_file_imu_back, df_arm_rotated, df_back_rotated)
'''

"-- Drop IMU data from mitch --"
df_fsr_left,    path_fsr_left   = extract_pressure_data(path_label_left, df_label_left)
df_fsr_right,   path_fsr_right  = extract_pressure_data(path_label_right, df_label_right)


"-- Remove idle --"
df_no_idle_arm,     path_no_idle_arm    = remove_idle(path_label_arm, df_label_arm)
df_no_idle_back,    path_no_idle_back   = remove_idle(path_label_back, df_label_back)
df_no_idle_left,    path_no_idle_left   = remove_idle(path_fsr_left, df_fsr_left)
df_no_idle_right,   path_no_idle_right  = remove_idle(path_fsr_right, df_fsr_right)


"-- Apply median filter --"
imu_columns = ["Axl.X", "Axl.Y", "Axl.Z",
               "Gyr.X", "Gyr.Y", "Gyr.Z",
               "Mag.X", "Mag.Y", "Mag.Z"]
fsr_columns = [f"Fsr.{str(sensor).zfill(2)}" for sensor in range(1, 17)]

df_median_arm,      path_median_arm     = median_filter_medfilt(path_no_idle_arm, imu_columns, kernel_size=21, df=df_no_idle_arm)
df_median_back,     path_median_back    = median_filter_medfilt(path_no_idle_back, imu_columns, kernel_size=21, df=df_no_idle_back)
df_median_left,     path_median_left    = median_filter_medfilt(path_no_idle_left, fsr_columns, kernel_size=9, df=df_no_idle_left)
df_median_right,    path_median_right   = median_filter_medfilt(path_no_idle_right, fsr_columns, kernel_size=9, df=df_no_idle_right)
"""

"-- Segment Repetitions --"
#df_start_stop_segm, path_start_stop_seg = make_start_stop_segmentation_file(path_median_arm, path_median_back, start_stop_labels, filepath_foler)

# Take a brake
# 1. Look at the start stop file just generated, change times if some dont stop and start again at same time, save changes
# 2. Comment out the part over this
# 3. Comment in the part below this
# 4. Add start stop filepaht below
# 5. run




start_stop_file     = ""
filter_file_arm     = ""
filter_file_back    = ""
filter_file_left    = ""
filter_file_right   = ""

#I tilfelle denne på kjøres på nytt:
#df_start_stop_segm, path_start_stop_seg = make_start_stop_segmentation_file(filter_file_arm, filter_file_back, start_stop_labels, filepath_foler)

df_segmented_arm, path_segmented_arm        = apply_repetition_label(filter_file_arm, start_stop_file, None, None)
df_segmented_back, path_segmented_back      = apply_repetition_label(filter_file_back, start_stop_file, None, None)
df_segmented_left, path_segmented_left      = apply_repetition_label(filter_file_left, start_stop_file, None, None)
df_segmented_right, path_segmented_right    = apply_repetition_label(filter_file_right, start_stop_file, None, None)


"-- Sort push and pull --"
df_ppsort_arm, path_ppsort_arm      = sort_push_pull(path_segmented_arm, df_segmented_arm)
df_ppsort_back, path_ppsort_back    = sort_push_pull(path_segmented_back, df_segmented_back)
df_ppsort_left, path_ppsort_left    = sort_push_pull(path_segmented_left, df_segmented_left)
df_ppsort_right, path_ppsort_right  = sort_push_pull(path_segmented_right, df_segmented_right)
