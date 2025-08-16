import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import mode

import labeling



def find_mag_peaks(activity_df, peak_distance, peak_height, filepath=None, num_segments=None):
    if activity_df is None:
        activity_df = pd.read_csv(filepath)

    activity_df["Axl_mag"] = np.sqrt(activity_df["Axl.X"]**2 + activity_df["Axl.Y"]**2 + activity_df["Axl.Z"]**2)
    
    # --- Detect Peaks ---
    peaks, _ = find_peaks(activity_df["Axl_mag"], distance=peak_distance, height=peak_height)

    # Check even number of peaks
    if len(peaks) % 2 != 0:
        peaks = peaks[:-1]  # Drop last peak if odd

    # Limit number of peaks if num_segments is set
    if num_segments is not None:
        peaks = peaks[:num_segments * 2]

    # --- Visualize Detected Peaks ---
    plt.figure(figsize=(12, 5))
    plt.plot(activity_df["ReconstructedTime"], activity_df["Axl_mag"], label="Accel Mag")
    plt.plot(activity_df["ReconstructedTime"].iloc[peaks], activity_df["Axl_mag"].iloc[peaks], "rx", label="Peaks")
    plt.title(f"Detected Peaks for {len(peaks)//2} Segments")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration Magnitude (mg)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return peaks, activity_df


def cut_repetitions_in_midle(peaks, activity_df, filepath, activity, save_file=False, save_start_stop=False):

    # Compute cut points between each peak pair
    cut_indices = [(peaks[i] + peaks[i + 1]) // 2 for i in range(1, len(peaks)-1, 2)]

    # --- Label entire DataFrame ---
    activity_df["rep_id"] = ""
    current_rep = 1
    last_cut = 0
    summary_rows = []

    for cut in cut_indices:
        rep_label = f"{activity}_{current_rep}"
        activity_df.loc[last_cut:cut, "rep_id"] = rep_label
        
        start_time = activity_df["ReconstructedTime"].iloc[last_cut]
        end_time = activity_df["ReconstructedTime"].iloc[cut]
        summary_rows.append([rep_label, round(start_time, 2), round(end_time, 2)])
        
        current_rep += 1
        last_cut = cut + 1

    # Label the remaining tail
    rep_label = f"{activity}_{current_rep}"
    activity_df.loc[last_cut:, "rep_id"] = rep_label

    start_time = activity_df["ReconstructedTime"].iloc[last_cut]
    end_time = activity_df["ReconstructedTime"].iloc[-1]
    summary_rows.append([rep_label, round(start_time, 2), round(end_time, 2)])

    summary_df = pd.DataFrame(summary_rows, columns=["label", "Start Time (s)", "End Time (s)"])

    if save_file:
        #activity_df.drop(columns=["Axl_mag"], inplace=True)
        output_path = filepath.replace(".csv", f"_{activity}_segm.csv")
        activity_df.to_csv(output_path, index=False)
        print(f"Labeled repetitions as '{activity}_N' and saved to:\n{output_path}")
    
    if save_start_stop:
        summary_path = filepath.replace(".csv", f"_{activity}_segments_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved labeled repetitions to:\n{output_path}")
        print(f"Saved segment summary to:\n{summary_path}")

    plt.figure(figsize=(12, 5))
    plt.plot(activity_df["ReconstructedTime"], 
             np.sqrt(activity_df["Axl.X"]**2 + activity_df["Axl.Y"]**2 + activity_df["Axl.Z"]**2), 
             label="Accel Magnitude")
    for cut in cut_indices:
        if cut < len(activity_df):
            plt.axvline(activity_df["ReconstructedTime"].iloc[cut], color='red', linestyle='--')
    plt.title(f"Repetition Segmentation: {activity}")
    plt.xlabel("Time")
    plt.ylabel("Accel Magnitude")
    plt.grid(True)
    plt.legend()
    plt.show()

    return activity_df, summary_df


def make_segments(filepath, activity):
    df_activity = labeling.extract_one_activity(filepath, activity, save_file=False)
    peaks,_ = find_mag_peaks(df_activity, 1500, 1100)
    df_segmented, df_summary = cut_repetitions_in_midle(peaks, df_activity, None, activity)
    return df_segmented, df_summary


def number_push_pull(filepath_start_stop_label):
    df_labels = pd.read_csv(filepath_start_stop_label)
    push_counter = 1
    pull_counter = 1
    new_rows = []

    already_segmented = {"hand_up_back", "hands_forward", "hands_up", "squatting", "lifting", "sitting", "standing", "walking"}

    for _, row in df_labels.iterrows():
        label = row["label"]
        
        if label in already_segmented:
            continue  # hopp over

        if row["label"] == "push":
            label = f"push_{push_counter}"
            push_counter += 1
        elif row["label"] == "pull":
            label = f"pull_{pull_counter}"
            pull_counter += 1
        
        new_rows.append([label, row["Start Time (s)"], row["End Time (s)"]])

    df_pushpull_numbered = pd.DataFrame(new_rows, columns=["label", "Start Time (s)", "End Time (s)"])
    return df_labels, df_pushpull_numbered


def make_start_stop_segmentation_file(filepath_arm, filepath_back, filepath_start_stop_labels, filepath_folder):
    _, df_hand_up_back_sum = make_segments(filepath_arm, "hand_up_back")
    _, df_hands_forward_sum = make_segments(filepath_arm, "hands_forward")
    _, df_hands_up_sum = make_segments(filepath_arm, "hands_up")

    df_labels, df_push_pull_num = number_push_pull(filepath_start_stop_labels)

    _, df_squatting_sum = make_segments(filepath_back, "squatting")
    _, df_lifting_sum = make_segments(filepath_back, "lifting")
    
    df_static_activities = df_labels.tail(3).copy()
    
    df_combined = pd.concat([df_hand_up_back_sum, 
                             df_hands_forward_sum, 
                             df_hands_up_sum,
                             df_push_pull_num,
                             df_squatting_sum,
                             df_lifting_sum,
                             df_static_activities], 
                             ignore_index=True)
    
    output_path = f"{filepath_folder}/sart_stop_segments.csv"
    df_combined.to_csv(output_path, index=False)

    return df_combined, output_path


def apply_repetition_label(sensor_file, start_stop_file, sensor_df, start_stop_df):
    if sensor_df is None:
        sensor_df = pd.read_csv(sensor_file)
    if start_stop_df is None:
        start_stop_df = pd.read_csv(start_stop_file)
    
    time_array = sensor_df["ReconstructedTime"].values
    sensor_df["rep_id"] = "idle"

    for _, row in start_stop_df.iterrows():
        start_time = row["Start Time (s)"]
        end_time = row["End Time (s)"]
        label = row["label"]

        # Find rows within the corrected time range
        start_idx = np.searchsorted(time_array, start_time, side="left")
        end_idx = np.searchsorted(time_array, end_time, side="right")

        sensor_df.loc[start_idx:end_idx, "rep_id"] = label

    output_filepath = f"{sensor_file.rstrip(".csv")}_segmented.csv"
    sensor_df.to_csv(output_filepath, index=False)

    return sensor_df, output_filepath
    


filepath_arm = ""
filepath_back = ""
#filepath_start_stop_labels = ""
filepath_foler = ""

#df_combined, output_path = make_start_stop_segmentation_file(filepath_arm, filepath_back, filepath_start_stop_labels, filepath_foler)

start_stop_file = ""
#apply_repetition_label(filepath_arm, start_stop_file, None, None)
apply_repetition_label(filepath_back, start_stop_file, None, None)


filepath_mitch_left = ""
filepath_mitch_right = ""
#apply_repetition_label(filepath_mitch_left, start_stop_file, None, None)
#apply_repetition_label(filepath_mitch_right, start_stop_file, None, None)