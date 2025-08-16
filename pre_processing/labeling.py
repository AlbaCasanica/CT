import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def detect_spikes(file_path):
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    df["acc_mag"] = np.sqrt(df["Axl.X"]**2 + df["Axl.Y"]**2 + df["Axl.Z"]**2)

    peaks, _ = find_peaks(df["acc_mag"], height=3000, distance=1000)

    return df, peaks


def init_label(df, peaks, file_path):
    df["label"] = "idle"  # default label
    buffer = 2      # Nuber to be excluded from labeling on each side of peak
    
    for i in range(len(peaks) - 1):
        start = peaks[i] + buffer + 1   # Exclude the peak
        end = peaks[i + 1] - buffer - 1 # Exclude the next peak
        df.loc[start:end, "label"] = f"Activity_{i+1}"
    
    df.to_csv(f"{file_path.rstrip(".txt").rstrip(".csv")}_init_labeled.csv", index=False)
    return df


def plot_signal_peaks(df_labeled, peaks, filepath_label=None):
    
    if filepath_label is not None:
        df_labeled = pd.read_csv(filepath_label)

    plt.figure(figsize=(15, 5))
    plt.plot(df_labeled["ReconstructedTime"], df_labeled["Axl.X"], label="Accelerometer X")
    plt.plot(df_labeled["ReconstructedTime"].iloc[peaks], df_labeled["Axl.X"].iloc[peaks], "rx", label="Detected Spikes")

    # Shade areas based on labels (excluding 'idle')
    unique_labels = df_labeled["label"].unique()
    colors = plt.get_cmap("hsv", len(unique_labels))

    for i, label in enumerate(unique_labels):
        if label == "idle":
            continue
        activity_df = df_labeled[df_labeled["label"] == label]
        if not activity_df.empty:
            start_time = activity_df["ReconstructedTime"].iloc[0]
            end_time = activity_df["ReconstructedTime"].iloc[-1]
            plt.axvspan(start_time, end_time, alpha=0.2, color=colors(i), label=label, zorder=0)

    #plt.legend()
    plt.title("labeled Segments Between Spikes (Different Colors)")
    plt.xlabel("Time")
    plt.ylabel("Sensor Value")
    plt.tight_layout()
    plt.show()


def plot_closeups_around_peaks(df, peaks, window_seconds=0.25, sampling_rate=800):
    window_size = int(window_seconds * sampling_rate)  # number of samples in window
    labels = df["label"].unique()
    colors = {label: plt.cm.hsv(i / len(labels)) for i, label in enumerate(labels)}

    for peak in peaks:
        start_idx = max(0, peak - window_size)
        end_idx = min(len(df) - 1, peak + window_size)
        segment = df.iloc[start_idx:end_idx]

        plt.figure(figsize=(10, 4))
        plt.plot(segment["ReconstructedTime"], segment["Axl.X"], label="Axl.X", zorder=1)
        plt.axvline(df["ReconstructedTime"].iloc[peak], color="red", linestyle="--", label="Peak", zorder=2)

        # Shade regions by label
        for label in segment["label"].unique():
            if label == "idle":
                continue
            label_segment = segment[segment["label"] == label]
            if not label_segment.empty:
                start_time = label_segment["ReconstructedTime"].iloc[0]
                end_time = label_segment["ReconstructedTime"].iloc[-1]
                plt.axvspan(start_time, end_time, color=colors[label], alpha=0.3, label=label, zorder=0)

        plt.title(f"Close-up Around Peak at {df['ReconstructedTime'].iloc[peak]:.3f}s")
        plt.xlabel("Time (s)")
        plt.ylabel("Axl.X")
        plt.legend()
        plt.tight_layout()
        plt.show()


def extract_activity_windows(df, output_path="activity_windows.csv", exclude_idle=True):
    activity_windows = []

    # Ensure labels are filled consistently
    df = df.copy()
    df["label"] = df["label"].fillna("idle")

    # Identify label boundaries
    current_label = None
    start_time = None

    for i, row in df.iterrows():
        label = row["rep_id"]
        time = row["ReconstructedTime"]

        if label != current_label:
            if current_label is not None:
                activity_windows.append({
                    "label": current_label,
                    "Start Time (s)": start_time,
                    "End Time (s)": prev_time
                })
            current_label = label
            start_time = time
        prev_time = time

    # Append last activity
    if current_label is not None:
        activity_windows.append({
            "label": current_label,
            "Start Time (s)": start_time,
            "End Time (s)": prev_time
        })

    activity_df = pd.DataFrame(activity_windows)

    if exclude_idle:
        activity_df = activity_df[activity_df["label"] != "idle"]
    
    activity_df.to_csv(f"{output_path}/sart_stop_segments.csv", index=False)
    print(f"Activity window summary saved to {output_path}")
    return activity_df


def apply_corrected_labels(new_time_file, start_stop_csv_path, df=None, rename_map=None):
    if df is None:
        df = pd.read_csv(new_time_file)
    
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    corrected = pd.read_csv(start_stop_csv_path)
    time_array = df["ReconstructedTime"].values
    df["label"] = "idle"

    if rename_map:
        corrected["label"] = corrected["label"].replace(rename_map)
    
    for _, row in corrected.iterrows():
        start_time = row["Start Time (s)"]
        end_time = row["End Time (s)"]
        label = row["label"]

        # Find rows within the corrected time range
        start_idx = np.searchsorted(time_array, start_time, side="left")
        end_idx = np.searchsorted(time_array, end_time, side="right")

        df.loc[start_idx:end_idx, "label"] = label

    output_filepath = f"{new_time_file.rstrip(".txt").rstrip(".csv")}_labeled.csv"
    df.to_csv(output_filepath, index=False)

    return df, output_filepath


def remove_idle(correct_label_filepath, correct_label_df=None):
    if correct_label_df is None:
        correct_label_df = pd.read_csv(correct_label_filepath)
    
    correct_label_df = correct_label_df.loc[:, ~correct_label_df.columns.str.contains("^Unnamed")]
    
    if "label" not in correct_label_df.columns:
        raise ValueError("The DataFrame must contain a 'label' column.")

    # Filter out idle samples
    labeled_df = correct_label_df[correct_label_df["label"] != "idle"].copy()

    output_path = f"{correct_label_filepath.rstrip(".txt").rstrip(".csv")}_no_idle.csv"
    labeled_df.to_csv(output_path, index=False)

    return labeled_df, output_path


def extract_one_activity(filepath, activity, save_file=True):
    df = pd.read_csv(filepath)

    if "label" not in df.columns:
        raise ValueError("The DataFrame must contain a 'label' column.")
    
    activity_df = df[df["label"] == activity].copy()
    if save_file == True:
        activity_df.to_csv(f"{filepath.rstrip(".txt").rstrip(".csv")}_{activity}.csv", index=False)
    return activity_df


def extract_pressure_data(filepath, df=None):
    if df is None:
        df = pd.read_csv(filepath)
    
    fsr_columns = [f"Fsr.{str(sensor).zfill(2)}" for sensor in range(1, 17)]
    fsr_data = df[fsr_columns].copy()
    fsr_data.insert(0, "ReconstructedTime", df["ReconstructedTime"], True)
    fsr_data["label"] = df["label"]

    output_path = f"{filepath.rstrip("_del_end_dupli_new_time_labeled.csv")}_pressure_labeled.csv"
    fsr_data.to_csv(output_path, index=False)

    return fsr_data, output_path


def sort_push_pull(filepath, df=None):
    if df is None:
        df = pd.read_csv(filepath)

    # Extract push and pull repetitions
    push_pull_df = df[df["label"].str.startswith("push") | df["label"].str.startswith("pull")].copy()

    # Extract label order using rep_id suffix (e.g., push_1 → 1)
    push_pull_df["rep_index"] = push_pull_df["rep_id"].str.extract(r"_(\d+)").astype(int)

    push_pull_df_sorted = push_pull_df.sort_values(by=["label", "rep_index"]).drop(columns=["rep_index"])

    other_df = df[~df.index.isin(push_pull_df_sorted.index)]

    df_sorted = pd.concat([other_df, push_pull_df_sorted], axis=0)

    output_path = f"{filepath.rstrip('.csv')}_ppsorted.csv"
    df_sorted.to_csv(output_path, index=False)

    return df_sorted, output_path


sensor_file = ""
#df, peaks = detect_spikes(sensor_file)
#df_init_label = init_label(df, peaks, sensor_file)
#plot_signal_peaks(df, peaks)

output_path_activity_file = ""
#plot_closeups_around_peaks(df, peaks, 2)
#extract_activity_windows(df, output_path_activity_file)


rename_map = {
    "Activity_2": "hand_up_back",
    "Activity_4": "hands_forward",
    "Activity_6": "hands_up",
    "Activity_8": "push",
    "Activity_9": "pull",
    "Activity_10": "push",
    "Activity_11": "pull",
    "Activity_12": "push",
    "Activity_13": "pull",
    "Activity_14": "push",
    "Activity_15": "pull",
    "Activity_16": "push",
    "Activity_17": "pull",
    "Activity_18": "push",
    "Activity_19": "pull",
    "Activity_21": "squat",
    "Activity_23": "lift",
    "Activity_25": "sit",
    "Activity_27": "stand",
    "Activity_29": "walk",
}

"--- Aply correct labels ---"
start_stop_label_file = ""
#df_labeled_correct, labeled_filepath = apply_corrected_labels(sensor_file, start_stop_label_file)
#plot_signal_peaks(df_labeled_correct, peaks)


"--- Remove idle ---"
labeled_filepath = ""
#df_no_idle = remove_idle(sensor_file)
#plot_signal_peaks(df_no_idle, peaks)

"--- Extract one activity ---"
filepath_one_activity = ""
#extract_one_activity(filepath_one_activity, "hand_up_back")

"--- Cut repetitions from one activity ---"
#filepath_activity = ""
#print(cut_repetitions(filepath_activity, 1500, 1090))

"--- Extract Pressure Data ---"
#filepath = ""
#extract_pressure_data(filepath)

"--- Sort Push and Pull ---"
#sort_push_pull("")