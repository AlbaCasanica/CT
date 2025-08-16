import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

from get_paths import get_feture_paths, get_one_file, get_one_foler_path

label_mapping_v3 = {
    "hand_up_back"  : "hands_up",
    "hands_forward" : "hands_up",
    "hands_up"      : "hands_up",
    "push"          : "push_pull_lift",
    "pull"          : "push_pull_lift",
    "squatting"     : "squat",
    "lifting"       : "push_pull_lift",
    "sitting"       : "sit",
    "standing"      : "stand_walk",
    "walking"       : "stand_walk"
}

label_mapping_v2 = {
    "hand_up_back"  : "hands_up",
    "hands_forward" : "hands_up",
    "hands_up"      : "hands_up",
    "push"          : "push_pull",
    "pull"          : "push_pull",
    "squatting"     : "squatting",
    "lifting"       : "lifting",
    "sitting"       : "sit_stand",
    "standing"      : "sit_stand",
    "walking"       : "walking"
}

def check_nr_windows_pr_label(norm_IMU, window_size, combine_labels=None):
    feature_files = get_feture_paths(window_length_sec=window_size, norm_IMU=norm_IMU, mean_fsr=True, hdr=False)
    
    all_data = []
    for test_id, path in feature_files.items():
        df = pd.read_csv(path)
    
        if 'label' not in df.columns:
            raise ValueError(f"No 'label' column found in file: {path}")

        if combine_labels is not None:
            df['label'] = df['label'].map(combine_labels).fillna(df['label'])
        
        label_counts = df['label'].value_counts()
        print(f"\nTest ID: {test_id}")
        print(label_counts)

        all_data.append(df)
    
    full_df = pd.concat(all_data, ignore_index=True)
    overall_counts = full_df['label'].value_counts()

    print("Numbers of windows per label:")
    print(overall_counts)

    return all_data, overall_counts

def plot_stacked_label_distribution(all_data, label_order):
    label_counts_per_test = {}
    
    for i, df in enumerate(all_data):
        test_id = f"test_{i+1}"
        label_counts = df['label'].value_counts()
        label_counts_per_test[test_id] = label_counts

    # Combine into a single DataFrame, missing values filled with 0
    df_counts = pd.DataFrame(label_counts_per_test).fillna(0).astype(int)
    df_counts = df_counts

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = [0] * df_counts.shape[0]
    x = df_counts.index
    colors = plt.cm.tab20.colors

    for i, test_id in enumerate(df_counts.columns):
        values = df_counts[test_id]
        ax.bar(x, values, bottom=bottom, label=test_id, color=colors[i % len(colors)])
        bottom = [sum(x) for x in zip(bottom, values)]

    ax.set_title("Stacked Bar Plot of Windows per Class per Test")
    ax.set_xlabel("Class Label")
    ax.set_ylabel("Number of Windows")
    plt.xticks(rotation=45)
    ax.legend(title="Test ID", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_total_windows_per_label(all_data, label_order, label_mapping=None):
    full_df = pd.concat(all_data, ignore_index=True)

    if label_mapping is not None:
        full_df['label'] = full_df['label'].map(label_mapping).fillna(full_df['label'])

    total_counts = full_df['label'].value_counts()
    total_counts = total_counts.reindex(label_order).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(4.6, 5))
    ax.bar(total_counts.index, total_counts.values, color='skyblue')

    ax.set_title("Total Number of Windows per Label (All Tests)")
    ax.set_xlabel("Class Label")
    ax.set_ylabel("Number of Windows")
    ax.set_ylim(0, 250)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def get_label_durations():
    all_durations = {}
    total_durations = defaultdict(float)

    for test_number in range(1, 13):
        path = get_one_file(test_number, "arm")
        df = pd.read_csv(path)
    
        df = df.sort_values('ReconstructedTime')

        durations = defaultdict(float)

        for (label, rep_id), group in df.groupby(['label', 'rep_id']):
            start_time = group['ReconstructedTime'].iloc[0]
            end_time = group['ReconstructedTime'].iloc[-1]
            duration = end_time - start_time

            durations[label] += duration
            total_durations[label] += duration

        all_durations[f"test_{test_number}"] = dict(durations)

    return all_durations, dict(total_durations)


def plot_total_duration_per_label(total_durations, label_order):
    values = [total_durations.get(label, 0) for label in label_order]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(label_order, values, color='skyblue', width=0.4)

    ax.set_title("Total Duration per Label (All Tests)")
    ax.set_xlabel("Class Label")
    ax.set_ylabel("Duration (seconds)")
    ax.set_ylim(0, 800)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def compute_repetition_durations():
    per_test_means = defaultdict(dict)  # test_id -> activity -> mean duration
    all_durations_by_activity = defaultdict(list)
    duration_sources = defaultdict(list) 

    for test_number in range(1, 13):
        test_id = f"test_{test_number}"
        folder_path = get_one_foler_path(test_number)
        test_path = folder_path + "/sart_stop_segments.csv"

        df = pd.read_csv(test_path)
        df["activity"] = df["label"].apply(lambda x: x.rsplit("_", 1)[0])
        df["duration"] = df["End Time (s)"] - df["Start Time (s)"]

        for activity, group in df.groupby("activity"):
            durations = group["duration"].tolist()
            mean_duration = np.mean(durations)

            per_test_means[test_id][activity] = mean_duration
            all_durations_by_activity[activity].extend(durations)
            duration_sources[activity].extend([(d, test_id) for d in durations])

    overall_stats = {}
    for activity, durations in all_durations_by_activity.items():
        mean = np.mean(durations)
        min_val = np.min(durations)
        max_val = np.max(durations)

        min_test = next(test for d, test in duration_sources[activity] if d == min_val)
        max_test = next(test for d, test in duration_sources[activity] if d == max_val)

        overall_stats[activity] = {
            "mean": np.mean(durations),
            "min": np.min(durations),
            "min_test": min_test,
            "max": np.max(durations),
            "max_test": max_test
        }
        

    print("\nMean repetition duration per activity (per test):")
    header = ["Test"] + sorted(overall_stats.keys())
    print("\t".join(header))
    for test_id in sorted(per_test_means.keys()):
        row = [test_id]
        for activity in sorted(overall_stats.keys()):
            dur = per_test_means[test_id].get(activity)
            row.append(f"{dur:.2f}" if dur else "–")
        print("\t".join(row))

    print("\nOverall stats (all tests combined):")
    print("Activity\tMean\tMin\t(from)\tMax\t(from)")
    for activity in sorted(overall_stats.keys()):
        stats = overall_stats[activity]
        print(f"{activity}\t{stats['mean']:.2f}\t{stats['min']:.2f}\t{stats['min_test']}\t{stats['max']:.2f}\t{stats['max_test']}")

    return per_test_means, overall_stats


def plot_repetition_duration_stats(overall_stats, label_order):
    means = [overall_stats[label]["mean"] for label in label_order]
    mins = [overall_stats[label]["min"] for label in label_order]
    maxs = [overall_stats[label]["max"] for label in label_order]
    errors = [np.array(means) - np.array(mins), np.array(maxs) - np.array(means)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(label_order, means, yerr=errors, capsize=5, color='skyblue', edgecolor='black')

    ax.set_title("Mean Repetition Duration per Activity")
    ax.set_ylabel("Duration (s)")
    ax.set_xlabel("Activity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



all_data,_ = check_nr_windows_pr_label(norm_IMU=False, window_size=8, combine_labels=label_mapping_v3)

plot_total_windows_per_label(all_data, label_order=list(label_mapping_v3.values()), label_mapping=label_mapping_v3)
