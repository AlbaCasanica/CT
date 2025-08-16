import matplotlib.pyplot as plt
import pandas as pd
import re

def make_df(muse_file_1, muse_file_2, mitch_file_1 = None, mitch_file_2 = None, muse_file_3 = None, muse_file_4 = None):
    muse_file_list = [muse_file_1, muse_file_2, mitch_file_1, mitch_file_2, muse_file_3, muse_file_4]

    df_list_muse = []
    for file in muse_file_list:
        if file:
            if file.endswith(".txt"):
                df_list_muse.append(pd.read_csv(file, delimiter="\t", skiprows=8, decimal=","))
            elif file.endswith(".csv"):
                df_list_muse.append(pd.read_csv(file))
        else:
            df_list_muse.append(None)

    return df_list_muse


def plot_axl_imu_labels(df_list):
    df_muse_1 = None
    if df_list[0] is not None:
        df_muse_1 = df_list[0]
    df_muse_2 = None
    if df_list[1] is not None:
        df_muse_2 = df_list[1]
    df_mitch_1 = None
    if df_list[2] is not None:
        df_mitch_1 = df_list[2]
    df_mitch_2 = None
    if df_list[3] is not None:
        df_mitch_2 = df_list[3]

    unique_labels = df_muse_1["label"].unique()
    colors = plt.get_cmap("hsv", len(unique_labels))
    
    # Plot Accelerometer X
    plt.figure(figsize=(12, 5))
    
    if df_mitch_1 is not None:
        plt.plot(df_mitch_1['ReconstructedTime'], df_mitch_1['Axl.X'], label='muse_3 - Axl X', alpha=0.7)
    if df_mitch_2 is not None:
        plt.plot(df_mitch_2['ReconstructedTime'], df_mitch_2['Axl.X'], label='muse_4 - Axl X', alpha=0.7)
    if df_muse_1 is not None:
        plt.plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.X'], label='muse_1 - Axl X', alpha=0.7)
    if df_muse_2 is not None:
        plt.plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.X'], label='muse_2 - Axl X', alpha=0.7)

    for i, label in enumerate(unique_labels):
        if label == "idle":
            continue
        activity_df = df_muse_1[df_muse_1["label"] == label]
        if not activity_df.empty:
            start_time = activity_df["ReconstructedTime"].iloc[0]
            end_time = activity_df["ReconstructedTime"].iloc[-1]
            plt.axvspan(start_time, end_time, alpha=0.2, color=colors(i), label=label, zorder=0)

    plt.legend()
    plt.title("Accelerometer X Over Time with labels")
    plt.xlabel("ReconstructedTime")
    plt.ylabel("Acceleration (mg)")
    plt.grid(True)
    plt.show()

    # Plot Accelerometer Y
    plt.figure(figsize=(12, 5))
    if df_muse_1 is not None:
        plt.plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.Y'], label='muse_1 - Axl Y', alpha=0.7)
    if df_muse_2 is not None:
        plt.plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.Y'], label='muse_2 - Axl Y', alpha=0.7)
    if df_mitch_1 is not None:
        plt.plot(df_mitch_1['ReconstructedTime'], df_mitch_1['Axl.Y'], label='muse_3 - Axl Y', alpha=0.7)
    if df_mitch_2 is not None:
        plt.plot(df_mitch_2['ReconstructedTime'], df_mitch_2['Axl.Y'], label='muse_4 - Axl Y', alpha=0.7)
    
    for i, label in enumerate(unique_labels):
        if label == "idle":
            continue
        activity_df = df_muse_1[df_muse_1["label"] == label]
        if not activity_df.empty:
            start_time = activity_df["ReconstructedTime"].iloc[0]
            end_time = activity_df["ReconstructedTime"].iloc[-1]
            plt.axvspan(start_time, end_time, alpha=0.2, color=colors(i), label=label, zorder=0)
    
    plt.legend()
    plt.title("Accelerometer Y Over Time")
    plt.xlabel("ReconstructedTime")
    plt.ylabel("Acceleration (mg)")
    plt.grid(True)
    plt.show()

    # Plot Accelerometer Z
    plt.figure(figsize=(12, 5))
    if df_muse_1 is not None:
        plt.plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.Z'], label='muse_1 - Axl Z', alpha=0.7)
    if df_muse_2 is not None:
        plt.plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.Z'], label='muse_2 - Axl Z', alpha=0.7)
    if df_mitch_1 is not None:
        plt.plot(df_mitch_1['ReconstructedTime'], df_mitch_1['Axl.Z'], label='muse_3 - Axl Z', alpha=0.7)
    if df_mitch_2 is not None:
        plt.plot(df_mitch_2['ReconstructedTime'], df_mitch_2['Axl.Z'], label='muse_4 - Axl Z', alpha=0.7)
    
    for i, label in enumerate(unique_labels):
        if label == "idle":
            continue
        activity_df = df_muse_1[df_muse_1["label"] == label]
        if not activity_df.empty:
            start_time = activity_df["ReconstructedTime"].iloc[0]
            end_time = activity_df["ReconstructedTime"].iloc[-1]
            plt.axvspan(start_time, end_time, alpha=0.2, color=colors(i), label=label, zorder=0)
    
    plt.legend()
    plt.title("Accelerometer Z Over Time")
    plt.xlabel("ReconstructedTime")
    plt.ylabel("Acceleration (mg)")
    plt.grid(True)
    plt.show()

    # Create subplots
    non_none_count = sum(x is not None for x in df_list)
    fig, axs = plt.subplots(non_none_count, 1, figsize=(14, 10), sharex=True)

    # Axl muse_1
    if df_muse_1 is not None:
        axs[0].plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.X'], label='Axl X', alpha=0.7)
        axs[0].plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.Y'], label='Axl Y', alpha=0.7)
        axs[0].plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.Z'], label='Axl Z', alpha=0.7)
        axs[0].set_ylabel("muse_1")
        axs[0].legend()
        axs[0].grid(True)
    
        for i, label in enumerate(unique_labels):
            if label == "idle":
                continue
            activity_df = df_muse_1[df_muse_1["label"] == label]
            if not activity_df.empty:
                start_time = activity_df["ReconstructedTime"].iloc[0]
                end_time = activity_df["ReconstructedTime"].iloc[-1]
                axs[0].axvspan(start_time, end_time, alpha=0.2, color=colors(i), label=label, zorder=0)

    # Axl muse_2
    if df_muse_2 is not None:
        axs[1].plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.X'], label='Axl X', alpha=0.7)
        axs[1].plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.Y'], label='Axl Y', alpha=0.7)
        axs[1].plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.Z'], label='Axl Z', alpha=0.7)
        axs[1].set_ylabel("muse_2")
        axs[1].legend()
        axs[1].grid(True)

        for i, label in enumerate(unique_labels):
            if label == "idle":
                continue
            activity_df = df_muse_2[df_muse_2["label"] == label]
            if not activity_df.empty:
                start_time = activity_df["ReconstructedTime"].iloc[0]
                end_time = activity_df["ReconstructedTime"].iloc[-1]
                axs[1].axvspan(start_time, end_time, alpha=0.2, color=colors(i), label=label, zorder=0)

    
    # Axl mitch 1
    if df_mitch_1 is not None:
        axs[2].plot(df_mitch_1['ReconstructedTime'], df_mitch_1['Axl.X'], label='Axl X', alpha=0.7)
        axs[2].plot(df_mitch_1['ReconstructedTime'], df_mitch_1['Axl.Y'], label='Axl Y', alpha=0.7)
        axs[2].plot(df_mitch_1['ReconstructedTime'], df_mitch_1['Axl.Z'], label='Axl Z', alpha=0.7)
        axs[2].set_ylabel("muse_3")
        axs[2].set_xlabel("ReconstructedTime")
        axs[2].legend()
        axs[2].grid(True)
    
    # Axl mitch 2
    if df_mitch_2 is not None:
        axs[3].plot(df_mitch_2['ReconstructedTime'], df_mitch_2['Axl.X'], label='Axl X', alpha=0.7)
        axs[3].plot(df_mitch_2['ReconstructedTime'], df_mitch_2['Axl.Y'], label='Axl Y', alpha=0.7)
        axs[3].plot(df_mitch_2['ReconstructedTime'], df_mitch_2['Axl.Z'], label='Axl Z', alpha=0.7)
        axs[3].set_ylabel("muse_4")
        axs[3].set_xlabel("ReconstructedTime")
        axs[3].legend()
        axs[3].grid(True)

    # Title and layout
    fig.suptitle("Accelerometer Data (X, Y, Z) from All Devices", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_axl_repetitions_muse(filepath):
    df = pd.read_csv(filepath)

    # Extract numeric rep numbers from rep_id like "hand_back_up_3"
    df["rep_num"] = df["rep_id"].str.extract(r"_(\d+)$").astype(int)

    rep_nrs = df["rep_num"].max()
    colors = plt.get_cmap("hsv", rep_nrs + 1)

    # Plot Accelerometer X
    plt.figure(figsize=(12, 5))    
    plt.plot(df['ReconstructedTime'], df['Axl.X'], label='Axl.X', alpha=0.7)

    for num in range(1, rep_nrs + 1):
        rep_df = df[df["rep_num"] == num]
        if not rep_df.empty:
            start_time = rep_df["ReconstructedTime"].iloc[0]
            end_time = rep_df["ReconstructedTime"].iloc[-1]
            rep_label = rep_df["rep_id"].iloc[0]
            plt.axvspan(start_time, end_time, alpha=0.2, color=colors(num), label=rep_label, zorder=0)

    plt.legend()
    plt.title("Accelerometer X Over Time with Repetition Labels")
    plt.xlabel("ReconstructedTime")
    plt.ylabel("Acceleration (mg)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_axl_one_activity_filter(file_unfiltered, file_filtered, activity):
    df_unfiltered = pd.read_csv(file_unfiltered).query("label == @activity")
    df_filtered = pd.read_csv(file_filtered).query("label == @activity")

    # Plot Accelerometer X
    plt.figure(figsize=(12, 5))
    plt.rcParams.update({'font.size': 16})    
    plt.plot(df_unfiltered['ReconstructedTime'], df_unfiltered['Axl.X'], label='Unfiltered', alpha=0.7)
    plt.plot(df_filtered['ReconstructedTime'], df_filtered['Axl.X'], label='Filtered', alpha=0.7)
    plt.legend()
    plt.title(f"Accelerometer X Over Time with and without median filter over {activity}")
    plt.xlabel("ReconstructedTime (s)")
    plt.ylabel("Acceleration (mg)")
    plt.grid(True)
    plt.show()

    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    axs[0].plot(df_unfiltered['ReconstructedTime'], df_unfiltered['Axl.X'], label='Axl X', alpha=0.7)
    axs[0].plot(df_unfiltered['ReconstructedTime'], df_unfiltered['Axl.Y'], label='Axl Y', alpha=0.7)
    axs[0].plot(df_unfiltered['ReconstructedTime'], df_unfiltered['Axl.Z'], label='Axl Z', alpha=0.7)
    axs[0].set_ylabel("Unfiltered")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(df_filtered['ReconstructedTime'], df_filtered['Axl.X'], label='Axl X', alpha=0.7)
    axs[1].plot(df_filtered['ReconstructedTime'], df_filtered['Axl.Y'], label='Axl Y', alpha=0.7)
    axs[1].plot(df_filtered['ReconstructedTime'], df_filtered['Axl.Z'], label='Axl Z', alpha=0.7)
    axs[1].legend()
    axs[1].grid(True)

    fig.suptitle(f"Accelerometer Data (X, Y, Z) for {activity}, filtered and unfiltered", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_axl_one_activity_rotation(file_unfiltered, file_filtered, activity):
    df_unfiltered = pd.read_csv(file_unfiltered).query("label == @activity")
    df_filtered = pd.read_csv(file_filtered).query("label == @activity")

    # Plot Accelerometer X
    plt.figure(figsize=(12, 5))    
    plt.plot(df_unfiltered['ReconstructedTime'], df_unfiltered['Axl.Z'], label='Original', alpha=0.7)
    plt.plot(df_filtered['ReconstructedTime'], df_filtered['Axl.Z'], label='Rotated', alpha=0.7)
    plt.legend()
    plt.title(f"Accelerometer Z Over Time with and without rotation {activity}")
    plt.xlabel("ReconstructedTime (s)")
    plt.ylabel("Acceleration (mg)")
    plt.grid(True)
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    axs[0].plot(df_unfiltered['ReconstructedTime'], df_unfiltered['Axl.X'], label='Axl X', alpha=0.7)
    axs[0].plot(df_unfiltered['ReconstructedTime'], df_unfiltered['Axl.Y'], label='Axl Y', alpha=0.7)
    axs[0].plot(df_unfiltered['ReconstructedTime'], df_unfiltered['Axl.Z'], label='Axl Z', alpha=0.7)
    axs[0].set_ylabel("Original")
    axs[0].set_xlabel("ReconstructedTime")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(df_filtered['ReconstructedTime'], df_filtered['Axl.X'], label='Axl X', alpha=0.7)
    axs[1].plot(df_filtered['ReconstructedTime'], df_filtered['Axl.Y'], label='Axl Y', alpha=0.7)
    axs[1].plot(df_filtered['ReconstructedTime'], df_filtered['Axl.Z'], label='Axl Z', alpha=0.7)
    axs[1].set_ylabel("Rotated")
    axs[1].set_xlabel("ReconstructedTime")
    axs[1].legend()
    axs[1].grid(True)

    fig.suptitle(f"Accelerometer Data (X, Y, Z) for {activity}, original and rotated", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_axl_imu_preassure(df_list, with_mitch_imu=0):
    df_muse_1 = None
    if df_list[0] is not None:
        df_muse_1 = df_list[0]
    df_muse_2 = None
    if df_list[1] is not None:
        df_muse_2 = df_list[1]
    df_mitch_1 = None
    if df_list[2] is not None:
        df_mitch_1 = df_list[2]
    df_mitch_2 = None
    if df_list[3] is not None:
        df_mitch_2 = df_list[3]

    unique_labels = df_muse_1["label"].unique()
    colors = plt.get_cmap("hsv", len(unique_labels))

    # Plot Accelerometer X
    plt.figure(figsize=(12, 5))    
    plt.plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.X']/100, label='muse_arm', alpha=0.7)
    plt.plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.X']/100, label='muse_back', alpha=0.7)
    if with_mitch_imu == 2:
        plt.plot(df_mitch_1['ReconstructedTime'], df_mitch_1['Axl.X']/100, label='mitch_left', alpha=0.7)
        plt.plot(df_mitch_2['ReconstructedTime'], df_mitch_2['Axl.X']/100, label='mitch_right', alpha=0.7)
    elif with_mitch_imu == 1:
        plt.plot(df_mitch_1['ReconstructedTime'], df_mitch_1['Axl.X']/100, label='mitch_right', alpha=0.7)

    for i, label in enumerate(unique_labels):
        if label == "idle":
            continue
        activity_df = df_muse_1[df_muse_1["label"] == label]
        if not activity_df.empty:
            start_time = activity_df["ReconstructedTime"].iloc[0]
            end_time = activity_df["ReconstructedTime"].iloc[-1]
            plt.axvspan(start_time, end_time, alpha=0.2, color=colors(i), label=label, zorder=0)
    
    plt.legend()
    plt.title(f"Accelerometer X Over Time")
    plt.xlabel("ReconstructedTime")
    plt.ylabel("Acceleration (mg)")
    plt.grid(True)
    plt.show()


    if df_mitch_2 is not None and df_mitch_1 is not None:
        fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    elif df_mitch_1 is not None or df_mitch_2 is not None:
        fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    else:
        fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axs[0].plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.X']/100, label='Axl X', alpha=0.7)
    axs[0].plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.Y']/100, label='Axl Y', alpha=0.7)
    axs[0].plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.Z']/100, label='Axl Z', alpha=0.7)
    axs[0].set_ylabel("Muse arm")
    axs[0].set_xlabel("ReconstructedTime")
    axs[0].legend()
    axs[0].grid(True)
    for i, label in enumerate(unique_labels):
            if label == "idle":
                continue
            activity_df = df_muse_1[df_muse_1["label"] == label]
            if not activity_df.empty:
                start_time = activity_df["ReconstructedTime"].iloc[0]
                end_time = activity_df["ReconstructedTime"].iloc[-1]
                axs[0].axvspan(start_time, end_time, alpha=0.2, color=colors(i), label=label, zorder=0)

    axs[1].plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.X']/100, label='Axl X', alpha=0.7)
    axs[1].plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.Y']/100, label='Axl Y', alpha=0.7)
    axs[1].plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.Z']/100, label='Axl Z', alpha=0.7)
    axs[1].set_ylabel("Muse back")
    axs[1].set_xlabel("ReconstructedTime")
    axs[1].legend()
    axs[1].grid(True)
    for i, label in enumerate(unique_labels):
            if label == "idle":
                continue
            activity_df = df_muse_2[df_muse_2["label"] == label]
            if not activity_df.empty:
                start_time = activity_df["ReconstructedTime"].iloc[0]
                end_time = activity_df["ReconstructedTime"].iloc[-1]
                axs[1].axvspan(start_time, end_time, alpha=0.2, color=colors(i), label=label, zorder=0)

    if df_mitch_1 is not None:
        fsr_columns_1 = [f"Fsr.{str(sensor).zfill(2)}" for sensor in range(1, 17)]
        fsr_data_1 = df_mitch_1[fsr_columns_1]
        average_1 = fsr_data_1.mean(axis=1)
        axs[2].plot(df_mitch_1['ReconstructedTime'], average_1, label=f"Avg Pressure")
        axs[2].set_ylabel("Mich left")
        axs[2].set_xlabel("ReconstructedTime")
        axs[2].legend()
        axs[2].grid(True)
        for i, label in enumerate(unique_labels):
                if label == "idle":
                    continue
                activity_df = df_mitch_1[df_mitch_1["label"] == label]
                if not activity_df.empty:
                    start_time = activity_df["ReconstructedTime"].iloc[0]
                    end_time = activity_df["ReconstructedTime"].iloc[-1]
                    axs[2].axvspan(start_time, end_time, alpha=0.2, color=colors(i), label=label, zorder=0)
    if df_mitch_2 is not None:
        fsr_columns_2 = [f"Fsr.{str(sensor).zfill(2)}" for sensor in range(1, 17)]
        fsr_data_2 = df_mitch_2[fsr_columns_2]
        average_2 = fsr_data_2.mean(axis=1)
        axs[3].plot(df_mitch_2['ReconstructedTime'], average_2, label=f"Avg Pressure")
        axs[3].set_ylabel("Mich right")
        axs[3].set_xlabel("ReconstructedTime")
        axs[3].legend()
        axs[3].grid(True)
        for i, label in enumerate(unique_labels):
                if label == "idle":
                    continue
                activity_df = df_mitch_2[df_mitch_2["label"] == label]
                if not activity_df.empty:
                    start_time = activity_df["ReconstructedTime"].iloc[0]
                    end_time = activity_df["ReconstructedTime"].iloc[-1]
                    axs[3].axvspan(start_time, end_time, alpha=0.2, color=colors(i), label=label, zorder=0)


    fig.suptitle(f"Accelerometer Data (X, Y, Z) and preassure data", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_axl_imu_pressure_segment_labels(df_list, with_mitch_imu=0):
    df_muse_1 = None
    if df_list[0] is not None:
        df_muse_1 = df_list[0]
    df_muse_2 = None
    if df_list[1] is not None:
        df_muse_2 = df_list[1]
    df_mitch_1 = None
    if df_list[2] is not None:
        df_mitch_1 = df_list[2]
    df_mitch_2 = None
    if df_list[3] is not None:
        df_mitch_2 = df_list[3]
    
    # Plot Accelerometer X
    plt.figure(figsize=(12, 5))
    plt.rcParams.update({'font.size': 16})    
    plt.plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.X'], label='Muse arm', alpha=0.7)
    plt.plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.X'], label='Muse back', alpha=0.7)
    if with_mitch_imu == 2:
        plt.plot(df_mitch_1['ReconstructedTime'], df_mitch_1['Axl.X'], label='mitch_left', alpha=0.7)
        plt.plot(df_mitch_2['ReconstructedTime'], df_mitch_2['Axl.X'], label='mitch_right', alpha=0.7)
    elif with_mitch_imu == 1:
        plt.plot(df_mitch_1['ReconstructedTime'], df_mitch_1['Axl.X'], label='mitch_right', alpha=0.7)

    df = df_muse_1
    if df is not None and "rep_id" in df.columns:
        rep_ids = df["rep_id"].unique()
        activity_names = sorted(set(re.sub(r'_\d+', '', rep_id) for rep_id in rep_ids))

        for rep_id in rep_ids:
            rep_df = df[df["rep_id"] == rep_id]
            if not rep_df.empty:
                activity = re.sub(r'_\d+', '', rep_id)
                activity_parts = rep_id.split("_")
                last_part = activity_parts[-1]

                if last_part.isdigit():
                    rep_num = int(last_part)
                else:
                    rep_num = 0

                if activity_names.index(activity) % 2 == 0 and rep_num % 2 == 0: color = "deepskyblue"
                elif activity_names.index(activity) % 2 == 0 and rep_num % 2 == 1: color = "lightskyblue"
                elif activity_names.index(activity) % 2 == 1 and rep_num % 2 == 0: color = "limegreen"
                elif activity_names.index(activity) % 2 == 1 and rep_num % 2 == 1: color = "lightgreen"

                start_time = rep_df["ReconstructedTime"].iloc[0]
                end_time = rep_df["ReconstructedTime"].iloc[-1]
                plt.axvspan(start_time, end_time, color=color, alpha=0.3, label=rep_id if (rep_num == 1 or rep_num == 0) else None)

    plt.legend()
    plt.title("Accelerometer X Over Time with labels")
    plt.xlabel("ReconstructedTime")
    plt.ylabel("Acceleration (g)")
    plt.grid(True)
    plt.show()

    # Plot Accelerometer Y
    plt.figure(figsize=(12, 5))    
    plt.plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.Y']/100, label='muse_arm', alpha=0.7)
    plt.plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.Y']/100, label='muse_back', alpha=0.7)
    if with_mitch_imu == 2:
        plt.plot(df_mitch_1['ReconstructedTime'], df_mitch_1['Axl.Y']/100, label='mitch_left', alpha=0.7)
        plt.plot(df_mitch_2['ReconstructedTime'], df_mitch_2['Axl.Y']/100, label='mitch_right', alpha=0.7)
    elif with_mitch_imu == 1:
        plt.plot(df_mitch_1['ReconstructedTime'], df_mitch_1['Axl.Y']/100, label='mitch_right', alpha=0.7)
  
    df = df_muse_1
    if df is not None and "rep_id" in df.columns:
        rep_ids = df["rep_id"].unique()
        activity_names = sorted(set(re.sub(r'_\d+', '', rep_id) for rep_id in rep_ids))

        for rep_id in rep_ids:
            rep_df = df[df["rep_id"] == rep_id]
            if not rep_df.empty:
                activity = re.sub(r'_\d+', '', rep_id)
                activity_parts = rep_id.split("_")
                last_part = activity_parts[-1]

                if last_part.isdigit():
                    rep_num = int(last_part)
                else:
                    rep_num = 0

                if activity_names.index(activity) % 2 == 0 and rep_num % 2 == 0: color = "deepskyblue"
                elif activity_names.index(activity) % 2 == 0 and rep_num % 2 == 1: color = "lightskyblue"
                elif activity_names.index(activity) % 2 == 1 and rep_num % 2 == 0: color = "limegreen"
                elif activity_names.index(activity) % 2 == 1 and rep_num % 2 == 1: color = "lightgreen"

                start_time = rep_df["ReconstructedTime"].iloc[0]
                end_time = rep_df["ReconstructedTime"].iloc[-1]
                plt.axvspan(start_time, end_time, color=color, alpha=0.3, label=rep_id if rep_num == 1 else None)
    
    plt.legend()
    plt.title("Accelerometer Y Over Time")
    plt.xlabel("ReconstructedTime")
    plt.ylabel("Acceleration (g)")
    plt.grid(True)
    plt.show()

    # Plot Accelerometer Z
    plt.figure(figsize=(12, 5))    
    plt.plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.Z']/100, label='muse_arm', alpha=0.7)
    plt.plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.Z']/100, label='muse_back', alpha=0.7)
    if with_mitch_imu == 2:
        plt.plot(df_mitch_1['ReconstructedTime'], df_mitch_1['Axl.Z']/100, label='mitch_left', alpha=0.7)
        plt.plot(df_mitch_2['ReconstructedTime'], df_mitch_2['Axl.Z']/100, label='mitch_right', alpha=0.7)
    elif with_mitch_imu == 1:
        plt.plot(df_mitch_1['ReconstructedTime'], df_mitch_1['Axl.Z']/100, label='mitch_right', alpha=0.7)
   
    df = df_muse_1
    if df is not None and "rep_id" in df.columns:
        rep_ids = df["rep_id"].unique()
        activity_names = sorted(set(re.sub(r'_\d+', '', rep_id) for rep_id in rep_ids))

        for rep_id in rep_ids:
            rep_df = df[df["rep_id"] == rep_id]
            if not rep_df.empty:
                activity = re.sub(r'_\d+', '', rep_id)
                activity_parts = rep_id.split("_")
                last_part = activity_parts[-1]

                if last_part.isdigit():
                    rep_num = int(last_part)
                else:
                    rep_num = 0

                if activity_names.index(activity) % 2 == 0 and rep_num % 2 == 0: color = "deepskyblue"
                elif activity_names.index(activity) % 2 == 0 and rep_num % 2 == 1: color = "lightskyblue"
                elif activity_names.index(activity) % 2 == 1 and rep_num % 2 == 0: color = "limegreen"
                elif activity_names.index(activity) % 2 == 1 and rep_num % 2 == 1: color = "lightgreen"

                start_time = rep_df["ReconstructedTime"].iloc[0]
                end_time = rep_df["ReconstructedTime"].iloc[-1]
                plt.axvspan(start_time, end_time, color=color, alpha=0.3, label=rep_id if rep_num == 1 else None)
    
    plt.legend()
    plt.title("Accelerometer Z Over Time")
    plt.xlabel("ReconstructedTime")
    plt.ylabel("Acceleration (g)")
    plt.grid(True)
    plt.show()

    # Create subplots
    non_none_count = sum(x is not None for x in df_list)
    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(non_none_count, 1, figsize=(14, 10), sharex=True)

    # Axl muse_1
    if df_muse_1 is not None:
        axs[0].plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.X'], label='Axl X', alpha=0.7)
        axs[0].plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.Y'], label='Axl Y', alpha=0.7)
        axs[0].plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.Z'], label='Axl Z', alpha=0.7)
        axs[0].set_ylabel("Muse arm")
        axs[0].legend()
        axs[0].grid(True)
    
        df = df_muse_1
        if df is not None and "rep_id" in df.columns:
            rep_ids = df["rep_id"].unique()
            activity_names = sorted(set(re.sub(r'_\d+', '', rep_id) for rep_id in rep_ids))

            for rep_id in rep_ids:
                rep_df = df[df["rep_id"] == rep_id]
                if not rep_df.empty:
                    activity = re.sub(r'_\d+', '', rep_id)
                    activity_parts = rep_id.split("_")
                    last_part = activity_parts[-1]

                    if last_part.isdigit():
                        rep_num = int(last_part)
                    else:
                        rep_num = 0

                    if activity_names.index(activity) % 2 == 0 and rep_num % 2 == 0: color = "deepskyblue"
                    elif activity_names.index(activity) % 2 == 0 and rep_num % 2 == 1: color = "lightskyblue"
                    elif activity_names.index(activity) % 2 == 1 and rep_num % 2 == 0: color = "limegreen"
                    elif activity_names.index(activity) % 2 == 1 and rep_num % 2 == 1: color = "lightgreen"

                    start_time = rep_df["ReconstructedTime"].iloc[0]
                    end_time = rep_df["ReconstructedTime"].iloc[-1]
                    axs[0].axvspan(start_time, end_time, color=color, alpha=0.3, label=rep_id if rep_num == 1 else None)


    # Axl muse_2
    if df_muse_2 is not None:
        axs[1].plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.X'], label='Axl X', alpha=0.7)
        axs[1].plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.Y'], label='Axl Y', alpha=0.7)
        axs[1].plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.Z'], label='Axl Z', alpha=0.7)
        axs[1].set_ylabel("Muse back")
        axs[1].legend()
        axs[1].grid(True)

        df = df_muse_2
        if df is not None and "rep_id" in df.columns:
            rep_ids = df["rep_id"].unique()
            activity_names = sorted(set(re.sub(r'_\d+', '', rep_id) for rep_id in rep_ids)) 

            for rep_id in rep_ids:
                rep_df = df[df["rep_id"] == rep_id]
                if not rep_df.empty:
                    activity = re.sub(r'_\d+', '', rep_id)
                    activity_parts = rep_id.split("_")
                    last_part = activity_parts[-1]

                    if last_part.isdigit():
                        rep_num = int(last_part)
                    else:
                        rep_num = 0
                    if activity_names.index(activity) % 2 == 0 and rep_num % 2 == 0: color = "deepskyblue"
                    elif activity_names.index(activity) % 2 == 0 and rep_num % 2 == 1: color = "lightskyblue"
                    elif activity_names.index(activity) % 2 == 1 and rep_num % 2 == 0: color = "limegreen"
                    elif activity_names.index(activity) % 2 == 1 and rep_num % 2 == 1: color = "lightgreen"

                    start_time = rep_df["ReconstructedTime"].iloc[0]
                    end_time = rep_df["ReconstructedTime"].iloc[-1]
                    axs[1].axvspan(start_time, end_time, color=color, alpha=0.3, label=rep_id if rep_num == 1 else None)

    
    # Axl mitch 1
    if df_mitch_1 is not None:
        fsr_columns_1 = [f"Fsr.{str(sensor).zfill(2)}" for sensor in range(1, 17)]
        fsr_data_1 = df_mitch_1[fsr_columns_1]
        average_1 = fsr_data_1.mean(axis=1)
        axs[2].plot(df_mitch_1['ReconstructedTime'], average_1, label=f"Avg Pressure")
        axs[2].set_ylabel("Yeti left")
        axs[2].legend()
        axs[2].grid(True)

        df = df_mitch_1  # Or whichever df you're working with
        if df is not None and "rep_id" in df.columns:
            rep_ids = df["rep_id"].unique()
            activity_names = sorted(set(re.sub(r'_\d+', '', rep_id) for rep_id in rep_ids))

            for rep_id in rep_ids:
                rep_df = df[df["rep_id"] == rep_id]
                if not rep_df.empty:
                    activity = re.sub(r'_\d+', '', rep_id)
                    activity_parts = rep_id.split("_")
                    last_part = activity_parts[-1]

                    if last_part.isdigit():
                        rep_num = int(last_part)
                    else:
                        rep_num = 0

                    if activity_names.index(activity) % 2 == 0 and rep_num % 2 == 0: color = "deepskyblue"
                    elif activity_names.index(activity) % 2 == 0 and rep_num % 2 == 1: color = "lightskyblue"
                    elif activity_names.index(activity) % 2 == 1 and rep_num % 2 == 0: color = "limegreen"
                    elif activity_names.index(activity) % 2 == 1 and rep_num % 2 == 1: color = "lightgreen"

                    start_time = rep_df["ReconstructedTime"].iloc[0]
                    end_time = rep_df["ReconstructedTime"].iloc[-1]
                    axs[2].axvspan(start_time, end_time, color=color, alpha=0.3, label=rep_id if rep_num == 1 else None)

    
    # Axl mitch 2
    if df_mitch_2 is not None:
        fsr_columns_1 = [f"Fsr.{str(sensor).zfill(2)}" for sensor in range(1, 17)]
        fsr_data_1 = df_mitch_2[fsr_columns_1]
        average_1 = fsr_data_1.mean(axis=1)
        axs[3].plot(df_mitch_2['ReconstructedTime'], average_1, label=f"Avg Pressure")
        axs[3].set_ylabel("Yeti right")
        axs[3].set_xlabel("ReconstructedTime (s)")
        axs[3].legend()
        axs[3].grid(True)

        df = df_mitch_2
        if df is not None and "rep_id" in df.columns:
            rep_ids = df["rep_id"].unique()
            activity_names = sorted(set(re.sub(r'_\d+', '', rep_id) for rep_id in rep_ids))

            for rep_id in rep_ids:
                rep_df = df[df["rep_id"] == rep_id]
                if not rep_df.empty:
                    activity = re.sub(r'_\d+', '', rep_id)
                    activity_parts = rep_id.split("_")
                    last_part = activity_parts[-1]

                    if last_part.isdigit():
                        rep_num = int(last_part)
                    else:
                        rep_num = 0

                    if activity_names.index(activity) % 2 == 0 and rep_num % 2 == 0: color = "deepskyblue"
                    elif activity_names.index(activity) % 2 == 0 and rep_num % 2 == 1: color = "lightskyblue"
                    elif activity_names.index(activity) % 2 == 1 and rep_num % 2 == 0: color = "limegreen"
                    elif activity_names.index(activity) % 2 == 1 and rep_num % 2 == 1: color = "lightgreen"

                    start_time = rep_df["ReconstructedTime"].iloc[0]
                    end_time = rep_df["ReconstructedTime"].iloc[-1]
                    axs[3].axvspan(start_time, end_time, color=color, alpha=0.3, label=rep_id if rep_num == 1 else None)

    # Title and layout
    fig.suptitle("Segmented Repetitions on All Devices", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_axl_one_activity_segmented(muse_arm, muse_back, activity):
    df_arm = pd.read_csv(muse_arm).query("label == @activity")
    if muse_back is not None:
        df_back = pd.read_csv(muse_back).query("label == @activity")

    df_arm["rep_num"] = df_arm["rep_id"].str.extract(r"_(\d+)$").astype(int)

    rep_nrs = df_arm["rep_num"].max()
    colors = plt.get_cmap("hsv", rep_nrs + 1)

    # Plot Accelerometer X
    plt.figure(figsize=(12, 5))    
    plt.plot(df_arm['ReconstructedTime'], df_arm['Axl.X'], label='arm', alpha=0.7)
    if muse_back is not None:
        plt.plot(df_back['ReconstructedTime'], df_back['Axl.X'], label='back', alpha=0.7)
    
    for num in range(1, rep_nrs + 1):
        rep_df = df_arm[df_arm["rep_num"] == num]
        if not rep_df.empty:
            start_time = rep_df["ReconstructedTime"].iloc[0]
            end_time = rep_df["ReconstructedTime"].iloc[-1]
            rep_label = rep_df["rep_id"].iloc[0]
            plt.axvspan(start_time, end_time, alpha=0.2, color=colors(num), label=rep_label, zorder=0)

    plt.legend()
    plt.title(f"Accelerometer X Over Time with and without median filter over {activity}")
    plt.xlabel("ReconstructedTime")
    plt.ylabel("Acceleration (g)")
    plt.grid(True)
    plt.show()

    if muse_back is not None:
        fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        axs[0].plot(df_arm['ReconstructedTime'], df_arm['Axl.X'], label='Axl X', alpha=0.7)
        axs[0].plot(df_arm['ReconstructedTime'], df_arm['Axl.Y'], label='Axl Y', alpha=0.7)
        axs[0].plot(df_arm['ReconstructedTime'], df_arm['Axl.Z'], label='Axl Z', alpha=0.7)
        axs[0].set_ylabel("arm")
        axs[0].set_xlabel("ReconstructedTime")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(df_back['ReconstructedTime'], df_back['Axl.X'], label='Axl X', alpha=0.7)
        axs[1].plot(df_back['ReconstructedTime'], df_back['Axl.Y'], label='Axl Y', alpha=0.7)
        axs[1].plot(df_back['ReconstructedTime'], df_back['Axl.Z'], label='Axl Z', alpha=0.7)
        axs[1].set_ylabel("back")
        axs[1].set_xlabel("ReconstructedTime")
        axs[1].legend()
        axs[1].grid(True)

        fig.suptitle(f"Accelerometer Data (X, Y, Z) for {activity}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    
    if muse_back is None:
        plt.figure(figsize=(12, 5))
        plt.plot(df_arm['ReconstructedTime'], df_arm['Axl.X'], label='Axl X', alpha=0.7)
        plt.plot(df_arm['ReconstructedTime'], df_arm['Axl.Y'], label='Axl Y', alpha=0.7)
        plt.plot(df_arm['ReconstructedTime'], df_arm['Axl.Z'], label='Axl Z', alpha=0.7)
        for num in range(1, rep_nrs + 1):
            rep_df = df_arm[df_arm["rep_num"] == num]
            if not rep_df.empty:
                start_time = rep_df["ReconstructedTime"].iloc[0]
                end_time = rep_df["ReconstructedTime"].iloc[-1]
                rep_label = rep_df["rep_id"].iloc[0]
                plt.axvspan(start_time, end_time, alpha=0.2, color=colors(num), label=rep_label, zorder=0)
        plt.ylabel("arm")
        plt.xlabel("ReconstructedTime")
        plt.legend()
        plt.grid(True)
        plt.show()
