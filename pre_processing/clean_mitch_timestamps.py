import pandas as pd
import numpy as np
import re



def check_irregular_timejums(cleaned_path):
    df = pd.read_csv(cleaned_path, sep=",")
    # Calculate timestamp differences
    timestamp_diffs = df['Timestamp'].diff()

    # Define thresholds for anomaly detection (100 Hz => ~10ms interval)
    mean_interval = timestamp_diffs[1:].mean()
    std_interval = timestamp_diffs[1:].std()
    threshold_upper = mean_interval + 3 * std_interval
    threshold_lower = mean_interval - 3 * std_interval

    # Identify jumps or outliers
    jumps = df[(timestamp_diffs > threshold_upper) | (timestamp_diffs < threshold_lower)]

    # Calculate difference between consecutive timestamps
    df['Timestamp_Diff'] = df['Timestamp'].diff()

    # Identify rows where timestamp jumps back (diff < 0)
    jumps_back = df[df['Timestamp_Diff'] < 0]

    # Return summary and some example rows
    return{
        "mean_interval_ms": mean_interval,
        "std_dev_ms": std_interval,
        "num_jumps": len(jumps),
        "num_jumps_back": len(jumps_back),
        "jumps_back": jumps_back[["Timestamp"]],
        "example_jumps": jumps[['Timestamp']].head(10)
    }



def remove_end_duplicates(file_path):
    
    if file_path.endswith(".txt"):
        df = pd.read_csv(file_path, delimiter="\t", skiprows=8, decimal=",")
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

    # Exclude timestamp column from comparison (assuming it's the first column)
    df_no_time = df.iloc[:, 1:]  # all columns except timestamp
    last_unique_index = len(df_no_time)  # default to keep all

    # Go bottom-up and compare rows to the previous one (excluding timestamp)
    for i in range(len(df_no_time) - 2, -1, -1):
        if df_no_time.iloc[i].equals(df_no_time.iloc[i + 1]):
            last_unique_index = i  # include first differing row

    df_clean = df.iloc[:last_unique_index]

    output_path = f"{file_path.rstrip(".csv").rstrip(".txt")}_del_end_dupli.csv"
    df_clean.to_csv(output_path, index=False)

    return df_clean, output_path


def check_sample(df):
    if not isinstance(df, pd.DataFrame):
        if df.endswith(".txt"):
            df = pd.read_csv(df, delimiter="\t", skiprows=8, decimal=",")
        elif df.endswith(".csv"):
            df = pd.read_csv(df) 
    
    sample_num = len(df["Timestamp"])
    recorded_time = float(df["Timestamp"].iloc[-1])-float(df["Timestamp"][0])
    recorded_time_sec = recorded_time/1000
    recorded_time_min = recorded_time/60000
    print(f"Sample number: {sample_num}\n"
          f"Recorded time sec: {recorded_time_sec}\n"
          f"Recorded time min: {recorded_time_min}")
    return sample_num, recorded_time_sec


def add_ReconstructedTime(cleaned_filepath, df=None):
    if df is None:
        df = pd.read_csv(cleaned_filepath)
    
    num_samples, recorded_time_sec = check_sample(df)

    df['ReconstructedTime'] = np.linspace(0, recorded_time_sec, num=num_samples)

    output_path = f"{cleaned_filepath.rstrip(".csv")}_new_time.csv"
    df.to_csv(output_path, index=False)

    return df, output_path


def fix_sensor_numbers(filepath, df=None):

    # Maps for changing numbers
    rename_map_left = {
        9:1,    10:2,   13:3,   16:4,
        14:5,   11:6,   12:7,   15:8,
        3:9,    2:10,   1:11,   5:12,
        7:13,   4:14,   8:15,   6:16
    }
    rename_map_right = {
        4:1,    8:2,    5:3,    2:4,
        3:5,    6:6,    7:7,    1:8,
        14:9,   16:10,  15:11,  13:12,
        12:13,  9:14,   10:15,  11:16
    }

    if df is None:
        if filepath.endswith(".txt"):
            df = pd.read_csv(filepath, delimiter="\t", skiprows=8, decimal=",")
        elif filepath.endswith(".csv"):
            df = pd.read_csv(filepath)

    if "left" in filepath:
        rename_map = rename_map_left
    elif "right" in filepath:
        rename_map = rename_map_right
    else:
        raise ValueError("Could not determine sensor type from filename. Include 'small'/'big' and 'left'/'right'.")
    
    # Step 3: Rename FSR columns
    new_columns = []
    for col in df.columns:
        match = re.match(r"Fsr\.(\d+)", col)
        if match:
            original_num = int(match.group(1))
            if original_num in rename_map:
                new_num = rename_map[original_num]
                new_columns.append(f"Fsr.{new_num:02d}")
            else:
                new_columns.append(col)
        else:
            new_columns.append(col)

    df.columns = new_columns

    output_path = f"{filepath.rstrip(".txt").rstrip(".csv")}_fsr_nums.csv"
    df.to_csv(output_path, index=False)

    return df, output_path


def run_fix_timestamp_mitch(filepath):
    df_dubli, path_dupli = remove_end_duplicates(filepath)
    print("after removed duplicates:")
    #check_sample(df_dubli)

    df_new_time, path_new_time = add_ReconstructedTime(path_dupli, df_dubli)

    df_f_sensors, path_f_sensors = fix_sensor_numbers(path_new_time, df_new_time)

    return df_f_sensors, path_f_sensors
