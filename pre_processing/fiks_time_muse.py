"""
Module: fiks_time_muse.py
Purpose:
    Utility functions to load, inspect, and adjust raw Muse (EEG/IMU) and Mitch (FSR) recordings.
    Provides helpers to:
        - Load raw files (.txt or .csv) into pandas DataFrames
        - Inspect number of samples and recording duration
        - Reconstruct time axis using a known sampling frequency

Input:
    File paths (.txt or .csv) from Muse or Mitch sensors.

Output:
    - List of pandas DataFrames with loaded sensor data
    - Sample counts and recording durations
    - CSV files with reconstructed time axis (when applying time correction)

Dependencies:
    - pandas
    - numpy
"""

import pandas as pd
import numpy as np


# =========================
# Data Loading
# =========================
def make_df(
    muse_file_1,
    muse_file_2=None,
    mitch_file_1=None,
    mitch_file_2=None,
    muse_file_3=None,
    muse_file_4=None,
):
    """
    Load Muse and Mitch recordings into pandas DataFrames.

    Parameters
    ----------
    muse_file_1 : str
        Path to first Muse file (required).
    muse_file_2, mitch_file_1, mitch_file_2, muse_file_3, muse_file_4 : str, optional
        Paths to additional sensor files.

    Returns
    -------
    list of pandas.DataFrame or None
        A list of DataFrames (or None if the file path is not provided).
    """
    file_list = [
        muse_file_1,
        muse_file_2,
        mitch_file_1,
        mitch_file_2,
        muse_file_3,
        muse_file_4,
    ]

    df_list = []
    for file in file_list:
        if file:
            if file.endswith(".txt"):
                df_list.append(
                    pd.read_csv(file, delimiter="\t", skiprows=8, decimal=",")
                )
            elif file.endswith(".csv"):
                df_list.append(pd.read_csv(file))
        else:
            df_list.append(None)

    return df_list


# =========================
# Data Inspection
# =========================
def check_samples(df_list):
    """
    Compute number of samples and recording duration per file.

    Parameters
    ----------
    df_list : list of pandas.DataFrame or None
        Sensor recordings loaded with `make_df`.

    Returns
    -------
    tuple of lists
        (sample_num_list, sample_time_list)
        - sample_num_list: number of rows in each DataFrame
        - sample_time_list: recording duration in minutes
    """
    sample_num_list = []
    sample_time_list = []
    for df in df_list:
        if df is not None:
            sample_num_list.append(len(df["Timestamp"]))
            record_time = float(df["Timestamp"].iloc[-1]) - float(df["Timestamp"][0])
            sample_time_list.append(record_time / 60000)
        else:
            sample_num_list.append(None)
            sample_time_list.append(None)
    return sample_num_list, sample_time_list


# =========================
# Time Correction
# =========================
def fix_time_muse_hz(df_list, index_file, file_path, frequency):
    """
    Reconstruct time axis for a Muse recording using known sampling frequency.

    Parameters
    ----------
    df_list : list of pandas.DataFrame
        List of sensor dataframes returned by `make_df`.
    index_file : int
        Index of the file in df_list to be processed.
    file_path : str
        Original path of the file to build the output filename.
    frequency : int or float
        Sampling frequency in Hz.

    Returns
    -------
    str
        Confirmation message after creating a new CSV file.
    """
    df = df_list[index_file]
    df["ReconstructedTime"] = np.arange(len(df)) * (1 / frequency)

    df.to_csv(f"{file_path.rstrip('.txt').rstrip('.csv')}_new_time_hz.csv")

    return "Time was reconstructed using frequency and saved to new file."


# =========================
# Example usage (debug/test)
# =========================
if __name__ == "__main__":
    muse_file_1 = ""
    muse_file_2 = ""
    mitch_file_1 = ""
    mitch_file_2 = ""

    df_list = make_df(
        muse_file_1=muse_file_1,
        muse_file_2=muse_file_2,
        mitch_file_1=None,
        mitch_file_2=None,
    )
    samples, time = check_samples(df_list)
    print(samples)
    print(time)
    # Example: fix_time_muse_hz(df_list, 0, muse_file_1, 800)
