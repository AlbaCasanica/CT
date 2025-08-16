import pandas as pd
import numpy as np


muse_file_1 = ""
muse_file_2 = ""
mitch_file_1 = ""
mitch_file_2 = ""


def make_df(muse_file_1, muse_file_2 = None, mitch_file_1 = None, mitch_file_2 = None, muse_file_3 = None, muse_file_4 = None):
    file_list = [muse_file_1, muse_file_2, mitch_file_1, mitch_file_2, muse_file_3, muse_file_4]

    df_list = []
    for file in file_list:
        if file:
            if file.endswith(".txt"):
                df_list.append(pd.read_csv(file, delimiter="\t", skiprows=8, decimal=","))
            elif file.endswith(".csv"):
                df_list.append(pd.read_csv(file))
        else:
            df_list.append(None)

    return df_list


def check_samples(df_list):
    sample_num_list = []
    sample_time_list = []
    for df in df_list:
        if df is not None:
            sample_num_list.append(len(df["Timestamp"]))
            record_time = float(df["Timestamp"].iloc[-1])-float(df["Timestamp"][0])
            sample_time_list.append(record_time/60000)
        else:
            sample_num_list.append(None)
            sample_time_list.append(None)
    return sample_num_list, sample_time_list


def fix_time_muse_hz(df_list, index_file, file_path, frequency):
    df = df_list[index_file]

    df['ReconstructedTime'] = np.arange(len(df)) * (1 / frequency)

    df.to_csv(f"{file_path.rstrip('.txt').rstrip('.csv')}_new_time_hz.csv")

    
    return "Time is fixed with hz and new files made"




df_list = make_df(muse_file_1=muse_file_1, muse_file_2=muse_file_2, mitch_file_1=None, mitch_file_2=None)
samples, time = check_samples(df_list)
print(samples)
print(time)
#fix_time_muse_hz(df_list, 0, muse_file_1, 800)
#fix_time_muse_hz(df_list, 1, muse_file_2, 800)
#fix_time_muse_hz(df_list, 2, mitch_file_1, 800)
#fix_time_muse_hz(df_list, 3, mitch_file_2, 800)
#fix_time_muse_hz(df_list, 4, muse_file_3, 800)