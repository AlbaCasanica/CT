import numpy as np
import pandas as pd
from get_Time_Domain_features_of_signal import get_Time_Domain_features_of_signal
from get_Freq_Domain_features_of_signal import get_Freq_Domain_features_of_signal

'Based on Royas code for extracting IMU features'

def ExtractPressure_Features(fsr_data, sensor_name, window_length, mean_fsr, fs):
    if not isinstance(fsr_data, pd.DataFrame):
        fsr_data = pd.read_csv(fsr_data)
    
    '''EXTRACT COLUMNS'''
    time_data   = fsr_data["ReconstructedTime"]

    fsr_columns = [f"Fsr.{str(sensor).zfill(2)}" for sensor in range(1, 17)]  # "Fsr.01" to "Fsr.16"
    fsr_data = fsr_data[fsr_columns]  # Select only the FSR columns

    # Define a list to store features for each window
    all_window_features = []

    # Calculate the number of windows
    num_samples = len(time_data)
    num_windows = num_samples // window_length
    print(f"Number of fsr  windows: {num_windows}")

    for i in range(num_windows):
        # Define the start and end index for the window
        start_idx = i * window_length
        end_idx = start_idx + window_length


        if mean_fsr == False:
            sum_fsr = fsr_data.sum(axis=1)

            window_fsr_sum = sum_fsr[start_idx:end_idx]

            window_features_fsr_sum_Time = get_Time_Domain_features_of_signal(window_fsr_sum, f"fsr_sum_{sensor_name}")
            window_features_fsr_sum_Freq = get_Freq_Domain_features_of_signal(window_fsr_sum, f"fsr_sum_{sensor_name}", fs)

            window_features = {**window_features_fsr_sum_Time, 
                               **window_features_fsr_sum_Freq}

        if mean_fsr == True:
            average_fsr = fsr_data.mean(axis=1)

            window_fsr_aver = average_fsr[start_idx:end_idx]

            window_features_fsr_aver_Time = get_Time_Domain_features_of_signal(window_fsr_aver, f"fsr_aver_{sensor_name}")
            window_features_fsr_aver_Freq = get_Freq_Domain_features_of_signal(window_fsr_aver, f"fsr_aver_{sensor_name}", fs)

            window_features = {**window_features_fsr_aver_Time, 
                               **window_features_fsr_aver_Freq}


        all_window_features.append(window_features)

    feature_df = pd.DataFrame(all_window_features)

    return feature_df
    