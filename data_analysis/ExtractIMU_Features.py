import numpy as np
import pandas as pd
from get_Time_Domain_features_of_signal import get_Time_Domain_features_of_signal
from get_Freq_Domain_features_of_signal import get_Freq_Domain_features_of_signal

'Based on the bachelor studens modification of Royas code'

def ExtractIMU_Features(imu_data, sensor_name, window_length, norm_IMU, fs, HDR=False):
    if not isinstance(imu_data, pd.DataFrame):
        imu_data = pd.read_csv(imu_data)

    ''' EXTRACT COLUMNS '''
    time_data   = imu_data["ReconstructedTime"]

    accel_X     = imu_data["Axl.X"]  
    accel_Y     = imu_data["Axl.Y"]  
    accel_Z     = imu_data["Axl.Z"]  

    gyro_X      = imu_data["Gyr.X"]  
    gyro_Y      = imu_data["Gyr.Y"]  
    gyro_Z      = imu_data["Gyr.Z"]  

    mag_X       = imu_data["Mag.X"]  
    mag_Y       = imu_data["Mag.Y"]  
    mag_Z       = imu_data["Mag.Z"]  

    if HDR:
        Hdr_X   = imu_data["Hdr.X"]
        Hdr_Y   = imu_data["Hdr.Y"]
        Hdr_Z   = imu_data["Hdr.Z"]

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
        # print(f"Getting features from window {start_idx} to {end_idx}") 

        if norm_IMU == True:
            # Extract acceleration and gyroscope data from the IMU dataset
            norm_accel  = np.sqrt( np.power(accel_X, 2) + np.power(accel_Y, 2) + np.power(accel_Z, 2))
            norm_gyro   = np.sqrt( np.power(gyro_X, 2)  + np.power(gyro_Y, 2)  + np.power(gyro_Z, 2))
            norm_mag    = np.sqrt( np.power(mag_X, 2)   + np.power(mag_Y, 2)   + np.power(mag_Z, 2))
            if HDR:
                norm_hdr = np.sqrt( np.power(Hdr_X, 2)   + np.power(Hdr_Y, 2)   + np.power(Hdr_Z, 2))

            # Remove gravity:
            '''
            g_constant = np.mean(norm_acceleration)
            # print(f"g constant: {g_constant}")
            gravless_norm = np.subtract(norm_acceleration, g_constant)  
            window_accel_Norm = gravless_norm[start_idx:end_idx]
            '''

            window_accel_Norm   = norm_accel[start_idx:end_idx]
            window_gyro_Norm    = norm_gyro[start_idx:end_idx]
            window_mag_Norm     = norm_mag[start_idx:end_idx]
            if HDR:
                window_hdr_Norm = norm_hdr[start_idx:end_idx]        

            window_features_accel_Norm_Time = get_Time_Domain_features_of_signal(window_accel_Norm, f"accel_Norm_{sensor_name}")
            window_features_gyro_Norm_Time  = get_Time_Domain_features_of_signal(window_gyro_Norm, f"gyro_Norm_{sensor_name}")
            window_features_mag_Norm_Time   = get_Time_Domain_features_of_signal(window_mag_Norm, f"mag_Norm_{sensor_name}")
            if HDR:
                window_features_hdr_Norm_Time   = get_Time_Domain_features_of_signal(window_hdr_Norm, f"hdr_Norm_{sensor_name}")

            window_features_accel_Norm_Freq = get_Freq_Domain_features_of_signal(window_accel_Norm, f"accel_Norm_{sensor_name}", fs)
            window_features_gyro_Norm_Freq  = get_Freq_Domain_features_of_signal(window_gyro_Norm, f"_gyro_Norm_{sensor_name}", fs)
            window_features_mag_Norm_Freq   = get_Freq_Domain_features_of_signal(window_mag_Norm, f"mag_Norm_{sensor_name}", fs)
            if HDR:
                window_features_hdr_Norm_Freq   = get_Freq_Domain_features_of_signal(window_hdr_Norm, f"hdr_Norm_{sensor_name}", fs)

            ## merge all
            if not HDR:
                window_features = {**window_features_accel_Norm_Time, 
                                **window_features_accel_Norm_Freq,
                                **window_features_gyro_Norm_Time,
                                **window_features_gyro_Norm_Freq,
                                **window_features_mag_Norm_Time,
                                **window_features_mag_Norm_Freq,
                                }
            if HDR:
                window_features = {**window_features_accel_Norm_Time, 
                                **window_features_accel_Norm_Freq,
                                **window_features_gyro_Norm_Time,
                                **window_features_gyro_Norm_Freq,
                                **window_features_mag_Norm_Time,
                                **window_features_mag_Norm_Freq,
                                **window_features_hdr_Norm_Time,
                                **window_features_hdr_Norm_Freq
                                }

            

        if norm_IMU == False:    
            window_accel_X = accel_X[start_idx:end_idx]
            window_accel_Y = accel_Y[start_idx:end_idx]
            window_accel_Z = accel_Z[start_idx:end_idx]

            window_gyro_X  = gyro_X[start_idx:end_idx]
            window_gyro_Y  = gyro_Y[start_idx:end_idx]
            window_gyro_Z  = gyro_Z[start_idx:end_idx]

            window_mag_X   = mag_X[start_idx:end_idx]
            window_mag_Y   = mag_Y[start_idx:end_idx]
            window_mag_Z   = mag_Z[start_idx:end_idx]

            if HDR:
                window_hdr_X   = Hdr_X[start_idx:end_idx]
                window_hdr_Y   = Hdr_Y[start_idx:end_idx]
                window_hdr_Z   = Hdr_Z[start_idx:end_idx]


            window_features_accel_X_Time = get_Time_Domain_features_of_signal(window_accel_X, f"accel_X_{sensor_name}")
            window_features_accel_Y_Time = get_Time_Domain_features_of_signal(window_accel_Y, f"accel_Y_{sensor_name}")
            window_features_accel_Z_Time = get_Time_Domain_features_of_signal(window_accel_Z, f"accel_Z_{sensor_name}")

            window_features_gyro_X_Time  = get_Time_Domain_features_of_signal(window_gyro_X, f"gyro_X_{sensor_name}")
            window_features_gyro_Y_Time  = get_Time_Domain_features_of_signal(window_gyro_Y, f"gyro_Y_{sensor_name}")
            window_features_gyro_Z_Time  = get_Time_Domain_features_of_signal(window_gyro_Z, f"gyro_Z_{sensor_name}")

            window_features_mag_X_Time   = get_Time_Domain_features_of_signal(window_mag_X, f"mag_X_{sensor_name}")
            window_features_mag_Y_Time   = get_Time_Domain_features_of_signal(window_mag_Y, f"mag_Y_{sensor_name}")
            window_features_mag_Z_Time   = get_Time_Domain_features_of_signal(window_mag_Z, f"mag_Z_{sensor_name}")

            if HDR:
                window_features_hdr_X_Time   = get_Time_Domain_features_of_signal(window_hdr_X, f"hdr_X_{sensor_name}")
                window_features_hdr_Y_Time   = get_Time_Domain_features_of_signal(window_hdr_Y, f"hdr_Y_{sensor_name}")
                window_features_hdr_Z_Time   = get_Time_Domain_features_of_signal(window_hdr_Z, f"hdr_Z_{sensor_name}")

            window_features_accel_X_Freq = get_Freq_Domain_features_of_signal(window_accel_X, f"accel_X_{sensor_name}", fs)
            window_features_accel_Y_Freq = get_Freq_Domain_features_of_signal(window_accel_Y, f"accel_Y_{sensor_name}", fs)
            window_features_accel_Z_Freq = get_Freq_Domain_features_of_signal(window_accel_Z, f"accel_Z_{sensor_name}", fs)

            window_features_gyro_X_Freq  = get_Freq_Domain_features_of_signal(window_gyro_X, f"gyro_X_{sensor_name}", fs)
            window_features_gyro_Y_Freq  = get_Freq_Domain_features_of_signal(window_gyro_Y, f"gyro_Y_{sensor_name}", fs)
            window_features_gyro_Z_Freq  = get_Freq_Domain_features_of_signal(window_gyro_Z, f"gyro_Z_{sensor_name}", fs)

            window_features_mag_X_Freq   = get_Freq_Domain_features_of_signal(window_mag_X, f"mag_X_{sensor_name}", fs)
            window_features_mag_Y_Freq   = get_Freq_Domain_features_of_signal(window_mag_Y, f"mag_Y_{sensor_name}", fs)
            window_features_mag_Z_Freq   = get_Freq_Domain_features_of_signal(window_mag_Z, f"mag_Z_{sensor_name}", fs)

            if HDR:
                window_features_hdr_X_Freq   = get_Freq_Domain_features_of_signal(window_hdr_X, f"hdr_X_{sensor_name}", fs)
                window_features_hdr_Y_Freq   = get_Freq_Domain_features_of_signal(window_hdr_Y, f"hdr_Y_{sensor_name}", fs)
                window_features_hdr_Z_Freq   = get_Freq_Domain_features_of_signal(window_hdr_Z, f"hdr_Z_{sensor_name}", fs)


            ## merge all
            if not HDR:
                window_features = {**window_features_accel_X_Time, 
                                **window_features_accel_Y_Time,
                                **window_features_accel_Z_Time,
                                **window_features_accel_X_Freq,
                                **window_features_accel_Y_Freq,
                                **window_features_accel_Z_Freq,
                                **window_features_gyro_X_Time,
                                **window_features_gyro_Y_Time,
                                **window_features_gyro_Z_Time,
                                **window_features_gyro_X_Freq,
                                **window_features_gyro_Y_Freq,
                                **window_features_gyro_Z_Freq,
                                **window_features_mag_X_Time,
                                **window_features_mag_Y_Time,
                                **window_features_mag_Z_Time,
                                **window_features_mag_X_Freq,
                                **window_features_mag_Y_Freq,
                                **window_features_mag_Z_Freq
                                }
            if HDR:
                window_features = {**window_features_accel_X_Time, 
                                **window_features_accel_Y_Time,
                                **window_features_accel_Z_Time,
                                **window_features_accel_X_Freq,
                                **window_features_accel_Y_Freq,
                                **window_features_accel_Z_Freq,
                                **window_features_gyro_X_Time,
                                **window_features_gyro_Y_Time,
                                **window_features_gyro_Z_Time,
                                **window_features_gyro_X_Freq,
                                **window_features_gyro_Y_Freq,
                                **window_features_gyro_Z_Freq,
                                **window_features_mag_X_Time,
                                **window_features_mag_Y_Time,
                                **window_features_mag_Z_Time,
                                **window_features_mag_X_Freq,
                                **window_features_mag_Y_Freq,
                                **window_features_mag_Z_Freq,
                                **window_features_hdr_X_Time,
                                **window_features_hdr_Y_Time,
                                **window_features_hdr_Z_Time,
                                **window_features_hdr_X_Freq,
                                **window_features_hdr_Y_Freq,
                                **window_features_hdr_Z_Freq
                                }                


        all_window_features.append(window_features)

    feature_df = pd.DataFrame(all_window_features)

    return feature_df
