from scipy.signal import medfilt
import pandas as pd


def median_filter_medfilt(filepath_no_idle, columns, kernel_size=3, df=None):
    if df is None:
        df = pd.read_csv(filepath_no_idle)

    filtered_df = df.copy()
    for col in columns:
        filtered_df[col] = medfilt(df[col], kernel_size=kernel_size)

    output_path = f"{filepath_no_idle.rstrip(".csv")}_median_filter.csv"
    filtered_df.to_csv(output_path, index=False)
    return filtered_df, output_path


imu_columns = ["Axl.X", "Axl.Y", "Axl.Z",
               "Gyr.X", "Gyr.Y", "Gyr.Z",
               "Mag.X", "Mag.Y", "Mag.Z"]

fsr_columns = [f"Fsr.{str(sensor).zfill(2)}" for sensor in range(1, 17)]

filepath_arm = ""
filepath_back = ""
#df_filtered_arm = median_filter_medfilt(filepath_arm, imu_columns, kernel_size=21)
#df_filtered_back = median_filter_medfilt(filepath_back, imu_columns, kernel_size=21)

filepath_left = ""
filepath_right = ""
#df_filtered_left = median_filter_medfilt(filepath_left, fsr_columns, 9)
#df_filtered_right = median_filter_medfilt(filepath_right, fsr_columns, 9)
