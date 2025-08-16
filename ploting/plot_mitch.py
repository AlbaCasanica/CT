import pandas as pd
import matplotlib.pyplot as plt

cableColors=[
    "#ff5252",
    "#ff9352",
    "#ffd452",
    "#e8ff52",
    "#a8ff52",
    "#67ff52",
    "#52ff7d",
    "#52ffbe",
    "#52ffff",
    "#52beff",
    "#527dff",
    "#6752ff",
    "#a852ff",
    "#e952ff",
    "#ff52d4",
    "#ff5293"
]

def plot_press_soles(file_path_mitch, start_sensor, stop_sensor, with_timestamp = True, with_average = True):

    if file_path_mitch.endswith(".txt"):
        df = pd.read_csv(file_path_mitch, delimiter="\t", skiprows=8, decimal=",")
    elif file_path_mitch.endswith(".csv"):
        df = pd.read_csv(file_path_mitch)

    # In case only some of the sensors on the soles should be plotted
    if not (start_sensor == 1 and stop_sensor == 16):
        for sensor in range (1, 17):
            if sensor < start_sensor or sensor > stop_sensor:
                df[f"Fsr.{str(sensor).zfill(2)}"] = 253

    # Extract FSR columns (assuming they are named "Fsr.01", "Fsr.02", ..., "Fsr.16")
    fsr_columns = [f"Fsr.{str(sensor).zfill(2)}" for sensor in range(1, 17)]  # "Fsr.01" to "Fsr.16"
    fsr_data = df[fsr_columns]  # Select only the FSR columns

    if with_timestamp:
        if not with_average:
            # Plot all 16 FSR elements over time
            plt.figure(figsize=(12, 6))
            for i in range(16):
                plt.plot(df['Timestamp'], fsr_data.iloc[:, i], label=f"FSR {i+1}")

            plt.legend()
            plt.title("FSR Pressure Data Over Time")
            plt.xlabel("Timestamp")
            plt.ylabel("Pressure Value")
            plt.show()
        elif with_average:
            average = fsr_data.mean(axis=1)
            plt.figure(figsize=(12, 6))
            plt.plot(df['ReconstructedTime'], average, label="Average Pressure", color='blue')
            plt.legend()
            plt.title("Average FSR Pressure Over Time")
            plt.xlabel("ReconstructedTime")
            plt.ylabel("Average Pressure")
            plt.grid(True)
            plt.show()

    else:
        if not with_average:
            # Plot all 16 FSR elements over time
            plt.figure(figsize=(12, 6))
            for i in range(16):
                plt.plot(fsr_data.iloc[:, i], label=f"FSR {i+1}", color=cableColors[i])

            plt.legend()
            plt.title("FSR Pressure Data, FSR number test")
            plt.xlabel("Timestep")
            plt.ylabel("Pressure Value")
            plt.show()
        elif with_average:
            average = fsr_data.mean(axis=1)
            plt.figure(figsize=(12, 6))
            plt.plot(average, label="Average Pressure", color='blue')
            plt.legend()
            plt.title("Average FSR Pressure Over Time")
            plt.xlabel("Timestep")
            plt.ylabel("Average Pressure")
            plt.grid(True)
            plt.show()


def plot_press_soles_two(file_paths, start_sensor=1, stop_sensor=16, with_timestamp=True, with_average=True):
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    dataframes = []
    for path in file_paths:
        if path.endswith(".txt"):
            df = pd.read_csv(path, delimiter="\t", skiprows=8, decimal=",")
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file type: {path}")
        
        for sensor in range(1, 17):
            if sensor < start_sensor or sensor > stop_sensor:
                df[f"Fsr.{str(sensor).zfill(2)}"] = 253
        dataframes.append(df)
    
    # Plotting
    plt.figure(figsize=(12, 6))

    for idx, df in enumerate(dataframes):
        label_prefix = f"Sensor {idx + 1}"

        fsr_columns = [f"Fsr.{str(sensor).zfill(2)}" for sensor in range(1, 17)]
        fsr_data = df[fsr_columns]

        if with_average:
            average = fsr_data.mean(axis=1)
            if with_timestamp:
                time_col = 'ReconstructedTime' if 'ReconstructedTime' in df.columns else 'Timestamp'
                plt.plot(df[time_col], average, label=f"{label_prefix} - Avg Pressure")
            else:
                plt.plot(average, label=f"{label_prefix} - Avg Pressure")
        else:
            for i in range(start_sensor - 1, stop_sensor):
                if with_timestamp:
                    time_col = 'ReconstructedTime' if 'ReconstructedTime' in df.columns else 'Timestamp'
                    plt.plot(df[time_col], fsr_data.iloc[:, i], label=f"{label_prefix} - FSR {i + 1}")
                else:
                    plt.plot(fsr_data.iloc[:, i], label=f"{label_prefix} - FSR {i + 1}")
    
    plt.legend()
    plt.title("FSR Pressure Data Over Time")
    plt.xlabel("Timestamp" if with_timestamp else "Timestep")
    plt.ylabel("Pressure Value" if not with_average else "Average Pressure")
    plt.grid(True)
    plt.show()


def plot_gyro_axl_mitch(file_path_mitch, plot_gyro = True, plot_axl = True, with_timestamp = True):
    
    if file_path_mitch.endswith(".txt"):
        df = pd.read_csv(file_path_mitch, delimiter="\t", skiprows=8, decimal=",")
    elif file_path_mitch.endswith(".csv"):
        df = pd.read_csv(file_path_mitch)

    if with_timestamp:

        if plot_gyro:
            # Plot Gyroscope
            plt.figure(figsize=(10, 5))
            plt.plot(df['Timestamp'], df['Gyr.X'], label='Gyro X', alpha=0.7)
            plt.plot(df['Timestamp'], df['Gyr.Y'], label='Gyro Y', alpha=0.7)
            plt.plot(df['Timestamp'], df['Gyr.Z'], label='Gyro Z', alpha=0.7)
            plt.title("Gyroscope Data Over Time")
            plt.xlabel("Timestamp")
            plt.ylabel("Angular Velocity")
            plt.show()
        
        if plot_axl:
            # Plot Accelerometer
            plt.figure(figsize=(10, 5))
            plt.plot(df['Timestamp'], df['Axl.X'], label='Axl X')
            plt.plot(df['Timestamp'], df['Axl.Y'], label='Axl Y')
            plt.plot(df['Timestamp'], df['Axl.Z'], label='Axl Z')
            plt.legend()
            plt.title("Accelerometer Data Over Time")
            plt.xlabel("Timestamp")
            plt.ylabel("Acceleration")
            plt.show()
    
    else:
        
        if plot_gyro:
            # Plot Gyroscope
            plt.figure(figsize=(10, 5))
            plt.plot(df['Gyr.X'], label='Gyro X', alpha=0.7)
            plt.plot(df['Gyr.Y'], label='Gyro Y', alpha=0.7)
            plt.plot(df['Gyr.Z'], label='Gyro Z', alpha=0.7)
            plt.title("Gyroscope Data Over Time")
            plt.xlabel("Timestep")
            plt.ylabel("Angular Velocity")
            plt.show()
        
        if plot_axl:
            # Plot Accelerometer
            plt.figure(figsize=(10, 5))
            plt.plot(df['Axl.X'], label='Axl X')
            plt.plot(df['Axl.Y'], label='Axl Y')
            plt.plot(df['Axl.Z'], label='Axl Z')
            plt.legend()
            plt.title("Accelerometer Data Over Time")
            plt.xlabel("Timestep")
            plt.ylabel("Acceleration")
            plt.show()
    

def preassure_plot_to_see_filter(df_list):
    df_mitch_1 = None
    if df_list[0] is not None:
        df_mitch_1 = df_list[0]
    df_mitch_2 = None
    if df_list[1] is not None:
        df_mitch_2 = df_list[1]
    df_mitch_3 = None
    if df_list[2] is not None:
        df_mitch_3 = df_list[2]
    df_mitch_4 = None
    if df_list[3] is not None:
        df_mitch_4 = df_list[3]
    
    # Extract FSR columns (assuming they are named "Fsr.01", "Fsr.02", ..., "Fsr.16")
    fsr_columns = [f"Fsr.{str(sensor).zfill(2)}" for sensor in range(1, 17)]  # "Fsr.01" to "Fsr.16"

    non_none_count = sum(x is not None for x in df_list)
    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(non_none_count, 1, figsize=(14, 10), sharex=True)
    
    if df_mitch_1 is not None:
        fsr_data = df_mitch_1[fsr_columns]  # Select only the FSR columns
        # Plot all 16 FSR elements over time
        for i in range(16):
            axs[0].plot(df_mitch_1['ReconstructedTime'], fsr_data.iloc[:, i], label=f"FSR {i+1}")
            axs[0].set_ylabel("Kernel size 11")

    if df_mitch_2 is not None:
        fsr_data = df_mitch_2[fsr_columns]  # Select only the FSR columns
        # Plot all 16 FSR elements over time
        for i in range(16):
            axs[1].plot(df_mitch_2['ReconstructedTime'], fsr_data.iloc[:, i], label=f"FSR {i+1}")
            axs[1].set_ylabel("Kernel size 9")

    if df_mitch_3 is not None:
        fsr_data = df_mitch_3[fsr_columns]  # Select only the FSR columns
        # Plot all 16 FSR elements over time
        for i in range(16):
            axs[2].plot(df_mitch_3['ReconstructedTime'], fsr_data.iloc[:, i], label=f"FSR {i+1}")
            axs[2].set_ylabel("Kernel size 7")
    
    if df_mitch_4 is not None:
        fsr_data = df_mitch_4[fsr_columns]  # Select only the FSR columns
        # Plot all 16 FSR elements over time
        for i in range(16):
            axs[3].plot(df_mitch_4['ReconstructedTime'], fsr_data.iloc[:, i], label=f"FSR {i+1}")
            axs[3].set_ylabel("No filter")
            axs[3].set_xlabel("ReconstructedTime (s)")
    

    fig.suptitle("Preassure with and without median filter", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


    fsr_columns = [f"Fsr.{str(sensor).zfill(2)}" for sensor in range(1, 17)]
    fsr_data_1 = df_mitch_1[fsr_columns]
    fsr_data_2 = df_mitch_2[fsr_columns]
    fsr_data_3 = df_mitch_3[fsr_columns]
    fsr_data_4 = df_mitch_4[fsr_columns]
    average_1 = fsr_data_1.mean(axis=1)
    average_2 = fsr_data_2.mean(axis=1)
    average_3 = fsr_data_3.mean(axis=1)
    average_4 = fsr_data_4.mean(axis=1)
    plt.rcParams.update({'font.size': 16})
    plt.plot(df_mitch_1['ReconstructedTime'], average_1, label=f"k11")
    plt.plot(df_mitch_2['ReconstructedTime'], average_2, label=f"k9")
    plt.plot(df_mitch_3['ReconstructedTime'], average_3, label=f"k7")
    plt.plot(df_mitch_4['ReconstructedTime'], average_4, label=f"No filter")
    plt.legend()
    plt.title("FSR Pressure Data Over Time")
    plt.xlabel("ReconstructedTime (s)")
    plt.ylabel("Pressure Value")
    plt.show()


def preassure_plots_average(df_list):
    df_mitch_1 = None
    if df_list[0] is not None:
        df_mitch_1 = df_list[0]
    df_mitch_2 = None
    if df_list[1] is not None:
        df_mitch_2 = df_list[1]
    df_mitch_3 = None
    if df_list[2] is not None:
        df_mitch_3 = df_list[2]
    df_mitch_4 = None
    if df_list[3] is not None:
        df_mitch_4 = df_list[3]
    
    # Extract FSR columns (assuming they are named "Fsr.01", "Fsr.02", ..., "Fsr.16")
    fsr_columns = [f"Fsr.{str(sensor).zfill(2)}" for sensor in range(1, 17)]  # "Fsr.01" to "Fsr.16"

    non_none_count = sum(x is not None for x in df_list)
    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(non_none_count, 1, figsize=(14, 10), sharex=True)
    
    if df_mitch_1 is not None:
        fsr_data = df_mitch_1[fsr_columns].mean(axis=1)  # Select only the FSR columns
        # Plot all 16 FSR elements over time
        for i in range(16):
            axs[0].plot(fsr_data, label=f"Avg Pressure")
            axs[0].set_ylabel("Left foot")

    if df_mitch_2 is not None:
        fsr_data = df_mitch_2[fsr_columns].mean(axis=1)  # Select only the FSR columns
        # Plot all 16 FSR elements over time
        for i in range(16):
            axs[1].plot(fsr_data, label=f"Avg Pressure")
            axs[1].set_ylabel("Right foot")
            axs[1].set_xlabel("Timestep")

    if df_mitch_3 is not None:
        fsr_data = df_mitch_3[fsr_columns]  # Select only the FSR columns
        # Plot all 16 FSR elements over time
        for i in range(16):
            axs[2].plot(df_mitch_3['ReconstructedTime'], fsr_data.iloc[:, i], label=f"FSR {i+1}")
            axs[2].set_ylabel("Kernel size 7")
            axs[2].set_xlabel("ReconstructedTime")
    
    if df_mitch_4 is not None:
        fsr_data = df_mitch_4[fsr_columns]  # Select only the FSR columns
        # Plot all 16 FSR elements over time
        for i in range(16):
            axs[3].plot(df_mitch_4['ReconstructedTime'], fsr_data.iloc[:, i], label=f"FSR {i+1}")
            axs[3].set_ylabel("No filter")
            axs[3].set_xlabel("ReconstructedTime")
    

    fig.suptitle("Pressure data from Yeti insoles, average of 16 pressure points", fontsize=24)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def preassure_plots_not_average(df_list):
    df_mitch_1 = None
    if df_list[0] is not None:
        df_mitch_1 = df_list[0]
    df_mitch_2 = None
    if df_list[1] is not None:
        df_mitch_2 = df_list[1]
    df_mitch_3 = None
    if df_list[2] is not None:
        df_mitch_3 = df_list[2]
    df_mitch_4 = None
    if df_list[3] is not None:
        df_mitch_4 = df_list[3]
    
    # Extract FSR columns (assuming they are named "Fsr.01", "Fsr.02", ..., "Fsr.16")
    fsr_columns = [f"Fsr.{str(sensor).zfill(2)}" for sensor in range(1, 17)]  # "Fsr.01" to "Fsr.16"

    non_none_count = sum(x is not None for x in df_list)
    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(non_none_count, 1, figsize=(14, 10), sharex=True)
    
    if df_mitch_1 is not None:
        fsr_data = df_mitch_1[fsr_columns]  # Select only the FSR columns
        # Plot all 16 FSR elements over time
        for i in range(16):
            axs[0].plot(fsr_data.iloc[:, i], label=f"FSR {i+1}")
            axs[0].set_ylabel("Left foot")

    if df_mitch_2 is not None:
        fsr_data = df_mitch_2[fsr_columns]  # Select only the FSR columns
        # Plot all 16 FSR elements over time
        for i in range(16):
            axs[1].plot(fsr_data.iloc[:, i], label=f"FSR {i+1}")
            axs[1].set_ylabel("Right foot")
            axs[1].set_xlabel("Timestep")

    if df_mitch_3 is not None:
        fsr_data = df_mitch_3[fsr_columns]  # Select only the FSR columns
        # Plot all 16 FSR elements over time
        for i in range(16):
            axs[2].plot(df_mitch_3['ReconstructedTime'], fsr_data.iloc[:, i], label=f"FSR {i+1}")
            axs[2].set_ylabel("Kernel size 7")
            axs[2].set_xlabel("ReconstructedTime")
    
    if df_mitch_4 is not None:
        fsr_data = df_mitch_4[fsr_columns]  # Select only the FSR columns
        # Plot all 16 FSR elements over time
        for i in range(16):
            axs[3].plot(df_mitch_4['ReconstructedTime'], fsr_data.iloc[:, i], label=f"FSR {i+1}")
            axs[3].set_ylabel("No filter")
            axs[3].set_xlabel("ReconstructedTime")
    

    fig.suptitle("Pressure data from Yeti insoles, all 16 pressure points", fontsize=24)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


    
