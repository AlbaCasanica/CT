import matplotlib.pyplot as plt
import pandas as pd


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


def plot_axl_muse(df_list):
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
    
    
    if df_mitch_1 is not None:
        plt.plot(df_mitch_1['Timestamp'], df_mitch_1['Axl.X'], label='Mitch_1 - Axl X', alpha=0.7)
    if df_mitch_2 is not None:
        plt.plot(df_mitch_2['Timestamp'], df_mitch_2['Axl.X'], label='Mitch_2 - Axl X', alpha=0.7)
    if df_muse_1 is not None:
        plt.plot(df_muse_1['Timestamp'], df_muse_1['Axl.X'], label='Muse_1 - Axl X', alpha=0.7)
    if df_muse_2 is not None:
        plt.plot(df_muse_2['Timestamp'], df_muse_2['Axl.X'], label='muse_2 - Axl X', alpha=0.7)
    plt.legend()
    plt.title("Accelerometer X Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Acceleration (mg)")
    plt.grid(True)
    plt.show()

    # Plot Accelerometer Y
    plt.figure(figsize=(12, 5))
    if df_muse_1 is not None:
        plt.plot(df_muse_1['Timestamp'], df_muse_1['Axl.Y'], label='Muse_1 - Axl Y', alpha=0.7)
    if df_muse_2 is not None:
        plt.plot(df_muse_2['Timestamp'], df_muse_2['Axl.Y'], label='muse_2 - Axl Y', alpha=0.7)
    if df_mitch_1 is not None:
        plt.plot(df_mitch_1['Timestamp'], df_mitch_1['Axl.Y'], label='Mitch_1 - Axl Y', alpha=0.7)
    if df_mitch_2 is not None:
        plt.plot(df_mitch_2['Timestamp'], df_mitch_2['Axl.Y'], label='Mitch_2 - Axl Y', alpha=0.7)
    plt.legend()
    plt.title("Accelerometer Y Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Acceleration (mg)")
    plt.grid(True)
    plt.show()

    # Plot Accelerometer Z
    plt.figure(figsize=(12, 5))
    if df_muse_1 is not None:
        plt.plot(df_muse_1['Timestamp'], df_muse_1['Axl.Z'], label='muse_1 - Axl Z', alpha=0.7)
    if df_muse_2 is not None:
        plt.plot(df_muse_2['Timestamp'], df_muse_2['Axl.Z'], label='muse_2 - Axl Z', alpha=0.7)
    if df_mitch_1 is not None:
        plt.plot(df_mitch_1['Timestamp'], df_mitch_1['Axl.Z'], label='Mitch_1 - Axl Z', alpha=0.7)
    if df_mitch_2 is not None:
        plt.plot(df_mitch_2['Timestamp'], df_mitch_2['Axl.Z'], label='Mitch_2 - Axl Z', alpha=0.7)
    plt.legend()
    plt.title("Accelerometer Z Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Acceleration (mg)")
    plt.grid(True)
    plt.show()

    # Create subplots
    non_none_count = sum(x is not None for x in df_list)
    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(non_none_count, 1, figsize=(14, 5), sharex=True)

    # Axl muse_1
    if df_muse_1 is not None:
        axs[0].plot(df_muse_1['Timestamp'], df_muse_1['Axl.X'], label='Axl X', alpha=0.7)
        axs[0].plot(df_muse_1['Timestamp'], df_muse_1['Axl.Y'], label='Axl Y', alpha=0.7)
        axs[0].plot(df_muse_1['Timestamp'], df_muse_1['Axl.Z'], label='Axl Z', alpha=0.7)
        axs[0].set_ylabel("Arm")
        axs[0].legend()
        axs[0].grid(True)

    # Axl muse_2
    if df_muse_2 is not None:
        axs[1].plot(df_muse_2['Timestamp'], df_muse_2['Axl.X'], label='Axl X', alpha=0.7)
        axs[1].plot(df_muse_2['Timestamp'], df_muse_2['Axl.Y'], label='Axl Y', alpha=0.7)
        axs[1].plot(df_muse_2['Timestamp'], df_muse_2['Axl.Z'], label='Axl Z', alpha=0.7)
        axs[1].set_ylabel("Back")
        axs[1].set_xlabel("Timestamp")
        axs[1].legend()
        axs[1].grid(True)

    
    # Axl mitch 1
    if df_mitch_1 is not None:
        axs[2].plot(df_mitch_1['Timestamp'], df_mitch_1['Axl.X']/100, label='Axl X', alpha=0.7)
        axs[2].plot(df_mitch_1['Timestamp'], df_mitch_1['Axl.Y']/100, label='Axl Y', alpha=0.7)
        axs[2].plot(df_mitch_1['Timestamp'], df_mitch_1['Axl.Z']/100, label='Axl Z', alpha=0.7)
        axs[2].set_ylabel("mitch_1")
        axs[2].set_xlabel("Timestamp")
        axs[2].legend()
        axs[2].grid(True)
    
    # Axl mitch 2
    if df_mitch_2 is not None:
        axs[3].plot(df_mitch_2['Timestamp'], df_mitch_2['Axl.X']/100, label='Axl X', alpha=0.7)
        axs[3].plot(df_mitch_2['Timestamp'], df_mitch_2['Axl.Y']/100, label='Axl Y', alpha=0.7)
        axs[3].plot(df_mitch_2['Timestamp'], df_mitch_2['Axl.Z']/100, label='Axl Z', alpha=0.7)
        axs[3].set_ylabel("mitch_2")
        axs[3].set_xlabel("Timestamp")
        axs[3].legend()
        axs[3].grid(True)
    


    # Title and layout
    fig.suptitle("Accelerometer Data (mg) from Muse sensors", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_axl_muse_rotation(df_list):
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
    
    
    if df_mitch_1 is not None:
        plt.plot(df_mitch_1['ReconstructedTime'], df_mitch_1['Axl.X'], label='Mitch_1 - Axl X', alpha=0.7)
    if df_mitch_2 is not None:
        plt.plot(df_mitch_2['ReconstructedTime'], df_mitch_2['Axl.X'], label='Mitch_2 - Axl X', alpha=0.7)
    if df_muse_1 is not None:
        plt.plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.X'], label='muse_1 - Axl X', alpha=0.7)
    if df_muse_2 is not None:
        plt.plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.X'], label='muse_2 - Axl X', alpha=0.7)
    plt.legend()
    plt.title("Accelerometer X Over Time")
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
        plt.plot(df_mitch_1['ReconstructedTime'], df_mitch_1['Axl.Y'], label='Mitch_1 - Axl Y', alpha=0.7)
    if df_mitch_2 is not None:
        plt.plot(df_mitch_2['ReconstructedTime'], df_mitch_2['Axl.Y'], label='Mitch_2 - Axl Y', alpha=0.7)
    plt.legend()
    plt.title("Accelerometer Y Over Time")
    plt.xlabel("ReconstructedTime")
    plt.ylabel("Acceleration (mg)")
    plt.grid(True)
    plt.show()

    # Plot Accelerometer Z
    plt.figure(figsize=(12, 5))
    if df_muse_1 is not None:
        plt.plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.Z']/1000, label='muse_1 - Axl Z', alpha=0.7)
    if df_muse_2 is not None:
        plt.plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.Z']/1000, label='muse_2 - Axl Z', alpha=0.7)
    if df_mitch_1 is not None:
        plt.plot(df_mitch_1['ReconstructedTime'], df_mitch_1['Axl.Z']/1000, label='Mitch_1 - Axl Z', alpha=0.7)
    if df_mitch_2 is not None:
        plt.plot(df_mitch_2['ReconstructedTime'], df_mitch_2['Axl.Z']/1000, label='Mitch_2 - Axl Z', alpha=0.7)
    plt.legend()
    plt.title("Accelerometer Z Over Time")
    plt.xlabel("ReconstructedTime")
    plt.ylabel("Acceleration (g)")
    plt.grid(True)
    plt.show()

    # Create subplots
    non_none_count = sum(x is not None for x in df_list)
    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(non_none_count, 1, figsize=(14, 5), sharex=True)

    # Axl muse_1
    if df_muse_1 is not None:
        axs[0].plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.X'], label='Axl X', alpha=0.7)
        axs[0].plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.Y'], label='Axl Y', alpha=0.7)
        axs[0].plot(df_muse_1['ReconstructedTime'], df_muse_1['Axl.Z'], label='Axl Z', alpha=0.7)
        axs[0].set_ylabel("Original Muse back")
        axs[0].legend()
        axs[0].grid(True)

    # Axl muse_2
    if df_muse_2 is not None:
        axs[1].plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.X'], label='Axl X', alpha=0.7)
        axs[1].plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.Y'], label='Axl Y', alpha=0.7)
        axs[1].plot(df_muse_2['ReconstructedTime'], df_muse_2['Axl.Z'], label='Axl Z', alpha=0.7)
        axs[1].set_ylabel("Rotated Muse back")
        axs[1].legend()
        axs[1].grid(True)

    
    # Axl mitch 1
    if df_mitch_1 is not None:
        axs[2].plot(df_mitch_1['ReconstructedTime'], df_mitch_1['Axl.X'], label='Axl X', alpha=0.7)
        axs[2].plot(df_mitch_1['ReconstructedTime'], df_mitch_1['Axl.Y'], label='Axl Y', alpha=0.7)
        axs[2].plot(df_mitch_1['ReconstructedTime'], df_mitch_1['Axl.Z'], label='Axl Z', alpha=0.7)
        axs[2].set_ylabel("Original Muse back")
        axs[2].set_xlabel("ReconstructedTime")
        axs[2].legend()
        axs[2].grid(True)
    
    # Axl mitch 2
    if df_mitch_2 is not None:
        axs[3].plot(df_mitch_2['ReconstructedTime'], df_mitch_2['Axl.X'], label='Axl X', alpha=0.7)
        axs[3].plot(df_mitch_2['ReconstructedTime'], df_mitch_2['Axl.Y'], label='Axl Y', alpha=0.7)
        axs[3].plot(df_mitch_2['ReconstructedTime'], df_mitch_2['Axl.Z'], label='Axl Z', alpha=0.7)
        axs[3].set_ylabel("Rotated Muse back")
        axs[3].set_xlabel("ReconstructedTime")
        axs[3].legend()
        axs[3].grid(True)

    # Title and layout
    fig.suptitle("Original vs. rotated accelerometer Data (X, Y, Z)", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



def plot_gyro_muse(df_list):
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

    # Plot Gyro X
    plt.figure(figsize=(12, 5))
    if df_muse_1 is not None:
        plt.plot(df_muse_1['Timestamp'], df_muse_1['Gyr.X'], label='muse_1 - Gyr X', alpha=0.7)
    if df_muse_2 is not None:
        plt.plot(df_muse_2['Timestamp'], df_muse_2['Gyr.X'], label='muse_2 - Gyr X', alpha=0.7)
    if df_mitch_1 is not None:
        plt.plot(df_mitch_1['Timestamp'], df_mitch_1['Gyr.X'], label='Mitch_1 - Gyr X', alpha=0.7)
    if df_mitch_2 is not None:
        plt.plot(df_mitch_2['Timestamp'], df_mitch_2['Gyr.X'], label='Mitch_2 - Gyr X', alpha=0.7)
    plt.legend()
    plt.title("Gyroscope X Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Angular Velocity (°/s)")
    plt.grid(True)
    plt.show()

    # Plot Gyro Y
    plt.figure(figsize=(12, 5))
    if df_muse_1 is not None:
        plt.plot(df_muse_1['Timestamp'], df_muse_1['Gyr.Y'], label='muse_1 - Gyr Y', alpha=0.7)
    if df_muse_2 is not None:
        plt.plot(df_muse_2['Timestamp'], df_muse_2['Gyr.Y'], label='muse_2 - Gyr Y', alpha=0.7)
    if df_mitch_1 is not None:
        plt.plot(df_mitch_1['Timestamp'], df_mitch_1['Gyr.Y'], label='Mitch_1 - Gyr Y', alpha=0.7)
    if df_mitch_2 is not None:
        plt.plot(df_mitch_2['Timestamp'], df_mitch_2['Gyr.Y'], label='Mitch_2 - Gyr Y', alpha=0.7)
    plt.legend()
    plt.title("Gyroscope Y Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Angular Velocity (°/s)")
    plt.grid(True)
    plt.show()

    # Plot Gyro Z
    plt.figure(figsize=(12, 5))
    if df_muse_1 is not None:
        plt.plot(df_muse_1['Timestamp'], df_muse_1['Gyr.Z'], label='muse_1 - Gyr Z', alpha=0.7)
    if df_muse_2 is not None:    
        plt.plot(df_muse_2['Timestamp'], df_muse_2['Gyr.Z'], label='muse_2 - Gyr Z', alpha=0.7)
    if df_mitch_1 is not None:
        plt.plot(df_mitch_1['Timestamp'], df_mitch_1['Gyr.Z'], label='Mitch_1 - Gyr Z', alpha=0.7)
    if df_mitch_2 is not None:
        plt.plot(df_mitch_2['Timestamp'], df_mitch_2['Gyr.Z'], label='Mitch_2 - Gyr Z', alpha=0.7)
    plt.legend()
    plt.title("Gyroscope Z Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Angular Velocity (°/s)")
    plt.grid(True)
    plt.show()

    # Create subplots
    non_none_count = sum(x is not None for x in df_list)
    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(non_none_count, 1, figsize=(14, 5), sharex=True)

    # Gyr. muse_1
    if df_muse_1 is not None:
        axs[0].plot(df_muse_1['Timestamp'], df_muse_1['Gyr.X'], label='Gyr X', alpha=0.7)
        axs[0].plot(df_muse_1['Timestamp'], df_muse_1['Gyr.Y'], label='Gyr Y', alpha=0.7)
        axs[0].plot(df_muse_1['Timestamp'], df_muse_1['Gyr.Z'], label='Gyr Z', alpha=0.7)
        axs[0].set_ylabel("Arm")
        axs[0].legend()
        axs[0].grid(True)

    # Gyr muse_2
    if df_muse_2 is not None:
        axs[1].plot(df_muse_2['Timestamp'], df_muse_2['Gyr.X'], label='Gyr X', alpha=0.7)
        axs[1].plot(df_muse_2['Timestamp'], df_muse_2['Gyr.Y'], label='Gyr Y', alpha=0.7)
        axs[1].plot(df_muse_2['Timestamp'], df_muse_2['Gyr.Z'], label='Gyr Z', alpha=0.7)
        axs[1].set_ylabel("Back")
        axs[1].set_xlabel("Timestamp")
        axs[1].legend()
        axs[1].grid(True)

    
    # Gyr mitch
    if df_mitch_1 is not None:
        axs[2].plot(df_mitch_1['Timestamp'], df_mitch_1['Gyr.X'], label='Gyr X', alpha=0.7)
        axs[2].plot(df_mitch_1['Timestamp'], df_mitch_1['Gyr.Y'], label='Gyr Y', alpha=0.7)
        axs[2].plot(df_mitch_1['Timestamp'], df_mitch_1['Gyr.Z'], label='Gyr Z', alpha=0.7)
        axs[2].set_ylabel("mitch_1")
        axs[2].set_xlabel("Timestamp")
        axs[2].legend()
        axs[2].grid(True)

    # Gyr mitch
    if df_mitch_2 is not None:
        axs[3].plot(df_mitch_2['Timestamp'], df_mitch_2['Gyr.X'], label='Gyr X', alpha=0.7)
        axs[3].plot(df_mitch_2['Timestamp'], df_mitch_2['Gyr.Y'], label='Gyr Y', alpha=0.7)
        axs[3].plot(df_mitch_2['Timestamp'], df_mitch_2['Gyr.Z'], label='Gyr Z', alpha=0.7)
        axs[3].set_ylabel("mitch_2")
        axs[3].set_xlabel("Timestamp")
        axs[3].legend()
        axs[3].grid(True)
    

    # Title and layout
    fig.suptitle("Gyroscope Data (mdps) from Muse sensors", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_mag_muse(df_list):
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

    # Create subplots
    non_none_count = sum(x is not None for x in df_list)
    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(non_none_count, 1, figsize=(14, 5), sharex=True)

    # Gyr. muse_1
    if df_muse_1 is not None:
        axs[0].plot(df_muse_1['Timestamp'], df_muse_1['Mag.X'], label='Mag X', alpha=0.7)
        axs[0].plot(df_muse_1['Timestamp'], df_muse_1['Mag.Y'], label='Mag Y', alpha=0.7)
        axs[0].plot(df_muse_1['Timestamp'], df_muse_1['Mag.Z'], label='Mag Z', alpha=0.7)
        axs[0].set_ylabel("Arm")
        axs[0].legend()
        axs[0].grid(True)

    # Gyr muse_2
    if df_muse_2 is not None:
        axs[1].plot(df_muse_2['Timestamp'], df_muse_2['Mag.X'], label='Mag X', alpha=0.7)
        axs[1].plot(df_muse_2['Timestamp'], df_muse_2['Mag.Y'], label='Mag Y', alpha=0.7)
        axs[1].plot(df_muse_2['Timestamp'], df_muse_2['Mag.Z'], label='Mag Z', alpha=0.7)
        axs[1].set_ylabel("Back")
        axs[1].set_xlabel("Timestamp")
        axs[1].legend()
        axs[1].grid(True)

    
    # Gyr mitch
    if df_mitch_1 is not None:
        axs[2].plot(df_mitch_1['Timestamp'], df_mitch_1['Gyr.X'], label='Gyr X', alpha=0.7)
        axs[2].plot(df_mitch_1['Timestamp'], df_mitch_1['Gyr.Y'], label='Gyr Y', alpha=0.7)
        axs[2].plot(df_mitch_1['Timestamp'], df_mitch_1['Gyr.Z'], label='Gyr Z', alpha=0.7)
        axs[2].set_ylabel("mitch_1")
        axs[2].set_xlabel("Timestamp")
        axs[2].legend()
        axs[2].grid(True)

    # Gyr mitch
    if df_mitch_2 is not None:
        axs[3].plot(df_mitch_2['Timestamp'], df_mitch_2['Gyr.X'], label='Gyr X', alpha=0.7)
        axs[3].plot(df_mitch_2['Timestamp'], df_mitch_2['Gyr.Y'], label='Gyr Y', alpha=0.7)
        axs[3].plot(df_mitch_2['Timestamp'], df_mitch_2['Gyr.Z'], label='Gyr Z', alpha=0.7)
        axs[3].set_ylabel("mitch_2")
        axs[3].set_xlabel("Timestamp")
        axs[3].legend()
        axs[3].grid(True)
    

    # Title and layout
    fig.suptitle("Magnetometer Data (mG) from Muse sensors", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_hdr_muse(df_list):
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

    # Create subplots
    non_none_count = sum(x is not None for x in df_list)
    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(non_none_count, 1, figsize=(14, 5), sharex=True)

    # Gyr. muse_1
    if df_muse_1 is not None:
        axs[0].plot(df_muse_1['Timestamp'], df_muse_1['Hdr.X'], label='Hdr X', alpha=0.7)
        axs[0].plot(df_muse_1['Timestamp'], df_muse_1['Hdr.Y'], label='Hdr Y', alpha=0.7)
        axs[0].plot(df_muse_1['Timestamp'], df_muse_1['Hdr.Z'], label='Hdr Z', alpha=0.7)
        axs[0].set_ylabel("Arm")
        axs[0].legend()
        axs[0].grid(True)

    # Gyr muse_2
    if df_muse_2 is not None:
        axs[1].plot(df_muse_2['Timestamp'], df_muse_2['Hdr.X'], label='Hdr X', alpha=0.7)
        axs[1].plot(df_muse_2['Timestamp'], df_muse_2['Hdr.Y'], label='Hdr Y', alpha=0.7)
        axs[1].plot(df_muse_2['Timestamp'], df_muse_2['Hdr.Z'], label='Hdr Z', alpha=0.7)
        axs[1].set_ylabel("Back")
        axs[1].set_xlabel("Timestamp")
        axs[1].legend()
        axs[1].grid(True)

    
    # Gyr mitch
    if df_mitch_1 is not None:
        axs[2].plot(df_mitch_1['Timestamp'], df_mitch_1['Gyr.X'], label='Gyr X', alpha=0.7)
        axs[2].plot(df_mitch_1['Timestamp'], df_mitch_1['Gyr.Y'], label='Gyr Y', alpha=0.7)
        axs[2].plot(df_mitch_1['Timestamp'], df_mitch_1['Gyr.Z'], label='Gyr Z', alpha=0.7)
        axs[2].set_ylabel("mitch_1")
        axs[2].set_xlabel("Timestamp")
        axs[2].legend()
        axs[2].grid(True)

    # Gyr mitch
    if df_mitch_2 is not None:
        axs[3].plot(df_mitch_2['Timestamp'], df_mitch_2['Gyr.X'], label='Gyr X', alpha=0.7)
        axs[3].plot(df_mitch_2['Timestamp'], df_mitch_2['Gyr.Y'], label='Gyr Y', alpha=0.7)
        axs[3].plot(df_mitch_2['Timestamp'], df_mitch_2['Gyr.Z'], label='Gyr Z', alpha=0.7)
        axs[3].set_ylabel("mitch_2")
        axs[3].set_xlabel("Timestamp")
        axs[3].legend()
        axs[3].grid(True)
    

    # Title and layout
    fig.suptitle("HDR accelerometer Data (mg) from Muse sensors", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
