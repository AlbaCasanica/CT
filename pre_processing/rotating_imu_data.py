import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Estimate mean gravity vectors from accelerometer
def compute_mean_gravity(file_standing):
    if isinstance(file_standing, str):
        df = pd.read_csv(file_standing)
        print("file_standing is string")
    elif isinstance(file_standing, pd.DataFrame):
        df = file_standing.copy()
        print("file_standing is DataFrame")
    else:
        print("Input is neither a filepath or a DataFram")

    acc = df[["Axl.X", "Axl.Y", "Axl.Z"]].values
    average_acc = np.mean(acc, axis = 0)
    norm_average_acc = average_acc / np.linalg.norm(average_acc)
    return norm_average_acc


# Compute rotation matrix to align with global Z axis (gravity)
def compute_rotation(g_vec):
    z_world = np.array([0, 0, -1])
    # Direction of rotation:
    rotation_axis = np.cross(g_vec, z_world) #takes corss product of two arrays
    # how far apart they are:
    cos_angle = np.dot(g_vec, z_world) #dot product of two arrays -- cosine_angle = cos(v) -> v angle between the vectors
    # How "twisted" they are
    sin_angle = np.linalg.norm(rotation_axis) # since |rotation_axis| = sin(v)

    # skew-symmetric matrix
    kmat = np.array([[                0, -rotation_axis[2],  rotation_axis[1]],
                     [ rotation_axis[2],                 0, -rotation_axis[0]],
                     [-rotation_axis[1],  rotation_axis[0],                 0]])
    
    # Rodrigues’ formula
    R = np.eye(3) + kmat + kmat @ kmat * ((1-cos_angle) / (sin_angle**2 + 1e-8))

    return R


def rotate_sensor_data(filepath, R, prefixes=["Axl", "Gyr", "Mag"]):
    df = pd.read_csv(filepath)
    
    df_rot = df.copy()
    for prefix in prefixes:
        data = df[[f"{prefix}.X", f"{prefix}.Y", f"{prefix}.Z"]].values
        df_rot[[f"{prefix}.X", f"{prefix}.Y", f"{prefix}.Z"]] = data @ R.T
    
    output_path = f"{filepath.rstrip(".csv")}_rotated.csv"
    df_rot.to_csv(output_path, index=False)

    return df_rot, output_path


def plot_z_axis_before_and_after(filepath_arm, filepath_back, df_arm_rotated, df_back_rotated):
    df_arm = pd.read_csv(filepath_arm)
    df_back = pd.read_csv(filepath_back)

    
    plt.figure(figsize=(14, 6))

    plt.subplot(2, 1, 1)
    plt.plot(df_back["ReconstructedTime"], df_back["Axl.Z"], label="Original Axl.Z (Back)")
    plt.plot(df_back["ReconstructedTime"], df_back_rotated["Axl.Z"], label="Rotated Axl.Z (Back)", linestyle='--')
    plt.title("Back IMU Accelerometer Z-Axis (Before and After Rotation)")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (mg)")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(df_arm["ReconstructedTime"], df_arm["Axl.Z"], label="Original Axl.Z (Arm)")
    plt.plot(df_arm["ReconstructedTime"], df_arm_rotated["Axl.Z"], label="Rotated Axl.Z (Arm)", linestyle='--')
    plt.title("Arm IMU Accelerometer Z-Axis (Before and After Rotation)")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (mg)")
    plt.legend()

    plt.tight_layout()
    plt.show()

"""
filepath_standing_arm = ""
filepath_standing_back = ""

g_arm = compute_mean_gravity(filepath_standing_arm)
g_back = compute_mean_gravity(filepath_standing_back)

R_arm = compute_rotation(g_arm)
R_back = compute_rotation(g_back)

filepath_arm = ""
filepath_back = ""

df_arm_rotated = rotate_sensor_data(filepath_arm, R_arm)
df_back_rotated = rotate_sensor_data(filepath_back, R_back)

plot_z_axis_before_and_after(filepath_arm, filepath_back, df_arm_rotated, df_back_rotated)
"""