import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_dtw_heatmap(df_results, title):
    # Get all unique rep_ids
    rep_ids = sorted(set(df_results['rep_i']).union(set(df_results['rep_j'])))
    
    matrix = pd.DataFrame(index=rep_ids, columns=rep_ids, data=np.nan)

    # Fill in DTW distances
    for _, row in df_results.iterrows():
        i, j, dtw = int(row['rep_i']), int(row['rep_j']), row['dtw']
        matrix.loc[i, j] = dtw
        matrix.loc[j, i] = dtw  # make symmetric
        matrix.loc[i, i] = 0
        matrix.loc[j, j] = 0

    plt.figure(figsize=(5, 4.3))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", annot_kws={"size": 10})
    plt.title(title)
    plt.xlabel("Repetition")
    plt.ylabel("Repetition")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_dtw_heatmap2(df_results, title):
    # Define desired sensor order
    sensor_order = ['IMU arm', 'IMU back', 'FSR left']
    
    # Create empty symmetric matrix
    matrix = pd.DataFrame(index=sensor_order, columns=sensor_order, data=np.nan)

    # Fill DTW values symmetrically
    for _, row in df_results.iterrows():
        i, j, dtw = row['sensor_i'], row['sensor_j'], row['dtw']
        matrix.loc[i, j] = dtw
        matrix.loc[j, i] = dtw  # symmetry

    # Plot
    plt.figure(figsize=(4, 3))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True, annot_kws={"size": 12})
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_dtw_boxplot(axl_df, gyro_df):
    axl_df['sensor'] = 'Accelerometer'
    gyro_df['sensor'] = 'Gyroscope'

    combined = pd.concat([axl_df[['dtw', 'sensor']], gyro_df[['dtw', 'sensor']]])

    plt.figure(figsize=(6, 5))
    sns.boxplot(x='sensor', y='dtw', data=combined)
    plt.title("DTW Norm Distance Distribution")
    plt.ylabel("DTW Distance")
    plt.show()

