from dtw import*
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd


def make_repetition_sequences(filepath, activity, sensor_cols):
    df = pd.read_csv(filepath)

    activity_df = df[df["label"] == activity].copy()

    nr_reps = len(activity_df["rep_id"].unique())
    print(f"nr_reps: {nr_reps}")
    

    # Makes sequence for each repetition
    reps = [activity_df[activity_df["rep_id"] == rep_id][sensor_cols].to_numpy() 
            for rep_id in activity_df["rep_id"].unique()]
    
    #df_reps = pd.DataFrame(reps)
    #df_reps.to_csv("test.csv", index=False)

    return reps, nr_reps, activity


def dtw_average_all_reps_norm(reps, nr_reps, activity, folder_path):

        #distance_matrix = np.zeros((nr_reps, nr_reps))
        results = []

        for i in range(nr_reps):
                for j in range(i + 1, nr_reps):
            
#                        alignment = dtw(reps[i], reps[j], distance_only=True)
                        alignment_norm = dtw(np.linalg.norm(reps[i], axis=1), np.linalg.norm(reps[j], axis=1), distance_only=True)
#                        alignment_mean = dtw(np.mean(reps[i], axis=1), np.mean(reps[j], axis=1), distance_only=True)
                        #distance_matrix[i,j] = distance_matrix[j, i] = alignment.distance

                        results.append({
                                'rep_i': i + 1,
                                'rep_j': j + 1,
                                #'dtw_distance': float(alignment.distance),
#                                'dtw_normalized': float(alignment.normalizedDistance),
#                                'dtw_distance_norm': float(alignment_norm.distance),
                                'dtw_normalized_norm': float(alignment_norm.normalizedDistance)

                        })
        
        df_results = pd.DataFrame(results)
        print(df_results)
        
        # Average (exclude NaN if some fails)
        #avg_dtw = df_results['dtw_distance'].mean(skipna=True)
        avg_dtw_normalized = df_results['dtw_normalized_norm'].mean(skipna=True)

        print(f"\nAverage: {avg_dtw_normalized:.2f}")

        
        dtw(np.linalg.norm(reps[2], axis=1), np.linalg.norm(reps[3], axis=1), keep_internals=True, 
                step_pattern=rabinerJuangStepPattern(6, "c")).plot(type="twoway",offset=-2)
#        dtw(reps[3],reps[4], keep_internals=True, 
#                step_pattern=rabinerJuangStepPattern(6, "c")).plot(type="twoway",offset=-2)
        plt.show()
        



cols_axl = ['Axl.X', 'Axl.Y', 'Axl.Z']
cols_gyro = ["Gyr.X", "Gyr.Y", "Gyr.Z"]
fsr_columns = [f"Fsr.{str(sensor).zfill(2)}" for sensor in range(1, 17)]

filepath_arm = ""
filepath_back = ""
filepath_left = ""
filepath_right = ""
activity = ""

reps, reps_nr,_ = make_repetition_sequences(filepath_arm, activity, cols_axl)
dtw_average_all_reps_norm(reps, reps_nr, activity, None)
