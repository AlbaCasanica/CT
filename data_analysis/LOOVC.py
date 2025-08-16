import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import os
import seaborn as sns
from matplotlib.patches import Rectangle

from get_paths import get_feture_paths
from run_SVC import run_SVC
from run_RFC import run_RFC
from run_NN import run_NN

def run_loocv_with_pca(label_mapping=None, clf_name = "", window_size=8, norm_IMU=True, mean_fsr=True, hdr=False, class_version=1):
    feature_files = get_feture_paths(window_length_sec=window_size, norm_IMU=norm_IMU, mean_fsr=mean_fsr, hdr=hdr)
    test_ids = list(feature_files.keys())

    all_accuracies = []
    all_f1 = []
    all_precision = []
    all_recall = []
    all_Y_true = []
    all_Y_pred = []

    start = time.time()
    for leave_out in test_ids:
        print(f"\n Testing on {leave_out}...")

        # Split training and test sets
        train_dfs = [pd.read_csv(path) for test_id, path in feature_files.items() if test_id != leave_out]
        test_df = pd.read_csv(feature_files[leave_out])

        # Combine training sets
        train_df = pd.concat(train_dfs, ignore_index=True)

        # Separate features and labels
        X_train = train_df.drop(columns=["label"])
        Y_train = train_df["label"]

        X_test = test_df.drop(columns=["label"])
        Y_test = test_df["label"]

        if label_mapping is not None:
            Y_train = train_df["label"].map(label_mapping)
            Y_test = test_df["label"].map(label_mapping)
            if label_mapping == label_mapping_v2:
                labels = ["hands_up", "push_pull", "squatting", "lifting", "sit_stand", "walking"]
            elif label_mapping == label_mapping_v3:
                labels = ["hands_up", "push_pull_lift", "squat", "sit", "stand_walk"]
        else:    
            labels = ["hand_up_back", "hands_forward", "hands_up", "push", "pull", "squatting", "lifting", "sitting", "standing", "walking"]

        # Scale data (SD of 1 and mean of 0)
        scaler = StandardScaler().set_output(transform="pandas")
        # Fit scaler on training data
        scaler.fit(X_train)
        # Transfor both train and test set with the scaler
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Apply pca
        pca = PCA(n_components=0.95)
        pca_fit = pca.fit(X_train_scaled)
        pca_components = pca.n_components_
        print (f"pca components: {pca_components}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum()}")

        X_train_pca = pca_fit.transform(X_train_scaled)
        X_test_pca = pca_fit.transform(X_test_scaled)

        if clf_name == "SVC":
            test_results, train_results = run_SVC(X_train_pca, Y_train, X_test_pca, Y_test, class_names=labels, CV_suffix=f"{leave_out}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}", opt=True, time_window=f"{window_size}_sec")
        elif clf_name == "RFC":
            test_results, train_results = run_RFC(X_train_pca, Y_train, X_test_pca, Y_test, class_names=labels, CV_suffix=f"{leave_out}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}", opt=True, time_window=f"{window_size}_sec")
        elif clf_name == "NN":
            test_results, train_results = run_NN(X_train_pca, Y_train, X_test_pca, Y_test, class_names=labels, CV_suffix=f"{leave_out}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}", opt=True, time_window=f"{window_size}_sec")
        else:
            print("Model name not possible, model set automatic to svc")
            test_results, train_results = run_SVC(X_train_pca, Y_train, X_test_pca, Y_test, class_names=labels, CV_suffix=f"{leave_out}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}", opt=True, time_window=f"{window_size}_sec")
        Y_test_fit = test_results[0]
        accuracy_test = test_results[1]
        f1_test = test_results[2]
        precision_test  = test_results[3]
        recall_test = test_results[4]

        accuracy_train = train_results[1]
        f1_test = test_results[2]
        precision_test  = test_results[3]
        recall_test = test_results[4]


        # Evaluate
        print(f"Accuracy for {leave_out}: {accuracy_test:.3f}")
        print(f"Precision for {leave_out}: {precision_test:.3f}")
        print(f"Recall for {leave_out}: {recall_test:.3f}")
        print(f"F1 for {leave_out}: {f1_test:.3f}")
        #print(f"Accuracy for train {leave_out}: {accuracy_train:.3f}")
        all_accuracies.append(accuracy_test)
        all_f1.append(f1_test)
        all_precision.append(precision_test)
        all_recall.append(recall_test)

        all_Y_true.extend(Y_test)
        all_Y_pred.extend(Y_test_fit)
    
        
    end = time.time()
    elapsed = end - start
    print(f"\n🕒 Done! Total time uesd: {elapsed:.2f} seconds")

    print(f"\n✅ Mean LOOCV accuracy: {np.mean(all_accuracies):.3f}")
    print(f"\n✅ Mean LOOCV f1: {np.mean(all_f1):.3f}")
    print(f"\n✅ Mean LOOCV precision: {np.mean(all_precision):.3f}")
    print(f"\n✅ Mean LOOCV recall: {np.mean(all_recall):.3f}")

    # Confusion matrix
    save_path="./plots_use"
    os.makedirs(save_path, exist_ok=True)
    
    cm = confusion_matrix(all_Y_true, all_Y_pred, labels=labels, normalize='true')
    print("\n🧮 Confusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation='vertical',cmap=plt.cm.Blues, values_format=".2f")
    for text in disp.ax_.texts:
        text.set_fontsize(8)
    plt.tight_layout()
    
    save_file1 = os.path.join(save_path, f"{clf_name}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}_{window_size}_sec_CM1.png")
    plt.savefig(save_file1, dpi=300)
    plt.close()
    
    cm2 = confusion_matrix(all_Y_true, all_Y_pred, labels=labels)
    df_cm2 = pd.DataFrame(cm2, index=labels, columns=labels)

    df_cm2['Total'] = df_cm2.sum(axis=1)
    totals_row = df_cm2.sum(axis=0)
    totals_row.name = 'Total'

    df_cm2 = pd.concat([df_cm2, totals_row.to_frame().T])
    
    n_rows, n_cols = df_cm2.shape

    mask = np.zeros_like(df_cm2, dtype=bool)
    mask[-1, :] = True   # last row (Total Pred)
    mask[:, -1] = True   # last column (Total True)

    if label_mapping is None:
        plt.figure(figsize=(7, 6))
    else:
        plt.figure(figsize=(5.5, 5)) 
    ax = sns.heatmap(df_cm2, annot=True, fmt='.0f', cmap='Blues', mask=mask, cbar=True)

    total_bg_color = '#e0eaf4'  # soft blue-gray to match 'Blues'
    for i in range(n_rows - 1):
        ax.add_patch(Rectangle((n_cols - 1, i), 1, 1, fill=True, color=total_bg_color, lw=0))
    for j in range(n_cols - 1):
        ax.add_patch(Rectangle((j, n_rows - 1), 1, 1, fill=True, color=total_bg_color, lw=0))
    ax.add_patch(Rectangle((n_cols - 1, n_rows - 1), 1, 1, fill=True, color=total_bg_color, lw=0))

    # Manually annotate the total row and column
    for i in range(n_rows - 1):  # all rows except last
        val = df_cm2.iat[i, -1]
        ax.text(n_cols - 0.5, i + 0.5, f'{val:.0f}', ha='center', va='center', color='black', fontsize=9)

    for j in range(n_cols - 1):  # all columns except last
        val = df_cm2.iat[-1, j]
        ax.text(j + 0.5, n_rows - 0.5, f'{val:.0f}', ha='center', va='center', color='black', fontsize=9)

    corner_val = df_cm2.iat[-1, -1]
    ax.text(n_cols - 0.5, n_rows - 0.5, f'{corner_val:.0f}', ha='center', va='center', color='black', fontsize=9)

    plt.title('Confusion Matrix with True and Predicted Totals')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    save_file2 = os.path.join(save_path, f"{clf_name}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}_{window_size}_sec_CM2.png")
    plt.tight_layout()
    plt.savefig(save_file2, dpi=300)
    plt.close()
    
    return all_accuracies

# class version 1: all classes, 2 and 3 are mapped below:
label_mapping_v2 = {
    "hand_up_back"  : "hands_up",
    "hands_forward" : "hands_up",
    "hands_up"      : "hands_up",
    "push"          : "push_pull",
    "pull"          : "push_pull",
    "squatting"     : "squatting",
    "lifting"       : "lifting",
    "sitting"       : "sit_stand",
    "standing"      : "sit_stand",
    "walking"       : "walking"
}

label_mapping_v3 = {
    "hand_up_back"  : "hands_up",
    "hands_forward" : "hands_up",
    "hands_up"      : "hands_up",
    "push"          : "push_pull_lift",
    "pull"          : "push_pull_lift",
    "squatting"     : "squat",
    "lifting"       : "push_pull_lift",
    "sitting"       : "sit",
    "standing"      : "stand_walk",
    "walking"       : "stand_walk"
}
