import pandas as pd

from ExtractIMU_Features import ExtractIMU_Features
from ExtractPressure_Features import ExtractPressure_Features
from get_paths import get_test_file_paths, get_one_foler_path
import time

"Windowing using all-or-nothing strategy"


def run_feature_extraction(df_muse_arm, 
                           df_muse_back, 
                           df_mitch_left, 
                           df_mitch_right, 
                           window_length_sec, 
                           norm_IMU, 
                           mean_fsr,
                           hdr,
                           output_path = None):

    window_length_samples_muse = int(window_length_sec * 800)
    window_length_samples_mitch = int(window_length_sec * 100)

    # Get features from each IMU source
    feat_muse_arm = ExtractIMU_Features(df_muse_arm, "arm", window_length_samples_muse, norm_IMU, 800, hdr)
    feat_muse_back = ExtractIMU_Features(df_muse_back, "back", window_length_samples_muse, norm_IMU, 800, hdr)

    # Get features from each FSR source
    feat_mitch_left = ExtractPressure_Features(df_mitch_left, "left", window_length_samples_mitch, mean_fsr, 100)
    feat_mitch_right = ExtractPressure_Features(df_mitch_right, "right", window_length_samples_mitch, mean_fsr, 100)

    # Label the windows using one of the IMU or FSR sources (assuming they are time-aligned)
    window_labels, total_windows, ambiguous_windows = label_windows(df_muse_arm, window_length_samples_muse)

    # Combine features
    min_len = min(len(feat_muse_arm), len(feat_muse_back), len(feat_mitch_left), len(feat_mitch_right), len(window_labels))
#    min_len = min(len(feat_muse_arm), len(feat_muse_back), len(window_labels))

    all_features = pd.concat([
        feat_muse_arm[:min_len],
        feat_muse_back[:min_len],
        feat_mitch_left[:min_len],
        feat_mitch_right[:min_len]
    ], axis=1)
    all_features['label'] = window_labels[:min_len]

    # Remove windows where label was ambiguous
    all_features = all_features.dropna(subset=['label']).reset_index(drop=True)

    # Count how many windows are dropped
    before_drop = len(all_features)
    all_features = all_features.dropna(subset=['label']).reset_index(drop=True)
    after_drop = len(all_features)
    dropped = before_drop - after_drop

    print(f"Total windows created: {total_windows}")
    print(f"Ambiguous windows skipped: {ambiguous_windows}")
    print(f"Windows kept (with clean labels): {after_drop}")

    if output_path is not None:
        all_features.to_csv(output_path, index=False)

    return all_features


def label_windows(data, window_length_samples):
    labels = []
    num_windows = len(data) // window_length_samples
    ambiguous_count = 0
    for i in range(num_windows):
        start_idx = i * window_length_samples
        end_idx = start_idx + window_length_samples
        window_labels = data['label'][start_idx:end_idx]
        unique_labels = window_labels.unique()
        if len(unique_labels) == 1:
            labels.append(unique_labels[0])
        else:
            labels.append(None)
            ambiguous_count += 1
    return labels, num_windows, ambiguous_count


def run_feature_extraction_for_all_tests(window_length_sec=2, norm_IMU=True, mean_fsr=False, hdr=False):
    file_dict = get_test_file_paths()
    start = time.time()
    for test_id, paths in file_dict.items():
        print(f"\n--- Running feature extraction for {test_id} ---")
        try:
            df_arm = pd.read_csv(paths["arm"])
            df_back = pd.read_csv(paths["back"])
            df_left = pd.read_csv(paths["left"])
            df_right = pd.read_csv(paths["right"])

            folder = get_one_foler_path(test_id)
            print(f"Folder to save features: {folder}")

            output_path = f"{folder}/{test_id}_features_4sensors_window{window_length_sec}_norm{'T' if norm_IMU else 'F'}_mean{'T' if mean_fsr else 'F'}_hdr{'T' if hdr else 'F'}.csv"
#            output_path = f"{folder}/{test_id}_features_4sensors_window{window_length_sec}_norm{'T' if norm_IMU else 'F'}_no_fsr_hdr{'T' if hdr else 'F'}.csv"
            
            run_feature_extraction(df_arm, df_back, df_left, df_right,
                                   window_length_sec, norm_IMU, mean_fsr, hdr,
                                   output_path=output_path)
#            run_feature_extraction(df_arm, df_back, None, None,
#                                   window_length_sec, norm_IMU, mean_fsr, hdr,
#                                   output_path=output_path)
        except Exception as e:
            print(f"Failed for {test_id}: {e}")

    end = time.time()
    elapsed = end - start
    print(f"\n🕒 Done! Total time uesd: {elapsed:.2f} seconds")



run_feature_extraction_for_all_tests(window_length_sec=8, norm_IMU=False, mean_fsr=True, hdr=False)

