import os

def get_test_file_paths():
    tests = {
        "test_1": {
            "arm": "",
            "back": "",
            "left": "",
            "right": ""
        },
        "test_2": {
            "arm": "",
            "back": "",
            "left": "",
            "right": ""
        },
        "test_3": {
            "arm": "",
            "back": "",
            "left": "",
            "right": ""
        },
        "test_4": {
            "arm": "",
            "back": "",
            "left": "",
            "right": ""
        },
        "test_5": {
            "arm": "",
            "back": "",
            "left": "",
            "right": ""
        },
        "test_6": {
            "arm": "",
            "back": "",
            "left": "",
            "right": ""
        },
        "test_7": {
            "arm": "",
            "back": "",
            "left": "",
            "right": ""
        },
        "test_8": {
            "arm": "",
            "back": "",
            "left": "",
            "right": ""
        },
        "test_9": {
            "arm": "",
            "back": "",
            "left": "",
            "right": ""
        },
        "test_10": {
            "arm": "",
            "back": "",
            "left": "",
            "right": ""
        },
        "test_11": {
            "arm": "",
            "back": "",
            "left": "",
            "right": ""
        },
        "test_12": {
            "arm": "",
            "back": "",
            "left": "",
            "right": ""
        }
    }
    return tests


def get_test_folder_paths():
    folders = {
        "test_1": "",
        "test_2": "",
        "test_3": "",
        "test_4": "",
        "test_5": "",
        "test_6": "",
        "test_7": "",
        "test_8": "",
        "test_9": "",
        "test_10": "",
        "test_11": "",
        "test_12": ""
    }
    return folders


def get_one_test(test_number):
    test_to_get = "test_" + str(test_number)
    return get_test_file_paths()[test_to_get]

def get_one_file(test_number, sensor):
    test_nr_to_get = "test_" + str(test_number)
    return get_test_file_paths()[test_nr_to_get][sensor]

def get_one_foler_path(test_number):
    if isinstance(test_number, int):
        test_number = "test_" + str(test_number)
    return get_test_folder_paths()[test_number]


def get_feture_paths(window_length_sec=4, norm_IMU=True, mean_fsr=False, hdr=False):
    folders = get_test_folder_paths()
    feature_files = {}

    for test_id, folder in folders.items():
        if mean_fsr is None:
            filename = f"{test_id}_features_4sensors_window{window_length_sec}_norm{'T' if norm_IMU else 'F'}_no_fsr_hdr{'T' if hdr else 'F'}.csv"
        else:
            filename = f"{test_id}_features_4sensors_window{window_length_sec}_norm{'T' if norm_IMU else 'F'}_mean{'T' if mean_fsr else 'F'}_hdr{'T' if hdr else 'F'}.csv"
        full_path = os.path.join(folder, filename)
        feature_files[test_id] = full_path

    return feature_files
