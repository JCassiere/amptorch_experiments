import numpy as np
import os
import time
import json


def save_to_numpy(data: np.ndarray, filename: str, prefix: str = "data") -> None:
    out_file = open("./{}/{}.npy".format(prefix, filename), "wb")
    np.save(out_file, data, allow_pickle=False)


def load_numpy_data(filename: str, prefix: str = "data") -> np.ndarray:
    in_file = open("./{}/{}.npy".format(prefix, filename), "rb")
    return np.load(in_file, allow_pickle=False)


def create_dir_if_not_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def save_experiment(title: str, subset: np.ndarray, experiment_metadata: json, second_level_dir: str = None):
    create_dir_if_not_exists("results/")
    timestamped_title = title + "_{}".format(time.time())
    if second_level_dir:
        create_dir_if_not_exists("results/" + second_level_dir)
        experiment_dir = "results/{}/{}".format(second_level_dir, timestamped_title)
    else:
        experiment_dir = "results/{}".format(timestamped_title)
    create_dir_if_not_exists(experiment_dir)
    subset_filename = experiment_metadata["subset_filename"]
    save_to_numpy(subset, subset_filename, experiment_dir)
    metadata_file = open(experiment_dir + "/metadata.json", 'w')
    json.dump(experiment_metadata, metadata_file, indent=4)
