import json
import numpy as np


def save_to_numpy(data: np.ndarray, filename: str):
    out_file = open("./data/{}.npy".format(filename), "wb")
    np.save(out_file, data, allow_pickle=False)


def load_numpy_data(filename: str) -> np.ndarray:
    in_file = open("./data/{}.npy".format(filename), "rb")
    return np.load(in_file, allow_pickle=False)