import json
import numpy as np


def save_to_numpy(data: np.ndarray, filename: str, prefix: str = "data") -> None:
    out_file = open("./{}/{}.npy".format(prefix, filename), "wb")
    np.save(out_file, data, allow_pickle=False)


def load_numpy_data(filename: str, prefix: str = "data") -> np.ndarray:
    in_file = open("./{}/{}.npy".format(prefix, filename), "rb")
    return np.load(in_file, allow_pickle=False)
