from data_exchange import save_to_numpy, load_numpy_data
from ccsubsample.subsampling.utils import *
import pickle

def load_qm9_data():
    data = pickle.load(open("./data/QM9_train_torch_data.p", "rb"))
    return data

def main():
    # data = load_qm9_data()
    # fingerprints, fingerprints_to_image_index = extract_fingerprints_with_image_indices(data)
    # save_to_numpy(fingerprints, "qm9_gmp_fingerprints")
    qm9_data = load_numpy_data("qm9_gmp_fingerprints")
    print(1)

if __name__ == "__main__":
    main()