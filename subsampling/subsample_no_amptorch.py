from ccsubsample.subsampling.utils import *
from ccsubsample.subsampling.subsampling import kdtree_subsample


def load_oc20_3k_data():
    data = pickle.load(open("../data/oc20_train_torch_data.p", "rb"))
    return data


def save_oc20_3k_fingerprints_to_json():
    data = load_oc20_3k_data()
    fingerprints, fingerprints_to_image_index = extract_fingerprints_with_image_indices(data)
    fp_list = fingerprints.tolist()
    fp_index_list = fingerprints_to_image_index.tolist()
    oc20_3k_fingerprints_out = open("../data/oc20_3k_fingerprints.json", "w")
    json.dump(fp_list, oc20_3k_fingerprints_out, indent=4)
    oc20_3k_fingerprints_index_out = open("../dataoc20_3k_fingerprints_index.json", "w")
    json.dump(fp_index_list, oc20_3k_fingerprints_index_out, indent=4)


if __name__ == "__main__":
    data = load_oc20_3k_data()
    fingerprints, fingerprints_to_image_index = extract_fingerprints_with_image_indices(data)
    reduced = reduce_dimensions_with_pca(fingerprints, max_components=6)
    scaled = scale_and_standardize_data(reduced)
    
    data_indices_to_keep = kdtree_subsample(scaled, cutoff_sig=0.3)
    image_indices_to_keep = get_image_indices_to_keep(data_indices_to_keep, fingerprints_to_image_index)
    