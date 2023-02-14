import time

import numpy as np

from data_exchange import save_to_numpy, load_numpy_data, save_experiment
from ccsubsample.subsampling.utils import *
from ccsubsample.subsampling.evaluation_metrics import *
from ccsubsample.subsampling.subsampling import faiss_flat_subsample, faiss_ivf_subsample, faiss_ivfpq_subsample, kmeans_subsample


def random_subsampling_experiment(scaled_data, title, num_points_desired):
    start = time.time()
    random_subsample = scaled_data[np.random.choice(np.arange(scaled_data.shape[0]), num_points_desired)]
    wallclock_time = time.time() - start
    subset_filename = title + "_subset"
    save_to_numpy(random_subsample, subset_filename, "results")
    num_points = number_of_points_remaining(random_subsample)
    _, std_dev = point_diversity_mean_std(random_subsample)
    experiment_metadata = {
        "experiment_title": title,
        "wallclock_time": wallclock_time,
        "num_points": num_points,
        "subset_filename": subset_filename,
        "std_dev": float(std_dev)
    }
    metadata_file = open("results/" + title + "_{}_metadata.json".format(time.time()), "w")
    json.dump(experiment_metadata, metadata_file, indent=4)


def kmeans_subsampling_experiment(scaled_data, title, num_points_desired):
    start = time.time()
    indices_to_keep = kmeans_subsample(scaled_data, num_points_desired)
    wallclock_time = time.time() - start
    points_to_keep = scaled_data[indices_to_keep]
    subset_filename = title + "_subset"
    save_to_numpy(points_to_keep, subset_filename, "results")
    num_points = number_of_points_remaining(points_to_keep)
    _, std_dev = point_diversity_mean_std(points_to_keep)
    experiment_metadata = {
        "experiment_title": title,
        "wallclock_time": wallclock_time,
        "num_points": num_points,
        "subset_filename": subset_filename,
        "std_dev": float(std_dev)
    }
    metadata_file = open("results/" + title + "_{}_metadata.json".format(time.time()), "w")
    json.dump(experiment_metadata, metadata_file, indent=4)


def faiss_subsampling_experiment(scaled_data, subsample_fn, title, cutoff):
    start = time.time()
    indices_to_keep = subsample_fn(scaled_data, cutoff_sig=cutoff, verbose=2)
    wallclock_time = time.time() - start
    points_to_keep = scaled_data[indices_to_keep]
    subset_filename = title + "_subset"
    num_points = number_of_points_remaining(points_to_keep)
    _, std_dev = point_diversity_mean_std(points_to_keep)
    experiment_metadata = {
        "experiment_title": title,
        "cutoff": cutoff,
        "wallclock_time": wallclock_time,
        "num_points": num_points,
        "subset_filename": subset_filename,
        "std_dev": float(std_dev)
    }
    save_experiment(title, points_to_keep, experiment_metadata)

# def load_oc20_3k_data():
#     data = pickle.load(open("./data/oc20_train_torch_data.p", "rb"))
#     return data
#
# def load_qm9_data():
#     data = pickle.load(open("./data/QM9_train_torch_data.p", "rb"))
#     return data


def dimensionality_reduction_experiment(scaled_data: np.ndarray, base_title: str,
                                        faiss_subsample_method=faiss_ivf_subsample, cutoff: float = 2):
    pca = PCA(svd_solver="randomized")
    data_pca = pca.fit_transform(scaled_data)
    explained_variance_ratio = pca.explained_variance_ratio_
    _, dim = data_pca.shape
    
    second_level_dir = base_title + "_dim_reduction_{}".format(time.time())
    for i in range(1, dim):
        start = time.time()
        indices_to_keep = faiss_subsample_method(np.array(data_pca[:, :i], order='C'), cutoff_sig=cutoff, verbose=2)
        wallclock_time = time.time() - start
        points_to_keep = scaled_data[indices_to_keep]
        title = base_title + "_{}_dimensions".format(i)
        subset_filename = title + "_subset"
        num_points = number_of_points_remaining(points_to_keep)
        _, std_dev = point_diversity_mean_std(points_to_keep)
        experiment_metadata = {
            "experiment_title": title,
            "num_dimensions_kept": i,
            "variance_retained": np.sum(explained_variance_ratio[:i]),
            "cutoff": cutoff,
            "wallclock_time": wallclock_time,
            "num_points": num_points,
            "subset_filename": subset_filename,
            "std_dev": float(std_dev)
        }
        save_experiment(title, points_to_keep, experiment_metadata, second_level_dir=second_level_dir)


def main():
    scaled_data = load_numpy_data("scaled_and_std_oc20_3k_gmp_fingerprints")
    dimensionality_reduction_experiment(scaled_data, "oc20_3k_gmp_ivf")
    random_subsampling_experiment(scaled_data, "oc20_3k_random_subsample", 20000)
    kmeans_subsampling_experiment(scaled_data, "oc20_3k_kmeans_subsample", 20000)
    faiss_subsampling_experiment(scaled_data, faiss_flat_subsample, "oc20_3k_faiss_flat_subsample", cutoff=2)
    faiss_subsampling_experiment(scaled_data, faiss_ivf_subsample, "oc20_3k_faiss_ivf_subsample", cutoff=2)
    faiss_subsampling_experiment(scaled_data, faiss_ivfpq_subsample, "oc20_3k_faiss_ivfpq_subsample", cutoff=2)


if __name__ == "__main__":
    main()