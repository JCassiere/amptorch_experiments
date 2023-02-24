import numpy as np

from data_exchange import load_numpy_data, save_experiment
from ccsubsample.subsampling.utils import *
from ccsubsample.subsampling.evaluation_metrics import *
from ccsubsample.subsampling.subsampling import faiss_flat_subsample, faiss_ivf_subsample, faiss_ivfpq_subsample, kmeans_subsample


def get_experiment_metadata(subsampled_data: np.ndarray, subsampled_indices: np.ndarray, outlier_indices: np.ndarray,
                            wallclock_time: float, experiment_title: str, outlier_cutoff_modifier: float = 2) -> dict:
    num_points = number_of_points_remaining(subsampled_data)
    _, std_dev = point_diversity_mean_std(subsampled_data)
    outlier_retention = calculate_outlier_retention(outlier_indices, subsampled_indices, outlier_cutoff_modifier)
    subset_filename = experiment_title + "_subset"
    experiment_metadata = {
        "experiment_title": experiment_title,
        "wallclock_time": wallclock_time,
        "num_points": num_points,
        "subset_filename": subset_filename,
        "std_dev": float(std_dev),
        "outlier_retention": outlier_retention
    }
    return experiment_metadata

    
def random_subsampling_experiment(scaled_data, experiment_title, outlier_indices, num_points_desired):
    start = time.time()
    random_indices = np.random.choice(np.arange(scaled_data.shape[0]), num_points_desired)
    random_subsample = scaled_data[random_indices]
    wallclock_time = time.time() - start
    experiment_metadata = get_experiment_metadata(random_subsample, random_indices, outlier_indices, wallclock_time, experiment_title)
    save_experiment(experiment_title, random_subsample, experiment_metadata)


def kmeans_subsampling_experiment(scaled_data, experiment_title, outlier_indices, num_points_desired):
    start = time.time()
    indices_to_keep = kmeans_subsample(scaled_data, num_points_desired)
    wallclock_time = time.time() - start
    points_to_keep = scaled_data[indices_to_keep]
    experiment_metadata = get_experiment_metadata(points_to_keep, indices_to_keep, outlier_indices, wallclock_time, experiment_title)
    save_experiment(experiment_title, points_to_keep, experiment_metadata)


def faiss_subsampling_experiment(scaled_data, subsample_fn, experiment_title, outlier_indices, cutoff):
    start = time.time()
    indices_to_keep = subsample_fn(scaled_data, cutoff_sig=cutoff, verbose=2)
    wallclock_time = time.time() - start
    points_to_keep = scaled_data[indices_to_keep]
    experiment_metadata = \
        get_experiment_metadata(points_to_keep, indices_to_keep, outlier_indices, wallclock_time, experiment_title) | {"cutoff": cutoff}
    save_experiment(experiment_title, points_to_keep, experiment_metadata)


def dimensionality_reduction_experiment(scaled_data: np.ndarray, base_title: str, outlier_indices: np.ndarray,
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
        additional_metadata = {
            "num_dimensions_kept": i,
            "variance_retained": np.sum(explained_variance_ratio[:i]),
            "cutoff": cutoff
        }
        experiment_metadata = \
            get_experiment_metadata(points_to_keep, indices_to_keep, outlier_indices, wallclock_time, title) | additional_metadata
        save_experiment(title, points_to_keep, experiment_metadata, second_level_dir=second_level_dir)
        

def main():
    # TODO - just find outliers ahead of time so you don't have to calculate it every time
    #  will need to refactor experiment functions
    scaled_data = load_numpy_data("qm9_train_gmp_fingerprints")
    outlier_indices = get_outlier_indices(scaled_data, outlier_cutoff_modifier=2)
    random_subsampling_experiment(scaled_data, "qm9_random_subsample", outlier_indices, 20000)
    kmeans_subsampling_experiment(scaled_data, "qm9_kmeans_subsample", outlier_indices, 20000)
    faiss_subsampling_experiment(scaled_data, faiss_flat_subsample, "qm9_faiss_flat_subsample", outlier_indices, cutoff=2)
    faiss_subsampling_experiment(scaled_data, faiss_ivf_subsample, "qm9_faiss_ivf_subsample", outlier_indices, cutoff=2)
    faiss_subsampling_experiment(scaled_data, faiss_ivfpq_subsample, "qm9_faiss_ivfpq_subsample", outlier_indices, cutoff=2)
    dimensionality_reduction_experiment(scaled_data, "qm9_gmp_ivf", outlier_indices)

    
if __name__ == "__main__":
    main()
