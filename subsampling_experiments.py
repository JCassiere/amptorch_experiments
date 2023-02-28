from data_exchange import load_numpy_data, save_experiment
from ccsubsample.subsampling.utils import *
from ccsubsample.subsampling.evaluation_metrics import *
from ccsubsample.subsampling.subsampling import faiss_flat_subsample, faiss_ivf_subsample, faiss_ivfpq_subsample, \
    kmeans_subsample


def get_experiment_metadata(subsampled_data: np.ndarray, wallclock_time: float, experiment_title: str) -> dict:
    num_points = number_of_points_remaining(subsampled_data)
    _, std_dev = point_diversity_mean_std(subsampled_data)
    subset_filename = experiment_title + "_subset"
    experiment_metadata = {
        "experiment_title": experiment_title,
        "wallclock_time": wallclock_time,
        "num_points": num_points,
        "subset_filename": subset_filename,
        "std_dev": float(std_dev),
    }
    return experiment_metadata


def get_experiment_metadata_with_outliers(subsampled_data: np.ndarray, outlier_indices: np.ndarray,
                                          subsampled_indices: np.ndarray,
                                          wallclock_time: float, experiment_title: str,
                                          outlier_cutoff_modifier: float = 2) -> dict:
    outlier_retention = calculate_outlier_retention(outlier_indices, subsampled_indices, outlier_cutoff_modifier)
    experiment_metadata = get_experiment_metadata(subsampled_data, wallclock_time, experiment_title) | {
        "outlier_retention": outlier_retention
    }
    return experiment_metadata


def full_dataset_metadata(scaled_data, dataset_description):
    experiment_title = dataset_description + "_full_dataset"
    experiment_metadata = get_experiment_metadata(scaled_data, 0, experiment_title)
    save_experiment(experiment_title, scaled_data, experiment_metadata)

def random_subsampling_experiment(scaled_data, experiment_title, outlier_indices, num_points_desired):
    start = time.time()
    random_indices = np.random.choice(np.arange(scaled_data.shape[0]), num_points_desired)
    random_subsample = scaled_data[random_indices]
    wallclock_time = time.time() - start
    experiment_metadata = get_experiment_metadata_with_outliers(random_subsample, outlier_indices, random_indices,
                                                                wallclock_time, experiment_title)
    save_experiment(experiment_title, random_subsample, experiment_metadata)


def kmeans_subsampling_experiment(scaled_data, experiment_title, outlier_indices, num_points_desired):
    start = time.time()
    indices_to_keep = kmeans_subsample(scaled_data, num_points_desired)
    wallclock_time = time.time() - start
    points_to_keep = scaled_data[indices_to_keep]
    experiment_metadata = get_experiment_metadata_with_outliers(points_to_keep, outlier_indices, indices_to_keep,
                                                                wallclock_time, experiment_title)
    save_experiment(experiment_title, points_to_keep, experiment_metadata)


def faiss_subsampling_experiment(scaled_data, subsample_fn, experiment_title, outlier_indices, cutoff):
    start = time.time()
    indices_to_keep = subsample_fn(scaled_data, cutoff_sig=cutoff, verbose=2)
    wallclock_time = time.time() - start
    points_to_keep = scaled_data[indices_to_keep]
    experiment_metadata = \
        get_experiment_metadata_with_outliers(points_to_keep, outlier_indices, indices_to_keep, wallclock_time,
                                              experiment_title) | {"cutoff": cutoff}
    save_experiment(experiment_title, points_to_keep, experiment_metadata)


def dimensionality_reduction_experiment(scaled_data: np.ndarray, base_title: str, outlier_indices: np.ndarray,
                                        faiss_subsample_method=faiss_ivf_subsample, cutoff: float = 0.07):
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
            get_experiment_metadata_with_outliers(points_to_keep, outlier_indices, indices_to_keep, wallclock_time,
                                                  title) | additional_metadata
        save_experiment(title, points_to_keep, experiment_metadata, second_level_dir=second_level_dir)


def cutoff_experiment(scaled_data: np.ndarray, base_title: str, faiss_subsample_method=faiss_ivf_subsample,
                      cutoffs: list = None):
    if not cutoffs:
        cutoffs = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10]
    second_level_dir = base_title + "_cutoff_variation_{}".format(time.time())
    for cutoff in cutoffs:
        start = time.time()
        indices_to_keep = faiss_subsample_method(scaled_data, cutoff_sig=cutoff, verbose=2)
        wallclock_time = time.time() - start
        points_to_keep = scaled_data[indices_to_keep]
        
        title = base_title + "_{}_cutoff".format(cutoff)
        experiment_metadata = \
            get_experiment_metadata(points_to_keep, wallclock_time, title) | {"cutoff": cutoff}
        save_experiment(title, points_to_keep, experiment_metadata, second_level_dir=second_level_dir)


def main():
    scaled_data = load_numpy_data("qm9_train_gmp_fingerprints")
    cutoff_experiment(scaled_data, "qm9_gmp_ivf")
    outlier_indices = get_outlier_indices(scaled_data, outlier_cutoff_modifier=0.07)
    full_dataset_metadata(scaled_data, "qm9")
    random_subsampling_experiment(scaled_data, "qm9_random_subsample", outlier_indices, 20000)
    kmeans_subsampling_experiment(scaled_data, "qm9_kmeans_subsample", outlier_indices, 20000)
    faiss_subsampling_experiment(scaled_data, faiss_flat_subsample, "qm9_faiss_flat_subsample", outlier_indices,
                                 cutoff=0.07)
    faiss_subsampling_experiment(scaled_data, faiss_ivf_subsample, "qm9_faiss_ivf_subsample", outlier_indices,
                                 cutoff=0.07)
    faiss_subsampling_experiment(scaled_data, faiss_ivfpq_subsample, "qm9_faiss_ivfpq_subsample", outlier_indices,
                                 cutoff=0.07)
    dimensionality_reduction_experiment(scaled_data, "qm9_gmp_ivf", outlier_indices)


if __name__ == "__main__":
    main()
