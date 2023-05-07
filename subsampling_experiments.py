import numpy as np

from data_exchange import load_numpy_data, save_experiment, save_to_numpy, create_dir_if_not_exists
from ccsubsample.subsampling.utils import *
from ccsubsample.subsampling.evaluation_metrics import *
from ccsubsample.subsampling.subsampling import faiss_flat_subsample, faiss_ivf_subsample, faiss_ivfpq_subsample, \
    kmeans_subsample, sobol_subsample, farthest_point_sampling, farthest_point_sampling_batched

CUTOFF_PERCENTILES = [0.01, 0.05, 0.1, 0.5, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.985, 0.99, 0.995, 0.999]
OUTLIER_PERCENTILES = [0.8, 0.9, 0.925, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 0.999]


def get_experiment_metadata(subsampled_data: np.ndarray,
                            wallclock_time: float,
                            experiment_title: str,
                            should_calc_spread_metrics: bool = True) -> dict:
    num_points = number_of_points_remaining(subsampled_data)
    subset_filename = experiment_title + "_subset"
    experiment_metadata = {
        "experiment_title": experiment_title,
        "wallclock_time": wallclock_time,
        "num_points": num_points,
        "subset_filename": subset_filename,
    }
    if should_calc_spread_metrics:
        discrepancy = calculate_discrepancy(subsampled_data)
        _, std_dev = point_diversity_mean_std(subsampled_data)
        experiment_metadata = experiment_metadata | {
            "discrepancy": discrepancy,
            "std_dev": float(std_dev)
        }
    return experiment_metadata


def get_experiment_metadata_with_outliers(subsampled_data: np.ndarray, outlier_indices: np.ndarray,
                                          subsampled_indices: np.ndarray,
                                          wallclock_time: float, experiment_title: str) -> dict:
    if len(outlier_indices) > 0:
        outlier_retention = calculate_outlier_retention(outlier_indices, subsampled_indices)
        experiment_metadata = get_experiment_metadata(subsampled_data, wallclock_time, experiment_title) | {
            "outlier_retention": outlier_retention
        }
    else:
        experiment_metadata = get_experiment_metadata(subsampled_data, wallclock_time, experiment_title)
    return experiment_metadata


def get_outliers_for_variance_experiment(data, dataset_title, outlier_percentiles: list = None):
    if not outlier_percentiles:
        outlier_percentiles = OUTLIER_PERCENTILES
    dir_name = "data/{}_outliers".format(dataset_title)
    create_dir_if_not_exists(dir_name)
    indices = np.arange(0, data.shape[0])
    distances = get_nearest_neighbor_distances(data)
    for percentile in outlier_percentiles:
        cutoff = np.quantile(distances, percentile)
        outlier_indices = indices[distances >= cutoff]
        save_to_numpy(
            outlier_indices,
            "{}_outlier_indices_{}_percentile".format(dataset_title, percentile),
            dir_name
        )


def outlier_retention_variance_experiment(subsampled_indices, dataset_title, algorithm_name, outlier_percentiles: list = None):
    if not outlier_percentiles:
        outlier_percentiles = OUTLIER_PERCENTILES
    dir_name = "data/{}_outliers".format(dataset_title)
    outlier_retentions = []
    for percentile in outlier_percentiles:
        filename = "{}_outlier_indices_{}_percentile".format(dataset_title, percentile)
        outlier_indices = load_numpy_data(filename, prefix=dir_name)
        outlier_retentions += [(calculate_outlier_retention(outlier_indices, subsampled_indices))]
    experiment_data = {
        "outlier_percentile": outlier_percentiles,
        "outlier_retention": outlier_retentions
    }
    experiment_dir = "results/{}_{}_outlier_retention_variance_{}".format(dataset_title, algorithm_name, time.time())
    create_dir_if_not_exists(experiment_dir)
    results_file = open(experiment_dir + "/outlier_retention.json", 'w')
    json.dump(experiment_data, results_file, indent=4)


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


def faiss_subsampling_experiment(scaled_data, subsample_fn, experiment_title, outlier_indices, cutoff_percentile):
    start = time.time()
    indices_to_keep = subsample_fn(scaled_data, cutoff_percentile=cutoff_percentile, verbose=2)
    wallclock_time = time.time() - start
    points_to_keep = scaled_data[indices_to_keep]
    experiment_metadata = \
        get_experiment_metadata_with_outliers(points_to_keep, outlier_indices, indices_to_keep, wallclock_time,
                                              experiment_title) | {"cutoff_percentile": cutoff_percentile}
    save_experiment(experiment_title, points_to_keep, experiment_metadata)


def dimensionality_reduction_experiment(scaled_data: np.ndarray, base_title: str, outlier_indices: np.ndarray,
                                        faiss_subsample_method=faiss_ivf_subsample, cutoff_percentile: float = 0.99):
    pca = PCA(svd_solver="randomized")
    data_pca = pca.fit_transform(scaled_data)
    explained_variance_ratio = pca.explained_variance_ratio_
    _, dim = data_pca.shape
    second_level_dir = base_title + "_dim_reduction_{}".format(time.time())
    for i in range(2, dim):
        start = time.time()
        indices_to_keep = faiss_subsample_method(np.array(data_pca[:, :i], order='C'), cutoff_percentile=cutoff_percentile, verbose=2)
        wallclock_time = time.time() - start
        points_to_keep = scaled_data[indices_to_keep]
        
        title = base_title + "_{}_dimensions".format(i)
        additional_metadata = {
            "num_dimensions_kept": i,
            "variance_retained": np.sum(explained_variance_ratio[:i]),
            "cutoff_percentile": cutoff_percentile
        }
        experiment_metadata = \
            get_experiment_metadata_with_outliers(points_to_keep, outlier_indices, indices_to_keep, wallclock_time,
                                                  title) | additional_metadata
        save_experiment(title, points_to_keep, experiment_metadata, second_level_dir=second_level_dir)


def cutoff_experiment(scaled_data: np.ndarray, base_title: str, faiss_subsample_method=faiss_ivf_subsample,
                      cutoff_percentiles: list = None):
    if not cutoff_percentiles:
        cutoff_percentiles = CUTOFF_PERCENTILES
    second_level_dir = base_title + "_cutoff_variation_{}".format(time.time())
    for percentile in cutoff_percentiles:
        start = time.time()
        indices_to_keep = faiss_subsample_method(scaled_data, cutoff_percentile=percentile, verbose=2)
        wallclock_time = time.time() - start
        points_to_keep = scaled_data[indices_to_keep]
        
        title = base_title + "_{}_cutoff_percentile".format(percentile)
        experiment_metadata = \
            get_experiment_metadata(points_to_keep, wallclock_time, title, should_calc_spread_metrics=False) | {"cutoff_percentile": percentile}
        save_experiment(title, points_to_keep, experiment_metadata, second_level_dir=second_level_dir)
        
        
    
def sobol_experiment(scaled_data, experiment_title, outlier_indices, num_points_desired):
    start = time.time()
    indices_to_keep = sobol_subsample(scaled_data, num_points_desired)
    wallclock_time = time.time() - start
    points_to_keep = scaled_data[indices_to_keep]
    experiment_metadata = get_experiment_metadata_with_outliers(points_to_keep, outlier_indices, indices_to_keep,
                                                                wallclock_time, experiment_title)
    save_experiment(experiment_title, points_to_keep, experiment_metadata)


def fps_experiment(scaled_data, experiment_title, outlier_indices, num_points_desired):
    start = time.time()
    indices_to_keep = farthest_point_sampling(scaled_data, num_points_desired)
    wallclock_time = time.time() - start
    points_to_keep = scaled_data[indices_to_keep]
    experiment_metadata = get_experiment_metadata_with_outliers(points_to_keep, outlier_indices, indices_to_keep,
                                                                wallclock_time, experiment_title)
    save_experiment(experiment_title, points_to_keep, experiment_metadata)


def fps_batched_experiment(scaled_data, experiment_title, outlier_indices, num_points_desired):
    start = time.time()
    indices_to_keep = farthest_point_sampling_batched(scaled_data, num_points_desired)
    wallclock_time = time.time() - start
    points_to_keep = scaled_data[indices_to_keep]
    experiment_metadata = get_experiment_metadata_with_outliers(points_to_keep, outlier_indices, indices_to_keep,
                                                                wallclock_time, experiment_title)
    save_experiment(experiment_title, points_to_keep, experiment_metadata)


def outlier_algorithm_comparison(scaled_data, dataset_title):
    faiss_ivf_indices = faiss_ivf_subsample(scaled_data, cutoff_percentile=0.99)
    outlier_retention_variance_experiment(faiss_ivf_indices, dataset_title, "faiss_ivf")
    num_points_desired = len(faiss_ivf_indices)
    faiss_ivfpq_indices = faiss_ivfpq_subsample(scaled_data, cutoff_percentile=0.99)
    outlier_retention_variance_experiment(faiss_ivfpq_indices, dataset_title, "faiss_ivfpq")
    kmeans_indices = kmeans_subsample(scaled_data, num_points_desired)
    outlier_retention_variance_experiment(kmeans_indices, dataset_title, "kmeans")
    sobol_indices = sobol_subsample(scaled_data, num_points_desired)
    outlier_retention_variance_experiment(sobol_indices, dataset_title, "sobol")
    fps_indices = farthest_point_sampling(scaled_data, num_points_desired)
    outlier_retention_variance_experiment(fps_indices, dataset_title, "fps")
    fps_batched_indices = farthest_point_sampling_batched(scaled_data, num_points_desired)
    outlier_retention_variance_experiment(fps_batched_indices, dataset_title, "fps_batched")
    

def isotropic_gaussian(num_points, num_dims):
    cov = np.identity(num_dims)
    mean = [0 for _ in range(num_dims)]
    x = np.random.default_rng().multivariate_normal(mean, cov, size=num_points)
    return x


def anisotropic_gaussian(num_points, num_dims):
    diag = [np.random.randint(0, 9) * (10 ** np.random.randint(0, 3)) for _ in range(num_dims)]
    cov = np.diag(diag)
    mean = [0 for _ in range(num_dims)]
    x = np.random.default_rng().multivariate_normal(mean, cov, size=num_points)
    return x


def exps(scaled_data, dataset_title, outlier_indices):
    # cutoff_experiment(scaled_data, "{}_gmp_ivf".format(dataset_title))
    # full_dataset_metadata(scaled_data, dataset_title)
    random_subsampling_experiment(scaled_data, "{}_random_subsample".format(dataset_title), outlier_indices, 20000)
    kmeans_subsampling_experiment(scaled_data, "{}_kmeans_subsample".format(dataset_title), outlier_indices, 20000)
    sobol_experiment(scaled_data, "{}_sobol".format(dataset_title), outlier_indices, 20000)
    # fps_experiment(scaled_data, "{}_fps".format(dataset_title), outlier_indices, 20000)
    fps_batched_experiment(scaled_data, "{}_fps_batched".format(dataset_title), outlier_indices, 20000)
    faiss_subsampling_experiment(scaled_data, faiss_flat_subsample, "{}_faiss_flat_subsample".format(dataset_title), outlier_indices,
                                 cutoff_percentile=0.99)
    faiss_subsampling_experiment(scaled_data, faiss_ivf_subsample, "{}_faiss_ivf_subsample".format(dataset_title), outlier_indices,
                                 cutoff_percentile=0.99)
    faiss_subsampling_experiment(scaled_data, faiss_ivfpq_subsample, "{}_faiss_ivfpq_subsample".format(dataset_title), outlier_indices,
                                 cutoff_percentile=0.99)
    dimensionality_reduction_experiment(scaled_data, "{}_ivf".format(dataset_title), outlier_indices)
    outlier_algorithm_comparison(scaled_data, dataset_title)


def oc20_3k_exps():
    dataset_title = "oc20_3k_gmp"
    scaled_data = load_numpy_data("scaled_and_std_oc20_3k_gmp_fingerprints")
    # get_outliers_for_variance_experiment(scaled_data, dataset_title)
    # TODO - test LOF outlier functions
    outlier_indices = load_numpy_data("oc20_3k_outlier_indices_99th_percentile")
    exps(scaled_data, dataset_title, outlier_indices)


def qm9_exps():
    dataset_title = "qm9_gmp"
    scaled_data = load_numpy_data("qm9_train_gmp_fingerprints")
    # get_outliers_for_variance_experiment(scaled_data, dataset_title)
    outlier_indices = load_numpy_data("qm9_gmp_outlier_indices_99th_percentile")
    exps(scaled_data, dataset_title, outlier_indices)


if __name__ == "__main__":
    # oc20_3k_exps()
    qm9_exps()
