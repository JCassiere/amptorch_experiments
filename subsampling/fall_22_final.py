from ccsubsample.subsampling.utils import *
from ccsubsample.subsampling.evaluation_metrics import *
from ccsubsample.subsampling.plotting import *
from ccsubsample.subsampling.subsampling import kdtree_subsample, faiss_subsample, ivf_index, ivfpq_index, hnsw_index, flat_index, kmeans_subsample
import pickle
import json


def load_oc20_3k_data():
    data = pickle.load(open("../data/oc20_train_torch_data.p", "rb"))
    return data


def load_qm9_data():
    data = pickle.load(open("../data/QM9_train_torch_data.p", "rb"))
    return data


def save_fingerprints_to_json(fingerprints, fingerprints_to_image_index, filename):
    fp_list = fingerprints.tolist()
    fp_index_list = fingerprints_to_image_index.tolist()
    fingerprints_out = open("../data/{}_fingerprints.json".format(filename), "w")
    json.dump(fp_list, fingerprints_out, indent=4)
    fingerprints_index_out = open("../data/{}_fingerprints_index.json".format(filename), "w")
    json.dump(fp_index_list, fingerprints_index_out, indent=4)
    
def subsample_and_kde(data, subsample_method, cutoff=0.24):
    fingerprints, fingerprints_to_image_index = extract_fingerprints_with_image_indices(data)
    reduced = reduce_dimensions_with_pca(fingerprints, max_components=6)
    scaled = scale_and_standardize_data(reduced)
    
    indices_to_keep = subsample_method(scaled, cutoff_sig=cutoff, verbose=2)
    points_to_keep = scaled[indices_to_keep]
    num_points = number_of_points_remaining(points_to_keep)
    print("Number of points: {}".format(num_points))
    x_axis_pts, log_dens = point_diversity_kde(points_to_keep)
    return x_axis_pts, log_dens
    
def oc20_3k_kde():
    data = load_oc20_3k_data()
    x_axis_pts, log_dens = subsample_and_kde(data, kdtree_subsample, cutoff=0.3)
    plot_kde(x_axis_pts, log_dens, "OC20 3k Standard KDTree")
    
def oc20_3k_random_subsample_kde():
    data = load_oc20_3k_data()
    fingerprints, fingerprints_to_image_index = extract_fingerprints_with_image_indices(data)
    reduced = reduce_dimensions_with_pca(fingerprints, max_components=6)
    scaled = scale_and_standardize_data(reduced)
    pts = scaled[np.random.choice(scaled.shape[0], size=9000, replace=False)]
    mean, std = point_diversity_mean_std(pts)
    print("std: {}".format(std))
    x_axis_pts, log_dens = point_diversity_kde(scaled[np.random.choice(scaled.shape[0], size=9000, replace=False)])
    plot_kde(x_axis_pts, log_dens, "OC20 3k Random Subsampling")
    
def kmeans_kde_no_dim_reduction(data, num_points, title):
    print("Beginning {}".format(title))
    fingerprints, fingerprints_to_image_index = extract_fingerprints_with_image_indices(data)
    scaled = scale_and_standardize_data(fingerprints)
    start = time.time()
    indices_to_keep = kmeans_subsample(scaled, num_points)
    print("Kmeans subsample completed - took {}".format(time.time() - start))
    points_to_keep = scaled[indices_to_keep]
    
    num_points = number_of_points_remaining(points_to_keep)
    print("Number of points for {}: {}".format(title, num_points))
    
    mean, std = point_diversity_mean_std(points_to_keep)
    print("std: {}".format(std))
    
    x_axis_pts, log_dens = point_diversity_kde(points_to_keep)
    plot_kde(x_axis_pts, log_dens, title + " {} points".format(num_points))

def kdtree_no_dim_reduction(data, title, cutoff=0.5):
    print("Beginning {}".format(title))
    fingerprints, fingerprints_to_image_index = extract_fingerprints_with_image_indices(data)
    scaled = scale_and_standardize_data(fingerprints)
    indices_to_keep = kdtree_subsample(scaled, cutoff_sig=cutoff, verbose=2)
    points_to_keep = scaled[indices_to_keep]
    
    num_points = number_of_points_remaining(points_to_keep)
    print("Number of points for {}: {}".format(title, num_points))
    
    mean, std = point_diversity_mean_std(points_to_keep)
    print("std: {}".format(std))
    
    x_axis_pts, log_dens = point_diversity_kde(points_to_keep)
    plot_kde(x_axis_pts, log_dens, title + " cutoff_sig={}, {} points".format(cutoff, num_points))


def faiss_no_dim_reduction(data, index_fn, title, cutoff=2):
    print("Beginning {}".format(title))
    fingerprints, fingerprints_to_image_index = extract_fingerprints_with_image_indices(data)
    scaled = scale_and_standardize_data(fingerprints)
    indices_to_keep = faiss_subsample(scaled, index_fn, cutoff_sig=cutoff, verbose=2)
    points_to_keep = scaled[indices_to_keep]
    
    num_points = number_of_points_remaining(points_to_keep)
    print("Number of points for {}: {}".format(title, num_points))
    
    mean, std = point_diversity_mean_std(points_to_keep)
    print("std: {}".format(std))

    x_axis_pts, log_dens = point_diversity_kde(points_to_keep)
    plot_kde(x_axis_pts, log_dens, title + " cutoff_sig={}, {} points".format(cutoff, num_points))


def full_kde(data, title):
    fingerprints, fingerprints_to_image_index = extract_fingerprints_with_image_indices(data)

    num_points = number_of_points_remaining(fingerprints)
    print("Number of points for {}: {}".format(title, num_points))
    x_axis_pts, log_dens = point_diversity_kde(fingerprints)
    plot_kde(x_axis_pts, log_dens, title + " {} points".format(num_points))


def kde_random_subsample(data, title, num_points):
    fingerprints, fingerprints_to_image_index = extract_fingerprints_with_image_indices(data)
    scaled = scale_and_standardize_data(fingerprints)
    random_subsample = scaled[np.random.choice(np.arange(fingerprints.shape[0]), num_points)]

    mean, std = point_diversity_mean_std(random_subsample)
    print("std: {}".format(std))

    print("Number of points for {}: {}".format(title, num_points))
    x_axis_pts, log_dens = point_diversity_kde(random_subsample)
    plot_kde(x_axis_pts, log_dens, title + " {} points".format(num_points))


def kde_experiments():
    oc20_3k_data = load_oc20_3k_data()
    qm9_data = load_qm9_data()

    #OC20 3K
    faiss_no_dim_reduction(oc20_3k_data, flat_index, "OC20 3K Flat Bandwidth=0.5")
    faiss_no_dim_reduction(oc20_3k_data, ivf_index, "OC20 3K IVF Bandwidth=0.5")
    faiss_no_dim_reduction(oc20_3k_data, ivfpq_index, "OC20 3K IVFPQ Bandwidth=0.5")
    faiss_no_dim_reduction(oc20_3k_data, hnsw_index, "OC20 3K HNSW Bandwidth=0.5")
    kmeans_kde_no_dim_reduction(oc20_3k_data, 20000, "OC20 3K Kmeans subsample")
    kdtree_no_dim_reduction(oc20_3k_data, "OC20 3K KD-Tree Bandwidth=0.03")
    
    
    #QM9
    faiss_no_dim_reduction(qm9_data, flat_index, "QM9 Flat Bandwidth=0.5")
    faiss_no_dim_reduction(qm9_data, ivf_index, "QM9 IVF Bandwidth=0.5")
    faiss_no_dim_reduction(qm9_data, ivfpq_index, "QM9 IVFPQ Bandwidth=0.5")
    faiss_no_dim_reduction(qm9_data, hnsw_index, "QM9 HNSW Bandwidth=0.5")
    kmeans_kde_no_dim_reduction(qm9_data, 20000, "QM9 Kmeans subsample")
    kdtree_no_dim_reduction(qm9_data, "QM9 KD-Tree Bandwidth=0.5")

    #Raw datasets
    full_kde(oc20_3k_data, "OC20 Full Dataset")
    kde_random_subsample(qm9_data, "QM9 Random Subsample", 20000)

if __name__ == "__main__":
    kde_experiments()
    