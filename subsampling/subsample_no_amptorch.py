from ccsubsample.subsampling.utils import *
from ccsubsample.subsampling.evaluation_metrics import *
from ccsubsample.subsampling.plotting import *
from ccsubsample.subsampling.subsampling import kdtree_subsample, ivfpq_subsample
# from ccsubsample.subsampling.subsampling import kdtree_subsample_centroids
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
    
    
def oc20_3k_kde_centroid():
    data = load_oc20_3k_data()
    x_axis_pts, log_dens = subsample_and_kde(data, kdtree_subsample_centroids, cutoff=0.26)
    plot_kde(x_axis_pts, log_dens, "OC20 3k Centroid KDTree")

def qm9_kde():
    data = load_qm9_data()
    x_axis_pts, log_dens = subsample_and_kde(data, kdtree_subsample, cutoff=0.3)
    plot_kde(x_axis_pts, log_dens, "QM9 Standard KDTree")

def qm9_kde_centroid():
    data = load_qm9_data()
    x_axis_pts, log_dens = subsample_and_kde(data, kdtree_subsample_centroids, cutoff=0.24)
    plot_kde(x_axis_pts, log_dens, "QM9 Centroid KDTree")
    
def oc20_kdtree_no_dim_reduction():
    data = load_oc20_3k_data()
    fingerprints, fingerprints_to_image_index = extract_fingerprints_with_image_indices(data)
    # reduced = reduce_dimensions_with_pca(fingerprints, max_components=6)
    scaled = scale_and_standardize_data(fingerprints)
    
    indices_to_keep = kdtree_subsample(scaled, cutoff_sig=0.25, verbose=2)
    print(1)


def faiss_no_dim_reduction():
    data = load_oc20_3k_data()
    # data = load_qm9_data()
    fingerprints, fingerprints_to_image_index = extract_fingerprints_with_image_indices(data)
    scaled = scale_and_standardize_data(fingerprints)
    #
    indices_to_keep = ivfpq_subsample(scaled, cutoff_sig=2, verbose=2)
    points_to_keep = fingerprints[indices_to_keep]
    num_points = number_of_points_remaining(points_to_keep)
    print("Number of points: {}".format(num_points))
    x_axis_pts, log_dens = point_diversity_kde(points_to_keep)
    plot_kde(x_axis_pts, log_dens, "OC20 3K Flat Bandwidth=0.03 Cutoff_sig=2")
    


def main():
    # oc20_3k_random_subsample_kde()
    # oc20_3k_kde()
    # oc20_3k_kde_cZDentroid()
    # oc20_kdtree_no_dim_reduction()
    faiss_no_dim_reduction()
    # qm9_kde()
    # qm9_kde_centroid()
    # data = load_qm9_data()
    # data = load_oc20_3k_data()
    # fingerprints, fingerprints_to_image_index = extract_fingerprints_with_image_indices(data)
    # # save_fingerprints_to_json(fingerprints, fingerprints_to_image_index, "QM9")
    # reduced = reduce_dimensions_with_pca(fingerprints, max_components=6)
    # scaled = scale_and_standardize_data(reduced)
    #
    # centroid_indices_to_keep = kdtree_subsample_centroids(scaled, cutoff_sig=0.24, verbose=2)
    # centroid_to_keep = fingerprints[centroid_indices_to_keep]
    # x_axis_pts, log_dens = point_diversity_kde(centroid_to_keep)
    # # plt.fill(centroid_to_keep, np.exp(log_dens), c='cyan')
    # # tst = np.exp(log_dens)
    # plt.fill(x_axis_pts, np.exp(log_dens))
    # plt.show()
    # centroid_mean, centroid_std = point_diversity_mean_std(centroid_to_keep)
    # centroid_histogram = point_diversity_histogram(centroid_to_keep)
    
    # data_indices_to_keep = kdtree_subsample(scaled, cutoff_sig=0.3, verbose=2)
    # data_to_keep = fingerprints[data_indices_to_keep]
    # mean, std = point_diversity_mean_std(data_to_keep)
    # histogram = point_diversity_histogram(data_to_keep)

    # image_indices_to_keep = get_image_indices_to_keep(data_indices_to_keep, fingerprints_to_image_index)
    
if __name__ == "__main__":
    main()