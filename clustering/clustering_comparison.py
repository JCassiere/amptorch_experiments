import numpy as np
import time
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from utils import load_qm9
from amptorch.subsampling import average_images_over_fingerprints, reduce_dimensions_with_pca, \
    scale_and_standardize_data, subsample_clustering


def calculate_silhouette_score_nn_subsample_clustering(clusters, average_fingerprints):
    assigned_clusters = np.zeros(average_fingerprints.shape[0])
    cluster_num = 0
    for cluster in clusters.values():
        assigned_clusters[np.array(cluster)] = cluster_num
        cluster_num += 1
    
    return silhouette_score(average_fingerprints, assigned_clusters, metric='euclidean')
    
def qm9_silhouette_score_nn_subsample_clustering():
    data, images = load_qm9()
    averaged_fingerprints, fingerprint_to_image_index = average_images_over_fingerprints(data)
    reduced_data = reduce_dimensions_with_pca(averaged_fingerprints, max_components=6)
    scaled_data = scale_and_standardize_data(reduced_data)
    start = time.time()
    clusters = subsample_clustering(scaled_data, max_clusters=20)
    print("NN subsample clustering took {} s".format(time.time() - start))
    score = calculate_silhouette_score_nn_subsample_clustering(clusters, scaled_data)
    print("NN subsample silhouette score: {}".format(score))


def qm9_silhouette_score_k_means():
    data, images = load_qm9()
    averaged_fingerprints, fingerprint_to_image_index = average_images_over_fingerprints(data)
    reduced_data = reduce_dimensions_with_pca(averaged_fingerprints, max_components=6)
    scaled_data = scale_and_standardize_data(reduced_data)
    start = time.time()
    km = KMeans(n_clusters=20)
    km.fit_predict(scaled_data)
    print("Kmeans clustering took {} s".format(time.time() - start))
    score = silhouette_score(scaled_data, km.labels_, metric='euclidean')
    print("Kmeans silhouette score: {}".format(score))

# def umap_plot_qm9_nn_subsample_clustering():


# TODO - try messing with the starting cutoff
if __name__ == "__main__":
    # qm9_silhouette_score_nn_subsample_clustering()
    qm9_silhouette_score_k_means()