import numpy as np
from amptorch.sampler import average_images_over_fingerprints, extract_fingerprints_with_image_indices, \
    reduce_dimensions_with_pca, scale_and_standardize_data, subsample_clustering
from utils import load_qm9, load_oc20_3k, get_clusters, images_to_formulas, clusters_to_formulas, output_clusters_to_json
from sklearn.cluster import KMeans

def cluster_and_output_oc20_image_average():
    data, images = load_oc20_3k()
    averaged_fingerprints, fingerprint_to_image_index = average_images_over_fingerprints(data)
    clusters = get_clusters(averaged_fingerprints)
    formulas = images_to_formulas(images)
    clustered_formulas = clusters_to_formulas(clusters, formulas)
    output_clusters_to_json(clustered_formulas, "oc20_3k")

def cluster_and_output_qm9_image_average():
    data, images = load_qm9()
    averaged_fingerprints, fingerprint_to_image_index = average_images_over_fingerprints(data)
    clusters = get_clusters(averaged_fingerprints)
    formulas = images_to_formulas(images)
    clustered_formulas = clusters_to_formulas(clusters, formulas)
    output_clusters_to_json(clustered_formulas, "qm9")

def cluster_qm9_fingerprints_to_elements():
    data, images = load_qm9()
    # data, images = load_oc20_3k()
    data = data[:500]
    per_image_fingerprints = []
    per_image_elements = []

    for i, torch_image in enumerate(data):
        image_fingerprint = torch_image.fingerprint.numpy()
        image_elements = torch_image.atomic_numbers.numpy()
        per_image_fingerprints.append(image_fingerprint)
        per_image_elements.append(image_elements)

    fingerprints = np.vstack(per_image_fingerprints)
    elements = np.concatenate(per_image_elements)
    # scaled_and_reduced_data = scale_and_standardize_data(reduce_dimensions_with_pca(fingerprints, max_components=6))
    scaled_data = scale_and_standardize_data(fingerprints)
    # clusters = subsample_clustering(scaled_and_reduced_data, max_clusters=100)
    clusters = subsample_clustering(scaled_data, max_clusters=70)
    clusters_by_element = []
    for cluster in clusters.values():
        cluster_elements = elements[np.array(cluster)]
        unique, counts = np.unique(cluster_elements, return_counts=True)
        clusters_by_element.append(dict(zip(unique, counts)))
        
    nn_cluster_precisions = []
    nn_cluster_counts = []
    for cluster in clusters_by_element:
        counts = np.array(list(cluster.values()))
        max_count = np.max(counts)
        tot = np.sum(counts)
        nn_cluster_precisions.append(float(max_count) / tot)
        nn_cluster_counts.append(tot)
    nn_cluster_counts = np.array(nn_cluster_counts)
    weighted_average_nn_precision = np.sum(nn_cluster_precisions * nn_cluster_counts) / np.sum(nn_cluster_counts)
    
    n_clusters = 50
    km = KMeans(n_clusters=n_clusters)
    km.fit_predict(scaled_data)
    km_clusters = [{} for _ in range(n_clusters)]
    for el, cluster in zip(elements, km.labels_):
        try:
            km_clusters[cluster][el] += 1
        except KeyError:
            km_clusters[cluster][el] = 1

    km_cluster_precisions = []
    km_cluster_counts = []
    for cluster in km_clusters:
        counts = np.array(list(cluster.values()))
        max_count = np.max(counts)
        tot = np.sum(counts)
        km_cluster_precisions.append(float(max_count) / tot)
        km_cluster_counts.append(tot)
    km_cluster_counts = np.array(km_cluster_counts)
    weighted_average_km_precision = np.sum(km_cluster_precisions * km_cluster_counts) / np.sum(km_cluster_counts)
    print(1)

if __name__ == "__main__":
    # cluster_and_output_oc20_image_average()
    # cluster_and_output_qm9_image_average()
    cluster_qm9_fingerprints_to_elements()
