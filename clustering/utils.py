from amptorch.sampler import subsample_clustering, reduce_dimensions_with_pca, scale_and_standardize_data
import pickle
import json
import numpy as np
from subsampling.GMP_subsample_profile import load_oc20_images, load_qm9_images

# def cluster_images_by_fingerprint_argmax(data, clusters, images, fingerprint_to_image_index):
#     cluster_image_bincounts = []
#     for cluster in clusters.values():
#         image_index_per_cluster_datapoint = fingerprint_to_image_index[np.array(cluster)]
#         bincount = np.bincount(image_index_per_cluster_datapoint)
#         bincount = np.append(bincount, np.zeros(len(data) - bincount.size))
#         cluster_image_bincounts.append(bincount)
#     cluster_image_bincounts = np.vstack(cluster_image_bincounts)
#
#     image_to_cluster = np.argmax(cluster_image_bincounts, axis=0)
#     image_formulas = [image.symbols.formula._formula for image in images]
#     cluster_formulas = [[] for _ in range(len(clusters.keys()))]
#     for image_index, image_cluster in enumerate(image_to_cluster):
#         cluster_formulas[image_cluster].append(image_formulas[image_index])
#
#     cluster_formulas_no_empties = [x for x in cluster_formulas if len(x) > 0]
#
#     return cluster_formulas_no_empties

# TODO - separate modules better
def load_qm9():
    data = pickle.load(open("data/QM9_train_torch_data.p", "rb"))
    images, _ = load_qm9_images()
    return data, images


def load_oc20_3k():
    data = pickle.load(open("data/oc20_train_torch_data.p", "rb"))
    images, _ = load_oc20_images()
    return data, images


def get_clusters(averaged_fingerprints):
    reduced_data = reduce_dimensions_with_pca(averaged_fingerprints, max_components=6)
    scaled_data = scale_and_standardize_data(reduced_data)
    clusters = subsample_clustering(scaled_data, max_clusters=35)
    return clusters


def images_to_formulas(images):
    return np.array([image.get_chemical_formula() for image in images])


def clusters_to_formulas(clusters, formulas):
    cluster_formulas = []
    for cluster_members in clusters.values():
        cluster_formulas.append(list(formulas[np.array(cluster_members)]))
    
    return cluster_formulas


def output_clusters_to_json(cluster_formulas, dataset_description):
    top_20 = [x[:20] for x in cluster_formulas]
    
    oc20_out = open(dataset_description + "_clusters.json", "w")
    json.dump(cluster_formulas, oc20_out, indent=4)
    oc20_top20_out = open(dataset_description + "_clusters_top20.json", "w")
    json.dump(top_20, oc20_top20_out, indent=4)