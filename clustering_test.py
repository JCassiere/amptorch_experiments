from amptorch.sampler import subsample_clustering, reduce_dimensions_with_pca, scale_and_standardize_data, extract_fingerprints_with_image_indices
import pickle
import json
import numpy as np
from GMP_subsample_profile import convert_and_pickle_atoms, load_oc20_images, load_qm9_images

# data = pickle.load(open("data/QM9_train_torch_data.p", "rb"))
# images, _ = load_qm9_images()
data = pickle.load(open("data/oc20_train_torch_data.p", "rb"))
images, _ = load_oc20_images()

fingerprints, fingerprint_to_image_index = extract_fingerprints_with_image_indices(data)
reduced_data = reduce_dimensions_with_pca(fingerprints, max_components=6)
scaled_data = scale_and_standardize_data(reduced_data)
clusters = subsample_clustering(scaled_data, max_clusters=35)

cluster_json = {}
cluster_image_bincounts = []
for cluster in clusters.values():
    image_index_per_cluster_datapoint = fingerprint_to_image_index[np.array(cluster)]
    bincount = np.bincount(image_index_per_cluster_datapoint)
    bincount = np.append(bincount, np.zeros(len(data) - bincount.size))
    cluster_image_bincounts.append(bincount)
cluster_image_bincounts = np.vstack(cluster_image_bincounts)

image_to_cluster = np.argmax(cluster_image_bincounts, axis=0)
image_formulas = [image.symbols.formula._formula for image in images]
cluster_formulas = [[] for i in range(len(clusters.keys()))]
for image_index, image_cluster in enumerate(image_to_cluster):
    cluster_formulas[image_cluster].append(image_formulas[image_index])
    
final_clusters = [x for x in cluster_formulas if len(x) > 0]
top_20 = [x[:20] for x in final_clusters]

# qm9_out = open("qm9_clusters.json", "w")
# json.dump(final_clusters, qm9_out, indent=4)
# qm9_top20_out = open("qm9_clusters_top20.json", "w")
# json.dump(top_20, qm9_top20_out, indent=4)
oc20_out = open("oc20_3k_clusters.json", "w")
json.dump(final_clusters, oc20_out, indent=4)
oc20_top20_out = open("oc20_3k_clusters_top20.json", "w")
json.dump(top_20, oc20_top20_out, indent=4)

