# from amptorch.subsampling import subsample_clustering, reduce_dimensions_with_pca, scale_and_standardize_data
import ase
import os
import ase.io
import pickle
import json
import numpy as np
from subsampling.GMP_subsample_profile import load_oc20_images, load_qm9_images
from amptorch.trainer import AtomsTrainer
from amptorch.preprocessing import AtomsToData, FeatureScaler, TargetScaler
from amptorch.descriptor.GMPOrderNorm import GMPOrderNorm
from ccsubsample.subsampling.utils import extract_fingerprints_with_image_indices, scale_and_standardize_data
from data_exchange import save_to_numpy

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

def load_oc20_images():
    images = ase.io.read("data/oc20_3k_train.traj", ":")
    test_images = ase.io.read("data/oc20_300_test.traj", ":")
    return images, test_images

def load_oc20_3k():
    data = pickle.load(open("data/oc20_train_torch_data.p", "rb"))
    images, _ = load_oc20_images()
    return data, images

def ase_atoms_to_gmp_torch_data(images, MCSHs_index=3, nsigmas=10):
    # use MCSHs_index = 3 for QM9
    sigmas = np.linspace(0, 2.0, nsigmas + 1, endpoint=True)[1:]
    MCSHs = {"orders": [i for i in range(MCSHs_index + 1)], "sigmas": sigmas}
    gaussians = {}
    dir = "./valence_gaussians"
    for file in os.listdir(dir):
        el = file.split("_")[0]
        gaussians[el] = dir + "/" + file

    MCSHs_descriptor = {
        "MCSHs": MCSHs,
        "atom_gaussians": gaussians,
        # "cutoff": 10.0,
        "square": False,
        "solid_harmonics": True
    }

    trainer = AtomsTrainer()
    elements = trainer.get_unique_elements(images)
    descriptor = GMPOrderNorm(MCSHs=MCSHs_descriptor, elements=elements)
    atoms_to_data = AtomsToData(
        descriptor=descriptor,
        r_energy=True,
        r_forces=False,
        save_fps=True,
        fprimes=False,
        cores=1
    )

    torch_data = atoms_to_data.convert_all(images)
    scaling = {"type": "normalize", "range": (0, 1), "threshold": 1e-6}
    feature_scaler = FeatureScaler(torch_data, False, scaling)
    target_scaler = TargetScaler(torch_data, False)
    feature_scaler.norm(torch_data)
    target_scaler.norm(torch_data)
    return torch_data

    
# def get_clusters(averaged_fingerprints):
#     reduced_data = reduce_dimensions_with_pca(averaged_fingerprints, max_components=6)
#     scaled_data = scale_and_standardize_data(reduced_data)
#     clusters = subsample_clustering(scaled_data, max_clusters=35)
#     return clusters


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
    