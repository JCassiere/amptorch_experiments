import ase.io
from utils import ase_atoms_to_gmp_torch_data
from ccsubsample.subsampling.utils import extract_fingerprints_with_image_indices, scale_and_standardize_data
from data_exchange import save_to_numpy


def ase_atoms_to_numpy_fingerprints(images, filename):
    torch_data = ase_atoms_to_gmp_torch_data(images)
    fingerprints, _ = extract_fingerprints_with_image_indices(torch_data)
    scale_and_standardize_data(fingerprints)
    save_to_numpy(fingerprints, filename)


def qm9_ase_data_to_numpy_gmp_fingerprints():
    train_images = ase.io.read("data/qm9_trial_120000/train_data_seed12345.extxyz", ":")
    ase_atoms_to_numpy_fingerprints(train_images, "qm9_train_gmp_fingerprints")

    test_images = ase.io.read("data/qm9_trial_120000/test_data_seed12345.extxyz", ":")
    ase_atoms_to_numpy_fingerprints(test_images, "qm9_test_gmp_fingerprints")
