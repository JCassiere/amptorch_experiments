import time
import ase.io
import numpy as np
import os
import cProfile
import pickle
import torch

# from amptorch.trainer import AtomsTrainer
# from amptorch.preprocessing import AtomsToData, FeatureScaler, TargetScaler
# from amptorch.descriptor.GMP impo
# from amptorch.sampler.NearestNeighbor import NearestNeighbor


# def convert_and_pickle_atoms(images, dump_name):
#     sigmas = [0.02, 0.2, 0.4, 0.69, 1.1, 1.66, 2.66, 4.4]
#
#     gaussians = {}
#     dir = "../valence_gaussians"
#     for file in os.listdir(dir):
#         el = file.split("_")[0]
#         gaussians[el] = dir + "/" + file
#
#     MCSHs = {
#         "MCSHs": {
#             "0": {"groups": [1], "sigmas": sigmas},
#             "1": {"groups": [1], "sigmas": sigmas},
#             "2": {"groups": [1, 2], "sigmas": sigmas},
#             "3": {"groups": [1, 2, 3], "sigmas": sigmas},
#         },
#         "atom_gaussians": gaussians,
#         "cutoff": 8,
#     }
#
#     trainer = AtomsTrainer()
#     elements = trainer.get_unique_elements(images)
#     descriptor = GMP(MCSHs=MCSHs, elements=elements)
#     atoms_to_data = AtomsToData(
#         descriptor=descriptor,
#         r_energy=True,
#         r_forces=False,
#         save_fps=True,
#         fprimes=False,
#         cores=1
#     )
#
#     torch_data = atoms_to_data.convert_all(images)
#     scaling = {"type": "normalize", "range": (0, 1), "threshold": 1e-6}
#     feature_scaler = FeatureScaler(torch_data, False, scaling)
#     target_scaler = TargetScaler(torch_data, False)
#     feature_scaler.norm(torch_data)
#     target_scaler.norm(torch_data)
#     pickle.dump(torch_data, open("{}.p".format(dump_name), "wb"))

def load_qm9_images():
    with open("data/QM9_train_120000_linear_fit.p", 'rb') as pickle_file:
        images = pickle.load(pickle_file)

    with open("data/QM9_test_120000_linear_fit.p", 'rb') as pickle_file:
        test_images = pickle.load(pickle_file)
    
    for i in range(len(images)):
        images[i].arrays['positions'] = images[i].arrays['positions'].astype('float32')
        images[i].calc.atoms.arrays['positions'] = images[i].calc.atoms.arrays['positions'].astype('float32')
        images[i].calc.atoms.positions = images[i].calc.atoms.positions.astype('float32')
        images[i].calc.results['energy'] = float(images[i].calc.results['energy'])
    
    for i in range(len(test_images)):
        test_images[i].arrays['positions'] = test_images[i].arrays['positions'].astype('float32')
        test_images[i].calc.atoms.arrays['positions'] = test_images[i].calc.atoms.arrays['positions'].astype('float32')
        test_images[i].calc.atoms.positions = test_images[i].calc.atoms.positions.astype('float32')
        test_images[i].calc.results['energy'] = float(test_images[i].calc.results['energy'])
        
    return images, test_images

# def load_and_pickle_qm9():
#     images, test_images = load_qm9_images()
#     convert_and_pickle_atoms(images, "data/QM9_train_torch_data")
#     convert_and_pickle_atoms(test_images, "data/QM9_test_torch_data")

def load_pickled_QM9_torch_data():
    data = pickle.load(open("data/QM9_train_torch_data.p", "rb"))
    return data

def load_oc20_images():
    images = ase.io.read("data/oc20_3k_train.traj", ":")
    test_images = ase.io.read("data/oc20_300_test.traj", ":")
    return images, test_images

# def pickle_oc20():
#     images, test_images = load_oc20_images()
#     convert_and_pickle_atoms(images, "data/oc20_train_torch_data")
#     convert_and_pickle_atoms(test_images, "data/oc20_test_torch_data")

def load_pickled_oc20_torch_data():
    data = pickle.load(open("data/oc20_train_torch_data.p", "rb"))
    return data

def get_config():
    """
    Need to manually add in images afterwards
    :return:
    """
    sigmas = [0.02, 0.2, 0.4, 0.69, 1.1, 1.66, 2.66, 4.4]
    
    gaussians = {}
    dir = "../valence_gaussians"
    for file in os.listdir(dir):
        el = file.split("_")[0]
        gaussians[el] = dir + "/" + file
    
    MCSHs = {
        "MCSHs": {
            "0": {"groups": [1], "sigmas": sigmas},
            "1": {"groups": [1], "sigmas": sigmas},
            "2": {"groups": [1, 2], "sigmas": sigmas},
            "3": {"groups": [1, 2, 3], "sigmas": sigmas},
        },
        "atom_gaussians": gaussians,
        "cutoff": 8,
    }

    nns_sampling = {
        "sampling_method": "nns",  # string. use nearest-neighbor sampling
        "sampling_params": sample_config(),
        "save": False,  # boolean. whether to save the sampling results.
    }
    
    # config = {
    #     "model": {
    #         "name": "singlenn",
    #         "get_forces": False,
    #         "num_layers": 3,
    #         "num_nodes": 20,
    #     },
    #     "optim": {
    #         "device": "cpu",
    #         "force_coefficient": 0.0,
    #         "lr": 1e-2,
    #         "batch_size": 10,
    #         "epochs": 100,
    #     },
    #     "dataset": {
    #         "raw_data": None,
    #         # "raw_data": "./data/oc20_3k_train.traj",
    #         "val_split": 0,
    #         "fp_scheme": "gmp",
    #         "fp_params": MCSHs,
    #         "save_fps": True,
    #         "sampling": nns_sampling,  # sampling config here
    #     },
    #     "cmd": {
    #         "debug": False,
    #         "run_dir": "./",
    #         "seed": 1,
    #         "identifier": "test",
    #         "verbose": True,
    #         "logger": False,
    #     },
    # }
    config = {
        "model": {
            "name":"singlenn",
            "get_forces": False,
            "hidden_layers": [64, 64, 64],
            #"elementwise":False,
            "activation": torch.nn.GELU,
            "batchnorm": True,
            "initialization": "xavier",
        },
        "optim": {
            "gpus": 0, # for cpu training
            "force_coefficient": 0.0,
            "lr": 5e-3,
            "batch_size": 128,
            "epochs": 4000,
            "loss": "mae",
            "scheduler": {
                "policy": "StepLR",
                "params": {
                    "step_size": 300,
                    "gamma": 0.7
                }
            }
        },
        "dataset": {
            "raw_data": None,
            # "raw_data": "./data/oc20_3k_train.traj",
            "val_split": 0.1,
            "fp_scheme": "gmp",
            "fp_params": MCSHs,
            "save_fps": True,
            "sampling": nns_sampling,  # sampling config here
        },
        "cmd": {
            "debug": False,
            "run_dir": "./",
            "seed": 1,
            "identifier": "test",
            "verbose": True,
            "logger": False,
            "dtype": torch.DoubleTensor,
            "early_stopping": True,
            "early_stopping_patience": 75
        }
    }
    
    return config

def sample_config():
    return {
        "cutoff": 0.25,  # float. cutoff defined in terms of standard deviation
        "image_average": False,  # boolean. using atomic fingerprints
        "preprocess": {
            "method": "PCA",  # performing preprocessing used by PCA
            "max_component": 6,  # int. number of PCs
            "target_variance": 0.99,  # float.
        },
    }
    
# def get_error_QM9():
#     images, test_images = load_qm9_images()
#     images = images
#
#     config = get_config()
#     config["dataset"]["raw_data"] = images
#
#     trainer = AtomsTrainer(config)
#     trainer.train()
#
#     predictions = trainer.predict(test_images)
#
#     true_energies = np.array([image.get_potential_energy() for image in test_images])
#     pred_energies = np.array(predictions["energy"])
#
#     print("Energy MSE:", np.mean((true_energies - pred_energies) ** 2))

# def error_qm9_10_runs():
#     errors = []
#     images, test_images = load_qm9_images()
#     for i in range(10):
#         config = get_config()
#         config["dataset"]["raw_data"] = images
#
#         trainer = AtomsTrainer(config)
#         trainer.train()
#
#         _, test_images = load_qm9_images()
#         predictions = trainer.predict(test_images)
#
#         true_energies = np.array([image.get_potential_energy() for image in test_images])
#         pred_energies = np.array(predictions["energy"])
#         error = np.mean((true_energies - pred_energies) ** 2)
#         errors.append(error)
#         print("Energy MSE:", error)
#
#     errors = np.array(errors)
#     print("Median Energy MSE:", np.median(errors))
#     print("Average Energy MSE:", np.mean(errors))

# def error_oc20_10_runs():
#     errors = []
#     for i in range(10):
#         config = get_config()
#         config["dataset"]["raw_data"] = "../data/oc20_3k_train.traj"
#
#         trainer = AtomsTrainer(config)
#         trainer.train()
#
#         test_images = ase.io.read("../data/oc20_300_test.traj", ":")
#         predictions = trainer.predict(test_images)
#
#         true_energies = np.array([image.get_potential_energy() for image in test_images])
#         pred_energies = np.array(predictions["energy"])
#         error = np.mean((true_energies - pred_energies) ** 2)
#         errors.append(error)
#         print("Energy MSE:", error)
#
#     errors = np.array(errors)
#     print("Median Energy MSE:", np.median(errors))
#     print("Average Energy MSE:", np.mean(errors))

# def error_oc20():
#     config = get_config()
#     config["dataset"]["raw_data"] = "../data/oc20_3k_train.traj"
#
#     trainer = AtomsTrainer(config)
#     trainer.train()
#
#     test_images = ase.io.read("../data/oc20_300_test.traj", ":")
#     predictions = trainer.predict(test_images)
#
#     true_energies = np.array([image.get_potential_energy() for image in test_images])
#     pred_energies = np.array(predictions["energy"])
#     error = np.mean((true_energies - pred_energies) ** 2)
#     print("Energy MSE:", error)
    
# def error_h2o():
#     config = get_config()
#     config["dataset"]["raw_data"] = "../data/water_data.traj"
#     trainer = AtomsTrainer(config)
#     trainer.train()
#
#     test_images = ase.io.read("../data/water_validation.traj", ":")
#     predictions = trainer.predict(test_images)
#
#     true_energies = np.array([image.get_potential_energy() for image in test_images])
#     pred_energies = np.array(predictions["energy"])
#     print("Energy MSE:", np.mean((true_energies - pred_energies) ** 2))

# def error_h2o_10_runs():
#     errors = []
#     for i in range(10):
#         config = get_config()
#         config["dataset"]["raw_data"] = "../data/water_data.traj"
#         # config["dataset"]["sampling"] = None
#
#         trainer = AtomsTrainer(config)
#         trainer.train()
#
#         test_images = ase.io.read("../data/water_validation.traj", ":")
#         predictions = trainer.predict(test_images)
#
#         true_energies = np.array([image.get_potential_energy() for image in test_images])
#         pred_energies = np.array(predictions["energy"])
#         error = np.mean((true_energies - pred_energies) ** 2)
#         errors.append(error)
#         print("Energy MSE:", error)
#
#     errors = np.array(errors)
#     print("Median Energy MSE:", np.median(errors))
#     print("Average Energy MSE:", np.mean(errors))

# def error_h2o_no_sample():
#     config = get_config()
#     config["dataset"]["raw_data"] = "./data/water_data.traj"
#     config["dataset"]["sampling"] = None
#     trainer = AtomsTrainer(config)
#     trainer.train()
#
#     test_images = ase.io.read("../data/water_validation.traj", ":")
#     predictions = trainer.predict(test_images)
#
#     true_energies = np.array([image.get_potential_energy() for image in test_images])
#     pred_energies = np.array(predictions["energy"])
#     print("Energy MSE:", np.mean((true_energies - pred_energies) ** 2))

# def error_QM9_no_sample():
#     images, test_images = load_qm9_images()
#
#     config = get_config()
#     config["dataset"]["raw_data"] = images
#     config["dataset"]["sampling"] = None
#
#     trainer = AtomsTrainer(config)
#     trainer.train()
#
#     predictions = trainer.predict(test_images)
#
#     true_energies = np.array([image.get_potential_energy() for image in test_images])
#     pred_energies = np.array(predictions["energy"])
#
#     print("Energy MSE:", np.mean((true_energies - pred_energies) ** 2))

# def error_oc20_no_sample():
#     config = get_config()
#     config["dataset"]["raw_data"] = "../data/oc20_3k_train.traj"
#     config["dataset"]["sampling"] = None
#     trainer = AtomsTrainer(config)
#     trainer.train()
#
#     test_images = ase.io.read("../data/oc20_300_test.traj", ":")
#     predictions = trainer.predict(test_images)
#
#     true_energies = np.array([image.get_potential_energy() for image in test_images])
#     pred_energies = np.array(predictions["energy"])
#     print("Energy MSE:", np.mean((true_energies - pred_energies) ** 2))

# TODO - completely eliminate amptorch from workflow
if __name__ == "__main__":
    start_time = time.time()
    # pickle_oc20()
    # load_and_pickle_qm9()
    # cProfile.run("error_oc20()", filename="subsampling.prof")
    # get_error_QM9()
    # error_qm9_10_runs()
    # error_h2o_10_runs()
    # error_QM9_no_sample()
    # error_oc20_10_runs()
    error_oc20()
    # error_h2o()
    # error_h2o_no_sample()
    error_oc20_no_sample()
    end_time = time.time()
    print("Total time elapsed: {}".format(end_time - start_time))