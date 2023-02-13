import time
from data_exchange import save_to_numpy, load_numpy_data
from ccsubsample.subsampling.utils import *
from ccsubsample.subsampling.evaluation_metrics import *
from ccsubsample.subsampling.subsampling import faiss_flat_subsample, faiss_ivf_subsample, faiss_ivfpq_subsample, kmeans_subsample


def kmeans_subsampling_experiment(scaled_data, title, num_points_desired):
    # need - wallclock time, std. dev, number of points, cutoff, and filename for
    #  resulting points in an experiment metadata json
    start = time.time()
    indices_to_keep = kmeans_subsample(scaled_data, num_points_desired)
    wallclock_time = time.time() - start
    points_to_keep = scaled_data[indices_to_keep]
    subset_filename = title + "_subset"
    save_to_numpy(points_to_keep, subset_filename, "results")
    num_points = number_of_points_remaining(points_to_keep)
    _, std_dev = point_diversity_mean_std(points_to_keep)
    experiment_metadata = {
        "experiment_title": title,
        "wallclock_time": wallclock_time,
        "num_points": num_points,
        "subset_filename": subset_filename,
        "std_dev": float(std_dev)
    }
    metadata_file = open("results/" + title + "_{}_metadata.json".format(time.time()), "w")
    json.dump(experiment_metadata, metadata_file, indent=4)


def faiss_subsampling_experiment(scaled_data, subsample_fn, title, cutoff):
    # need - wallclock time, std. dev, number of points, cutoff, and filename for
    #  resulting points in an experiment metadata json
    start = time.time()
    indices_to_keep = subsample_fn(scaled_data, cutoff_sig=cutoff, verbose=2)
    wallclock_time = time.time() - start
    points_to_keep = scaled_data[indices_to_keep]
    subset_filename = title + "_subset"
    save_to_numpy(points_to_keep, subset_filename, "results")
    num_points = number_of_points_remaining(points_to_keep)
    _, std_dev = point_diversity_mean_std(points_to_keep)
    experiment_metadata = {
        "experiment_title": title,
        "cutoff": cutoff,
        "wallclock_time": wallclock_time,
        "num_points": num_points,
        "subset_filename": subset_filename,
        "std_dev": float(std_dev)
    }
    metadata_file = open("results/" + title + "_{}_metadata.json".format(time.time()), "w")
    json.dump(experiment_metadata, metadata_file, indent=4)

def load_oc20_3k_data():
    data = pickle.load(open("./data/oc20_train_torch_data.p", "rb"))
    return data

def load_qm9_data():
    data = pickle.load(open("./data/QM9_train_torch_data.p", "rb"))
    return data

def main():
    scaled_data = load_numpy_data("scaled_and_std_oc20_3k_gmp_fingerprints")
    kmeans_subsampling_experiment(scaled_data, "oc20_3k_kmeans_subsample", 20000)
    faiss_subsampling_experiment(scaled_data, faiss_flat_subsample, "oc20_3k_faiss_flat_subsample", cutoff=2)
    faiss_subsampling_experiment(scaled_data, faiss_ivf_subsample, "oc20_3k_faiss_ivf_subsample", cutoff=2)
    faiss_subsampling_experiment(scaled_data, faiss_ivfpq_subsample, "oc20_3k_faiss_ivfpq_subsample", cutoff=2)
    print(1)

if __name__ == "__main__":
    main()