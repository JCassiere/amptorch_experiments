import matplotlib.pyplot as plt
import json
import os
# EXPERIMENTS = [
#     "oc20_3k_full_dataset_1677525998.11669",
#     "oc20_3k_random_subsample_1676735829.709882",
#     "oc20_3k_kmeans_subsample_1676735999.871988",
#     "oc20_3k_faiss_flat_subsample_1676736119.6817799",
#     "oc20_3k_faiss_ivf_subsample_1676736181.562736",
#     "oc20_3k_faiss_ivfpq_subsample_1676736261.211222"
# ]
EXPERIMENTS = [
    "qm9_full_dataset_1677563436.0786748",
    "qm9_random_subsample_1677359075.102741",
    "qm9_kmeans_subsample_1677359793.545002",
    "qm9_faiss_flat_subsample_1677365442.3335621",
    "qm9_faiss_ivf_subsample_1677365592.81146",
    "qm9_faiss_ivfpq_subsample_1677365728.198956"
]

EXPERIMENT_TITLES = [
    "Full Dataset",
    "Random",
    "Kmeans",
    "Faiss - Flat",
    "Faiss - IVF",
    "Faiss - IVFPQ"
]


def get_bar_chart_data():
    std_devs = []
    wallclock_times = []
    num_points = []
    for exp in EXPERIMENTS:
        with open("results/" + exp + "/metadata.json") as f:
            metadata = json.load(f)
            std_devs.append(metadata["std_dev"])
            wallclock_times.append(metadata["wallclock_time"])
            num_points.append(metadata["num_points"])
    return std_devs, wallclock_times, num_points


def bar_chart(x_labels, y_label, data, title):
    plt.bar(x_labels, data, color='red')
    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    
def cutoff_scatter_plot(x, y, title, y_label, log_y=True):
    plt.plot(x, y, '-o')
    plt.semilogx()
    plt.xlabel("Cutoff Modifier")
    plt.ylabel(y_label)
    plt.title(title)
    if log_y:
        plt.semilogy()
    plt.show()

    
    
def cutoff_chart(cutoff_exp):
    cutoff_folder = "results/" + cutoff_exp + "/"
    directory = os.fsencode(cutoff_folder)
    cutoffs_with_metrics = []
    
    for folder in os.listdir(directory):
        folder_name = os.fsdecode(folder)
        with open(cutoff_folder + folder_name + "/metadata.json") as f:
            metadata = json.load(f)
            cutoffs_with_metrics.append({
                "cutoff": metadata["cutoff"],
                "wallclock_time": metadata["wallclock_time"],
                "num_points": metadata["num_points"],
                "std_dev": metadata["std_dev"]
            })
    cutoffs_with_metrics.sort(key=lambda x: x["cutoff"])
    cutoffs = [x["cutoff"] for x in cutoffs_with_metrics]
    wallclock_times = [x["wallclock_time"] for x in cutoffs_with_metrics]
    num_points = [x["num_points"] for x in cutoffs_with_metrics]
    std_devs = [x["std_dev"] for x in cutoffs_with_metrics]
    cutoff_scatter_plot(cutoffs, wallclock_times, "Cutoff Modifier vs. Wallclock Time", "Wallclock Time (s)", log_y=False)
    cutoff_scatter_plot(cutoffs, num_points, "Cutoff Modifier vs. Number of Points", "Number of Points")
    cutoff_scatter_plot(cutoffs, std_devs, "Cutoff Modifier vs. Nearest Neighbor Distance Standard Deviation", "Std. Dev.")
    print(1)





# for subdir, dirs, files in os.walk(cutoff_folder):
    #     for file in files:
    #         print(os.path.join(subdir, file))
    # with open("results/" + cutoff_exp + "/metadata.json") as f:
    #     metadata = json.load(f)


def main():
    # std_devs, wallclock_times, num_points = get_bar_chart_data()
    # bar_chart(EXPERIMENT_TITLES, "Std. Dev.", std_devs, "Subsampled Dataset Nearest Neighbor Distance Standard Deviation")
    # bar_chart(EXPERIMENT_TITLES, "Wallclock Times (s)", wallclock_times, "Algorithm Wallclock Times")
    # bar_chart(EXPERIMENT_TITLES, "Num Points", num_points, "Subsampled Dataset Number of Data Points")
    cutoff_chart("qm9_gmp_ivf_cutoff_variation_1677349163.449655")
    
    
if __name__ == "__main__":
    main()
