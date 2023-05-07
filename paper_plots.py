import matplotlib.pyplot as plt
import json
import os
EXPERIMENTS = [
    "oc20_3k_gmp_full_dataset_1681685394.594069",
    "oc20_3k_gmp_random_subsample_1681685407.232893",
    "oc20_3k_gmp_kmeans_subsample_1681685699.9315782",
    "oc20_3k_gmp_sobol_1681685714.9607081",
    "oc20_3k_gmp_fps_1681686650.4540448",
    "oc20_3k_gmp_fps_batched_1681686677.3794692",
    "oc20_3k_gmp_faiss_flat_subsample_1681686865.0633442",
    "oc20_3k_gmp_faiss_ivf_subsample_1681686890.514122",
    "oc20_3k_gmp_faiss_ivfpq_subsample_1681686921.022644"
]
# EXPERIMENTS = [
#     "qm9_full_dataset_1677563436.0786748",
#     "qm9_random_subsample_1677359075.102741",
#     "qm9_kmeans_subsample_1677359793.545002",
#     # TODO - add sobol and fps
#     "qm9_faiss_flat_subsample_1677365442.3335621",
#     "qm9_faiss_ivf_subsample_1677365592.81146",
#     "qm9_faiss_ivfpq_subsample_1677365728.198956"
# ]

EXPERIMENT_TITLES = [
    "Full Dataset",
    "Random",
    "Kmeans",
    "Sobol",
    "FPS",
    "FPS Batched",
    "Faiss - Flat",
    "Faiss - IVF",
    "Faiss - IVFPQ"
]

OUTLIER_RETENTION_EXPERIMENTS = [
    "oc20_3k_gmp_faiss_ivf_outlier_retention_variance_1681688171.0504801",
    "oc20_3k_gmp_faiss_ivfpq_outlier_retention_variance_1681688192.213641",
    "oc20_3k_gmp_fps_outlier_retention_variance_1681688846.191094",
    "oc20_3k_gmp_fps_batched_outlier_retention_variance_1681688861.991261",
    "oc20_3k_gmp_kmeans_outlier_retention_variance_1681688483.312287",
    "oc20_3k_gmp_sobol_outlier_retention_variance_1681688485.829708"
]

OUTLIER_RETENTION_TITLES = [
    "NN Subsampling - IVF",
    "NN Subsampling - IVFPQ",
    "Farthest Point Sampling",
    "FPS Batched",
    "Kmeans",
    "Sobol"
]


def get_bar_chart_data():
    std_devs = []
    wallclock_times = []
    num_points = []
    outlier_retentions = []
    discrepancies = []
    for exp in EXPERIMENTS:
        with open("results/" + exp + "/metadata.json") as f:
            metadata = json.load(f)
            std_devs.append(metadata["std_dev"])
            wallclock_times.append(metadata["wallclock_time"])
            num_points.append(metadata["num_points"])
            outlier_retentions.append(metadata.get("outlier_retention", 1))
            discrepancies.append(metadata["discrepancy"])
    return std_devs, wallclock_times, num_points, outlier_retentions, discrepancies


def bar_chart(x_labels, y_label, data, title):
    plt.bar(x_labels, data, color='red')
    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
def double_bar_chart(x_labels, y1, y2, y1_label, y2_label, title):
    ind = list(range(len(x_labels)))
    fig, host = plt.subplots()
    par1 = host.twinx()
    width = 0.3
    
    color1 = plt.cm.viridis(0)
    color2 = plt.cm.viridis(0.5)
    # Plotting
    p1 = host.bar(ind, y1, width, color=color1, label=y1_label)
    p2 = par1.bar([i + width for i in ind], y2, width, color=color2, label=y2_label)
    host.set_ylim(0, 30000)

    # plt.xlabel('Here goes x-axis label')
    # plt.ylabel('Here goes y-axis label')
    # plt.title('Here goes title of the plot')
    
    # xticks()
    # First argument - A list of positions at which ticks should be placed
    # Second argument -  A list of labels to place at the given locations
    plt.xticks([i + width / 2 for i in ind], x_labels)
    
    # Finding the best position for legends and putting it
    lns = [p1, p2]
    host.legend(handles=lns, loc='best')
    plt.show()
    
    
    
    # fig, host = plt.subplots()
    #
    # color1 = plt.cm.viridis(0)
    # color2 = plt.cm.viridis(0.5)
    # par1 = host.twinx()
    # p1 = host.bar(x_labels, y1, color=color1, label=y1_label)
    # p2 = par1.bar(x_labels, y2, color=color2, label=y2_label)
    # lns = [p1, p2]
    # host.legend(handles=lns, loc='best')
    # plt.title(title)
    # plt.show()



def single_line_plot(x, y, title, x_label,
                     y_label, log_x=True, log_y=True):
    plt.plot(x, y, '-o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if log_x:
        plt.semilogx()
    if log_y:
        plt.semilogy()
    plt.show()
    
def double_line_plot(x, y1, y2, title, x_label, y1_label, y2_label, y1_range=None, y2_range=None):
    fig, host = plt.subplots()
    
    par1 = host.twinx()
    
    # host.set_xlim(0, 2)
    if y1_range:
        host.set_ylim(y1_range[0], y1_range[1])
    if y2_range:
        par1.set_ylim(y2_range[0], y2_range[1])
    
    host.set_xlabel(x_label)
    host.set_ylabel(y1_label)
    par1.set_ylabel(y2_label)
    
    color1 = plt.cm.viridis(0)
    color2 = plt.cm.viridis(0.5)

    p1, = host.plot(x, y1, color=color1, label=y1_label)
    p2, = par1.plot(x, y2, color=color2, label=y2_label)

    lns = [p1, p2]
    host.legend(handles=lns, loc='best')
    plt.title(title)
    plt.show()
    
def cutoff_charts(cutoff_exp):
    cutoff_folder = "results/" + cutoff_exp + "/"
    directory = os.fsencode(cutoff_folder)
    cutoffs_with_metrics = []
    
    for folder in os.listdir(directory):
        folder_name = os.fsdecode(folder)
        with open(cutoff_folder + folder_name + "/metadata.json") as f:
            metadata = json.load(f)
            cutoffs_with_metrics.append({
                "cutoff_percentile": metadata["cutoff_percentile"],
                "wallclock_time": metadata["wallclock_time"],
                "num_points": metadata["num_points"],
                # "std_dev": metadata["std_dev"]
            })
    cutoffs_with_metrics.sort(key=lambda x: x["cutoff_percentile"])
    cutoffs = [x["cutoff_percentile"] for x in cutoffs_with_metrics]
    wallclock_times = [x["wallclock_time"] for x in cutoffs_with_metrics]
    num_points = [x["num_points"] for x in cutoffs_with_metrics]
    # std_devs = [x["std_dev"] for x in cutoffs_with_metrics]
    single_line_plot(cutoffs, wallclock_times, "Cutoff Percentile vs. Wallclock Time", "Cutoff Percentile", "Wallclock Time (s)", log_x=False, log_y=False)
    single_line_plot(cutoffs, num_points, "Cutoff Percentile vs. Number of Points", "Cutoff Percentile", "Number of Points", log_x=False)
    # single_line_plot(cutoffs, std_devs, "Cutoff Modifier vs. Nearest Neighbor Distance Standard Deviation", "Cutoff Modifier", "Std. Dev.")


def dim_reduction_charts(dim_reduction_exp):
    dim_reduction_folder = "results/" + dim_reduction_exp + "/"
    directory = os.fsencode(dim_reduction_folder)
    num_dims_with_metrics = []
    
    for folder in os.listdir(directory):
        folder_name = os.fsdecode(folder)
        with open(dim_reduction_folder + folder_name + "/metadata.json") as f:
            metadata = json.load(f)
            num_dims_with_metrics.append({
                "num_dimensions_kept": metadata["num_dimensions_kept"],
                "wallclock_time": metadata["wallclock_time"],
                "num_points": metadata["num_points"],
                "std_dev": metadata["std_dev"],
                "variance_retained": metadata["variance_retained"],
                "discrepancy": metadata["discrepancy"]
                # "outlier_retention": metadata["outlier_retention"]
            })
    num_dims_with_metrics.sort(key=lambda x: x["num_dimensions_kept"])
    num_dims = [x["num_dimensions_kept"] for x in num_dims_with_metrics]
    wallclock_times = [x["wallclock_time"] for x in num_dims_with_metrics]
    num_points = [x["num_points"] for x in num_dims_with_metrics]
    std_devs = [x["std_dev"] for x in num_dims_with_metrics]
    discrepancies = [x["discrepancy"] for x in num_dims_with_metrics]
    variance_retained = [x["variance_retained"] for x in num_dims_with_metrics]
    # outlier_retention = [x["outlier_retention"] for x in num_dims_with_metrics]
    # double_line_plot(num_dims, variance_retained, outlier_retention, "Variance Retained and Outlier Retention with Dimensionality Reduction",
    #                  "Number of Principal Comzponents Kept", "Variance Retained", "Outlier Retention",
    #                  y1_range=[0, 1], y2_range=[0, 1])
    double_line_plot(num_dims, num_points, discrepancies, "Num Points and Discrepancy. with Dim. Reduction",
                     "Number of Principal Components Kept", "Number of Data Points", "Discrepancy")
    single_line_plot(num_dims, wallclock_times, "Wallclock Times with Dimensionality Reduction",
                     "Number of Principal Components Kept", "Wallclock Time (s)", log_x=False, log_y=False)
    

def outlier_retention_plots():
    exps = []
    for i in range(len(OUTLIER_RETENTION_EXPERIMENTS)):
        folder = "results/" + OUTLIER_RETENTION_EXPERIMENTS[i] + "/"
        with open(folder + "outlier_retention.json") as f:
            outlier_retention = json.load(f)
            results = {
                "title": OUTLIER_RETENTION_TITLES[i],
                "percentile": outlier_retention["outlier_percentile"],
                "retention": outlier_retention["outlier_retention"]
            }
            exps.append(results)
    for exp in exps:
        plt.plot(exp["percentile"], exp["retention"], '-o', label=exp["title"])
        
    
    plt.xlabel("Outlier Definition (Nearest Neighbor Distance Percentile)")
    plt.ylabel("Outlier Retention")
    plt.title("Outlier Retention with Outlier Definition Variance")
    plt.legend()
    plt.show()


# for subdir, dirs, files in os.walk(cutoff_folder):
    #     for file in files:
    #         print(os.path.join(subdir, file))
    # with open("results/" + cutoff_exp + "/metadata.json") as f:
    #     metadata = json.load(f)


def main():
    std_devs, wallclock_times, num_points, outlier_retentions, discrepancies = get_bar_chart_data()
    # bar_chart(EXPERIMENT_TITLES, "Std. Dev.", std_devs, "Subsampled Dataset Nearest Neighbor Distance Standard Deviation")
    # bar_chart(EXPERIMENT_TITLES, "Outlier Retention", outlier_retentions, "Subsampled Dataset Outlier Retention")
    # double_bar_chart(EXPERIMENT_TITLES, num_points, std_devs, "Num Points", "Std. Dev.", "Subsampled Dataset Nearest Neighbor Distance Standard Deviation/Number of Points")
    # bar_chart(EXPERIMENT_TITLES, "Wallclock Times (s)", wallclock_times, "Algorithm Wallclock Times")
    # bar_chart(EXPERIMENT_TITLES, "Num Points", num_points, "Subsampled Dataset Number of Data Points")
    # bar_chart(EXPERIMENT_TITLES, "Discrepancy", discrepancies, "Subsampled Dataset Discrepancy")
    # cutoff_charts("oc20_3k_gmp_ivf_cutoff_variation_1681683497.948586")
    dim_reduction_charts("oc20_3k_gmp_ivf_dim_reduction_1681686922.518399")
    # outlier_retention_plots()
    # dim_reduction_charts("pre_apr16_restart/oc20_3k_gmp_ivf_dim_reduction_1676736262.799958")

if __name__ == "__main__":
    main()
