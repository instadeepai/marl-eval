import json
import os

import matplotlib.pyplot as plt

from marl_eval.plotting_tools.plotting import (
    aggregate_scores,
    performance_profiles,
    plot_single_task,
    probability_of_improvement,
    sample_efficiency_curves,
)
from marl_eval.utils.data_processing_utils import (
    create_matrices_for_rliable,
    data_process_pipeline,
)

ENV_NAME = "Matrax"
SAVE_PDF = True

data_dir = "./concatenated_json_files/metrics.json"
png_plot_dir = "./plots/png/"
pdf_plot_dir = "./plots/pdf/"

legend_map = {
    "ff_ippo": "FF IPPO",
    "ff_mappo": "FF MAPPO",
    "rec_ippo": "REC IPPO",
    "rec_mappo": "REC MAPPO",
}

##############################
# Read in and process data
##############################
METRICS_TO_NORMALIZE = ["mean_episode_return"]

with open(data_dir) as f:
    raw_data = json.load(f)

custom_max = {
    "climbing-stateless-v0": 275.0,
    "penalty-0-stateless-v0": 250.0,
    "penalty-25-stateless-v0": 250.0,
    "penalty-50-stateless-v0": 250.0,
    "penalty-75-stateless-v0": 250.0,
    "penalty-100-stateless-v0": 250.0,
    "noconflict-15-stateless-v0": 100.0,
    "noconflict-16-stateless-v0": 100.0,
    "noconflict-17-stateless-v0": 100.0,
    "noconflict-18-stateless-v0": 100.0,
    "noconflict-19-stateless-v0": 100.0,
    "noconflict-20-stateless-v0": 100.0,
}

custom_min = {
    "climbing-stateless-v0": -750.0,
    "penalty-0-stateless-v0": 0.0,
    "penalty-25-stateless-v0": -625.0,
    "penalty-50-stateless-v0": -1250.0,
    "penalty-75-stateless-v0": -1875.0,
    "penalty-100-stateless-v0": -2500.0,
    "noconflict-15-stateless-v0": 25.0,
    "noconflict-16-stateless-v0": 25.0,
    "noconflict-17-stateless-v0": 37.5,
    "noconflict-18-stateless-v0": 50.0,
    "noconflict-19-stateless-v0": 37.5,
    "noconflict-20-stateless-v0": 37.5,
}

processed_data = data_process_pipeline(
    raw_data=raw_data,
    metrics_to_normalize=METRICS_TO_NORMALIZE,
    custom_max=custom_max,
    custom_min=custom_min,
)

environment_comparison_matrix, sample_efficiency_matrix = create_matrices_for_rliable(
    data_dictionary=processed_data,
    environment_name=ENV_NAME,
    metrics_to_normalize=METRICS_TO_NORMALIZE,
)

# Create folder for storing plots
if not os.path.exists(png_plot_dir):
    os.makedirs(png_plot_dir)
if not os.path.exists(pdf_plot_dir):
    os.makedirs(pdf_plot_dir)

##############################
# Plot episode return data
##############################

# Get all tasks
tasks = list(processed_data[ENV_NAME.lower()].keys())

# Aggregate data over a single task
for task in tasks:
    fig = plot_single_task(
        processed_data=processed_data,
        environment_name=ENV_NAME,
        task_name=task,
        metric_name="mean_episode_return",
        metrics_to_normalize=METRICS_TO_NORMALIZE,
        legend_map=legend_map,
    )

    fig.figure.savefig(
        f"{png_plot_dir}matrax_{task}_agg_return.png", bbox_inches="tight"
    )
    if SAVE_PDF:
        fig.figure.savefig(
            f"{pdf_plot_dir}matrax_{task}_agg_return.pdf", bbox_inches="tight"
        )

    # Close the figure object
    plt.close(fig.figure)

# Aggregate data over all environment tasks.

fig, _, _ = sample_efficiency_curves(  # type: ignore
    sample_efficiency_matrix,
    metric_name="mean_episode_return",
    metrics_to_normalize=METRICS_TO_NORMALIZE,
    legend_map=legend_map,
)
fig.figure.savefig(
    f"{png_plot_dir}return_sample_efficiency_curve.png", bbox_inches="tight"
)
if SAVE_PDF:
    fig.figure.savefig(
        f"{pdf_plot_dir}return_sample_efficiency_curve.pdf", bbox_inches="tight"
    )

# probability of improvement
fig = probability_of_improvement(
    environment_comparison_matrix,
    metric_name="mean_episode_return",
    metrics_to_normalize=METRICS_TO_NORMALIZE,
    algorithms_to_compare=[
        ["ff_ippo", "ff_mappo"],
        ["ff_ippo", "rec_ippo"],
        ["ff_ippo", "rec_mappo"],
        ["ff_mappo", "rec_ippo"],
        ["ff_mappo", "rec_mappo"],
        ["rec_ippo", "rec_mappo"],
    ],
    legend_map=legend_map,
)
fig.figure.savefig(f"{png_plot_dir}prob_of_improvement.png", bbox_inches="tight")
if SAVE_PDF:
    fig.figure.savefig(f"{pdf_plot_dir}prob_of_improvement.pdf", bbox_inches="tight")

# aggregate scores
fig, _, _ = aggregate_scores(  # type: ignore
    environment_comparison_matrix,
    metric_name="mean_episode_return",
    metrics_to_normalize=METRICS_TO_NORMALIZE,
    save_tabular_as_latex=True,
    legend_map=legend_map,
)
fig.figure.savefig(f"{png_plot_dir}aggregate_scores.png", bbox_inches="tight")
if SAVE_PDF:
    fig.figure.savefig(f"{pdf_plot_dir}aggregate_scores.pdf", bbox_inches="tight")

# performance profiles
fig = performance_profiles(
    environment_comparison_matrix,
    metric_name="mean_episode_return",
    metrics_to_normalize=METRICS_TO_NORMALIZE,
    legend_map=legend_map,
)
fig.figure.savefig(f"{png_plot_dir}performance_profile.png", bbox_inches="tight")
if SAVE_PDF:
    fig.figure.savefig(f"{pdf_plot_dir}performance_profile.pdf", bbox_inches="tight")
