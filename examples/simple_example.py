# python3
# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

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
from marl_eval.json_tools.json_utils import concatenate_json_files, pull_neptune_data
import matplotlib.pyplot as plt
import colorcet as cc
import seaborn as sns
import numpy as np

# Run in terminal: export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNjg2NjQ4OC01ZDNhLTQzZTYtODBlNS04NTBlZGQ0YmFkNjYifQ=="
# pull_neptune_data(project_name="Instadeep/Mava",tags=["decay_rware_memory"])
# concatenate_json_files("downloaded_json_data")

METRICS_TO_NORMALIZE = []
metric_name="episode_return"
legend_map = {
    "retmat": "Sable",
    "mat": "MAT",    
    "rec_mappo": "MAPPO",
    "ff_ippo": "IPPO",
    "happo": "HAPPO",
    "rec_qmix": "QMIX",
    "rec_iql": "IQL",
     "ff_masac": "MASAC",
    "ff_hasac": "HASAC",
}

algorithms = list(legend_map.values())
colors = dict(zip(algorithms, sns.color_palette(cc.glasbey_category10)))

env_name = "LevelBasedForaging" # RobotWarehouse LevelBasedForaging Mabrax Smax
with open("concatenated_json_files/metrics.json") as f:
    raw_data = json.load(f)

processed_data = data_process_pipeline(
    raw_data=raw_data, metrics_to_normalize=METRICS_TO_NORMALIZE
)

environment_comparison_matrix, sample_effeciency_matrix = create_matrices_for_rliable(
    data_dictionary=processed_data,
    environment_name=env_name,
    metrics_to_normalize=METRICS_TO_NORMALIZE,
)

# Create folder for storing plots
if not os.path.exists("examples/plots/"):
    os.makedirs("examples/plots/")

##############################
# Aggregated Results
##############################
fig, _, _ = sample_efficiency_curves(  # type: ignore
    sample_effeciency_matrix,
    metric_name=metric_name,
    metrics_to_normalize=METRICS_TO_NORMALIZE,
    legend_map=legend_map,
    colors=colors,
)
# legend = plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.9), prop={'size': 15}, ncol=7, bbox_transform=plt.gcf().transFigure, borderaxespad=0.2, frameon=True)
# fig  = legend.figure
# fig.canvas.draw()
# bbox  = legend.get_window_extent()
# bbox = bbox.from_extents(*(bbox.extents + np.array([-4,-4,4,4])))
# bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
# fig.savefig('legend.png', dpi=1200, bbox_inches=bbox)
# plt.hlines(y=2, xmin=0, xmax=4e7, colors='gray', linestyles='--', label='Threshold')
# plt.legend()
fig.figure.savefig(f"examples/plots/{env_name}_sample_effeciency_curve.pdf", bbox_inches="tight")

# # Aggregate data over all environment tasks.
# fig = performance_profiles(
#     environment_comparison_matrix,
#     metric_name=metric_name,
#     metrics_to_normalize=METRICS_TO_NORMALIZE,
#     legend_map=legend_map,
#     colors=colors,
# )
# # plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.9), prop={'size': 17}, ncol=5, bbox_transform=plt.gcf().transFigure, borderaxespad=0.2, frameon=True)
# fig.figure.savefig("examples/plots/return_performance_profile.pdf", bbox_inches="tight")

# fig, _, _ = aggregate_scores(  # type: ignore
#     environment_comparison_matrix,
#     metric_name=metric_name,
#     metrics_to_normalize=METRICS_TO_NORMALIZE,
#     save_tabular_as_latex=True,
#     legend_map=legend_map,
# )
# fig.figure.savefig( "examples/plots/return_aggregate_scores.pdf", bbox_inches="tight")

# fig = probability_of_improvement(
#     environment_comparison_matrix,
#     metric_name=metric_name,
#     metrics_to_normalize=METRICS_TO_NORMALIZE,
#     algorithms_to_compare=[
#         ["ff_mappo", "ff_ippo"],
#         ["rec_mappo", "rec_ippo"],
#         ["ff_mappo", "rec_mappo"],
#         ["ff_ippo", "rec_ippo"],
#     ],
#     legend_map=legend_map,
# )
# fig.figure.savefig("examples/plots/return_prob_of_improvement.pdf", bbox_inches="tight")

##############################
# Single Task Plots
##############################

# Aggregate data over a single task

for task in processed_data[env_name.lower()].keys(): 
    fig = plot_single_task(
        processed_data=processed_data,
        environment_name=env_name,
        task_name=task,
        metric_name=metric_name,
        metrics_to_normalize=METRICS_TO_NORMALIZE,
        legend_map=legend_map,
        colors=colors,
    )

    fig.figure.savefig(f"examples/plots/{env_name}_{task}_agg_return.pdf", bbox_inches="tight")
