# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
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

from marl_tools.plotting_tools.plotting import (
    aggregate_scores,
    performance_profiles,
    probability_of_improvement,
    sample_efficiency_curves,
)
from marl_tools.utils.data_processing_utils import (
    create_matrices_for_rliable,
    data_process_pipeline,
)

# Read in and process data
METRICS_TO_NORMALIZE = ["return"]

with open("examples/example_results.json", "r") as f:
    raw_data = json.load(f)

processed_data = data_process_pipeline(
    raw_data=raw_data, metrics_to_normalize=METRICS_TO_NORMALIZE
)

with open("examples/processed_example_results.json", "w+") as f:
    json.dump(processed_data, f, indent=4)

environment_comparison_matrix, sample_effeciency_matrix = create_matrices_for_rliable(
    data_dictionary=processed_data,
    environment_name="env_1",
    metrics_to_normalize=METRICS_TO_NORMALIZE,
)

# Create folder for storing plots
if not os.path.exists("examples/plots/"):
    os.makedirs("examples/plots/")

# Plot success rate data
fig = performance_profiles(
    environment_comparison_matrix,
    metric_name="success_rate",
    metrics_to_normalize=METRICS_TO_NORMALIZE,
)
fig.figure.savefig(
    "examples/plots/success_rate_performance_profile.png", bbox_inches="tight"
)

fig, _, _ = aggregate_scores(  # type: ignore
    environment_comparison_matrix,
    metric_name="success_rate",
    metrics_to_normalize=METRICS_TO_NORMALIZE,
)
fig.figure.savefig(
    "examples/plots/success_rate_aggregate_scores.png", bbox_inches="tight"
)

fig = probability_of_improvement(
    environment_comparison_matrix,
    metric_name="success_rate",
    algorithms_to_compare=[
        ["algo_1", "algo_2"],
        ["algo_1", "algo_3"],
        ["algo_2", "algo_4"],
    ],
)
fig.figure.savefig(
    "examples/plots/success_rate_prob_of_improvement.png", bbox_inches="tight"
)

fig, _, _ = sample_efficiency_curves(
    sample_effeciency_matrix,
    metric_name="success_rate",
    metrics_to_normalize=METRICS_TO_NORMALIZE,
)
fig.figure.savefig(
    "examples/plots/success_rate_sample_effeciency_curve.png", bbox_inches="tight"
)

# Plot episode return data
fig = performance_profiles(
    environment_comparison_matrix,
    metric_name="norm_return",
    metrics_to_normalize=METRICS_TO_NORMALIZE,
)
fig.figure.savefig("examples/plots/return_performance_profile.png", bbox_inches="tight")

fig, _, _ = aggregate_scores(  # type: ignore
    environment_comparison_matrix,
    metric_name="norm_return",
    metrics_to_normalize=METRICS_TO_NORMALIZE,
)
fig.figure.savefig("examples/plots/return_aggregate_scores.png", bbox_inches="tight")

fig = probability_of_improvement(
    environment_comparison_matrix,
    metric_name="norm_return",
    algorithms_to_compare=[
        ["algo_1", "algo_2"],
        ["algo_1", "algo_3"],
        ["algo_2", "algo_4"],
    ],
)
fig.figure.savefig("examples/plots/return_prob_of_improvement.png", bbox_inches="tight")

fig, _, _ = sample_efficiency_curves(
    sample_effeciency_matrix,
    metric_name="norm_return",
    metrics_to_normalize=METRICS_TO_NORMALIZE,
)
fig.figure.savefig(
    "examples/plots/return_sample_effeciency_curve.png", bbox_inches="tight"
)
