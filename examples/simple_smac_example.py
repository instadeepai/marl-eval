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

METRICS_TO_NORMALIZE = ["return"]

with open("examples/smac_dummy_data.json", "r") as f:
    raw_data = json.load(f)

processed_data = data_process_pipeline(
    raw_data=raw_data, metrics_to_normalize=METRICS_TO_NORMALIZE
)

with open("examples/smac_processed_dummy_data.json", "w+") as f:
    json.dump(processed_data, f, indent=4)

environment_comparison_matrix, sample_effeciency_matrix = create_matrices_for_rliable(
    data_dictionary=processed_data,
    environment_name="SMAC",
    metrics_to_normalize=METRICS_TO_NORMALIZE,
)

fig = performance_profiles(
    environment_comparison_matrix,
    metric_name="return",
    metrics_to_normalize=METRICS_TO_NORMALIZE,
)
fig.figure.savefig("examples/demo_return_performance_profile.png", bbox_inches="tight")

fig, _, _, _ = aggregate_scores(
    environment_comparison_matrix,
    metric_name="return",
    metrics_to_normalize=METRICS_TO_NORMALIZE,
)
fig.figure.savefig("examples/demo_return_aggregate_scores.png", bbox_inches="tight")

fig = probability_of_improvement(
    environment_comparison_matrix,
    metric_name="win_rate",
    algorithms_to_compare=[["QMIX", "VDN"], ["VDN", "MAPPO"], ["MADQN", "QMIX"]],
)
fig.figure.savefig("examples/demo_return_prob_of_improvement.png", bbox_inches="tight")

fig, _, _ = sample_efficiency_curves(
    sample_effeciency_matrix,
    metric_name="win_rate",
    metrics_to_normalize=METRICS_TO_NORMALIZE,
)
fig.figure.savefig(
    "examples/demo_win_rate_sample_effeciency_curve.png", bbox_inches="tight"
)
