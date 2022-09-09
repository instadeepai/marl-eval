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

"""Simple script for generating synthetic Smac experiment data"""

import json
from typing import Any, Dict, List

import numpy as np


def generate_win_rate(
    max_value: float,
    min_value: float,
    mean: float,
    num_vals: int,
) -> List[float]:
    """Generates fake win rate data"""

    array = np.random.normal(loc=mean, scale=0.5, size=num_vals)
    array = np.clip(a=array, a_min=min_value, a_max=max_value)
    list_data = array.tolist()
    return list_data


def generate_return(
    max_value: float, min_value: float, mean: float, num_vals: int
) -> List[float]:
    """Generates fake episode return data"""
    array = np.random.normal(loc=mean, scale=10.0, size=num_vals)
    array = np.clip(a=array, a_min=min_value, a_max=max_value)
    list_data = array.tolist()
    return list_data


def generate_win_rate_absolute_metric(
    max_value: float, min_value: float, mean: float, num_vals: int
) -> List[float]:
    """Generates fake absolute win rate data"""
    array = np.random.normal(loc=mean, scale=0.25, size=num_vals)
    array = np.clip(a=array, a_min=min_value, a_max=max_value)
    list_data = array.tolist()
    return list_data


def generate_return_absolute_metric(
    max_value: float, min_value: float, mean: float, num_vals: int
) -> List[float]:
    """Generates fake absolute episode return data"""
    array = np.random.normal(loc=mean, scale=5.0, size=num_vals)
    array = np.clip(a=array, a_min=min_value, a_max=max_value)
    list_data = array.tolist()
    return list_data


def get_max_return(algorithm: str, scenario: str) -> float:
    """Gives the maximum possible return value for a \
        given algorithm and scenario.

    Args:
        algorithm: algorithm name
        scenario: scenarion name

    Returns:
        maximum value as a float
    """
    if scenario == "3m":
        if algorithm == "QMIX":
            return 20.0
        if algorithm == "MADQN":
            return 20.0
        if algorithm == "VDN":
            return 20.0
        if algorithm == "MAPPO":
            return 20.0

    if scenario == "8m":
        if algorithm == "QMIX":
            return 20.0
        if algorithm == "MADQN":
            return 16.0
        if algorithm == "VDN":
            return 19.0
        if algorithm == "MAPPO":
            return 19.5

    if scenario == "3s5z":
        if algorithm == "QMIX":
            return 20.0
        if algorithm == "MADQN":
            return 12.0
        if algorithm == "VDN":
            return 16.0
        if algorithm == "MAPPO":
            return 18.5


def get_max_win_rate(algorithm: str, scenario: str) -> float:
    """Gives the maximum possible win rate for a \
        given algorithm and scenario.

    Args:
        algorithm: algorithm name
        scenario: scenarion name

    Returns:
        maximum value as a float
    """
    if scenario == "3m":
        if algorithm == "QMIX":
            return 1.0
        if algorithm == "MADQN":
            return 1.0
        if algorithm == "VDN":
            return 1.0
        if algorithm == "MAPPO":
            return 1.0

    if scenario == "8m":
        if algorithm == "QMIX":
            return 1.0
        if algorithm == "MADQN":
            return 0.8
        if algorithm == "VDN":
            return 0.9
        if algorithm == "MAPPO":
            return 0.95

    if scenario == "3s5z":
        if algorithm == "QMIX":
            return 0.85
        if algorithm == "MADQN":
            return 0.6
        if algorithm == "VDN":
            return 0.75
        if algorithm == "MAPPO":
            return 0.8


runs_dictionary: Dict[str, Any] = {}

environments = ["SMAC"]
scenarios = ["3m", "3s5z", "8m"]
runs = [f"RUN_{i}" for i in range(1, 11)]
algorithms = ["QMIX", "MADQN", "VDN", "MAPPO"]

metrics = ["return", "win_rate"]

eval_length = 32
num_eval_steps = 200

for env in environments:
    if env not in runs_dictionary.keys():
        runs_dictionary[env] = {}

    for scenario in scenarios:
        if scenario not in runs_dictionary[env].keys():
            runs_dictionary[env][scenario] = {}

        for algorithm in algorithms:
            if algorithm not in runs_dictionary[env][scenario].keys():
                runs_dictionary[env][scenario][algorithm] = {}
                max_win_rate = get_max_win_rate(algorithm, scenario)
                max_return = get_max_return(algorithm, scenario)

            for run in runs:
                if run not in runs_dictionary[env][scenario][algorithm].keys():
                    runs_dictionary[env][scenario][algorithm][run] = {}

                for step in range(1, num_eval_steps + 1):
                    current_return = max_return * (step / num_eval_steps)
                    current_win_rate = max_win_rate * (step / num_eval_steps)
                    runs_dictionary[env][scenario][algorithm][run][f"STEP_{step}"] = {}
                    runs_dictionary[env][scenario][algorithm][run][f"STEP_{step}"][
                        "step_count"
                    ] = int(round(step * 10000 + np.random.random() * 10, 0))
                    runs_dictionary[env][scenario][algorithm][run][f"STEP_{step}"][
                        "return"
                    ] = generate_return(
                        max_value=max_return,
                        min_value=0.0,
                        mean=current_return,
                        num_vals=eval_length,
                    )
                    runs_dictionary[env][scenario][algorithm][run][f"STEP_{step}"][
                        "win_rate"
                    ] = generate_win_rate(
                        max_value=max_win_rate,
                        min_value=0.0,
                        mean=current_win_rate,
                        num_vals=eval_length,
                    )

                if (
                    "ABSOLUTE_METRICS"
                    not in runs_dictionary[env][scenario][algorithm][run]
                ):
                    runs_dictionary[env][scenario][algorithm][run][
                        "ABSOLUTE_METRICS"
                    ] = {}

                runs_dictionary[env][scenario][algorithm][run]["ABSOLUTE_METRICS"][
                    "return"
                ] = generate_return_absolute_metric(
                    max_value=max_return,
                    min_value=0.0,
                    mean=max_return,
                    num_vals=eval_length * 10,
                )
                runs_dictionary[env][scenario][algorithm][run]["ABSOLUTE_METRICS"][
                    "win_rate"
                ] = generate_win_rate_absolute_metric(
                    max_value=max_win_rate,
                    min_value=0.0,
                    mean=max_win_rate,
                    num_vals=eval_length * 10,
                )

# Store dummy data
import json

with open("smac_dummy_data.json", "w+") as f:
    json.dump(runs_dictionary, f, indent=4)
