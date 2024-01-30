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
from collections import defaultdict
from typing import Dict, Tuple


def _read_json_files(directory: str) -> list:
    """Reads all JSON files in a directory and returns a list of JSON objects."""
    json_data = []

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)
                with open(file_path) as file:
                    json_data.append(json.load(file))

    return json_data


def _get_seed_number(seed_str: str) -> Tuple[str, int]:
    """Get the seed number from the seed string."""
    if seed_str.isnumeric():
        return "", int(seed_str)
    else:
        try:
            seed_string, seed_number = seed_str.split("_")
            return seed_string, int(seed_number)
        except ValueError:
            raise ValueError(
                f"Seed number {seed_str} is not in the correct format.\
                It should be an integer or a string with the format 'seed_number'"
            )


def _check_seed(concatenated_data: Dict, algo_data: Dict, seed_number: str) -> str:
    """Function to check if seed is already in concatenated_data and algo_data."""
    if seed_number in (concatenated_data.keys() or algo_data.keys()):
        seed_string, seed_n = _get_seed_number(seed_number)
        seed_number = (
            f"{seed_string}_{seed_n+1}" if seed_string != "" else str(seed_n + 1)
        )
        return _check_seed(concatenated_data, algo_data, seed_number)
    else:
        return seed_number


def concatenate_files(
    input_directory: str, output_json_path: str = "concatenated_json_files/"
) -> Dict:
    """Concatenate all json files in a directory and save the result in a json file."""
    # Read all json files in a input_directory
    json_data = _read_json_files(input_directory)

    # Create target folder
    if not os.path.exists(output_json_path):
        os.makedirs(output_json_path)

    # Using defaultdict for automatic handling of missing keys
    concatenated_data: Dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for data in json_data:
        for env_name, envs in data.items():
            for scenario_name, scenarios in envs.items():
                for algo_name, algos in scenarios.items():
                    concatenated_data[env_name][scenario_name][algo_name]
                    for seed_number, algo_data in algos.items():
                        # Get seed number
                        seed_n = _check_seed(
                            concatenated_data[env_name][scenario_name][algo_name],
                            algo_data,
                            seed_number,
                        )
                        concatenated_data[env_name][scenario_name][algo_name][
                            seed_n
                        ] = algo_data

    # Save concatenated data in a json file
    if output_json_path[-1] != "/":
        output_json_path += "/"
    with open(f"{output_json_path}metrics.json", "w") as f:
        json.dump(concatenated_data, f, indent=4)

    return concatenated_data
