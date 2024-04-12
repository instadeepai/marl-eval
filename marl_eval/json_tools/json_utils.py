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
import logging
import os
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import neptune
from colorama import Fore, Style
from tqdm import tqdm


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


def concatenate_json_files(
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

    print(
        f"{Fore.CYAN}{Style.BRIGHT}Concatenated data saved in "
        + f"{output_json_path}metrics.json successfully!{Style.RESET_ALL}"
    )
    return concatenated_data


def pull_neptune_data(
    project_name: str,
    tags: List[str],
    store_directory: str = "./downloaded_json_data",
    neptune_data_key: str = "metrics",
    disable_progress_bar: bool = False,
) -> None:
    """Downloads logs from a Neptune project based on provided tags.

    Args:
        project_name (str): Name of the Neptune project.
        tags (List[str]): List of tags associated with the desired experiments.
        store_directory (str, optional): Directory to store the downloaded logs.
            Default is "./downloaded_json_data".
        neptune_data_key (str, optional): Key for the Neptune data to download.
            Default is "metrics".
        disable_progress_bar (bool, optional): Whether to hide a progress bar.
            Default is False.

    Raises:
        ValueError: If the provided project name or tags are invalid.
    """
    # Create the log directory if it doesn't exist
    os.makedirs(store_directory, exist_ok=True)

    # Disable Neptune logging
    neptune_logger = logging.getLogger("neptune")
    neptune_logger.setLevel(logging.ERROR)

    # Initialize the Neptune project
    try:
        project = neptune.init_project(project=project_name)
    except Exception as e:
        raise ValueError(f"Invalid project name '{project_name}': {e}")

    # Fetch runs based on provided tags
    try:
        runs_table_df = project.fetch_runs_table(
            state="inactive", columns=["sys/id"], tag=tags, sort_by="sys/id"
        ).to_pandas()
    except Exception as e:
        raise ValueError(f"Invalid tags {tags}: {e}")

    run_ids = runs_table_df["sys/id"].values.tolist()

    # Download logs concurrently
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                _download_and_extract_data,
                project_name,
                run_id,
                store_directory,
                neptune_data_key,
            )
            for run_id in run_ids
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloading JSON logs",
            disable=disable_progress_bar,
        ):
            future.result()

    # Restore neptune logger level
    neptune_logger.setLevel(logging.INFO)
    print(f"{Fore.CYAN}{Style.BRIGHT}Data downloaded successfully!{Style.RESET_ALL}")


def _download_and_extract_data(
    project_name: str, run_id: str, store_directory: str, neptune_data_key: str
) -> None:
    try:
        with neptune.init_run(
            project=project_name, with_id=run_id, mode="read-only"
        ) as run:
            for j, data_key in enumerate(
                run.get_structure()[neptune_data_key].keys(), start=1
            ):
                file_path = f"{store_directory}/{run_id}"
                if j > 1:
                    file_path += f"_{j}"
                run[f"{neptune_data_key}/{data_key}"].download(destination=file_path)
                _extract_zip_file(file_path)
    except Exception as e:
        print(f"Error downloading data for run {run_id}: {e}")


def _extract_zip_file(file_path: str) -> None:
    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            for member in zip_ref.infolist():
                if not member.is_dir():
                    target_path = Path(f"{file_path}{Path(member.filename).suffix}")
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zip_ref.open(member) as src, target_path.open("wb") as dest:
                        dest.write(src.read())
            # Remove the zip file
            os.remove(file_path)
    except zipfile.BadZipFile:
        # If the file is not zipped, no action is required
        pass
    except Exception as e:
        print(f"Error while unzipping or storing JSON data at path {file_path}: {e}")
