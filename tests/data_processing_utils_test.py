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

"""Tests for data processing utils"""

import json
from typing import Any, Dict, Mapping

import jax
import numpy as np
import pytest
from expected_test_data import (
    matrix_1_expected_data,
    matrix_1_expected_data_single_algorithm,
    matrix_1_expected_data_single_task,
    sample_efficiency_matrix_expected_data,
    sample_efficiency_matrix_expected_data_single_algorithm,
    sample_efficiency_matrix_expected_data_single_task,
)

from marl_eval.utils.data_processing_utils import (
    create_matrices_for_rliable,
    data_process_pipeline,
)


@pytest.fixture
def raw_data() -> Mapping[str, Dict[str, Any]]:
    """Fixture for raw experiment data."""
    with open("tests/mock_data_test.json", "r") as f:
        read_in_data = json.load(f)

    return read_in_data


def test_data_processing_pipeline(raw_data: Mapping[str, Dict[str, Any]]) -> None:
    """Tests whether data processing pipeline runs."""

    processed_data = data_process_pipeline(  # noqa
        raw_data=raw_data, metrics_to_normalize=["return"]
    )


def test_matrices_for_rliable_full_environment_dataset(
    raw_data: Mapping[str, Dict[str, Any]]
) -> None:
    """Tests that arrays for rliable are created correctly for \
        a full dataset containing multiple algorithms and tasks \
            for a given envionment."""

    processed_data = data_process_pipeline(
        raw_data=raw_data, metrics_to_normalize=["return"]
    )

    m1, m2 = create_matrices_for_rliable(
        data_dictionary=processed_data,
        environment_name="env_1",
        metrics_to_normalize=["return"],
    )

    # delete extra param from m2
    del m2["extra"]  # type: ignore

    # Test that all arrays are equal.
    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_allclose(x, y, rtol=0.0, atol=1e-05),
        m1,
        matrix_1_expected_data,
    )
    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_allclose(x, y, rtol=0.0, atol=1e-05),
        m2,
        sample_efficiency_matrix_expected_data,
    )


def test_matrices_for_rliable_single_environment_task(
    raw_data: Mapping[str, Dict[str, Any]]
) -> None:
    """Tests that arrays for rliable are created correctly for \
        a dataset containing multiple algorithms but only a single task."""

    # Select only one task (task_1) from the full environment (env_1)
    filtered_raw_data: Dict[Any, Any] = {}
    filtered_raw_data["env_1"] = {}
    filtered_raw_data["env_1"]["task_1"] = {}
    filtered_raw_data["env_1"]["task_1"] = raw_data["env_1"]["task_1"]
    raw_data = filtered_raw_data

    processed_data = data_process_pipeline(
        raw_data=raw_data, metrics_to_normalize=["return"]
    )

    m1, m2 = create_matrices_for_rliable(
        data_dictionary=processed_data,
        environment_name="env_1",
        metrics_to_normalize=["return"],
    )

    # delete extra param from m2
    del m2["extra"]  # type: ignore

    # Test that all arrays are equal.
    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_allclose(x, y, rtol=0.0, atol=1e-05),
        m1,
        matrix_1_expected_data_single_task,
    )
    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_allclose(x, y, rtol=0.0, atol=1e-05),
        m2,
        sample_efficiency_matrix_expected_data_single_task,
    )


def test_matrices_for_rliable_single_algorithm(
    raw_data: Mapping[str, Dict[str, Any]]
) -> None:
    """Tests that arrays for rliable are created correctly for \
        a dataset containing a single algorithms but multiple tasks."""

    # select only one algorithm (algo_1) over multiple tasks
    filtered_raw_data: Dict[Any, Any] = {}
    filtered_raw_data["env_1"] = {}
    filtered_raw_data["env_1"]["task_1"] = {}
    filtered_raw_data["env_1"]["task_1"]["algo_1"] = raw_data["env_1"]["task_1"]["algo_1"]

    filtered_raw_data["env_1"]["task_2"] = {}
    filtered_raw_data["env_1"]["task_2"]["algo_1"] = raw_data["env_1"]["task_2"]["algo_1"]

    filtered_raw_data["env_1"]["task_3"] = {}
    filtered_raw_data["env_1"]["task_3"]["algo_1"] = raw_data["env_1"]["task_3"]["algo_1"]
    raw_data = filtered_raw_data

    processed_data = data_process_pipeline(
        raw_data=raw_data, metrics_to_normalize=["return"]
    )

    m1, m2 = create_matrices_for_rliable(
        data_dictionary=processed_data,
        environment_name="env_1",
        metrics_to_normalize=["return"],
    )

    # delete extra param from m2
    del m2["extra"]  # type: ignore

    # Test that all arrays are equal.
    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_allclose(x, y, rtol=0.0, atol=1e-05),
        m1,
        matrix_1_expected_data_single_algorithm,
    )
    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_allclose(x, y, rtol=0.0, atol=1e-05),
        m2,
        sample_efficiency_matrix_expected_data_single_algorithm,
    )
