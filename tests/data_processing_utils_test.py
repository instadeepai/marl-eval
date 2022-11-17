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
from typing import Any, Dict

import jax
import numpy as np
import pytest
from expected_test_data import (
    expected_processed_data,
    expected_single_task_ci_data_returns,
    expected_single_task_ci_data_win_rates,
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
    get_and_aggregate_data_single_task,
)


@pytest.fixture
def raw_data() -> Dict[str, Dict[str, Any]]:
    """Fixture for raw experiment data."""
    with open("tests/mock_data_test.json", "r") as f:
        read_in_data = json.load(f)

    return read_in_data


@pytest.fixture
def processed_data(raw_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Fixture for processed experiment data"""

    processed_data = data_process_pipeline(
        raw_data=raw_data, metrics_to_normalize=["return"]
    )

    return processed_data


def test_data_processing_pipeline(processed_data: Dict[str, Dict[str, Any]]) -> None:
    """Tests whether data processing pipeline runs and aggregates \
        data correctly."""

    assert processed_data == expected_processed_data


def test_matrices_for_rliable_full_environment_dataset(
    raw_data: Dict[str, Dict[str, Any]]
) -> None:
    """Tests that arrays for rliable are created correctly for \
        a full dataset containing multiple algorithms and tasks \
            for a given envionment."""

    processed_data = data_process_pipeline(
        raw_data=raw_data, metrics_to_normalize=["return"]
    )

    m1, m2 = create_matrices_for_rliable(
        data_dictionary=processed_data,
        environment_name="SMAC",
        metrics_to_normalize=["return"],
    )

    # delete extra param from m2
    del m2["extra"]

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
    raw_data: Dict[str, Dict[str, Any]]
) -> None:
    """Tests that arrays for rliable are created correctly for \
        a dataset containing multiple algorithms but only a single task."""

    # Select only one task (3m) from the full environment (SMAC)
    filtered_raw_data: Dict[Any, Any] = {}
    filtered_raw_data["SMAC"] = {}
    filtered_raw_data["SMAC"]["3m"] = {}
    filtered_raw_data["SMAC"]["3m"] = raw_data["SMAC"]["3m"]
    raw_data = filtered_raw_data

    processed_data = data_process_pipeline(
        raw_data=raw_data, metrics_to_normalize=["return"]
    )

    m1, m2 = create_matrices_for_rliable(
        data_dictionary=processed_data,
        environment_name="SMAC",
        metrics_to_normalize=["return"],
    )

    # delete extra param from m2
    del m2["extra"]

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
    raw_data: Dict[str, Dict[str, Any]]
) -> None:
    """Tests that arrays for rliable are created correctly for \
        a dataset containing a single algorithms but multiple tasks."""

    # select only one algorithm (QMIX) over multiple tasks
    filtered_raw_data: Dict[Any, Any] = {}
    filtered_raw_data["SMAC"] = {}
    filtered_raw_data["SMAC"]["3m"] = {}
    filtered_raw_data["SMAC"]["3m"]["QMIX"] = raw_data["SMAC"]["3m"]["QMIX"]

    filtered_raw_data["SMAC"]["3s5z"] = {}
    filtered_raw_data["SMAC"]["3s5z"]["QMIX"] = raw_data["SMAC"]["3s5z"]["QMIX"]

    filtered_raw_data["SMAC"]["8m"] = {}
    filtered_raw_data["SMAC"]["8m"]["QMIX"] = raw_data["SMAC"]["8m"]["QMIX"]
    raw_data = filtered_raw_data

    processed_data = data_process_pipeline(
        raw_data=raw_data, metrics_to_normalize=["return"]
    )

    m1, m2 = create_matrices_for_rliable(
        data_dictionary=processed_data,
        environment_name="SMAC",
        metrics_to_normalize=["return"],
    )

    # delete extra param from m2
    del m2["extra"]

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


def test_single_task_data_aggregation(
    processed_data: Dict[str, Dict[str, Any]]
) -> None:
    """Tests that single task aggregation is done correctly."""

    # Test for return calculations of single task.
    task_return_ci_data = get_and_aggregate_data_single_task(
        processed_data=processed_data,
        metric_name="return",
        metrics_to_normalize=["return"],
        environment_name="SMAC",
        task_name="3m",
    )

    del task_return_ci_data["extra"]

    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_allclose(x, y, rtol=0.0, atol=1e-05),
        task_return_ci_data,
        expected_single_task_ci_data_returns,
    )

    # Test for win rate calculations of single task.
    task_win_rate_ci_data = get_and_aggregate_data_single_task(
        processed_data=processed_data,
        metric_name="win_rate",
        metrics_to_normalize=["return"],
        environment_name="SMAC",
        task_name="8m",
    )

    del task_win_rate_ci_data["extra"]

    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_allclose(x, y, rtol=0.0, atol=1e-05),
        task_win_rate_ci_data,
        expected_single_task_ci_data_win_rates,
    )
