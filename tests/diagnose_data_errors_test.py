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

"""Unit tests for the DiagnoseData component"""

import json
from typing import Any, Dict

import pytest

from marl_eval.utils.diagnose_data_errors import DiagnoseData


@pytest.fixture
def valid_raw_data() -> Dict[str, Dict[str, Any]]:
    """Fixture for raw experiment data."""
    with open("tests/mock_data_test.json", "r") as f:
        read_in_data = json.load(f)

    return read_in_data


@pytest.fixture
def unvalid_algo_raw_data() -> Dict[str, Dict[str, Any]]:
    """Fixture for raw experiment data."""
    with open("tests/mock_data_test.json", "r") as f:
        read_in_data = json.load(f)

    del read_in_data["env_1"]["task_1"]["algo_1"]
    return read_in_data


@pytest.fixture
def unvalid_metrics_raw_data() -> Dict[str, Dict[str, Any]]:
    """Fixture for raw experiment data."""
    with open("tests/mock_data_test.json", "r") as f:
        read_in_data = json.load(f)

    del read_in_data["env_1"]["task_1"]["algo_1"]["43289"]["STEP_1"]["return"]
    return read_in_data


@pytest.fixture
def unvalid_runs_raw_data() -> Dict[str, Dict[str, Any]]:
    """Fixture for raw experiment data."""
    with open("tests/mock_data_test.json", "r") as f:
        read_in_data = json.load(f)

    del read_in_data["env_1"]["task_2"]["algo_1"]["43289"]
    return read_in_data


@pytest.fixture
def unvalid_steps_raw_data() -> Dict[str, Dict[str, Any]]:
    """Fixture for raw experiment data."""
    with open("tests/mock_data_test.json", "r") as f:
        read_in_data = json.load(f)

    del read_in_data["env_1"]["task_1"]["algo_1"]["42"]["STEP_1"]
    return read_in_data


@pytest.fixture
def unvalid_raw_data() -> Dict[str, Dict[str, Any]]:
    """Fixture for raw experiment data."""
    with open("tests/mock_data_test.json", "r") as f:
        read_in_data = json.load(f)

    del read_in_data["env_1"]["task_2"]["algo_1"]
    del read_in_data["env_1"]["task_2"]["algo_2"]["43289"]
    del read_in_data["env_1"]["task_1"]["algo_1"]["42"]["STEP_1"]
    del read_in_data["env_1"]["task_1"]["algo_1"]["43289"]["STEP_2"]["return"]
    return read_in_data


def test_valid_data(valid_raw_data: Dict[str, Dict[str, Any]]) -> None:
    """Test valid data"""
    data_diag_tools = DiagnoseData(raw_data=valid_raw_data)
    check_data_results = data_diag_tools.check_data()["env_1"]
    assert check_data_results == {
        "valid_algorithms": True,
        "valid_runs": True,
        "valid_steps": True,
        "valid_metrics": True,
    }


def test_unvalid_algo_data(unvalid_algo_raw_data: Dict[str, Dict[str, Any]]) -> None:
    """Test unvalid data"""
    data_diag_tools = DiagnoseData(raw_data=unvalid_algo_raw_data)
    check_data_results = data_diag_tools.check_data()["env_1"]
    assert check_data_results == {
        "valid_algorithms": False,
        "valid_runs": True,
        "valid_steps": True,
        "valid_metrics": True,
    }


def test_unvalid_runs_data(unvalid_runs_raw_data: Dict[str, Dict[str, Any]]) -> None:
    """Test unvalid data"""
    data_diag_tools = DiagnoseData(raw_data=unvalid_runs_raw_data)
    check_data_results = data_diag_tools.check_data()["env_1"]
    assert check_data_results == {
        "valid_algorithms": True,
        "valid_runs": False,
        "valid_steps": True,
        "valid_metrics": True,
    }


def test_unvalid_metrics_data(
    unvalid_metrics_raw_data: Dict[str, Dict[str, Any]]
) -> None:
    """Test unvalid data"""
    data_diag_tools = DiagnoseData(raw_data=unvalid_metrics_raw_data)
    check_data_results = data_diag_tools.check_data()["env_1"]
    assert check_data_results == {
        "valid_algorithms": True,
        "valid_runs": True,
        "valid_steps": True,
        "valid_metrics": False,
    }


def test_unvalid_data(unvalid_raw_data: Dict[str, Dict[str, Any]]) -> None:
    """Test unvalid data"""
    data_diag_tools = DiagnoseData(raw_data=unvalid_raw_data)
    check_data_results = data_diag_tools.check_data()["env_1"]
    assert check_data_results == {
        "valid_algorithms": False,
        "valid_runs": False,
        "valid_steps": False,
        "valid_metrics": False,
    }
