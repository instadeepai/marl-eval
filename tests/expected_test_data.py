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

import numpy as np

# Full dataset tests
matrix_1_expected_data = {
    "mean_norm_return": {
        "QMIX": np.array(
            [[0.40277778, 0.51388889, 0.76388889], [0.20833333, 0.33333333, 0.375]]
        ),
        "MADQN": np.array(
            [[0.36111111, 0.36111111, 0.43055556], [0.33333333, 0.43055556, 0.43055556]]
        ),
        "VDN": np.array(
            [[0.43055556, 0.40277778, 0.76388889], [0.47222222, 0.58333333, 0.5]]
        ),
    },
    "mean_win_rate": {
        "QMIX": np.array([[0.2, 0.5, 0.7], [0.8, 0.7, 0.6]]),
        "MADQN": np.array([[0.6, 0.9, 0.3], [0.4, 0.8, 0.3]]),
        "VDN": np.array([[0.1, 0.5, 0.2], [0.1, 0.2, 0.1]]),
    },
}

sample_efficiency_matrix_expected_data = {
    "mean_norm_return": {
        "QMIX": np.array(
            [
                [
                    [0.27777778, 0.16666667, 0.30555556],
                    [0.19444444, 0.38888889, 0.27777778],
                    [0.36111111, 0.91666667, 0.69444444],
                ],
                [
                    [0.13888889, 0.27777778, 0.38888889],
                    [0.5, 0.80555556, 0.61111111],
                    [0.08333333, 0.63888889, 0.63888889],
                ],
            ]
        ),
        "MADQN": np.array(
            [
                [
                    [0.08333333, 0.38888889, 0.5],
                    [0.38888889, 0.47222222, 0.36111111],
                    [0.02777778, 0.36111111, 0.91666667],
                ],
                [
                    [0.55555556, 0.25, 0.52777778],
                    [0.72222222, 0.5, 0.33333333],
                    [0.19444444, 0.16666667, 0.66666667],
                ],
            ]
        ),
        "VDN": np.array(
            [
                [
                    [0.05555556, 0.11111111, 0.19444444],
                    [0.69444444, 0.58333333, 0.94444444],
                    [0.86111111, 0.41666667, 0.69444444],
                ],
                [
                    [0.72222222, 0.80555556, 0.25],
                    [0.72222222, 0.38888889, 0.41666667],
                    [0.66666667, 0.55555556, 0.63888889],
                ],
            ]
        ),
    },
    "mean_win_rate": {
        "QMIX": np.array(
            [
                [[0.8, 0.5, 0.4], [0.2, 0.5, 0.1], [0.2, 0.6, 0.8]],
                [
                    [0.7, 0.8, 0.4],
                    [0.5, 0.1, 0.2],
                    [0.4, 0.3, 0.2],
                ],
            ]
        ),
        "MADQN": np.array(
            [
                [[0.4, 0.7, 0.4], [0.9, 0.1, 0.1], [0.1, 0.3, 0.1]],
                [[0.2, 0.8, 0.1], [0.2, 0.8, 0.8], [0.2, 0.3, 0.5]],
            ]
        ),
        "VDN": np.array(
            [
                [
                    [
                        0.1,
                        0.3,
                        0.5,
                    ],
                    [
                        0.6,
                        0.2,
                        0.2,
                    ],
                    [
                        0.3,
                        0.3,
                        0.2,
                    ],
                ],
                [[0.3, 0.2, 0.7], [0.1, 0.4, 0.5], [0.3, 0.3, 0.3]],
            ]
        ),
    },
}

# Single task test data

matrix_1_expected_data_single_task = {
    "mean_norm_return": {
        "QMIX": np.array([[0.40277778], [0.20833333]]),
        "MADQN": np.array([[0.36111111], [0.33333333]]),
        "VDN": np.array([[0.43055556], [0.47222222]]),
    },
    "mean_win_rate": {
        "QMIX": np.array([[0.2], [0.8]]),
        "MADQN": np.array([[0.6], [0.4]]),
        "VDN": np.array([[0.1], [0.1]]),
    },
}

sample_efficiency_matrix_expected_data_single_task = {
    "mean_norm_return": {
        "QMIX": np.array(
            [
                [[0.27777778, 0.16666667, 0.30555556]],
                [[0.13888889, 0.27777778, 0.38888889]],
            ]
        ),
        "MADQN": np.array(
            [[[0.08333333, 0.38888889, 0.5]], [[0.55555556, 0.25, 0.52777778]]]
        ),
        "VDN": np.array(
            [[[0.05555556, 0.11111111, 0.19444444]], [[0.72222222, 0.80555556, 0.25]]]
        ),
    },
    "mean_win_rate": {
        "QMIX": np.array([[[0.8, 0.5, 0.4]], [[0.7, 0.8, 0.4]]]),
        "MADQN": np.array([[[0.4, 0.7, 0.4]], [[0.2, 0.8, 0.1]]]),
        "VDN": np.array([[[0.1, 0.3, 0.5]], [[0.3, 0.2, 0.7]]]),
    },
}

# Single algorithm multiple task test data
matrix_1_expected_data_single_algorithm = {
    "mean_norm_return": {
        "QMIX": np.array(
            [[0.453125, 0.51388889, 0.76388889], [0.234375, 0.33333333, 0.375]]
        ),
    },
    "mean_win_rate": {
        "QMIX": np.array([[0.2, 0.5, 0.7], [0.8, 0.7, 0.6]]),
    },
}

sample_efficiency_matrix_expected_data_single_algorithm = {
    "mean_norm_return": {
        "QMIX": np.array(
            [
                [
                    [0.3125, 0.1875, 0.34375],
                    [0.19444444, 0.38888889, 0.27777778],
                    [0.36111111, 0.91666667, 0.69444444],
                ],
                [
                    [0.15625, 0.3125, 0.4375],
                    [0.5, 0.80555556, 0.61111111],
                    [0.08333333, 0.63888889, 0.63888889],
                ],
            ]
        ),
    },
    "mean_win_rate": {
        "QMIX": np.array(
            [
                [[0.8, 0.5, 0.4], [0.2, 0.5, 0.1], [0.2, 0.6, 0.8]],
                [[0.7, 0.8, 0.4], [0.5, 0.1, 0.2], [0.4, 0.3, 0.2]],
            ]
        ),
    },
}
