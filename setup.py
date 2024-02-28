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

"""Install script for setuptools."""

import os
from importlib import util as import_util
from typing import List

from setuptools import find_packages, setup

spec = import_util.spec_from_file_location("_metadata", "marl_eval/_metadata.py")
_metadata = import_util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(_metadata)  # type: ignore

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

long_description = """`marl_eval` is a repo implementing experiment data processing
that goes along with the work done by Gorsane et al. (2022) on standardising
the way in which multi-agent reinforcement learning evaluation is done.
This repo builds on the work done by Agarwal et al. (2022) with the `rliable`
repo which may be found at [https://github.com/google-research/rliable].
What `marl_eval` adds is extra data processing functionality on top of the
`rliable` tools for particular use in multi-agent reinforcement learning.
Given data in the correct json format `marl_eval` will process all data for
downstream statistical aggregation by `rliable`. `marl-eval` will also plot
all aggregated data and produce tabular results which may be easily used
by researchers in order to present clear work which may be easily compared
to by others. For for information please see the associated paper
[https://arxiv.org/abs/2209.10485] and repo
[https://github.com/instadeepai/marl-eval].
"""

# Get the version from metadata.
version = _metadata.__version__  # type: ignore


def _parse_requirements(path: str) -> List[str]:
    """Returns content of given requirements file."""
    with open(os.path.join(_CURRENT_DIR, path)) as f:
        return [
            line.rstrip() for line in f if not (line.isspace() or line.startswith("#"))
        ]


setup(
    name="id-marl-eval",
    version=version,
    description="A Python library for Multi-Agent Reinforcement Learning evaluation.",
    long_description=open(os.path.join(_CURRENT_DIR, "README.md")).read(),
    long_description_content_type="text/markdown",
    author="InstaDeep Ltd",
    license="Apache License, Version 2.0",
    keywords="multi-agent reinforcement-learning python machine learning",
    packages=find_packages(),
    install_requires=_parse_requirements("requirements/requirements.txt"),
    extras_require={
        "dev": _parse_requirements("requirements/requirements-dev.txt"),
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
