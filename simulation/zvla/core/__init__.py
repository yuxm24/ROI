# Copyright 2026 Limx Dynamics
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

"""Core abstractions for the Zvla training framework.

This module defines the minimal set of interfaces that decouple models,
distributed strategies, and the training loop:

    VLAForwardOutput  – standard model forward output
    VLAModel          – abstract base for all VLA models
    BaseStrategy      – abstract base for distributed training strategies
"""

from .forward_output import VLAForwardOutput
from .batch_contracts import (
    FAM_ABSOLUTE_POSE_GEOMETRY_ROBOT_SAMPLE_KEYS,
    FAM_ABSOLUTE_POSE_ROBOT_SAMPLE_KEYS,
    FAM_GEOMETRY_ROBOT_SAMPLE_KEYS,
    FAM_STATE_TRAJECTORY_GEOMETRY_ROBOT_SAMPLE_KEYS,
    FAM_STATE_TRAJECTORY_ROBOT_SAMPLE_KEYS,
    FAM_PAIRED_SAMPLE_KEYS,
    FAM_ROBOT_SAMPLE_KEYS,
    FAM_VIDEO_SAMPLE_KEYS,
)
from .model import OptimizerGroupSpec, VLAModel
from .strategy import BaseStrategy

__all__ = [
    'VLAForwardOutput',
    'FAM_ABSOLUTE_POSE_GEOMETRY_ROBOT_SAMPLE_KEYS',
    'FAM_ABSOLUTE_POSE_ROBOT_SAMPLE_KEYS',
    'FAM_GEOMETRY_ROBOT_SAMPLE_KEYS',
    'FAM_STATE_TRAJECTORY_GEOMETRY_ROBOT_SAMPLE_KEYS',
    'FAM_STATE_TRAJECTORY_ROBOT_SAMPLE_KEYS',
    'FAM_ROBOT_SAMPLE_KEYS',
    'FAM_VIDEO_SAMPLE_KEYS',
    'FAM_PAIRED_SAMPLE_KEYS',
    'VLAModel',
    'OptimizerGroupSpec',
    'BaseStrategy',
]
