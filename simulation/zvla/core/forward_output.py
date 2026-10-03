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

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch.utils import _pytree


@dataclass
class VLAForwardOutput:
    """Standard forward-output container for all VLA models.

    The trainer's loop interacts with models *only* through this interface.
    Any per-dataset or per-task metrics should be placed in ``metrics`` so
    that the trainer does not need to hard-code field names.

    Args:
        loss: Scalar loss tensor (``loss.backward()`` is called on this).
        metrics: Optional dict of detached scalar values for logging.
        predictions: Optional model predictions (e.g. actions, tokens).
    """

    loss: torch.Tensor
    metrics: Dict[str, Any] = field(default_factory=dict)
    predictions: Optional[torch.Tensor] = None

    def __post_init__(self):
        if not isinstance(self.loss, torch.Tensor):
            raise TypeError(
                f'VLAForwardOutput.loss must be a torch.Tensor, '
                f'got {type(self.loss)}'
            )
        if self.loss.ndim != 0:
            raise ValueError(
                f'VLAForwardOutput.loss must be a 0-dim scalar tensor, '
                f'got shape {self.loss.shape}'
            )


def _flatten_vla_forward_output(value: VLAForwardOutput):
    return [value.loss, value.metrics, value.predictions], None


def _unflatten_vla_forward_output(values, context):
    del context
    return VLAForwardOutput(*values)


_pytree.register_pytree_node(
    VLAForwardOutput,
    _flatten_vla_forward_output,
    _unflatten_vla_forward_output,
)
