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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Type

import torch
import torch.nn as nn

from .forward_output import VLAForwardOutput


@dataclass(frozen=True)
class OptimizerGroupSpec:
    """An explicit, identity-based optimizer parameter group."""

    name: str
    parameters: Tuple[nn.Parameter, ...]
    lr_scale: float = 1.0
    update_after_step: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("optimizer group name cannot be empty")
        if self.lr_scale <= 0:
            raise ValueError("optimizer group lr_scale must be positive")
        if self.update_after_step < 0:
            raise ValueError("optimizer group update_after_step cannot be negative")
        if not self.parameters:
            raise ValueError(f"optimizer group {self.name!r} cannot be empty")


class VLAModel(nn.Module, ABC):
    """Abstract base class for all VLA models in Zvla.

    ``VLAModel`` is task-agnostic: it works for generative VLAs *and*
    action-prediction VLAs (e.g. Flow Matching).
    """

    @abstractmethod
    def forward(self, **batch: Any) -> VLAForwardOutput:
        """Forward pass.

        Args:
            **batch: The dictionary returned by the collator.  The model
                decides which keys to consume.

        Returns:
            VLAForwardOutput: Standardised output container.
        """
        ...

    def get_fsdp_wrap_policy(self) -> Optional[Callable]:
        """Return an FSDP auto-wrap policy (FSDP1 style) or ``None``.

        For FSDP2 the trainer extracts the layer classes from this policy
        via ``_extract_transformer_layer_classes``.  If this returns
        ``None`` the root module is wrapped only.

        Returns:
            A ``partial(transformer_auto_wrap_policy, ...)`` or ``None``.
        """
        return None

    def ddp_trainable_module(self) -> Optional[nn.Module]:
        """Return a trainable submodule that DDP may wrap independently.

        Models with a large frozen carrier can override this hook so that
        distributed gradient synchronization covers only the small module
        updated by the experiment.  ``None`` keeps ordinary whole-model DDP.
        """
        return None

    def install_ddp_trainable_wrapper(self, wrapper: nn.Module) -> None:
        """Install the wrapper returned for :meth:`ddp_trainable_module`."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support partial DDP wrapping")

    def freeze_backbones(self) -> List[str]:
        """Freeze/unfreeze backbones and return trainable module keys.

        The default implementation returns the names of all parameters that
        require gradients.  Subclasses may override this for finer control
        (e.g. freezing vision backbone while training the action head).

        Returns:
            List of trainable module key names (used for checkpoint
            splitting and logging).
        """
        return [
            name for name, p in self.named_parameters() if p.requires_grad
        ]

    def load_pretrained_weights(self, checkpoint_path: Optional[str] = None) -> None:
        """Load pretrained weights.  Default no-op; override in subclasses."""
        pass

    def get_gradient_checkpointing_layer_classes(self) -> Set[Type[nn.Module]]:
        """Return layer classes that should receive activation checkpointing.

        The FSDP strategies call this before wrapping so that each model
        can declare its own transformer / attention layer types without
        hard-coding attribute names in the strategy.

        Returns:
            Set of ``nn.Module`` subclasses to checkpoint.  Empty set means
            no custom layers (the strategy may still add generic ones such
            as ``timm`` vision blocks).
        """
        return set()

    def optimizer_group_specs(self) -> Tuple[OptimizerGroupSpec, ...]:
        """Return exact trainable parameter identities and LR multipliers."""
        parameters = tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad
        )
        if not parameters:
            raise RuntimeError(f"{type(self).__name__} has no trainable parameters")
        return (OptimizerGroupSpec("default", parameters, 1.0),)

    def training_metric_parameter_groups(
        self,
    ) -> Mapping[str, Tuple[nn.Parameter, ...]]:
        """Return disjoint semantic parameter groups for gradient telemetry.

        This interface is deliberately separate from optimizer groups: an
        optimizer group describes learning-rate policy, whereas these groups
        describe architectural ownership.  Subclasses may therefore split one
        optimizer group into several interpretable W&B curves.
        """
        parameters = tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad
        )
        return {"model": parameters} if parameters else {}

    def training_metric_group_roles(
        self,
    ) -> Mapping[str, Tuple[str, ...]]:
        """Declare aggregate telemetry roles for every semantic group.

        ``training_metric_parameter_groups`` owns the exact parameter
        partition.  This second, explicit mapping says which aggregate curves
        a group contributes to.  A group may be both ``world`` and ``vision``;
        ``action`` is kept separate.  Concrete multimodal models should
        override this method instead of relying on parameter-name heuristics.
        """
        return {
            name: ("world", )
            for name in self.training_metric_parameter_groups()
        }

    def set_train_step_context(
        self,
        step: int,
        *,
        collect_metrics: bool,
        collect_expensive_metrics: bool = False,
        collect_source_metrics: bool = False,
        total_steps: Optional[int] = None,
    ) -> None:
        """Tell the model which diagnostics are required for this forward.

        Detailed metrics can contain reductions over millions of K/V entries.
        The trainer enables them only on logging steps.  Direct model calls
        retain the convenient default of collecting regular metrics.
        """
        if step < 0:
            raise ValueError("training step cannot be negative")
        if collect_expensive_metrics and not collect_metrics:
            raise ValueError("expensive metrics require regular metrics")
        if collect_source_metrics and not collect_metrics:
            raise ValueError("source metrics require regular metrics")
        if total_steps is not None and total_steps <= 0:
            raise ValueError("total training steps must be positive")
        self._train_metric_step = int(step)
        if total_steps is not None:
            self._train_total_steps = int(total_steps)
        self._collect_train_metrics = bool(collect_metrics)
        self._collect_expensive_train_metrics = bool(collect_expensive_metrics)
        self._collect_source_train_metrics = bool(collect_source_metrics)

    @property
    def collect_train_metrics(self) -> bool:
        return bool(getattr(self, "_collect_train_metrics", True))

    @property
    def collect_expensive_train_metrics(self) -> bool:
        return bool(getattr(self, "_collect_expensive_train_metrics", False))

    @property
    def collect_source_train_metrics(self) -> bool:
        """Whether this logging step requests per-source packed telemetry."""
        return bool(getattr(self, "_collect_source_train_metrics", False))

    def after_optimizer_step(self, step: int) -> None:
        """Run model-owned state updates once per real optimizer step."""

    def checkpoint_modules(self) -> Mapping[str, nn.Module]:
        """Persistent train-state modules; external frozen teachers are omitted."""
        return {"model": self}

    def model_artifact_modules(self) -> Mapping[str, nn.Module]:
        """Modules exported for inference or the next training stage.

        A resumable training checkpoint and a stage hand-off artifact have
        different jobs.  The former must retain every trainable/optimizer
        state, while the latter should contain only weights consumed by the
        next stage.  Models may override this mapping to avoid exporting
        redundant live/EMA copies.  Names remain part of the public artifact
        contract and therefore use the same flat ``name.tensor`` layout as
        :meth:`checkpoint_modules`.
        """
        return self.checkpoint_modules()

    def checkpoint_state_dict(self) -> Dict[str, torch.Tensor]:
        modules = self.checkpoint_modules()
        if not modules:
            raise RuntimeError("checkpoint_modules() cannot be empty")
        state: Dict[str, torch.Tensor] = {}
        for module_name, module in modules.items():
            if not module_name or "." in module_name:
                raise ValueError(
                    "checkpoint module name must be a non-empty path component: "
                    f"{module_name!r}"
                )
            for key, value in module.state_dict().items():
                flat_key = f"{module_name}.{key}"
                if flat_key in state:
                    raise RuntimeError(f"duplicate checkpoint key {flat_key!r}")
                state[flat_key] = value
        return state

    def _checkpoint_module_path(self, target: nn.Module) -> str:
        matches = [
            name for name, module in self.named_modules() if module is target
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"checkpoint module {type(target).__name__} must occur exactly once "
                f"in the model tree, found paths={matches}"
            )
        return matches[0]

    def checkpoint_state_from_full_model_state(
        self, full_state: Mapping[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Select persistent modules from an FSDP/DDP full model state."""
        return self._named_module_state_from_full_model_state(
            full_state, self.checkpoint_modules(), label="checkpoint")

    def model_artifact_state_from_full_model_state(
        self, full_state: Mapping[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Select the compact model-only hand-off state from a full state."""
        return self._named_module_state_from_full_model_state(
            full_state, self.model_artifact_modules(), label="model artifact")

    def _named_module_state_from_full_model_state(
        self,
        full_state: Mapping[str, torch.Tensor],
        modules: Mapping[str, nn.Module],
        *,
        label: str,
    ) -> Dict[str, torch.Tensor]:
        if not modules:
            raise RuntimeError(f"{label} modules cannot be empty")
        selected: Dict[str, torch.Tensor] = {}
        for checkpoint_name, module in modules.items():
            if not checkpoint_name or "." in checkpoint_name:
                raise ValueError(
                    f"{label} module name must be a non-empty path component: "
                    f"{checkpoint_name!r}"
                )
            module_path = self._checkpoint_module_path(module)
            internal_prefix = f"{module_path}." if module_path else ""
            found = 0
            for internal_key, value in full_state.items():
                if not internal_key.startswith(internal_prefix):
                    continue
                suffix = internal_key[len(internal_prefix) :]
                checkpoint_key = f"{checkpoint_name}.{suffix}"
                if checkpoint_key in selected:
                    raise RuntimeError(f"duplicate checkpoint key {checkpoint_key!r}")
                selected[checkpoint_key] = value
                found += 1
            if not found:
                raise RuntimeError(
                    f"{label} module {checkpoint_name!r} at {module_path!r} "
                    "has no entries in the full model state"
                )
        return selected

    def merge_checkpoint_into_full_model_state(
        self,
        checkpoint_state: Mapping[str, torch.Tensor],
        full_state: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Strictly map a persistent checkpoint back into a complete state."""
        expected = self.checkpoint_state_from_full_model_state(full_state)
        missing = sorted(set(expected) - set(checkpoint_state))
        extra = sorted(set(checkpoint_state) - set(expected))
        shape_errors = sorted(
            key
            for key in set(expected) & set(checkpoint_state)
            if tuple(expected[key].shape) != tuple(checkpoint_state[key].shape)
        )
        if missing or extra or shape_errors:
            raise RuntimeError(
                "strict checkpoint mismatch: "
                f"missing={missing[:20]}, extra={extra[:20]}, "
                f"shape_errors={shape_errors[:20]}"
            )
        merged = dict(full_state)
        for checkpoint_name, module in self.checkpoint_modules().items():
            module_path = self._checkpoint_module_path(module)
            internal_prefix = f"{module_path}." if module_path else ""
            checkpoint_prefix = f"{checkpoint_name}."
            for checkpoint_key, value in checkpoint_state.items():
                if checkpoint_key.startswith(checkpoint_prefix):
                    suffix = checkpoint_key[len(checkpoint_prefix) :]
                    merged[f"{internal_prefix}{suffix}"] = value
        return merged

    def load_checkpoint_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        modules = self.checkpoint_modules()
        expected_prefixes = tuple(f"{name}." for name in modules)
        unexpected = sorted(
            key
            for key in state
            if not any(key.startswith(prefix) for prefix in expected_prefixes)
        )
        if unexpected:
            raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:20]}")
        for module_name, module in modules.items():
            prefix = f"{module_name}."
            module_state = {
                key[len(prefix) :]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            expected = module.state_dict()
            missing = sorted(set(expected) - set(module_state))
            extra = sorted(set(module_state) - set(expected))
            shape_errors = sorted(
                key
                for key in set(expected) & set(module_state)
                if tuple(expected[key].shape) != tuple(module_state[key].shape)
            )
            if missing or extra or shape_errors:
                raise RuntimeError(
                    f"strict checkpoint mismatch for {module_name}: "
                    f"missing={missing[:20]}, extra={extra[:20]}, "
                    f"shape_errors={shape_errors[:20]}"
                )
            module.load_state_dict(module_state, strict=True)
