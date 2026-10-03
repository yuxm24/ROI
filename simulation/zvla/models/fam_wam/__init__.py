"""FaM-WAM v5.2: Wan2.2 world prefill + VGGT geometry + G4D + action Reader."""

from .config import FAM_WAM_V52_TEACHER_ARCHITECTURE_VERSION, FaMWAMConfig
from .deployment import (
    ARTIFACT_FORMAT,
    export_deployment_artifact,
    load_deployment_artifact,
)
from .kv import DenseKV, LayerKV
from .model import (
    ARTIFACT_MODULES,
    NEURAL_MODULES,
    FaMWAMPolicy,
    FaMWAMPostTrainModel,
    load_policy_from_model_artifact,
)

__all__ = [
    "ARTIFACT_FORMAT",
    "ARTIFACT_MODULES",
    "DenseKV",
    "FAM_WAM_V52_TEACHER_ARCHITECTURE_VERSION",
    "FaMWAMConfig",
    "FaMWAMPolicy",
    "FaMWAMPostTrainModel",
    "LayerKV",
    "NEURAL_MODULES",
    "export_deployment_artifact",
    "load_deployment_artifact",
    "load_policy_from_model_artifact",
]
