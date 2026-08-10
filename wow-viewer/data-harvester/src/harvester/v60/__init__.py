"""v60 unified dataset package (Specs 134 and 139).

The v60 line is a distinct dataset family from v50. It consolidates all harvested
builds into a single unified Zarr store with the decomposed terrain signals
(terrain_shadow_256, signal_class, surviving_height_levels). Code that is
v60-specific lives here, not in ``harvester.v50``.  The clean-signal modules add the
deployment-safe v7-inspired observation and target contracts without changing historical imports.
"""

from harvester.v60.clean_signal_corpus import validate_clean_signal_corpus
from harvester.v60.clean_signal_inputs import (
    CleanObservationPackage,
    build_clean_observation,
    validate_clean_observation,
)
from harvester.v60.clean_signal_losses import (
    CLEAN_SIGNAL_LOSS_PROFILES,
    CleanSignalLossError,
    V7GuidanceConfig,
    clean_signal_loss,
    get_clean_signal_loss_config,
)
from harvester.v60.clean_signal_model import (
    CleanSignalModel,
    CleanSignalModelError,
    CleanSignalPredictions,
    build_clean_signal_model,
    build_clean_signal_model_from_identity,
)
from harvester.v60.clean_signal_targets import (
    StructuralTarget,
    decompose_relative_height,
    recompose_height,
)
from harvester.v60.clean_signal_train import (
    CleanSignalRow,
    CleanSignalSplit,
    CleanSignalTrainConfig,
    build_clean_signal_split,
    evaluate_clean_signal_model,
    select_clean_signal_training_rows,
    train_clean_signal_model,
)

__all__ = [
    "CleanObservationPackage",
    "CleanSignalModel",
    "CleanSignalModelError",
    "CleanSignalPredictions",
    "CleanSignalLossError",
    "CLEAN_SIGNAL_LOSS_PROFILES",
    "CleanSignalRow",
    "CleanSignalSplit",
    "CleanSignalTrainConfig",
    "StructuralTarget",
    "V7GuidanceConfig",
    "build_clean_observation",
    "build_clean_signal_model",
    "build_clean_signal_model_from_identity",
    "decompose_relative_height",
    "clean_signal_loss",
    "get_clean_signal_loss_config",
    "build_clean_signal_split",
    "evaluate_clean_signal_model",
    "select_clean_signal_training_rows",
    "train_clean_signal_model",
    "recompose_height",
    "validate_clean_observation",
    "validate_clean_signal_corpus",
]
