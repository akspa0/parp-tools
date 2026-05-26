"""V18 terrain model definitions.

Each model predicts ONE terrain signal from minimap. Models train
independently; they chain via data dependencies, not shared weights.

V18 re-exports the V16.1 model architectures under the V18 namespace.
The V16.1 files remain the canonical implementation. This module is the
public surface for all V18 model work.

Usage:
    from harvester.v18_models import V18NormalModel, V18HeightModel, ...
"""

from harvester.v16_1_models import (
    V161HeightModel as V18HeightModel,
    V161NormalModel as V18NormalModel,
    V161NormalHeightModel as V18NormalHeightModel,
    V161NormalHeightCombinedModel as V18NormalHeightCombinedModel,
    V161NormalRefiner as V18NormalRefiner,
    V161HolesModel as V18HolesModel,
    V161LiquidModel as V18LiquidModel,
    V161TexcompModel as V18TexcompModel,
    recompose_from_mcly_alpha,
    compute_compositor_weights_torch,
)

__all__ = [
    "V18HeightModel",
    "V18NormalModel",
    "V18NormalHeightModel",
    "V18NormalHeightCombinedModel",
    "V18NormalRefiner",
    "V18HolesModel",
    "V18LiquidModel",
    "V18TexcompModel",
    "recompose_from_mcly_alpha",
    "compute_compositor_weights_torch",
]
