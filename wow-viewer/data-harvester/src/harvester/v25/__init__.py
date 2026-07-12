"""V25 Terrain Convergence and Decompiler Library."""

from harvester.v25.segformer import V25SegformerDecompiler, TerrainInpaintHead, ObjectPlacementHead
from harvester.v25.solver import BatchedSylvesterSolver
from harvester.v25.lapnet import V25StageBPredictor
from harvester.v25.prior import V25StageAPredictor, WdlDownsampler
from harvester.v25.pm4_guide import V25Pm4GuideHandler
from harvester.v25.fractal import DifferentiableFractalGenerator, FractalParameterHead
from harvester.v25.texture import MtexPredictor, MclyDecoder
from harvester.v25.losses import V25UnifiedLoss
from harvester.v25.dataset import (
    V25TileSource,
    attach_holes_bits,
    attach_pm4_segments,
    attach_tileset_images,
    build_v25_dataset,
    load_pm4_segment_records,
    write_prediction_store,
)

__all__ = [
    "V25TileSource",
    "attach_holes_bits",
    "attach_pm4_segments",
    "attach_tileset_images",
    "build_v25_dataset",
    "load_pm4_segment_records",
    "write_prediction_store",
    "V25SegformerDecompiler",
    "TerrainInpaintHead",
    "ObjectPlacementHead",
    "BatchedSylvesterSolver",
    "V25StageBPredictor",
    "V25StageAPredictor",
    "WdlDownsampler",
    "V25Pm4GuideHandler",
    "DifferentiableFractalGenerator",
    "FractalParameterHead",
    "MtexPredictor",
    "MclyDecoder",
    "V25UnifiedLoss",
]
