"""PM4 Asset Matching — Python/Zarr signal-store lane.

Provides data models, JSON import, Zarr signal store, scorer, and placement
synthesizer for PM4 asset matching.  Mirrors the C# matching pipeline in
``WowViewer.Core.PM4.Matching``.
"""

from .json_import import import_asset_corpus, import_match_report, import_segment_export
from .models import (
    Pm4AssetMatchCandidate,
    Pm4AssetMatchStatus,
    Pm4AssetReferenceSignalRecord,
    Pm4Bounds3,
    Pm4ReplacementPlacementProposal,
    Pm4SegmentAnchorSignals,
    Pm4SegmentHeightStats,
    Pm4SegmentMatchResult,
    Pm4SegmentSignalRecord,
    Pm4SegmentTopologyStats,
)
from .placement_synthesizer import synthesize_placements
from .scorer import score_segment, score_segments
from .signal_store import (
    read_asset_references_zarr,
    read_segment_signals_zarr,
    write_asset_references_zarr,
    write_segment_signals_zarr,
)

__all__ = [
    "Pm4AssetMatchCandidate",
    "Pm4AssetMatchStatus",
    "Pm4AssetReferenceSignalRecord",
    "Pm4Bounds3",
    "Pm4ReplacementPlacementProposal",
    "Pm4SegmentAnchorSignals",
    "Pm4SegmentHeightStats",
    "Pm4SegmentMatchResult",
    "Pm4SegmentSignalRecord",
    "Pm4SegmentTopologyStats",
    "import_asset_corpus",
    "import_match_report",
    "import_segment_export",
    "read_asset_references_zarr",
    "read_segment_signals_zarr",
    "score_segment",
    "score_segments",
    "synthesize_placements",
    "write_asset_references_zarr",
    "write_segment_signals_zarr",
]
