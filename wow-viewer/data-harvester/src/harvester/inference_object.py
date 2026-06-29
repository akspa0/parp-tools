"""Spec 077 Phase 5 (US4) inference object explanation contracts.

The runtime path for development-map and PM4-only tiles must explain object
coverage and asset identity from raw minimap alone — no ADT placement
arrays at inference time. The contracts here are the data shape that the
runtime path produces and that downstream restorers consume.

Spec 077 data-model.md §4.1 (InferenceObjectHypothesis) and §4.2
(RecoveredObjectPlacement) live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class ObjectMaskPrediction:
    """Per-tile object-coverage prediction emitted by the object-mask lane.

    ``mask`` is a 256×256 uint8 binary/soft mask in tile-local pixel
    space (top-left origin). ``mask_confidence`` is the lane's
    confidence in the prediction (0..1, scalar).
    """

    tile_id: int
    mask: bytes  # serialized np.ndarray (256, 256) uint8
    mask_confidence: float
    model_build: str
    model_checkpoint: str


@dataclass(frozen=True)
class AssetCandidate:
    """One candidate match for a predicted object instance.

    The candidate list is sorted by ``score`` descending. ``pose_xy`` is
    the (x, y) tile-local center in pixels; ``pose_yaw`` is the
    rotation in radians around the up axis (spec 077 FR-018).
    """

    asset_path: str
    library_id: str
    score: float
    pose_xy: tuple[float, float]
    pose_yaw: float
    bbox_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class InferenceObjectHypothesis:
    """A grouped set of candidate matches for one predicted object instance.

    Spec 077 data-model.md §4.1: this is the unit the runtime pipeline
    emits per connected component in the predicted object mask. The
    downstream restorer can convert it to a RecoveredObjectPlacement by
    adding the Z height from the predicted terrain.
    """

    tile_id: int
    instance_id: int
    mask_bbox: tuple[int, int, int, int]
    mask_confidence: float
    asset_candidate_paths: tuple[str, ...]
    asset_candidate_scores: tuple[float, ...]
    asset_candidate_library_ids: tuple[str, ...] = field(default_factory=tuple)
    pose_xy: tuple[float, float] = (0.0, 0.0)
    pose_yaw: float = 0.0
    pose_z_from_terrain: float | None = None

    def top_candidate(self) -> AssetCandidate | None:
        if not self.asset_candidate_paths:
            return None
        idx = max(range(len(self.asset_candidate_paths)), key=lambda i: self.asset_candidate_scores[i])
        return AssetCandidate(
            asset_path=self.asset_candidate_paths[idx],
            library_id=self.asset_candidate_library_ids[idx] if self.asset_candidate_library_ids else "",
            score=self.asset_candidate_scores[idx],
            pose_xy=self.pose_xy,
            pose_yaw=self.pose_yaw,
            bbox_xyxy=self.mask_bbox,
        )

    def ranked_candidates(self) -> list[AssetCandidate]:
        ranked = sorted(
            range(len(self.asset_candidate_paths)),
            key=lambda i: self.asset_candidate_scores[i],
            reverse=True,
        )
        out: list[AssetCandidate] = []
        for i in ranked:
            lib_id = (
                self.asset_candidate_library_ids[i]
                if i < len(self.asset_candidate_library_ids)
                else ""
            )
            out.append(
                AssetCandidate(
                    asset_path=self.asset_candidate_paths[i],
                    library_id=lib_id,
                    score=float(self.asset_candidate_scores[i]),
                    pose_xy=self.pose_xy,
                    pose_yaw=self.pose_yaw,
                    bbox_xyxy=self.mask_bbox,
                )
            )
        return out


@dataclass(frozen=True)
class RecoveredObjectPlacement:
    """A reconstructed placement ready to be written back to a placement table.

    Spec 077 data-model.md §4.2. Only the first-required pose fields are
    emitted: ``x``, ``y``, ``z_from_terrain``, ``yaw``, ``confidence``.
    Pitch / roll / scale are optional and default to None.
    """

    asset_path: str
    x: float
    y: float
    z_from_terrain: float
    yaw: float
    confidence: float
    pitch: float | None = None
    roll: float | None = None
    scale: float | None = None
    source_tile_id: int | None = None


def hypothesis_to_recovered(
    hypothesis: InferenceObjectHypothesis,
    *,
    terrain_z: float,
) -> RecoveredObjectPlacement:
    """Lift a hypothesis into a recovered placement using terrain Z.

    Pitch / roll / scale remain None — spec 077 FR-018 says they may be
    deferred; the recovery only emits XY, yaw, and a terrain-derived Z.
    """
    top = hypothesis.top_candidate()
    if top is None:
        raise ValueError(
            f"Hypothesis for tile {hypothesis.tile_id} instance "
            f"{hypothesis.instance_id} has no asset candidates"
        )
    return RecoveredObjectPlacement(
        asset_path=top.asset_path,
        x=float(hypothesis.pose_xy[0]),
        y=float(hypothesis.pose_xy[1]),
        z_from_terrain=float(terrain_z),
        yaw=float(hypothesis.pose_yaw),
        confidence=float(top.score),
        pitch=None,
        roll=None,
        scale=None,
        source_tile_id=int(hypothesis.tile_id),
    )


def collect_hypotheses(
    hypotheses: Iterable[InferenceObjectHypothesis],
) -> list[InferenceObjectHypothesis]:
    """Stable sort: highest top-candidate score first; tiebreak by tile_id."""
    items = list(hypotheses)
    items.sort(
        key=lambda h: (
            -float(max(h.asset_candidate_scores)) if h.asset_candidate_scores else 0.0,
            int(h.tile_id),
            int(h.instance_id),
        )
    )
    return items
