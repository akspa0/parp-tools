"""Rosetta Vision Catalog and Overhead Object Asset Indexer (Spec 262).

Loads exported Rosetta overhead catalog metadata, provides spatial tile queries,
and formats bounding boxes and centroid prompts for SAM 3.1 object sieving.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class OverheadPlacementBox:
    tile_x: int
    tile_y: int
    world_pos: Tuple[float, float, float]
    cell_u: float
    cell_v: float
    cell_size: float
    norm_min_u: float
    norm_min_v: float
    norm_max_u: float
    norm_max_v: float
    pixel_min_x: int
    pixel_min_y: int
    pixel_max_x: int
    pixel_max_y: int
    unique_id: int

    @property
    def pixel_width(self) -> int:
        return max(1, self.pixel_max_x - self.pixel_min_x)

    @property
    def pixel_height(self) -> int:
        return max(1, self.pixel_max_y - self.pixel_min_y)

    @property
    def pixel_center(self) -> Tuple[int, int]:
        return (
            (self.pixel_min_x + self.pixel_max_x) // 2,
            (self.pixel_min_y + self.pixel_max_y) // 2,
        )

    def to_comfy_box_dict(self) -> Dict[str, int]:
        """Format as ComfyUI BOUNDING_BOX socket dictionary."""
        return {
            "x": self.pixel_min_x,
            "y": self.pixel_min_y,
            "width": self.pixel_width,
            "height": self.pixel_height,
        }


@dataclass(frozen=True, slots=True)
class OverheadExhibit:
    asset_id: str
    asset_path: str
    kind: str
    extent_x: float
    extent_y: float
    extent_z: float
    footprint_area: float
    bounds_min: Tuple[float, float, float]
    bounds_max: Tuple[float, float, float]
    placements: List[OverheadPlacementBox]


class RosettaVisionCatalog:
    """Query index for Rosetta overhead exhibits and bounding boxes."""

    def __init__(self, map_name: str, exhibits: List[OverheadExhibit]):
        self.map_name = map_name
        self.exhibits = exhibits
        self._by_id: Dict[str, OverheadExhibit] = {e.asset_id: e for e in exhibits}
        self._by_path: Dict[str, OverheadExhibit] = {e.asset_path.lower(): e for e in exhibits}

    @classmethod
    def load_json(cls, json_path: Path | str) -> RosettaVisionCatalog:
        """Load catalog from exported JSON file."""
        p = Path(json_path)
        data = json.loads(p.read_text(encoding="utf-8"))

        map_name = data.get("MapName", "Rosetta")
        raw_exhibits = data.get("Exhibits", [])

        exhibits: List[OverheadExhibit] = []
        for raw in raw_exhibits:
            b_min = raw.get("BoundsMin", {})
            b_max = raw.get("BoundsMax", {})
            bounds_min = (float(b_min.get("X", 0)), float(b_min.get("Y", 0)), float(b_min.get("Z", 0)))
            bounds_max = (float(b_max.get("X", 0)), float(b_max.get("Y", 0)), float(b_max.get("Z", 0)))

            placements: List[OverheadPlacementBox] = []
            for p_raw in raw.get("Placements", []):
                pos = p_raw.get("pos", {})
                w_pos = (float(pos.get("X", 0)), float(pos.get("Y", 0)), float(pos.get("Z", 0)))

                box = OverheadPlacementBox(
                    tile_x=int(p_raw.get("TileX", 0)),
                    tile_y=int(p_raw.get("TileY", 0)),
                    world_pos=w_pos,
                    cell_u=float(p_raw.get("CellU", 0)),
                    cell_v=float(p_raw.get("CellV", 0)),
                    cell_size=float(p_raw.get("CellSize", 0)),
                    norm_min_u=float(p_raw.get("NormalizedMinU", 0)),
                    norm_min_v=float(p_raw.get("NormalizedMinV", 0)),
                    norm_max_u=float(p_raw.get("NormalizedMaxU", 0)),
                    norm_max_v=float(p_raw.get("NormalizedMaxV", 0)),
                    pixel_min_x=int(p_raw.get("PixelMinX", 0)),
                    pixel_min_y=int(p_raw.get("PixelMinY", 0)),
                    pixel_max_x=int(p_raw.get("PixelMaxX", 0)),
                    pixel_max_y=int(p_raw.get("PixelMaxY", 0)),
                    unique_id=int(p_raw.get("UniqueId", 0)),
                )
                placements.append(box)

            exhibit = OverheadExhibit(
                asset_id=raw.get("AssetId", ""),
                asset_path=raw.get("AssetPath", ""),
                kind=raw.get("Kind", "Model"),
                extent_x=float(raw.get("ExtentX", 0)),
                extent_y=float(raw.get("ExtentY", 0)),
                extent_z=float(raw.get("ExtentZ", 0)),
                footprint_area=float(raw.get("FootprintArea", 0)),
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                placements=placements,
            )
            exhibits.append(exhibit)

        return cls(map_name, exhibits)

    def get_placements_for_tile(self, tile_x: int, tile_y: int) -> List[Tuple[OverheadExhibit, OverheadPlacementBox]]:
        """Retrieve all placement exhibits located within a given minimap tile."""
        results: List[Tuple[OverheadExhibit, OverheadPlacementBox]] = []
        for exhibit in self.exhibits:
            for p in exhibit.placements:
                if p.tile_x == tile_x and p.tile_y == tile_y:
                    results.append((exhibit, p))
        return results

    def get_tile_prompt_boxes(self, tile_x: int, tile_y: int) -> List[Dict[str, int]]:
        """Get bounding boxes for a tile formatted for ComfyUI SAM3."""
        return [p.to_comfy_box_dict() for _, p in self.get_placements_for_tile(tile_x, tile_y)]

    def get_tile_point_prompts(self, tile_x: int, tile_y: int) -> List[Dict[str, int]]:
        """Get centroid point coordinates for all objects on the tile formatted as JSON prompt."""
        points: List[Dict[str, int]] = []
        for _, p in self.get_placements_for_tile(tile_x, tile_y):
            cx, cy = p.pixel_center
            points.append({"x": cx, "y": cy})
        return points
