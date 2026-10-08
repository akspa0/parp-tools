"""Building Foundation Plateau Carver (Spec 264 Phase 3).

Carves level foundation plateaus under detected or placed WMO building footprints
with smooth Hermite perimeter blending, preventing flat pancake or sunken pit artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

from harvester.v60.development_ground_truth import WmoPlacement


@dataclass
class FoundationCarveResult:
    carved_height_257: np.ndarray  # Shape (257, 257) float32
    foundation_mask: np.ndarray  # Shape (257, 257) bool
    plateau_elevations: List[Tuple[str, float]]  # [(building_name, base_elevation)]


class BuildingFoundationCarver:
    """Carves level foundation plateaus for WMO buildings into terrain elevation meshes."""

    def __init__(self, blend_margin: int = 3):
        self.blend_margin = max(1, blend_margin)

    def carve_foundations(
        self,
        terrain_height_257: np.ndarray,
        wmo_placements: List[WmoPlacement],
    ) -> FoundationCarveResult:
        """Carve level foundation plateaus under building footprints with smooth perimeter transitions."""
        out_height = terrain_height_257.copy().astype(np.float32)
        total_foundation_mask = np.zeros((257, 257), dtype=bool)
        plateaus: List[Tuple[str, float]] = []

        if not wmo_placements:
            return FoundationCarveResult(
                carved_height_257=out_height,
                foundation_mask=total_foundation_mask,
                plateau_elevations=plateaus,
            )

        for wmo in wmo_placements:
            px_min, py_min, px_max, py_max = wmo.pixel_box
            # Rescale from 256x256 to 257x257 grid
            gx_min = int(np.clip(px_min * (257.0 / 256.0), 0, 256))
            gx_max = int(np.clip(px_max * (257.0 / 256.0), 0, 256))
            gy_min = int(np.clip(py_min * (257.0 / 256.0), 0, 256))
            gy_max = int(np.clip(py_max * (257.0 / 256.0), 0, 256))

            if gx_max <= gx_min or gy_max <= gy_min:
                continue

            # Build binary footprint mask for this building
            building_footprint = np.zeros((257, 257), dtype=bool)
            building_footprint[gy_min : gy_max + 1, gx_min : gx_max + 1] = True

            # Determine perimeter sampling boundary (1-2 pixels outside footprint)
            dilated = ndimage.binary_dilation(building_footprint, iterations=2)
            perimeter = dilated & (~building_footprint)

            if np.any(perimeter):
                perimeter_heights = out_height[perimeter]
                # Foundation elevation is median perimeter height or authored placement Z
                if abs(wmo.pos[2]) > 0.1:
                    foundation_z = float(wmo.pos[2])
                else:
                    foundation_z = float(np.median(perimeter_heights))
            else:
                foundation_z = float(wmo.pos[2]) if abs(wmo.pos[2]) > 0.1 else float(np.mean(out_height[building_footprint]))

            plateaus.append((wmo.name, foundation_z))

            # Distance transform for smooth blending zone
            # Distance inside and outside footprint
            dist_outside = ndimage.distance_transform_edt(~building_footprint)
            blend_zone = (dist_outside <= self.blend_margin) & (~building_footprint)

            # Apply level foundation inside footprint
            out_height[building_footprint] = foundation_z
            total_foundation_mask |= building_footprint

            # Smooth Hermite transition in blend zone
            if np.any(blend_zone):
                t = np.clip(dist_outside[blend_zone] / float(self.blend_margin), 0.0, 1.0)
                # Cubic smoothstep: 3*t^2 - 2*t^3
                smooth_w = t * t * (3.0 - 2.0 * t)
                natural_z = out_height[blend_zone]
                out_height[blend_zone] = (1.0 - smooth_w) * foundation_z + smooth_w * natural_z

        return FoundationCarveResult(
            carved_height_257=out_height,
            foundation_mask=total_foundation_mask,
            plateau_elevations=plateaus,
        )
