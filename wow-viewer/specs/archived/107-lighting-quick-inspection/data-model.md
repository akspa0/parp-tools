# Data Model

- **Scene far plane**: `max(fogEnd + padding, minimum)` bounded by the renderer maximum. The minimum is only enough for a valid projection, not a world-distance override.
- **Hover precision**: `HoveredAssetInfo.IsPreciseRayHit` is true only for a nearest scene/PM4 ray hit. Brush-derived candidates remain selectable but do not produce a tooltip.
