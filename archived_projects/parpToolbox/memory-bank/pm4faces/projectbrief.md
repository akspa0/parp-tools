# PM4Faces Project Brief

- Purpose: Export PM4 MSUR surface geometry to OBJ/GLTF with correct planar snapping and grouping.
- Primary Goal: Fix broken or non-planar faces by optionally snapping vertices to each surface plane during export.
- Constraints: Minimal complexity. Default behavior unchanged unless explicitly enabled.
- Key Knobs:
  - --snap-to-plane (off by default)
  - --height-scale <float> (only when snapping; default 1.0)
- Success Criteria:
  - Baseline export unaffected when not snapping.
  - When snapping, planar surfaces close, cracks reduced, no "exploded" geometry.
  - Real data validation across multiple tiles.
