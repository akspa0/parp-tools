# Progress — MdxViewer

This file is intentionally compressed. Keep only the latest compatibility milestones and open risks.

## Current Role

- `MdxViewer` is the legacy compatibility lane, not the primary destination for new `wow-viewer` ownership.

## Recent Compatibility Milestones

### Apr 16, 2026 - weak-signal terrain restore modes landed

- Added whole-tile and per-chunk restore modes.
- Added selected-chunk targeting and texture-tied sub-cell guidance.
- Proof level: compile validation only.

### Apr 16, 2026 - runtime-backed M2 path became the default successful viewer route

- `MdxViewer` now defaults to the pure `wow-viewer`-backed M2 renderer for successful runtime loads.
- Viewer-side skeletal animation playback is proven for the bounded wolf repro.
- Character customization and projected-heavy object fixes improved the bounded runtime path.

## Open Risks

- Full M2 parity is still open.
- Terrain restore remains heuristic and not broadly runtime-validated.
- Prefer moving new ownership into `wow-viewer` instead of extending this lane.
