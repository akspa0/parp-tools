# Active Context

- **Current Focus**: Improve WoWRollback viewer overlays for the Phase 0 time-travel feature. Priorities: resolve shadow-map overlay 404s, replace heavy JSON overlays with lightweight rasters (PNG with opacity) for map view, and retain detailed per-tile data for upcoming timeline workflows.
- **Recent Changes**:
  - Implemented MCNK-reader enhancements and added `LkAdtTerrainReader` so overlay builders can pull authoritative data from cached LK ADTs.
  - Updated `rebuild-and-regenerate.ps1` to rerun AlphaWdtAnalyzer terrain/shadow extraction even when cached ADTs already exist, ensuring CSV parity.
  - Identified viewer fetch mismatch for `shadow_map` tiles and confirmed JSON payload size (>2 MB) is a performance blocker for whole-map view.
- **Next Steps**:
  - Align viewer fetch paths with generated shadow overlay assets and add diagnostics to prevent silent 404s.
  - Design and implement PNG-based overlay generation (shadow, terrain, liquids, holes) with opacity controls while keeping JSON detail for per-tile inspection.
  - Update viewer overlay layers to consume rasters for map view and lazy-load JSON only when zooming to tiles (supporting Time Travel plan in `docs/planning/03_Rollback_TimeTravel_Feature.md`).
  - After overlay refactor, circle back to remaining map-lock validation work (forced-zero tables, DBCTool sweeps) to keep mapping stable.
- **Decisions**:
  - Mapping safeguards stay in place but are background tasks while viewer performance work is active.
  - Maintain library-first direction and existing coding standards (FourCC handling, immutable domains) during overlay refactor.
