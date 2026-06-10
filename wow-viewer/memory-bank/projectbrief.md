# Project Brief — wow-viewer

## Mission
WoW format analysis, cross-era terrain reconstruction, and PM4-based automated object matching. The project targets **Alpha 0.5.3**, **0.6.0**, **LK 3.3.5**, and **Cataclysm-era PM4** tiles.

## Primary Project: WoWViewer
A .NET 10 / OpenGL world viewer (`wow-viewer/src/viewer/WoWViewer/`) that renders terrain, WMOs, M2/MDX models, liquids, PM4 overlays, and DBC-driven data from staged game clients. This is the sole active development target. The legacy `MdxViewer` (`gillijimproject_refactor/src/MdxViewer/`) is read-only reference.

## Scope
- **WoWViewer**: Real-time 3D rendering, PM4 overlay analysis, WMO/M2 matching, headless validation capture
- **Read/Write**: WDT, ADT, WMO, M2/MDX/MDL, BLP, PM4, DBC/DB2 for all supported versions
- **Convert**: Modern → Alpha retroporting, WMO/M2 ↔ MDX, terrain format round-tripping
- **Tools**: PM4 analysis & matching, M2 animation pose farm, Zarr-based dataset building, terrain training-pipeline orchestration
- **Data Harvesting**: NPZ/Zarr terrain tensor extraction, multi-build corpus building, V18 dataset generation

## Current Reality (June 2026)
- WoWViewer renders terrain, WMOs, M2/MDX models, liquids, and PM4 overlays
- PM4 overlay has per-file dual cache (in-memory + on-disk), MSCN/MSPV visualization, WMO group matching
- M2 animation pose farm library (`WowViewer.Core.Anim`) has model loading and path normalization — keyframe extraction ready for Phase 2
- V18 terrain training pipeline is the active ML track
- The legacy `MdxViewer` in `gillijimproject_refactor` is read-only reference only

## Key Architecture Decisions
- **Library-first**: Format readers live in `WowViewer.Core.*` libraries, tools are thin CLI wrappers
- **Spec Kit**: Features start with a spec (`specs/NNN-name/`) before planning or implementation
- **One phase at a time**: Each phase ends with validation against staged game client data
- **No H:\CLIENTS**: All game client access goes through `output/tmp/wowarchive-clients/`

## Active Specs (important)
- 046: PM4 asset matching (consolidated, absorbs old 050/052)
- 051: PM4 MSCN/MSPV visualization and signature extraction
- 053: M2/MDX animation pose farm (Phase 0-1 done)
- 054: PM4 camera window cache (nearly complete)
