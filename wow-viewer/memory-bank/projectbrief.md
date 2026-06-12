# Project Brief — wow-viewer

## Why this project exists

The viewer, the format tools, and the dataset work all serve one purpose: **the PM4 and ADT data is a recoverable historical record of an art pipeline, and the project is the toolchain for reading it.**

Four empirical claims anchor that purpose:

1. **The PM4 is a giant map object addressed by a program or database system above it in a larger hierarchy.** It is not a self-contained format blob; it is a queryable index into something larger. The 24-bit CK24, the MSHD fields, the MSUR/MSCN/MSPV/links are all table entries, not arbitrary metadata. The byte-stratification work, the matching tool, and the writer all derive from this framing.

2. **The ADT UniqueID system is a sediment record of art in development.** The same `uniqueId` value persists across multiple ADTs and across build versions, and the way those values stack up over time is a record of the art pipeline's evolution. Peeling back the uniqueId layers is temporal archaeology, not present-state render. The same is true of the alpha-mask bands inside each ADT — every MCAL chunk is a record of "what did the terrain look like at this version of the art pipeline," not just "what does the terrain look like now."

3. **There are countless historical artifacts in these files that have not yet been characterized.** "We don't know what MSHD field 0x1C means" is not a known unknown; it is a known known that the project is responsible for narrowing. Every byte in MSHD, MSUR, MSLK, MPRL, MSCN, MSPV is on the table.

4. **The weak signal amplifier is the proof that the data is recoverable after the format's 33.334× downscale.** A curious eraser function scaled the source data down by exactly 33.334× at some point in the pipeline's history. The weak signal amplifier is the empirical demonstration that the resulting data still contains enough signal to reconstruct the original. The project is not a viewer for a partial dataset; the project's toolchain is the demonstration that the downscale was lossy-but-recoverable, and that what we are working with is closer to the original than its size suggests.

The project explicitly is not a viewer for the live game, not a publication vehicle for findings, and not a bot or exploit framework. The findings are the side effect; the value is the toolchain. The toolchain exists to make the bytes visible and the byte relationships queryable, and to keep doing that as the project's understanding of the format deepens.

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
