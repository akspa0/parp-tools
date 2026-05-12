# Alpha WDT Unified Ownership And Extraction Status

## Status

This migration slice is no longer a plan to create a shared Alpha reader. The shared alphaWDT stack already exists in `wow-viewer` and is the canonical owner of alphaWDT file semantics.

Implemented shared surfaces:

- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaWdtReader.cs`
- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaWdtWriter.cs`
- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaTerrainAdapter.cs`
- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaToLkConverter.cs`
- `wow-viewer/src/core/WowViewer.Core.IO/Maps/LkToAlphaConverter.cs`
- `wow-viewer/src/core/WowViewer.Core/Maps/AlphaTileData.cs`

Current proof anchors:

- staged `0.5.3` / `0.5.5` harvest and tensor-pack extraction through `AlphaWdtReader`
- focused `LkToAlphaRoundTripTests` for structural alphaWDT write parity, placements, and liquid round-trip
- staged `4.0.0 -> 0.5.3` Azeroth conversion validated in `MdxViewer`
- staged `4.0.0 -> 0.5.3` Kalimdor conversion now has full real-data WMO bundle proof in temp output with `311` converted WMOs, `1` missing root, and `0` converter failures after the WMO downgrade lane absorbed Alpha group-count and legacy batch-index limits
- Ghidra-backed format notes in `docs/architecture/alpha-wdt-ghidra-research-2026-05-10.md` and `docs/architecture/alpha-placement-coordinate-transforms-2026-05-09.md`

## Canonical Ownership

alphaWDT read and write behavior belongs in `wow-viewer` shared I/O.

- `MdxViewer` is a compatibility/runtime host, not the design owner for alphaWDT parsing or writing.
- Future alphaWDT fixes must land in `AlphaWdtReader`, `AlphaWdtWriter`, or the shared converters first.
- If `MdxViewer` needs alphaWDT behavior later, it should consume the shared domain models and bridges instead of creating another byte-level parser or writer.

This means `AlphaEmbeddedAdtReader` is compatibility-only. Keep it aligned until all consumers move to the shared reader, but do not deepen it as a second alphaWDT implementation.

## Current Shared Contract

- `MAIN` tile indexing is row-major: `tileY * 64 + tileX`
- Alpha embedded tiles always emit all `256` MCNKs
- `MCRF` stays FourCC-wrapped and contiguous inside MCNK payloads
- odd-sized top-level chunks are contiguous; do not insert pad bytes between top-level chunks
- placements use the shared round-trip-safe raw rotation convention `Rotation = (fileRotX, fileRotZ, fileRotY)`
- alpha doodads are single-owner in `MCRF`; containing chunk wins unless a preserved LK source ref stays in the same local `3x3` neighborhood
- preserved LK per-chunk `DoodadRefs` and `WorldModelRefs` must be remapped into the filtered placement-table index space before write; rebuilding placement tables without that remap collapses Alpha chunk ownership
- WMOs still use overlap-based multi-chunk references
- the Alpha client enforces an `MCNK` payload hard limit below `15000` bytes; `AlphaWdtWriter` treats that as a hard compatibility ceiling and trims only duplicate non-anchor WMO refs when a chunk would exceed budget
- target-client asset presence is determined from target archives, wrapper scan, and loose files only

## What Still Remains

- move remaining `MdxViewer` alpha consumers off `AlphaEmbeddedAdtReader` onto shared `AlphaWdtReader` / `AlphaTerrainAdapter` surfaces
- keep any future alphaWDT writer needs in `AlphaWdtWriter` rather than adding app-side write logic
- continue broad real-data LK/Cata corpus validation for `LkToAlpha`
- keep the shared alphaWDT docs current when format discoveries or consumer boundaries change
