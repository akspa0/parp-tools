# MSUR Refactor Checkpoint – Build Fallout and Next Actions (2025-08-10)

## Summary
- MSUR.Entry confirmed 32 bytes.
  - 0x00: GroupKey (aka SurfaceGroupKey alias kept)
  - 0x01–0x02: IndexCount (u16 LE, stored as int)
  - 0x03: Unknown03 (SurfaceAttributeMask alias kept)
  - 0x04–0x0F: Nx, Ny, Nz (floats)
  - 0x10: Float10 (Height getter alias kept; semantics unknown)
  - 0x14: MsviFirstIndex (uint)
  - 0x18: MdosIndex (uint)
  - 0x1C: CompositeKey (aka SurfaceKey)
- Helpers:
  - SurfaceKeyHigh16 / SurfaceKeyLow16
  - SurfaceKeyHi12 / SurfaceKeyLo12 (12+12 split of low 24 bits of CompositeKey)
- Compatibility aliases preserved (read-only): SurfaceGroupKey, SurfaceAttributeMask, IsM2Bucket, IsLiquidCandidate, Height(get).
- Neutral stance on unknown semantics for 0x03 and 0x10.

## Build errors to resolve (legacy usages)
Replace the following in call-sites:
- AttributeMask -> SurfaceAttributeMask (or Unknown03 if raw byte preferred)
- Padding_0x03 -> Unknown03 (we no longer maintain a padding field)
- Writes to Height -> write to Float10 (Height is getter-only alias)

Observed from last build log (file:line):
- Services/PM4/Core/Pm4FieldMappingService.cs:42 – AttributeMask
- Services/PM4/Pm4ChunkCombinationTester.cs:93 – AttributeMask
- Services/PM4/Pm4CsvDumper.cs:82 – AttributeMask
- Services/PM4/Pm4GlobalTileLoader.cs:166 – AttributeMask
- Services/PM4/Pm4GlobalTileLoader.cs:167 – Padding_0x03
- Services/PM4/Pm4GlobalTileLoader.cs:171 – assignment to Height (should write Float10)
- Services/PM4/Pm4GroupingTester.cs:576 – AttributeMask

Warnings are non-blocking (async without await, nullability hints) and can be deferred.

## Orientation baseline
- MSCN is ground-truth; default X-flip applied unless `legacyParity` disables.
- Exporters (regular and per-tile) must apply object-level X flip consistently.
- Remove tile-level rotation/mirroring paths; keep coordinates global.
- Validate overlay with `--no-remap` baseline.

## Tile coordinates and grouping
- Always use original tileX/tileY from source (filenames/PM4) for filenames/metadata — no fixups.
- Continue wiring `TileCoord` and `Scene.TileCoordByTileId` into assemblers/exporters.
- Persist `_Oxx` suffix as `superObjectId` for provenance and grouping.

## CompositeKey 12+12 usage
- `SurfaceKeyHi12` / `SurfaceKeyLo12` available on `MSUR.Entry` for auditing and potential grouping signals.
- Keep usage exploratory; do not assert semantics beyond observed utility for naming/audit.

## Next session plan
1) Update call-sites listed above
   - AttributeMask -> SurfaceAttributeMask/Unknown03
   - Padding_0x03 -> Unknown03
   - Writes to Height -> Float10
2) Rebuild solution (Release, minimal verbosity)
3) Quick validation export (small region) with MSCN + per-tile overlay; verify alignment with object-level X-flip only
4) Proceed to minimal MSPI span-based triangle inclusion to reduce mesh holes

## Notes
- Keep knobs minimal. Avoid adding new diagnostics unless explicitly requested.
- Do not speculate about unknown fields; name neutrally and document observations.
