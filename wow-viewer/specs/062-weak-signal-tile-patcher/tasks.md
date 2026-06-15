# Tasks: Weak Signal Tile Patcher Export

**Feature**: 062-weak-signal-tile-patcher

## Phase 1 — Extract Weak Signal Detection

- [x] 1.1 Create `WeakSignalDetector.cs` in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/`. Detection methods: `Analyze()`, `EstimateAmplificationFactor()`, `EstimateFallbackFactor()`, `EstimateFactorFromRanges()`, `AmplifyHeightmap()`, `SnapFactor()`. Static and stateless.
- [x] 1.2 Create `WeakSignalAnalysis.cs` data class: `TileX`, `TileY`, `IsWeakSignalCandidate`, `HeightRange`, `MinHeight`, `MaxHeight`, `AmplificationFactor`, `AnchorHeight`, `FactorSource`, `Severity`, `WasPatched`.
- [x] 1.3 Create `WeakSignalOptions.cs`: `MaxHeightRange`, `MinHeightBand`, `MaxHeightBand`, `UseAutoFactor`, `ManualFactor` with sensible defaults.
- [ ] 1.4 Refactor `ViewerApp.cs` to call shared methods (deferred — viewer has complex stateful hooks; shared library is usable by converter without viewer changes).
- [ ] 1.5 Write unit tests (deferred).

## Phase 2 — Build the Tile Patcher Command

- [x] 2.1 Create `TerrainWeakSignalPatchCommand.cs` in converter tool. CLI args: `--map-path`, `--output-dir`, `--format`, threshold overrides (`--max-height-range`, `--min-height-band`, `--max-height-band`), factor override (`--amplification-factor`), `--no-copy-family`.
- [x] 2.2 Implement map loading: `IndexRootAdts()` scans directory for `_root.adt` files.
- [x] 2.3 Implement detection pass: call `WeakSignalDetector.Analyze()` per tile, collects `WeakSignalTileReport`.
- [x] 2.4 Implement amplification pass: WDL or fallback factor estimation, `WeakSignalDetector.AmplifyHeightmap()`, snap factor.
- [x] 2.5 Implement LK ADT output: copy original ADT, call `AdtTerrainWriter.Write()` to patch heights/normals. Copy tile family.
- [x] 2.6 Alpha WDT output deferred — `AlphaWdtWriter` is frozen (Rule 10). Requires reading Alpha WDT tiles and building corrected data.
- [x] 2.7 Write `weak_signal_patch_report.json` with per-tile details.
- [x] 2.8 `dotnet build` passes (Core.Runtime + Converter).
- [x] 2.9 Add `--client-root <dir> --map <name>` support for in-memory MPQ reading via `NativeMpqService`. Supports both Alpha `.wdt.MPQ` and LK split ADT formats. Real-data proof: 1.12.1 EmeraldDream (38 patched), 0.5.3 Azeroth (722 patched).

## Phase 3 — Full Overlay Copy

- [ ] 3.1 Copy unpatched tile families to output directory (already done via `CopyTileFamilyToOutput`).
- [ ] 3.2 WDL file copied to output directory (already done).
- [ ] 3.3 Output directory is self-contained — requires testing with real map data.

## Phase 4 — End-to-End Validation

- [x] 4.1 End-to-end test with real map data: 0.5.3 Azeroth (722 patched), 0.5.5 Azeroth (586 patched), 1.12.1 EmeraldDream (38 patched) via `--client-root`.
- [ ] 4.2 Performance comparison (pending).
- [ ] 4.3 Alpha WDT format support (deferred to follow-up).
