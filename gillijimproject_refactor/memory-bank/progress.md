# Progress

- Works:
  - 1:1 C++ → C# parity achieved for Alpha WDT → Wrath ADT conversion; outputs validated with 010 templates.
  - Core primitives: `Utilities.cs`, `WowChunkedFormat.cs`, `Chunk.cs` stable.
  - Alpha: `WdtAlpha` parsing; `AdtAlpha` to `AdtLk` conversion pipeline.
  - LK: `AdtLk` assembly and serialization; MHDR/MCIN offsets stable for current assets.
  - Fixed bugs and correctness items:
    - MMID/MWID off-by-one eliminated (no extra final offset).
    - MCVT forward FourCC in memory; on-disk reversed correctly.
    - MH2O omitted when empty; MCNK writes `MCLQ` last.
    - Index and offset handling consolidated in `Chunk` and helpers.

- Pending:
  - Refactor into `GillijimProject.Core` (Class Library, net9.0) + `GillijimProject.Cli` (thin wrapper).
  - Define public API for parse/convert/write; add XML docs.
  - Add test project and smoke/integration tests with fixtures.
  - Integrate Warcraft.NET writer APIs via adapters/facades where applicable.
  - Prepare NuGet packaging for the core library; basic CI later.

## Session Updates (DBCTool.V2 + AlphaWDTAnalysisTool)

- Works:
  - CompareArea V2 implemented in `DBCTool.V2/Cli/CompareAreaV2Command.cs` with strict map-locked matching, optional 0.6.0 pivot, and stable CSV outputs (`mapping`/`unmatched`/`patch`/`patch_via060`/`trace`).
  - Crosswalks resolved for 0.5.x → 3.3.5 and via 0.6.0 when requested; path composition uses `contResolved`.
  - `053-viz/csv/*` generated (asset listings and ID-range summaries).
  - CSV schema now exposes `tgt_child_ids` / `tgt_child_names`, enabling LK child hierarchy inspection.
  - YAML exports `Area_hierarchy_335.yaml`, `Area_hierarchy_mid_0.6.0.yaml`, and `Area_hierarchy_src_<alias>.yaml` added to inspect canonical zone/sub-zone relationships per map.
  - `AdtWotlkWriter.PatchMcnkAreaIdsOnDiskV2()` now skips writing when no LK mapping exists, preserving alpha `zone<<16|sub` values.
  - `DbcPatchMapping` stores mid entries keyed by `(src_mapId, src_areaNumber)` and `(mid_mapId, mid_areaId)` to prevent cross-continent collisions during pivot lookups.
  - `Area_crosswalk_v3` / `Area_crosswalk_via060` now include canonical source and pivot names plus per-map splits to aid Alpha→LK reconciliation.

- Known Issues
  - `053-viz/viz/maps/Azeroth/index.html` is static and renders only AreaID 0; it is not wired to DBCTool outputs.
  - `DeadminesInstance` shows AreaIDs as 0 in visualization despite expected matches in 0.6.0 and 3.3.5.
  - `Area_crosswalk_v3`/`Area_crosswalk_via060` still contain `-1` mid entries; requires name-based inference before unified outputs can be generated.
  - Existing tooling still relies on per-map CSVs; YAML-informed crosswalks need integration into `DbcPatchMapping`/`AdtWotlkWriter`.
  - Many `mid060_*` entries remain `-1`; requires name-based inference before unified outputs can be generated.
  - Need helper plumbing (`TryResolveSourceMapId`, shared mid entry lookup) in `DbcPatchMapping` and `AdtWotlkWriter` to finalize map-locked mid usage.

- Next Steps
  - Implement mid-ID inference fallback driven by canonical names, then emit unified per-map CSV/YAML linking source → mid → target.
  - Wire `DbcPatchMapping` helpers into `AdtWotlkWriter` so mid lookups always carry the correct source map ID.
  - Rewire visualization to consume DBCTool V2 CSV outputs for coloring instead of ADT-embedded area metadata.
  - Add focused tests for instances and oddities to prevent regressions once exporter logic stabilizes.
