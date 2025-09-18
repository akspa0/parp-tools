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

- Known Issues / Follow-ups:
  - Broader asset coverage and performance profiling TBD.
  - Document edge cases and invariants (e.g., optional chunks, alignment/padding) as we expand tests.

## Session Updates (DBCTool.V2 + AlphaWDTAnalysisTool)

- Works
  - CompareArea V2 implemented in `DBCTool.V2/Cli/CompareAreaV2Command.cs` with strict map-locked matching, optional 0.6.0 pivot, and stable CSV outputs (`mapping`/`unmatched`/`patch`/`patch_via060`/`trace`).
  - Crosswalks resolved for 0.5.x → 3.3.5 and via 0.6.0 when requested; path composition uses `contResolved`.
  - `053-viz/csv/*` generated (asset listings and ID-range summaries).

- Known Issues
  - `053-viz/viz/maps/Azeroth/index.html` is static and renders only AreaID 0; it is not wired to DBCTool outputs.
  - `DeadminesInstance` shows AreaIDs as 0 in visualization despite expected matches in 0.6.0 and 3.3.5.

- Next Steps
  - Inspect `trace.csv` for Deadmines rows to verify `method`, `depth`, and `mapIdX`.
  - Validate 0.6.0 pivot behavior and child indices for instance maps; confirm LK re-parenting rules where applicable.
  - Rewire visualization to consume DBCTool V2 CSV outputs for coloring instead of ADT-embedded area metadata.
  - Add focused tests for instances and oddities to prevent regressions.
