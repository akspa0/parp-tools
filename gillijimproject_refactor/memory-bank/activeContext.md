# Active Context

- Current Focus: Refactor the working parity implementation into a reusable library (`GillijimProject.Core`) and integrate Warcraft.NET for safer, more efficient ADT writing. Maintain LK output compatibility.
- Recent Changes:
  - Achieved 1:1 C++ → C# parity for Alpha WDT → Wrath ADT conversion.
  - Fixed MMID/MWID off-by-one (no spurious final offset); corrected MCVT forward FourCC usage; ensured MH2O is omitted when empty; enforced MCLQ written last in MCNK.
  - Stabilized ADT outputs validated with 010 Editor templates.
- Next Steps:
  - Create `GillijimProject.Core` (Class Library, net9.0) and move core parsing/conversion/writing logic behind a public API.
  - Keep `GillijimProject.Cli` as a thin wrapper over the library.
  - Add test project for smoke/integration tests over representative Alpha WDT/ADT samples.
  - Introduce Warcraft.NET as writer backend where applicable; add adapters/facades as needed.
  - Document the public API (XML docs) and prepare for NuGet packaging.
- Decisions:
  - Library-first architecture; CLI delegates to library.
  - Forward FourCC in memory; reversed on disk by serializer.
  - Exceptions for error handling; immutable domain where practical; builders/writers handle mutation.

## Session Updates (DBCTool.V2 + AlphaWDTAnalysisTool)

- Recent Changes
  - Implemented CompareArea V2 pipeline in `DBCTool.V2/Cli/CompareAreaV2Command.cs`:
    - Built indices for source zones and optional 0.6.0 pivot.
    - Enforced strict map lock with `TryMatchChainExact(mapIdX, chain, ...)` only.
    - Added optional 0.6.0 pivot resolution with forced-parent overrides and minimal fuzzy only for declared oddities.
    - Composed `path` as `mapNameX/zone[/sub]` using `contResolved`/crosswalk for the map.
    - Emitted stable CSVs: `mapping.csv`, `unmatched.csv`, `patch.csv`, `patch_via060.csv`, `trace.csv`.
  - Crosswalks built for 0.5.x → 3.3.5 and optional 0.6.0 pivot; global rename fallbacks disabled when `chainVia060`.
  - Generated `053-viz/csv/*` artifacts; static viz currently shows only AreaID 0.

- Current Focus
  - Investigate `DeadminesInstance` AreaIDs reported as 0 despite good 0.6.0 and 3.3.5 matches.

- Next Steps
  - Instrument and inspect `trace.csv` rows for Deadmines: verify `method`, `depth`, and `mapIdX`.
  - Validate pivot map selection and child resolution (`idxTgtChildrenByZone`) for instance flows.
  - Rewire visualization to consume DBCTool V2 CSV outputs (e.g., `mapping.csv`/`trace.csv`) instead of ADT-embedded area fields.
  - Add targeted unit tests for instance/oddity paths and LK chain re-parenting.
