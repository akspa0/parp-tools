# Active Context

- Current Focus: Restore sub-zone AreaID fidelity for Alpha (0.5.x) → LK (3.3.5) exports by enriching DBCTool crosswalks with child hierarchies and reintroducing the 0.6.0 pivot while keeping prior parity work intact.
- Recent Changes:
  - Achieved 1:1 C++ → C# parity for Alpha WDT → Wrath ADT conversion.
  - Fixed MMID/MWID off-by-one, MCVT FourCC orientation, MH2O omission, and MCNK `MCLQ` ordering; outputs validated with 010 templates.
  - Updated `CompareAreaV2Command` to emit `tgt_child_ids` / `tgt_child_names` so LK hierarchies are preserved in CSV outputs.
- Next Steps:
  - Re-enable the 0.6.0 pivot inside DBCTool mapping so 0.5.x collisions can flow through canonical mid IDs before selecting LK targets.
  - Extend `DbcPatchMapping`/`AdtWotlkWriter` to consume pivot and child data (new lookups, verbose logging) and prefer sub-zone targets when Alpha tiles only expose parent IDs.
  - Regenerate crosswalks and verify instance maps (Deadmines, Wailing Caverns, etc.) patch to the correct LK child IDs.
  - Continue long-term refactor toward `GillijimProject.Core` / `GillijimProject.Cli` once mapping stability is achieved.
- Decisions:
  - Library-first architecture; CLI delegates to library.
  - Forward FourCC in memory; reversed on disk by serializer.
  - Exceptions for error handling; immutable domain where practical; builders/writers handle mutation.

## Session Updates (DBCTool.V2 + AlphaWDTAnalysisTool)

- Recent Changes
  - `CompareAreaV2Command` now writes LK child listings to the mapping CSVs, capturing full hierarchies per zone.
  - Added verbose ADT exporter logging to trace all 256 MCNK assignments, including chosen methods and zone/sub components.

- Current Focus
  - Plan and implement the 0.6.0 pivot revival and downstream consumer updates required to restore sub-zone mapping.

- Next Steps
  - Update DBCTool schema/output to include mid-build columns and regenerate compare artifacts.
  - Teach `DbcPatchMapping` to map `src → mid → LK`, exposing helpers for exporter fallbacks.
  - Adjust `AdtWotlkWriter` lookup order to leverage the new pivot/child data before falling back to parent zones.
  - Verify results on outdoor + instance tiles and document in planning / regression notes.
