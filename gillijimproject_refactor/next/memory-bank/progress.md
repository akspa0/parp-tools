# Progress (Next)

- Works:
  - LiquidsConverter implemented (MH2O↔MCLQ) with precedence and flags mapping.
  - CLI parses liquids flags and builds LiquidsOptions.
  - Alpha→LK pipeline signature updated to accept LiquidsOptions and an IAlphaLiquidsExtractor.
  - AdtLk extended to carry per-MCNK Mh2oChunk array (256 entries).
  - Convert command wired to call AlphaToLkConverter with AlphaMclqExtractor and prints conversion summary.
  - Docs updated (architecture/CLI) to reflect extractor behavior.
  - Next.Core builds cleanly after fixing CS0246 by adding `using System.Collections.Generic;` to `Services/UniqueIdAnalyzer.cs` and `Transform/AlphaToLkConverter.cs`.
  - Fixed `AlphaMclqExtractorTests` builder compile issue by simplifying `BuildMcnkHeader` (no out-params) and preserving `OfsLiquid` (100) / `SizeLiquid` (104) patch positions.
- Pending:
  - Unit tests for MCLQ parsing (synthetic water/ocean/magma) and optional fixture-based integration tests.
  - Validation & logging (size vs actual, offset heuristics, tile normalization, all-none short-circuit).
  - Round-trip and integration tests once IO paths are complete.
- Known Issues / Follow-ups:
  - LVF Case 1/3 (UVs) deferred; green-lava mapping behavior TBD.
- WDL Parsing:
  - Completed: Domain split to `Domain/Wdl.cs`, parser hardening (MVER tolerance, overflow-safe bounds, padding), MAHO holes parsing (incl. reversed `OHAM`), tests (synthetic normal + reversed + missing MAHO + odd-size padding) and fixture skip-if-missing.
  - Pending: Implement `WdlWriter` and minimal CLI verbs (`wdl-dump`, `wdl-build`) with roundtrip tests.
