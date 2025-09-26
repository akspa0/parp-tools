# Plan: Alpha WDT Follow-ups (Next Session)

## Goals
- Stop unmapped Alpha maps from inheriting unrelated LK AreaIDs.
- Restore actionable verbose logging with low overhead.
- Produce privacy-safe, per-map export indexes.
- Consolidate project memory bank into a single maintained location.

## Tasks
1. **Guard CSV fallback mappings**
   - Add `allowCsvFallback` gate in `AlphaWdtAnalyzer.Core/Export/AdtWotlkWriter.cs`.
   - Only execute numeric patch lookups when a zone candidate exists or `currentMapId` is known.
   - Re-run Kalidar to confirm AreaIDs remain `0xFFFFFFFF`.
2. **Fix verbose logging**
   - Flush `verboseLog` contents to `awdt_run_<timestamp>.log` when `--verbose` is set.
   - Skip log buffer allocation when verbose mode is off to regain performance.
3. **Per-map index outputs**
   - Emit `index.json` under `output_root/maps/<MapName>/` and use paths relative to `output_root/`.
   - Update any consumers that expect the previous global file.
4. **Memory bank cleanup**
   - Archive `memory-bank/`, `DBCTool.V2/memory-bank/`, and `AlphaWDTAnalysisTool/memory-bank/` into an `archive/` folder.
   - Create a single master memory bank directory with current context summaries.

## Notes
- Run targeted Kalidar/Kalimdor exports after code changes to verify no unexpected AreaID assignments.
- Document new logging behavior in `README` once implemented.
- Coordinate memory bank consolidation with future documentation updates.
