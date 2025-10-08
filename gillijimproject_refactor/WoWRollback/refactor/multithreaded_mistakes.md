# Multithreaded Conversion: What Went Wrong, What To Do Next

## Scope
- **Goal**: Speed up Alpha→LK ADT conversion (tile write + AreaID patching) safely.
- **Non-goal**: Changing the analysis pipeline (it was fine); mixing outputs for multiple maps.

## Current Symptoms
- **Slow conversion**: Hour+ for ~900MB despite small per-file sizes.
- **Build breakages**: Attempts to inline MT changes in `AdtExportPipeline.cs` caused compile errors.
- **Incorrect asset fixups**: MDX→M2 wasn’t normalized first; BLP specular variants not mapped.
- **Output confusion**: Analysis CSVs duplicated under `03_adts` and `04_analysis`.

## Root Causes (Technical)
- **Shared logger/policy**: `AdtExportPipeline.ExportSingle()` used a single `FixupLogger`/`AssetFixupPolicy`, making parallel tiles unsafe.
- **Parallel gating too strict**: `canParallel = !EnableFixups && !TrackAssets` disabled parallelism in normal runs.
- **Inline patching fragility**: Edits applied without exact context anchors → syntax errors (`fixupLogger` no longer in scope, tuple deconstruction errors).
- **Resolver gaps**:
  - MDX paths not normalized to `.m2` when present.
  - Tileset textures sometimes have only `_s` specular in later data.

## What We Tried (and why it failed)
- **Directly editing `AdtExportPipeline.cs`** for per-tile fixups and gating:
  - Failed to match exact code anchors; partial replacements left the file in an inconsistent state.
- **PowerShell text surgery**: brittle replacements, quoting issues → parser errors.
- **Dropping a new MT wrapper (`ConvertPipelineMT`)**:
  - Added parallel conversion, but didn’t fully remove or align with existing pipeline, causing build/runtime dissonance.

## Correct Approach (Minimal-Risk Plan)
1. **Stabilize current codebase**
   - Revert/undo partial edits to `AdtExportPipeline.cs` that removed `fixup`/`fixupLogger` references (bring file back to a compiling state).
   - Keep analysis changes isolated; don’t touch analysis while fixing conversion.

2. **Implement per‑tile fixups (safely) in conversion**
   - In `AdtExportPipeline.ExportSingle()` and `ExportBatch()`:
     - Create `FixupLogger` and `AssetFixupPolicy` INSIDE `ProcessTile`/`ProcessTileB` for each tile.
     - Use file paths `asset_fixups_{x}_{y}.csv` to avoid contention.
     - Pass `ctx.Fixup = tileFixup` to `AdtWotlkWriter.WriteBinary(ctx)`.
     - After tiles processed, merge `asset_fixups_*.csv` → `asset_fixups.csv`.
   - Change parallel gating to: `canParallel = !opts.TrackAssets`.

3. **Resolver improvements before fuzzy**
   - MDX→M2 normalization: If `.mdx` is referenced, but `.m2` exists in listfile or disk, map to `.m2` with method `normalize_ext:(primary|secondary)`.
   - Tileset BLP specular: For missing non-`_s` texture, try `_s` variant first.

4. **Disk usage cleanup**
   - After analysis completes, keep only one copy of terrain/shadow CSVs (prefer `04_analysis`).
   - Option: add `--cleanup-03-analysis` flag to delete `03_adts/<ver>/analysis` CSVs for the session.

5. **Verification**
   - Run conversion on a small subset (3–5 tiles) with verbose on; verify:
     - CPU usage near cores-1, no shared logger exceptions.
     - `asset_fixups.csv` merged, no missing deconstruction vars.
     - Two example fixes resolved as expected:
       - `world/azeroth/elwynn/passivedoodads/bellow/bellow.mdx` → `.m2`
       - `world/generic/human/passive doodads/statues/utherstatue.mdx` → `.m2`
       - `Tileset/Duskwood/DuskwoodRockGravelAlpha.blp` → `tileset/duskwood/duskwoodrockgravel_s.blp`

## Concrete Edit Steps (Surgical)
- `AlphaWdtAnalyzer.Core/Export/AdtExportPipeline.cs`:
  - In `ExportSingle()` tile function (`ProcessTile`):
    - Add before building `WriteContext`:
      - `var tileLogPath = Path.Combine(logDir, $"asset_fixups_{x}_{y}.csv");`
      - `using var tileLogger = new FixupLogger(tileLogPath);`
      - `var tileFixup = new AssetFixupPolicy(resolver, ..., tileLogger, inventory, opts.LogExact);`
    - In context: `Fixup = tileFixup` instead of shared fixup.
    - Wrap `BeginTile/EndTile` calls on `tileLogger` not the shared logger.
  - After tile loop, merge per-tile CSVs into `asset_fixups.csv`.
  - Change `bool canParallel = !opts.TrackAssets;`.
- `AlphaWdtAnalyzer.Core/Export/AssetFixupPolicy.cs`:
  - Before fuzzy, attempt `.mdx`→`.m2` normalization.
  - For tileset `.blp` when missing non-`_s`, try `_s` variant.

## Rollback Plan (if issues arise)
- Keep a copy of the pre-edit `AdtExportPipeline.cs`.
- Toggle parallelism off by forcing `canParallel = false` to compare runtimes/behavior quickly.
- Disable cleanup to inspect intermediates if analysis diffs are needed.

## Ownership & Guardrails
- All edits are confined to `AlphaWdtAnalyzer.Core` conversion code (not DBCTool V2).
- No changes to analysis output schema.
- One map per run integration test; avoid changing upstream listfile behavior beyond normalization.

## Success Criteria
- Parallel tile conversion with fixups enabled; no threading exceptions.
- MDX→M2 and BLP `_s` mappings fixed; those lines disappear from `fuzzy` fixups and appear as `normalize_ext` or `tileset_variant`.
- Runtime reduction: target <10–15 minutes for ~900MB (machine dependent), CPU near full utilization.
- Single authoritative copy of terrain/shadow CSVs at `04_analysis`.

## Next Actions (blocked until approval)
- Restore `AdtExportPipeline.cs` to a compiling baseline.
- Apply the surgical per-tile logger/policy patch and gating change.
- Re-run a small-map test, then full map with `MaxDegreeOfParallelism` tuned (start with cores-1).
