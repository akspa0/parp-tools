# ADT Workflow Guardrails & IDE Rules (2025-10-01)

## 1. Purpose
- **[goal]** Prevent repeat regressions when working with ADT ingestion, conversion, and visualization in `WoWRollback`.
- **[scope]** Applies to all pipelines that read Alpha/LK ADTs, generate coordinate CSVs, or render map imagery for 0.5.x–0.6.x datasets.

## 2. Non-Negotiable Rules (IDE Hints)
- **[no-temp-sources]** Never point `ConvertedAdtDir` or related settings at personal/temp folders. All scripts must target `rollback_outputs/converted_adts/<version>/<map>/`.
- **[plan-before-code]** For any workflow tweak >20 LOC, update `docs/planning/plan-001-adt-and-viewer-repair.md` (or successor) before editing code.
- **[smoke-test-first]** Run `dotnet run --project WoWRollback.Cli -- analyze-adt-dump` on a known tile after any reader change. Fail the commit if `MDDF`/`MODF` counts drop to 0.
- **[feature-flag]** Gate WIP features (e.g., split MODF/MDDM plotting) behind CLI switches; default behavior must match last stable release (`bd759de0`).
- **[doc-sync]** Any IDE rule edit requires updating this file and the relevant runbook before merging.

## 3. ADT Reading Consistency Rules
- **[common-path]** All readers (`AlphaWdtAnalyzer.Core.AdtScanner`, `WoWRollback.Core.Services.LkAdtReader`, etc.) must route through a shared abstraction that selects Alpha vs LK parsing by source version. Do not fork logic per caller.
- **[chunk-handshake]** Maintain identical chunk validation (FourCC reversal, size guards) across implementations. If one reader tolerates padding or missing subchunks, all should.
- **[version-resolution]** Map version aliases (`0.5.3.3368`, `0.6.0.3592`) to canonical source format (`alpha`, `lk`) in a single `VersionFormatRegistry`. Scripts must query the registry instead of hard-coding assumptions.
- **[logging]** On parse failure, emit structured logs including map, tile, chunk, byte offset. Never swallow exceptions silently.

## 4. Handling Empty or Zero-Sized Chunks
- **[detect]** If `chunkSize == 0`, log once per tile and classify the tile as `EmptyChunk`. Do not treat it as success.
- **[escalate]** When `EmptyChunk` tiles exceed threshold (config default: 1) for a `{version,map}`, abort the pipeline and surface an actionable error.
- **[fallback]** Provide a `--allow-empty-chunks` switch (default off) for exploratory runs; when enabled, tag resulting CSV rows with `source_chunk=EMPTY`.
- **[metrics]** Emit aggregate counts (`total_chunks`, `empty_chunks`, `decoded_chunks`) into the diagnostics CSV to aid regression tracking.

## 5. Conversion & Caching Workflow
- **[single-entry]** Enforce one PowerShell entry point (`rebuild-and-regenerate.ps1`) that calls `regenerate-with-coordinates.ps1`.
- **[cache-layout]** Converted ADTs must live under `rollback_outputs/converted_adts/<version>/<map>/tile_rXX_cYY.adt`. Scripts may only read/write inside this root.
- **[rebuild-switch]** Provide `-ForceConvert` to rebuild cache and `-CleanConverted` to wipe it. Default behavior is reuse-with-validation.
- **[validation]** After conversion, run `analyze-adt-dump` for a sample tile and assert `MDDF + MODF > 0` unless `--allow-empty-chunks` is active.
- **[version-sync]** Include the conversion command set in `docs/planning/runbook-adt-regeneration.md` (to be authored) so newcomers can reproduce from scratch.

## 6. Data Output & Plotting Expectations
- **[split-layers]** Separate MODF/MDDM plotting must produce two distinct CSV layers and viewer overlays. If not implemented, the feature flag remains off.
- **[schema-stability]** CSV headers (`world_x`, `world_y`, `world_z`, etc.) cannot change without updating consumers and documenting the change.
- **[diagnostics]** Collect per-tile statistics (counts, empty flags) and persist alongside outputs for diffing.

## 7. Image Conversion Service Concept
- **[service-goal]** Replace ad-hoc BLP → JPG/WebP converters with a dedicated service (`TileImageService`).
- **[inputs]** Accept raw `.blp` files and optional overlays (paths, bounding boxes).
- **[outputs]** Emit SVG tiles by default, with fallback to WebP when vectorization fails.
- **[architecture]** Implement as a library (`WoWRollback.Imaging`) exposing methods `RenderTileSvg()` and `RenderTileBitmap()`. CLI scripts call the service instead of invoking Python or shell utilities.
- **[caching]** Store rendered SVGs under `rollback_outputs/tiles/svg/<version>/<map>/` with cache-busting via file hashes.
- **[extensibility]** Keep palette conversion, lighting, and fallback assets configurable via JSON in `docs/config/tile-image-service.json` (new file).

## 8. Review & Enforcement
- **[code-review]** Require at least one reviewer to verify adherence to this document before merging ADT-related changes.
- **[checklist]** Add a PR template section referencing this doc: "Does the change alter ADT ingestion/conversion? If yes, link to updated runbook/tests.".
- **[automation]** Introduce CI job running a minimal `regenerate-with-coordinates.ps1 -Maps DeadminesInstance -Versions 0.5.3.3368` to guard against regressions.
- **[updates]** When expanding the workflow (e.g., new image service), log decisions here and in `docs/planning/plan-001-adt-and-viewer-repair.md`.
