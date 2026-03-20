# Progress

### Mar 19, 2026 - Terrain Texture Transfer Command (Backend Slice)

- Added first backend/library + CLI slice for mapped terrain texture transfer:
	- command: `terrain-texture-transfer`
	- payload scope: `MTEX`, `MCLY`, `MCAL`, `MCSH`, and MCNK holes
	- mapping modes: explicit `--pair` and auto `--global-delta`
	- supports `dry-run` manifests and `apply` output ADT writing
- Added split-ADT resilience for the active development dataset:
	- if `SplitAdtMerger` serialization fails, command now composes transferable texture payload from root + `_tex0.adt`
	- MCNK subchunk parsing now tolerates headerless tex0 MCNK payloads
	- top-level chunk walk/rebuild now handles odd-size boundary variance seen in split files
	- merge path now skips `obj0`-only sidecars (without `_tex0`) and uses root bytes directly for terrain-texture transfer
- Real-data validation performed (fixed path):
	- source/target: `test_data/development/World/Maps/development`
	- dry-run sample: `development_0_0 -> development_0_0` (chunk pairs=256, copied flags true for MTEX/MCLY/MCAL/MCSH/holes)
	- apply sample: same pair wrote output ADT + summary/tile manifests
	- non-identity sample: `development_0_0 -> development_1_0` succeeded in both dry-run and apply with full payload transfer and no manual-review flags
	- small global-delta batch (`--global-delta 1,0 --tile-limit 3`) completed; 2 tiles clean, 1 tile (`development_0_1 -> development_1_1`) still flagged manual-review due one target MCNK with no parseable subchunks
- Validation limits:
	- no viewer runtime visual signoff yet for transferred outputs in this pass
	- no new automated tests added in this pass

### Mar 19, 2026 - MdxViewer Thin UI Hook For Terrain Texture Transfer

- Added a thin UI entry in `MdxViewer` (`ViewerApp`) for the backend terrain texture transfer flow:
	- File menu item: `Terrain Texture Transfer...`
	- dialog supports source/target/output folders, dry-run/apply toggle, explicit-pair or global-delta mapping, chunk offsets, payload toggles, and optional manifest path
	- execution runs asynchronously via the existing app-thread pattern and surfaces summary + warnings in an in-dialog log panel
- Build validation passed for the viewer after wiring:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Validation limits:
	- no new runtime visual validation in the viewer yet for this dialog path
	- this UI slice does not resolve the known `development_0_1 -> development_1_1` target MCNK parse edge case from backend validation

### Mar 19, 2026 - Canonical Fresh-Output Pass For 3.3.5 Development Map

- Executed a full-map identity transfer pass to materialize a fresh canonical output folder for viewer use:
	- command: `terrain-texture-transfer --source-dir ...335-dev... --target-dir ...335-dev... --global-delta 0,0 --mode apply`
	- output root: `output/development-335-canonical-texture-transfer`
- Real-data result summary:
	- tiles planned/processed/written: 2303 / 2303 / 2303
	- manual review: 0
	- chunk pairs applied: 589,568
	- missing source/out-of-range chunk remaps: 0 / 0
	- summary manifest: `output/development-335-canonical-texture-transfer/manifests/summary.json`
	- companion `development.wdt` and `development.wdl` copied into output root
- Operational guidance:
	- this is now a viable "open the generated folder in MdxViewer" workflow for the tested 3.3.5 development dataset
	- this does not replace targeted non-identity remap validation when using non-zero global deltas or explicit cross-tile mappings

### Mar 19, 2026 - Development Repair WL Attribution + Texture Payload Manifests

- Reworked `DevelopmentRepairService` WL ingestion so repair no longer assumes tile-named `*.wl*` files.
	- new behavior pre-indexes all map-level WL files (`.wlw/.wlm/.wlq/.wll`) once, converts to MH2O by world position, and applies per-tile liquids from that coordinate-attributed index
	- tile manifests now record the actual WL source file paths used (for example `Clayton Test.wlw`) instead of synthetic `tileName.wlw` expectations
- Expanded per-tile JSON payload (`TextureData`) with terrain texturing data modeled after the VLM chunk-layer shape:
	- includes MTEX texture list
	- includes per-chunk layers with texture id/path, flags, alpha offset, effect id, plus optional base64 alpha bytes and byte count
	- extractor now chooses the richest source among output ADT, `_tex0.adt`, and root ADT so split-source tiles can still emit texture payload data
- Real-data validation performed on fixed paths:
	- command: `development-repair --mode repair --input-dir test_data/development/World/Maps/development --tile-limit 50`
	- observed manifests with `WlLiquidsConverted=true` and map-level WL source filenames attached to those tiles
	- reference check only: `development-repair --mode repair --input-dir test_data/WoWMuseum/335-dev/World/Maps/development --tile-limit 1` (used only to inspect payload shape, not as canonical pipeline input)
	- policy now enforced in code: `development-repair` rejects WoWMuseum `335-dev` input and requires building clean outputs from `test_data/development/World/Maps/development` constituent parts
- Validation limits:
	- this pass did not include viewer-side visual validation of generated MH2O/texturing results
	- no new automated regression tests were added in this pass

## Mar 17, 2026 - Recovery Branch Checkpoint (v0.4.0 base)

- Active branch reset in main tree: recovery/v0.4.0-surgical-main-tree (base 343dadf).
- Restored .github customization stack from main and committed as 845748b.
- Build from this branch passes in primary tree environment.
- Terrain alpha decode profile routing is now staged in code:
	- TerrainAlphaDecodeMode in AdtProfile
	- LichKingStrict for 3.x profiles
	- LegacySequential for 0.x profiles
	- StandardTerrainAdapter alpha extraction routes by profile mode

### Critical Pending Validation

- Runtime terrain checks still required on both families:
	- Alpha-era terrain
	- LK 3.3.5 terrain
- Do not mark terrain safety complete until these real-data checks are done.

### Immediate Next Work

1. Finalize commit state for the profile/decode changes (if still local).
2. Run manual runtime spot-checks for alpha decode output.
3. Resume surgical commit intake from v0.4.0..main in SAFE-first order.

### Mar 17, 2026 - Intake Triage Update

- Reviewed queued commits `177f961`, `d50cfe7`, `326e6f8`, `4e2f681`, `37f669c`, `39799bf`, and `62ecf64` against the recovery branch and terrain-alpha guardrails.
- Marked `177f961` and `37f669c` as RISKY and out of scope for safe-first intake.
- Marked `d50cfe7`, `326e6f8`, `4e2f681`, `39799bf`, and `62ecf64` as MIXED; only isolated helper/tooling slices are candidates.
- Selected first SAFE extraction: corrected `TerrainImageIo` alpha-atlas helper from `62ecf64` only.
- Explicitly rejected the earlier `d50cfe7` `TerrainImageIo` version because it hardcoded atlas edge remapping that the recovery notes already identified as changing shipped data.
- No claim of terrain safety from this triage alone; runtime real-data validation is still required.

### Mar 17, 2026 - First SAFE Batch Applied

- Added `src/MdxViewer/Export/TerrainImageIo.cs` from the corrected `62ecf64` implementation only.
- Kept ViewerApp, TerrainRenderer, WorldScene, test-project, and terrain decode heuristic changes out of this batch.
- Build gate passed: `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- Runtime terrain validation remains pending; build-only status is not sufficient for terrain signoff.

### Mar 18, 2026 - Rendering Recovery Batch

- Applied the `WorldAssetManager` renderer-residency fix from main so placed MDX/WMO renderers are no longer evicted out from under live world instances.
- `GetMdx` / `GetWmo` now lazy-load missing models and cached failed loads can be retried.
- Added the minimal skybox backdrop path from main:
	- route skybox-like MDX/M2 placements into a dedicated list
	- render the nearest skybox as a camera-anchored backdrop before terrain
	- added `ModelRenderer.RenderBackdrop(...)` with forced no-depth state for all layers
- Verified that the recovery branch already contained the reflective M2 depth-flag fix and env-map backface guard, so those regressions were not reintroduced here.
- Build gate passed again: `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- Runtime validation still required; build success does not prove:
	- doodad/WMO reload correctness after moving away and back
	- correct skybox classification on real map data
	- MH2O liquid correctness on LK 3.3.5 tiles

### Mar 18, 2026 - MCCV + MPQ Recovery Batch

- Restored MCCV terrain color support on the active chunk-based terrain path.
- `TerrainChunkData` now carries MCCV bytes, `StandardTerrainAdapter` populates them, `TerrainMeshBuilder` uploads them, and `TerrainRenderer` applies them in shader.
- Initial MCCV fix improved output but did not fully match runtime behavior.
- Applied the isolated `NativeMpqService` recovery slice from the mixed MPQ commits:
	- expanded patch archive ordering for locale/custom patch names
	- full normalized path encrypted-key derivation with basename fallback
	- compression bitmask handling for MPQ sectors
	- BZip2 support via SharpZipLib
- Build gates passed:
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Runtime validation still required; build success does not prove:
	- patched 1.x+ MPQ read correctness on real patch chains
	- encrypted later-version MPQ entry reads on real data
	- MCCV highlight/tint correctness on real 3.x terrain

### Mar 18, 2026 - MCCV + Patch-Letter Follow-up

- Reworked MCCV semantics after user runtime feedback showed the first shader heuristic was still wrong.
- Current interpretation now matches the repo's own MCCV writer comments:
	- bytes are treated as BGRA, not RGBA
	- neutral/no-tint values are mid-gray (`127`) rather than white
	- terrain tint uses RGB remapped around mid-gray, not MCCV alpha strength
- Extended `NativeMpqService.LoadArchives(...)` to discover MPQs recursively so nested/custom `patch-[A-Z].mpq` archives are included in the patch chain.
- Kept Alpha single-asset wrapper archives (`.wmo.mpq`, `.wdt.mpq`, `.wdl.mpq`) out of the generic recursive scan because they are handled separately by the viewer data source.
- Build gates passed again after this follow-up:
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Runtime validation is still the blocker:
	- confirm 3.x MCCV transparent/neutral regions no longer darken to black
	- confirm maps stored inside `patch-[A-Z].mpq` are now discovered and load through normal WDT/ADT lookup paths

### Mar 18, 2026 - 3.x Alpha Offset-0 Experiment Reverted

- The recent LK offset-0 fallback change in `StandardTerrainAdapter.ExtractAlphaMaps(...)` was reverted after runtime validation showed it was wrong.
- Updated conclusion:
	- treating `AlphaMapOffset == 0` as a valid relaxed fallback for the active 3.x terrain path is not the correct fix
	- keep the revert and continue investigating the real 3.x alpha decode/sourcing failure separately
- Validation status:
	- normal `dotnet build .../MdxViewer.sln -c Debug` still conflicts with the running viewer process locking `bin/Debug`
	- use the alternate-output build for compile validation while the viewer stays open

### Mar 18, 2026 - 3.x Profile-Driven Alpha Recovery

- Investigated the remaining 3.x terrain failure after the offset-0 revert.
- Confirmed the active recovery branch was still missing rollback-era handling for:
	- MPHD/WDT big-alpha mask `0x4 | 0x80`
	- split `*_tex0.adt` sourcing for textures/layers/alpha/shadow data
	- stronger MCAL decode semantics for compressed alpha, big alpha, and do-not-fix chunks
- Applied the recovery batch:
	- `FormatProfileRegistry`: added `BigAlphaFlagsMask` and `PreferTex0ForTextureData`; 3.0.1 and 3.3.5 profiles now use `0x4 | 0x80` and prefer `*_tex0.adt`
	- `StandardTerrainAdapter`: can read MTEX + MCNK data from `*_tex0.adt`, route layer/alpha/shadow sourcing through that file, pass the MCNK `0x8000` do-not-fix flag into alpha decode, and infer big-alpha per chunk
	- `WoWMapConverter.Core/Formats/LichKing/Mcal.cs`: replaced the broken/simple decoder with the stronger compressed / big-alpha / 4-bit implementation
- Build validation passed:
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`
	- `dotnet build "I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="I:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
- Runtime validation is still the blocker:
	- no claim yet that 3.x alpha blending is correct on real user data
	- next check is whether the failing 3.x sample now uses more than one alpha layer and stops looking like 4-bit Alpha-era decode

### Mar 18, 2026 - Terrain Runtime Validation Update

- User runtime validation now confirms the current terrain alpha recovery on two real data families:
	- Alpha 0.5.3 terrain renders correctly again after restoring the alpha-era edge fix in `AlphaTerrainAdapter`
	- 3.0.1 alpha-build terrain renders correctly on the profile-driven strict 3.x path
- Earlier runtime feedback also reported the 3.3.5 sample looked correct before the 0.5.3 regression was fixed.
- Status change:
	- terrain validation is no longer build-only for the tested 0.5.3 and 3.0.1 samples
	- broader signoff across more 3.x maps is still pending, so do not generalize this to all LK-era terrain yet

### Mar 18, 2026 - Remaining ModelRenderer Slice From 39799bf

- Applied the last model-side hunk from `39799bf` after the MPQ reader work was already in place.
- `ModelRenderer` now skips particle rendering on the world-scene batched render path only.
- Standalone model viewing still renders particles as before.
- Reason: per-instance transforms are not yet propagated into particle simulation for placed models, and leaving them enabled there can produce visibly wrong camera-locked effects.
- Build gate passed: `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.

### Mar 18, 2026 - WDL Preview Warmup + Texture Reuse Batch

- Ported the missing `main` WDL preview cache support into the recovery branch:
	- added `WdlPreviewCacheService`
	- `ViewerApp` now warms discovered WDL previews in the background and opens the preview dialog through the cache-aware path
	- `ViewerApp_WdlPreview` now shows warmup/error state instead of only a synchronous failure dialog
- Added a targeted model-load performance slice in `ModelRenderer`:
	- per-model texture diagnostic logs are now opt-in via `PARP_MDX_TEXTURE_DIAG`
	- BLP/PNG textures now use a shared refcounted GL texture cache so repeated world doodads do not decode/upload the same texture once per instance
- Build validation passed: `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- Runtime validation is still required before claiming:
	- WDL preview warmup/cache behavior is correct on the user's real map set
	- M2 load time is materially improved in the real world scene

### Mar 18, 2026 - WDL Parser Recovery + Transparency Heuristic Follow-up

- Addressed the newly reported WDL read failure after the preview-cache port:
	- `WoWMapConverter.Core/VLM/WdlParser.cs` no longer rejects all non-`0x12` WDL versions up front
	- parser now scans the WDL chunk stream for `MAOF` and accepts MAOF offsets that reference either `MARE` headers or direct height payloads
- Unified active viewer WDL reads through `src/MdxViewer/Terrain/WdlDataSourceResolver.cs` so both preview warmup and `WdlTerrainRenderer` use the same `.wdl` / `.wdl.mpq` + file-set lookup path.
- Closed a remaining 3.x model-path gap in `WmoRenderer` by extending doodad extension fallback from only `.mdx`/`.mdl` to also include `.m2`.
- Adjusted `ModelRenderer` transparency routing:
	- shared texture cache entries now retain simple alpha-shape metadata
	- classic non-M2 `Transparent` layer-0 materials only use hard cutout when the texture alpha is binary
	- textures with intermediate alpha now stay on the blended path
- Build validation passed: `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- No automated tests were added or run.
- No runtime real-data validation has been performed yet for this batch.

### Mar 18, 2026 - Standalone 3.x Model Load Freeze Follow-up

- Addressed the reported freeze / non-load behavior when opening individual 3.x `.mdx` files in the viewer.
- Root cause on the active standalone path was different from the world/WMO loaders:
	- standalone container probing only recognized `MD20`, not `MD21`
	- standalone M2 adaptation eagerly scanned the full `.skin` file list on the UI thread before trying the obvious same-basename candidates
	- standalone file loads also lacked the world path's canonical model-path recovery and MD20 converter fallback
- Current fix in `src/MdxViewer/ViewerApp.cs`:
	- standalone probe now routes both `MD20` and `MD21` through the M2-family path even when the file extension is `.mdx`
	- standalone M2 loads now resolve a canonical model path through MPQ file-set indexes before skin lookup
	- predictable `.skin` candidates are tried first, and the broader `.skin` file-list search is only used as a fallback with a per-session cache
	- standalone MD20 loads now also have the same M2->MDX converter fallback used elsewhere when direct adaptation cannot complete
	- standalone skin-path cache is cleared when a new MPQ data source is loaded
- Build validation passed: `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- No automated tests were added or run.
- No runtime real-data validation has been performed yet for this batch.

### Mar 18, 2026 - M2 Empty-Fallback Guardrail

- Follow-up after runtime feedback that some M2-family models still "load" into an empty viewport with `0` geosets / vertices.
- Current conclusion:
	- at least some failures are not clean adapter failures; the raw `MD20` converter fallback can produce an `MDX` shell that parses but has no renderable geometry
	- that state is misleading in the UI because it looks like a loaded model rather than an unsupported / failed conversion
- Current fix:
	- `WarcraftNetM2Adapter` now exposes shared renderable-geometry checks
	- standalone `ViewerApp`, world `WorldAssetManager`, and WMO doodad `WmoRenderer` now reject converted fallback models unless they contain at least one renderable geoset
	- rejected fallback loads now preserve/log the underlying failure instead of silently treating an empty converted model as success
- Build validation passed: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`.
- No automated tests were added or run.
- No runtime real-data validation has been performed yet for this batch.
- This is a diagnostics/correctness guardrail, not proof that pre-release `3.0.1` M2 layouts are fully supported.

### Mar 18, 2026 - Pre-release 3.0.1 M2 Scope Clarified

- User runtime verification after the guardrail patch indicates most remaining M2 problems are specific to the pre-release `3.0.1` model family rather than the later `3.3.5` family.
- Current working assumption:
	- pre-release `3.0.1` model files may be a transitional or hybrid `MDX` + `M2` variant
	- later-WotLK assumptions should not be silently reused for that path
- Separate runtime issue remains open across both model families:
	- neon-pink transparent surfaces still appear on both `MDX` and M2-family assets
	- treat that as a shared renderer/material/shader problem, not proof of a model-parser-only defect
- Resulting investigation split for the next pass:
	1. add true version/profile-aware handling for pre-release `3.0.1` model structure
	2. audit shared transparent-surface handling, texture resolution, and blend/shader parity independently of format parsing
- No new code changes were made in this note-only follow-up.
- Runtime evidence came from the user's real data, not fixtures.

### Mar 19, 2026 - Pre-release 3.0.1 Model Profile Guardrail

- Live `wow.exe` decompilation for build `3.0.1.8303` confirmed the client-side model gate is stricter than the active generic adapter path:
	- required root magic is `MD20`
	- accepted version range is `0x104..0x108`
	- parser behavior splits structurally at `0x108`
- Active viewer code now routes that profile knowledge into all three shared M2-family entry points:
	- standalone `ViewerApp.LoadM2FromBytes(...)`
	- world `WorldAssetManager.LoadMdxModel(...)`
	- WMO doodad `WmoRenderer.LoadM2DoodadRenderer(...)`
- `WorldScene` / `WorldAssetManager` now receive the build string at construction time so constructor-time manifest loads use the same profile guard instead of waiting for later `SetDbcCredentials(...)`.
- `WarcraftNetM2Adapter` now fails fast on build/profile mismatches before `.skin` search or fallback conversion:
	- `3.0.1.8303` and unknown `3.0.x` profiles reject `MD21` roots and out-of-range MD20 versions
	- `3.3.5.12340` currently keeps `MD21` container allowance to avoid broad later-branch regression while the parser path remains shared
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
	- no new runtime real-data validation yet for the guarded `3.0.1` model path
	- do not claim this as a full pre-release `3.0.1` render fix; it is a profile-routing/compatibility guardrail
- Separate shared renderer issue is still open:
	- neon-pink transparent surfaces remain a Track B problem across classic `MDX` and M2-family assets

### Mar 19, 2026 - Standalone Data-Source M2 Read-Path Fix

- The new user-visible `Failed to read: ...` symptom on standalone/browser-loaded M2-family assets was not a parser error.
- Root cause:
	- `ViewerApp.LoadFileFromDataSource(...)` still did an exact `_dataSource.ReadFile(virtualPath)` and returned early
	- M2-family assets in the file browser can appear under alias paths that need the same canonical resolution logic already used later in the standalone M2 path
- Current fix:
	- data-source loads for `.mdx` / `.mdl` / `.m2` now resolve through `ResolveStandaloneCanonicalModelPath(...)`
	- browser-side model reads now use `ReadStandaloneFileData(...)` before giving up
	- successful reads now carry the resolved virtual path into the later container-probe path
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
	- runtime retry on the user's actual 3.0.1 data is still required to confirm the failure moved from read-time to the next real blocker

### Mar 19, 2026 - Pre-release 3.0.1 wow.exe Documentation Pass

- Shifted from speculative code changes to binary-backed documentation after the user reported that models still do not load.
- New documented `wow.exe` facts for build `3.0.1.8303`:
	- common loader chain is `FUN_0077e2c0` -> `FUN_0077d3c0` -> `FUN_0079bc70` -> `FUN_0079bc50` -> `FUN_0079bb30` -> `FUN_0079a8c0`
	- accepted model-family extensions are normalized to `.m2` before parse/bootstrap continues
	- high-level failure falls back to `Spells\\ErrorCube.mdx`
	- root parser is `MD20`-only with version range `0x104..0x108`
	- parser layout splits at `0x108`
	- confirmed validator families now include shared span strides `1`, `2`, `4`, `8`, `0x0C`, `0x30`, `0x44` and nested record families `0x70`, `0x2C`, `0x38`, `0xD4`, `0x7C`
	- version split families are legacy `0xDC` + `0x1F8` versus later `0xE0` + `0x234`
- New artifacts created for fresh chats:
	- `documentation/pre-release-3.0.1-m2-wow-exe-guide.md`
	- `.github/prompts/pre-release-3-0-1-m2-implementation-plan.prompt.md`
	- `.github/prompts/pre-release-3-0-1-m2-ghidra-followup.prompt.md`
	- `.github/prompts/pre-release-3-0-1-m2-runtime-triage.prompt.md`
- Validation status for this pass:
	- no automated tests were added or run
	- no new build was needed because this pass only added documentation and prompts
	- no runtime real-data validation was performed

### Mar 19, 2026 - 3.0.1 Pre-release Profile Routing Broadening

- Follow-up after the wow.exe-backed profile guardrail: the active registry no longer binds the pre-release `3.0.1` profile only to exact build `3.0.1.8303`.
- Current behavior:
	- any parsed `3.0.1.x` build now resolves to the same pre-release `3.0.1` ADT, WMO, and M2 profiles
	- other `3.0.x` builds still fall back to the generic unknown `3.0.x` profile until there is binary evidence for a narrower mapping
- Why this matters:
	- standalone model loads, world doodads, WMO doodads, and terrain/WMO profile routing now stay on the pre-release path for the whole `3.0.1` family instead of silently downgrading non-`8303` builds to the generic `3.0.x` profile
- Validation status:
	- build validation pending for this specific routing change
	- no automated tests were added or run
	- no runtime real-data validation was performed

### Mar 19, 2026 - 3.0.1 Pre-release M2 Parser + Fallback Alignment

- Follow-up after the routing-only fix was not enough: active model loading now includes a dedicated pre-release `MD20` parse path in `WarcraftNetM2Adapter` instead of sending raw `3.0.1` files through Warcraft.NET's later-layout `MD21` assumptions.
- Current viewer-side behavior:
	- standalone, world, and WMO doodad adapter loads normalize pre-release `MD20` data through a local parsed-model abstraction
	- the old forced profile-specific `.skin` parser path was disabled because the wow.exe-derived `0x70` / `0x2C` family sizes were not proven `.skin` submesh / batch strides
	- converter fallback now receives the active build version and avoids hard-parsing later-layout animation / bone tables for pre-release `3.0.1`
	- converter skin fallback keeps only the index / triangle tables required for geometry conversion instead of forcing nonessential fixed-stride submesh / texture-unit tables
- Why this matters:
	- the primary runtime path and the fallback conversion path no longer disagree about pre-release `3.0.1` model-family assumptions
	- non-`8303` `3.0.1.x` builds now reach both the right profile and a compatible loader path
- Validation status for this pass:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
	- no automated tests were added or run
	- no runtime real-data validation was performed

### Mar 19, 2026 - Standalone Alias Recovery + Unsuffixed Skin Candidates

- Follow-up after fresh runtime errors still showed two model-load gaps:
	- standalone/browser `DataSourceRead` failures could still stop at the unresolved `.mdx` alias path even when the world-model path already had broader file-set heuristics
	- companion skin discovery only tried `00`-`03` suffixed names, not the unsuffixed `.skin` form some transitional assets may use
- Current fix:
	- `ViewerApp` standalone canonical resolution and data-source reads now reuse the broader candidate set already proven useful on the world path: exact path, extension aliases, bare filename aliases, and `Creature\Name\Name.{mdx|m2|mdl}` guesses
	- standalone resolution now also probes guessed candidates through `FileExists` / `ReadFile` instead of depending only on the prebuilt file index
	- shared `WarcraftNetM2Adapter.BuildSkinCandidates(...)` now includes unsuffixed `.skin` candidates before the numbered `00`-`03` forms
- Why this matters:
	- user-visible `Failed to read requested='...mdx'` errors can now recover through the same alias breadth the world loader already had
	- `Missing companion .skin for M2` can now recover when the sidecar is present under the base `.skin` name instead of only numbered variants
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
	- no automated tests were added or run
	- runtime validation on the specific failing assets is still pending

### Mar 19, 2026 - Cocoon Optional-Span Parser Follow-up

- Fresh runtime log from `Creature\Cocoon\Cocoon.mdx` showed the profiled pre-release parser was now reached, but it still failed before geometry extraction because an unresolved optional table span (`colors`, stride `0x2C`) was treated as fatal.
- Current fix:
	- `WarcraftNetM2Adapter.ParseProfiledMd20Model(...)` now hard-validates only the spans the runtime model builder actually dereferences for viewer geometry

### Mar 19, 2026 - MCNK Index Repair Hook For Development ADT Export

- Added a rollback-CLI `repair-mcnk-indices` command that audits or rewrites root ADT `MCNK` header `IndexX` / `IndexY` values.
- `development-repair` now runs the same fixup in-memory on exported root ADTs by default; disable with `--repair-mcnk-indices false` if raw output is needed.
- Repair logic prefers `MCIN` order when present and otherwise falls back to top-level `MCNK` scan order.
- Real-data audit on the loose source folder `test_data/development/World/Maps/development` found:
	- 466 root ADT filenames
	- 114 zero-byte placeholders
	- 352 non-empty roots with chunk data
	- 0 detected `MCNK` index mismatches under scan-order validation on those raw loose roots
- Validation limits:
	- this does not prove generated WDL-derived / repaired export sets are clean because the referenced `PM4ADTs/*` outputs are not present in this workspace
	- `dotnet run/build` for `WoWRollback.Cli` is still blocked here by pre-existing missing `WoWFormatLib` / `CascLib` references under `WoWRollback.AnalysisModule`, so end-to-end CLI execution was not revalidated in this environment
	- optional / unresolved table families now use a nonfatal validator that logs and skips invalid spans instead of rejecting the entire model
	- per-texture filename spans are also treated as optional so a bad embedded name table does not abort the whole model
- Why this matters:
	- `Cocoon.mdx` was failing in the parser before any real geometry read was attempted
	- this keeps the wow.exe-backed strictness for required geometry tables while avoiding false rejects from still-unmapped optional families on `0x104..0x107` models
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
	- no automated tests were added or run
	- no runtime real-data validation was performed after the fix

### Mar 19, 2026 - Classic 0.5.3 MDX Regression Closed; 3.0.1 Still Open

- User runtime validation now confirms the classic Alpha `0.5.3` MDX rendering regression is fixed.
- Confirmed repair stack in `src/MdxViewer/Rendering/ModelRenderer.cs`:
	- direct-path replaceable fallback is restricted to `_isM2AdapterModel`
	- wrap/clamp interpretation is split between classic MDX and M2-adapted models
	- classic `Layer 0 + Transparent` once again always uses alpha-cutout
- A new direct-asset diagnostic path was added in `src/MdxViewer/AssetProbe.cs` and wired through `src/MdxViewer/Program.cs`:
	- `--probe-mdx` loads an asset from a real client path, prints parsed materials, and reports decoded BLP alpha statistics
	- this was used on `DuskwoodTree07.mdx` to prove the remaining canopy failure was in renderer behavior after decode, not in TEXS parsing or BLP decode
- Current status change:
	- classic `0.5.3` MDX should be treated as restored for the tested runtime sample
	- pre-release `3.0.1` rendering is still buggy and remains the active unresolved model-family track

### Mar 19, 2026 - PM4 Coordinate Validation Command

- Added `WoWMapConverter.Core/Formats/PM4/Pm4CoordinateService.cs` as the first authoritative PM4 placement helper set in active core code.
- Added `WoWMapConverter.Core/Formats/PM4/Pm4CoordinateValidator.cs` to validate transformed `MPRL` refs against real `_obj0.adt` placements from the fixed development dataset.
- Added CLI command: `wowmapconverter pm4-validate-coords [--input-dir <dir>] [--tile-limit <n>] [--threshold <units>] [--json <path>]`.
- Important scope limit:
	- this is a real-data validation path for `MPRL` only
	- it does not yet validate MSCN semantics
	- it does not yet build the cross-tile CK24 registry
- Validation status at this note:
	- initial real-data slice showed `MPRL` is already in ADT placement order, not tile-local
	- broadened sample run on 100 validated tiles reported 38,133 refs in expected tile bounds (100.0%) and 36,070 refs within 32 units of a nearest `_obj0.adt` placement (94.6%)
	- average nearest-placement distance on that sample was 10.86 units
	- broader work is still pending for CK24 aggregation and MSCN semantics

### Mar 20, 2026 - PM4 Viewer Overlay Diagnostics/Grouping/Winding Pass

- Added active PM4 overlay rendering + diagnostics in `src/MdxViewer/Terrain/WorldScene.cs` and `src/MdxViewer/ViewerApp.cs`.
- Added PM4 color modes for structural inspection (`CK24` type/object/key, tile, dominant group/attribute, height).
- Added optional PM4 3D markers (`MPRL` refs and object centroids).
- Added CK24 decomposition controls for disjoint geometry:
	- split by shared vertex connectivity
	- optional split by dominant `MSUR.MdosIndex` before connectivity
- Added per-object planar transform solve and winding parity correction:
	- candidate swap/invert U/V planar transforms scored against nearest `MPRL` anchors
	- mirrored parity now flips triangle winding order to avoid backward-wound faces
- Validation status:
	- repeated `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
	- no automated tests were added or run
	- runtime real-data signoff still pending for merged/disjoint PM4 object cases
- Scope boundary:
	- this does not replace the pending map-level CK24 registry or finalize MSCN semantics
	- current PM4 reconstruction should be treated as viewer debug instrumentation + heuristics, not final export-grade identity mapping

## ✅ Working

### Mar 19, 2026 - 4.x Split ADT No-MCIN Fallback

- Real-data audit of the fixed `test_data/development/World/Maps/development` loose roots confirmed the current 4.x load failure is primarily a no-`MCIN` issue, not an `MCNK.IndexX/IndexY` issue:
	- 466 root ADT filenames
	- 114 zero-byte placeholders
	- 352 non-empty roots
	- 0 non-empty roots with `MCIN`
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` now falls back to top-level `MCNK` scan order when a root ADT omits `MCIN`.
- `src/WoWMapConverter/WoWMapConverter.Core/Converters/LkToAlphaConverter.cs` now uses the same root fallback so later split roots can flow into the existing Alpha conversion path instead of throwing immediately on missing `MCIN`.
- Scope limit:
	- this is a geometry/chunk-order recovery step first
	- full 4.x `_tex0.adt` texture-layer parity is still not claimed
	- the converter only consumes split texture companions when they expose LK-style `MCNK` payloads large enough for the current Alpha builder
- Validation status at this note:
	- code edits landed
	- build/runtime validation still pending after this patch

### MdxViewer (3D World Viewer) — Primary Project
- **Alpha 0.5.3 WDT terrain**: ✅ Monolithic format, 256 MCNK per tile, async streaming
- **0.6.0 split ADT terrain**: ✅ StandardTerrainAdapter, MCNK with header offsets (Feb 11)
- **0.6.0 WMO-only maps**: ✅ MWMO+MODF parsed from WDT (Feb 11)
- **Terrain liquid (MCLQ)**: ✅ Per-vertex sloped heights, absolute world Z, waterfall support (Feb 11)
- **WMO v14 rendering**: ✅ 4-pass: opaque → doodads → liquids → transparent
- **WMO liquid (MLIQ)**: ✅ matId-based type detection, correct positioning (Feb 11)
- **WMO doodad culling**: ✅ Distance (500u) + cap (64) + nearest-first sort + fog passthrough
- **WMO doodad loading**: ✅ FindInFileSet case-insensitive + mdx/mdl swap → 100% load rate
- **MDX rendering**: ✅ Two-pass opaque/transparent, alpha cutout, specular highlights, sphere env map
- **MDX GEOS version compatibility**: ✅ Ported version-routed GEOS parser behavior from `wow-mdx-viewer` (v1300/v1400 strict path + v1500 strict path + guarded fallback)
- **MDX SEQS name compatibility**: ✅ Counted 0x8C named-record detection broadened to reduce fallback `Seq_{animId}` names on playable models
- **MDX PRE2/RIBB parsing parity**: ✅ Expanded parser coverage for PRE2 and RIBB payload/tail animation chunks (runtime visual verification pending)
- **MDX animation engine**: ✅ BONE/PIVT/HELP parsing, keyframe interpolation, bone hierarchy (Feb 12)
- **Full-load mode**: ✅ `--full-load` (default) loads all tiles at startup with progress (Feb 11)
- **MCSH shadow maps**: ✅ 64×64 bitmask applied to all terrain layers
- **AOI streaming**: ✅ 9×9 tiles, directional lookahead, persistent tile cache, MPQ throttling (Feb 11)
- **Frustum culling**: ✅ View-frustum + distance + fade
- **AreaID lookup**: ✅ Low 16-bit extraction + low byte fallback for MapID mismatch
- **DBC Lighting**: ✅ LightService loads Light.dbc + LightData.dbc, zone-based ambient/fog/sky colors
- **Replaceable Textures**: ✅ DBC CDI variant validation against MPQ + model dir scan fallback
- **Minimap overlay**: ✅ From minimap tile images
- **PM4 debug overlay (viewer-side)**: 🔧 In progress — color modes, 3D markers, CK24 split modes, and parity-aware winding fixes landed; runtime signoff still pending

### Model Parsers & Tools
- **MDX-L_Tool**: ✅ Core parsing and Archaeology logic complete.
- **GEOS Chunk (Alpha)**: ✅ Robust scanner for Version 1300 validated.
- **Texture Export**: ✅ DBC-driven `ReplaceableId` resolution working.
- **OBJ Splitter**: ✅ Geoset-keyed export verified on complex creatures.
- **0.5.3 Alpha WDT/ADT**: ✅ Monolithic format, sequential MCNK.
- **WMO v14/v17 converter**: ✅ Both directions implemented.
- **BLP**: ✅ BlpResizer complete.

### Data Generation
- **VLM Datasets (Alpha)**: ✅ Azeroth v10 (685 tiles).

## ⚠️ Partial / In Progress

### MdxViewer — Rendering Quality & Performance
- **3.3.5 ADT loading freeze**: Needs investigation
- **WMO culling too aggressive**: Objects outside WMO not visible from inside
- **MDX GPU skinning**: Bone matrices computed per-frame but not yet applied in vertex shader (needs BIDX/BWGT vertex attributes)
- **MDX animation UI**: Sequence selection combo box in ImGui panel not yet wired
- **MDX per-geoset color/alpha**: Only static alpha used; animated GeosetAnims not wired
- **MDX particles/ribbons**: Parser coverage expanded; runtime behavior verification still pending on effect-heavy assets
- **MDX texture UV animation**: Not implemented
- **MDX billboard bones**: Not implemented
- **WMO lighting**: v14-16 grayscale lightmap + v17 MOCV vertex colors not implemented
- **Vulkan RenderManager**: Research phase — `IRenderBackend` abstraction for Silk.NET Vulkan

### Build & Release Infrastructure
- **GitHub Actions**: ✅ `.github/workflows/release-mdxviewer.yml` — tag push or manual dispatch
- **WoWDBDefs bundling**: ✅ 1315 `.dbd` files copied to output via csproj Content items
- **Self-contained publish**: ✅ `dotnet publish -c Release -r win-x64 --self-contained` verified

### MDX-L_Tool Enhancements
- **M2 Export (v264)**: 🔧 Implementing binary writer.

## ❌ Known Issues

### MdxViewer Rendering Bugs (Feb 12, 2026)

#### MDX Sphere Env / Specular Orientation (Feb 14, 2026)
- **Symptom**: Reflective/specular surfaces (e.g., dome-like geometry) appeared inward-facing on some two-sided materials.
- **Fix Applied**: Fragment shader now flips normals/view-space normals on backfaces before env UV generation and lighting/specular.
- **Status**: 🔧 Patched in code, pending visual confirmation on Dalaran dome repro.

#### WMO Semi-Transparent Window Materials
- **Symptom**: Stormwind WMO maps blue/gold stained glass textures to white marble columns instead of window frames
- **Hypothesis 1**: Secondary MOTV chunk not skipped → MOBA batch parsing misalignment
- **Fix Attempt 1**: Added `reader.BaseStream.Position += chunkSize;` when secondary MOTV encountered in `WmoV14ToV17Converter.ParseMogp` (line 922)
- **Result**: ❌ FAILED — window materials still map to wrong geometry
- **Status**: Root cause still unknown. May not be MOTV-related. Need to check console logs to verify if secondary MOTV is even present in Stormwind groups.

#### MDX Cylindrical Texture Stretching
- **Symptom**: Barrels, tree trunks show single wood plank stretched around entire circumference instead of tiled texture
- **Hypothesis 1**: Texture wrap mode incorrectly clamping both S and T axes when only one should clamp
- **Fix Attempt 1**: Changed `ModelRenderer.LoadTextures` to use per-axis clamp flags (clampS/clampT) based on `tex.Flags & 0x1` and `tex.Flags & 0x2` (lines 778-779)
- **Result**: ❌ FAILED — textures still stretched on cylindrical objects
- **Status**: Root cause still unknown. May not be wrap mode related. Need to check console logs to verify texture flags and investigate UV coordinates.

### AdtModfInjector
- **Problem**: Appends MWMO/MODF chunks to end of file; result is Noggit-incompatible.

## Key Technical Insights

### MCLQ Liquid Heights (Feb 11, 2026)
- MCLQ per-vertex heights (81 entries × 8 bytes) are absolute world Z values
- Heights can slope for waterfalls — adjacent water planes at different Z levels
- MH2O (3.3.5) was overwriting valid MCLQ data with garbage on 0.6.0 ADTs
- Fix: Skip MH2O when MCLQ liquid already found; never overwrite existing MCLQ
- WMO MLIQ liquid type: use `matId & 0x03` from MLIQ header, NOT tile flag bits

### Performance Tuning (Feb 11, 2026)
- AOI: 9×9 tiles (radius 4), forward lookahead 3, GPU uploads 8/frame
- MPQ read throttling: `SemaphoreSlim(4)` prevents I/O saturation
- Persistent tile cache: `TileLoadResult` stays in memory, re-entry is instant
- Dedup sets removed: objects always reload correctly after tile unload/reload

### WMO/MDX Coordinate System (Feb 9, 2026)
- WoW: right-handed (X=North, Y=West, Z=Up), Direct3D CW winding
- OpenGL: CCW winding for front faces
- **Fix**: Reverse winding at GPU upload + 180° Z rotation in placement
- MDX rotations: `rx = Rotation.X`, `ry = Rotation.Y` — NO axis swap
- WMO-only maps: raw WoW world coords (no MapOrigin conversion)

### WMO MLIQ Liquid Positioning (Feb 9, 2026)
- MLIQ data has inherent 90° CW misrotation (wowdev wiki)
- Fix: `axis0 = cornerX - j * tileSize`, `axis1 = cornerY + i * tileSize`
- Tile visibility: bit 3 (0x08) = hidden
- GroupLiquid=15 always → magma (old WMO "green lava" type)

### Replaceable Texture Resolution (Feb 10, 2026)
- Try ALL CDI variants, validate each resolved texture exists in MPQ
- If no DBC variant validates, fall through to model directory scan
