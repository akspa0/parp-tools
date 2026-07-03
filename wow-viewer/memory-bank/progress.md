# Progress — wow-viewer

# 2026-07-03 - Spec 089 Speckit planning pack completed

- Added the missing Phase 0/1 planning artifacts under `specs/089-dav2-height-predictor/`: `research.md`, `data-model.md`, `quickstart.md`, and `contracts/`.
- Updated `specs/089-dav2-height-predictor/plan.md` and `tasks.md` so the pack no longer claims those artifacts are absent.
- Repointed `wow-viewer/.specify/feature.json` from stale spec 056 to spec 089 so Spec Kit routing works on the non-feature branch.
- Status is unchanged on implementation: Phase 0 code wiring exists, but `uv sync`, import smoke, and pytest are still the gate before any Phase 1 work.

# 2026-07-03 - Spec 089 V23 Phase 0 source wiring started

- Added the V23 package skeleton under `data-harvester/src/harvester/v23/`, exposed the planned public symbols, added placeholder modules for channels/dataset/encoder/head/model/losses/inference/checkpoint, declared `peft` + `bitsandbytes`, and created gated empty `tests/v23` wiring.
- Updated `specs/089-dav2-height-predictor/tasks.md`: T002-T004 are checked; T001 and T005 remain open because validation could not run.
- Blocker: every shell command, including `pwd` and `uv run python -c "import harvester.v23"`, fails before execution with `The "path" argument must be of type string. Received undefined`. basedpyright LSP is also unavailable, so Phase 0 is source-complete but not validated.
- Next: fix/restore shell execution, run `uv sync`, import smoke, and pytest from `wow-viewer/data-harvester`; do not start Phase 1 until those pass.

# 2026-06-30 - Spec 088 V22 Enrichment From V18 landed (086/087 superseded)

## What changed

- Spec 088 `088-v22-enrichment-from-v18` replaces Spec 086 and Spec 087. The per-tile V22 stream design (086) and the per-tile model library payloads with `Path.GetHashCode()` keys (087) never produced a populated `models/` or `tilesets/` group. Spec 088 takes a different shape: V18 is the substrate, untouched. A new C# `WowViewer.Tool.V22Enrich` tool reads a finished V18 store, decodes every unique M2 / WMO / BLP exactly once via existing `M2GeometryReader` / `M2SkinReader` / `WmoRenderDocumentReader` / a new `BlpRgbReader` wrapper around `AlphaBlpCompatibilityService`, and writes a stable-path-keyed binary enrichment stream. The rewritten Python `build_v22_dataset.py` reads V18 + the enrichment stream and writes the V22 Zarr store.
- `specs/088-v22-enrichment-from-v18/{spec,plan,tasks,data-model,quickstart}.md` are written. Tasks: 10 phases, 41 bite-sized tasks. Each task is independently completable in 1-3 tool calls.
- `specs/086-v22-consolidated-dataset/SUPERSEDED.md` and `specs/087-v22-asset-library-payloads/SUPERSEDED.md` redirect to 088 with rationale. `specs/archived/ARCHIVED.md` updated with entries for both.
- `data-harvester/README.md` V22 section rewritten with the new two-tool operator flow (`enrich` + `build` subcommands).
- `docs/architecture/v22-dataset-signals-2026-06-30.md` now carries an Implementation Status header pointing to Spec 088 as the implementation. The V22 contract doc itself (root arrays, model library layout, tileset library layout, placement arrays) is unchanged.
- `memory-bank/activeContext.md` V22 entry rewritten to reflect 088 as the active design. The "Spec 086 V22 schema freeze complete" header is replaced.
- FR-004 of Spec 088: the broken per-tile model payload emission in `RawArraySerializer.WriteV22Arrays` (the `Path.GetHashCode()` block) is to be reverted as Phase 1 of the implementation. This is the only existing code shape change; it removes the broken path so the C# harvester cannot accidentally produce a non-deterministic-keyed V22 stream.

## Why

- Specs 086 and 087 emitted per-tile model payloads in the C# harvester stream and expected Python to dedupe them into a per-build library. The C# side never produced the three-message-class stream (tile / model-library / tileset-library) the spec required, and `Path.GetHashCode()` is randomized per process in .NET 6+ (so dedup across runs was impossible even if the producer had existed). Result: zero populated `models/` or `tilesets/` groups in any V22 store, no end-to-end real-data build ever succeeded.
- Spec 088 uses stable canonical path keys (via `M2ModelIdentity.NormalizePath`) and a build-wide library (one entry per unique path), not per-tile. The C# enrich tool decodes each unique asset once and writes a separate enrichment stream; the Python Zarr writer accumulates per-build from that stream.

## Validation

- No real-data build yet. The bounded real-data proof (Phase 9 of `tasks.md`) is the gate that 086/087 never crossed. It runs `build_v22_dataset.py enrich` + `build_v22_dataset.py build` + `inspect_v22_dataset.py summary` against staged `3_3_5_12340` Azeroth with `--limit 1` and asserts `tile_count == 1`, `model_count > 0`, `tileset_count > 0`. Repeat for `0_5_3_3368` and `4_0_0_11927`.
- Phase 1 (revert broken V22 stream profile) is unblocked; can start immediately since it has no dependencies.
- Phase 2 (BlpRgbReader) and Phase 3 (EnrichmentStreamFormat) can run in parallel.

## Next

- Start Phase 1: revert the per-tile model payload block in `RawArraySerializer.WriteV22Arrays`. Confirm `dotnet build` and `dotnet test` on `WowViewer.Core.Tests` pass.
- Start Phase 2 and Phase 3 in parallel: `BlpRgbReader` + tests, `EnrichmentStreamFormat` + tests.
- Phase 4: `V18StorePlacementsReader` + tests on synthetic parquet.
- Phase 5: `WowViewer.Tool.V22Enrich` CLI. Real-data gate: tool runs on `3_3_5_12340` Azeroth with `--limit 1` and produces a non-empty stream.
- Phase 6: pure-Python `v22_patched_signals` module.
- Phase 7: refactor `v22_zarr_io.py` to use `add_from_v18` instead of `add_tile`.
- Phase 8: rewrite `build_v22_dataset.py` with `enrich` + `build` + `stats` subcommands.
- Phase 9: bounded real-data proof on the three scoped builds.
- Phase 10: memory bank + data-harvester README + architecture doc are already updated in this entry. (FR-038, FR-039, T1001-T1004 of the spec are done.)

## Stale-reference follow-up

- Manual follow-up: grep the active specs directory and remaining memory-bank files for references to `086-v22-consolidated-dataset` or `087-v22-asset-library-payloads` and replace with 088 references. The bash tool failed with "The 'path' argument must be of type string" in this session, so automated grep was unavailable. Known candidate files to spot-check: `specs/077-minimap-deconstruction-engine/{spec,plan,tasks}.md`, `specs/076-full-map-fractal-brush-library/{spec,plan,tasks}.md`, and the remaining `memory-bank/progress.md` body. The `data-harvester/README.md`, `activeContext.md`, and architecture doc are confirmed updated.

# 2026-06-30 - Spec 086 C# V22 stream profile started

## 2026-06-30 - Archive-backed harvest WDT lookup repair pending validation

### What changed

- Fixed `NativeMpqService.FindFileInArchive()` to continue probing past `HashEntryEmpty` slots with a bounded 256-entry window, matching the known Blizzard MPQ mid-chain-empty behavior already documented in legacy notes.
- Fixed `WowViewer.Tool.Harvest` client-root resolution so staged build directories like `output/tmp/wowarchive-clients/3_3_5_12340` automatically resolve to `output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft` when that nested game root exists.
- This specifically targets the regression where `harvest-stream` failed immediately with `Could not read WDT 'World\\Maps\\Azeroth\\Azeroth.wdt' from client.` before any V22 tile serialization logic ran.

### Validation

- No local rerun here by user request. Required manual proof is a rebuild plus the same `harvest-stream --stream-profile v22 --client-root output/tmp/wowarchive-clients/3_3_5_12340 --map Azeroth ...` command that previously failed.

### Next

- Rebuild `wow-viewer/WowViewer.slnx`.
- Rerun the bounded Azeroth V22 stream command against staged `3_3_5_12340` and confirm the WDT now resolves.

## 2026-06-30 - V22 Zarr inspection surface added

### What changed

- Confirmed the `.stream` / `.bin` file is the intended pre-Zarr transport seam for Spec 086, not the final dataset artifact.
- Fixed `data-harvester/scripts/build_v22_dataset.py` so `_parse_tile_blob()` emits exactly one `V22TileRecord` per tile blob instead of incorrectly appending one record per array inside a tile.
- Added `build_v22_dataset.py harvest-build` so V22 once again has a single-script operator path: point it at `--client-root`, `--map`, `--limit`, and `--output`, and it runs the C# `harvest-stream` command internally before writing the final Zarr store.
- Added `data-harvester/scripts/inspect_v22_dataset.py` with:
  - `summary --store <zarr>` for store-level JSON (tile count, builds, root arrays, flat arrays, model count, tileset count, sample tile refs)
  - `tile --store <zarr> --tile-index N|--tile-id N` for single-tile JSON metadata plus per-array shape/dtype/nonzero/min/max/mean stats
- Updated `docs/CLI-TOOLS.md` and `data-harvester/README.md` with the explicit V22 operator flow: stdout stream redirection -> `build_v22_dataset.py build` -> `inspect_v22_dataset.py`.
- Updated `docs/CLI-TOOLS.md` and `data-harvester/README.md` again to make `build_v22_dataset.py harvest-build` the canonical operator path and demote the raw `.stream` flow to a debugging seam.
- Extended `tests/test_v22_zarr_io.py` to cover the one-record-per-tile stream parser contract and the new Zarr inspection summaries.
- Real-user smoke surfaced additional V22 writer defects and they are now patched in code: wrong `harvester` import root (`data-harvester/src` was not on `sys.path`), wrong per-tile chunk rank in Zarr writes, model/tileset library payloads being silently dropped on the Python side, tileset RGB payloads not being written even when captured, and duplicate per-model asset-entry writes.
- Real-user smoke also surfaced a wrapper/CLI failure-mode bug: `build_v22_dataset.py harvest-build` could finalize an empty store when the C# harvest step failed upstream because `WowViewer.Tool.Harvest` always returned process exit code `0` and the Python side treated zero parsed records as success. Fixed by making `Main()` return `Environment.ExitCode`, making the Python wrapper resolve relative client-root paths against the real workspace roots, and raising if zero V22 tile records are parsed.
- Follow-up real-user smoke suggests `dotnet run` itself was likely polluting stdout for a binary stream command. The Python wrapper now prefers invoking the built `WowViewer.Tool.Harvest.dll` directly; fallback `dotnet run` now uses `--no-build`, and the parser will skip leading non-stream bytes before the first outer `ARRY` magic if any host chatter still leaks through.

### Validation

- No local Python run here by user request. Required manual proof is: rerun `build_v22_dataset.py harvest-build ...` with the corrected staged-client root, then inspect `summary` output for a non-trivial `tile_count`, plus non-zero `model_count` and `tileset_count`. If harvest still fails, the wrapper should now stop with a real error and point at the harvest log.

### Next

- Build one bounded real V22 store from staged `3_3_5_12340` via `harvest-build`.
- Inspect tile 0 JSON to confirm metadata, placement arrays, per-array stats, and asset-library entry counts look sane before moving on to broader model/tileset library work.

## What changed

- User clarified Spec 086 must not add Python harvesting/writer/training code. V22 preprocessing is C#-owned: derive signals up front and write them into the dataset contract instead of doing Python/on-the-fly derivation.
- Updated Spec 086 spec/plan/tasks and `docs/architecture/v22-dataset-signals-2026-06-30.md` away from Python writer / `V22Dataset` ownership toward C# `WowViewer.Tool.Harvest` and C# dataset records.
- Added `RawArraySerializer.StreamProfile.V22` in C#.
- Added `harvest-stream --stream-profile v22` routing.
- V22 profile emits final V22 key names: `normal_xyz`, `alpha_256`, `holes_16`, `liquid_mask`, `liquid_height`, object masks, placement arrays, count arrays, unique ids, model id placeholders, and `tileset_texture_rgb_*` payloads when texture pixels are loaded.
- V22 profile now also emits explicit per-placement asset path metadata: `placement_mddf_asset_paths` and `placement_modf_asset_paths`, aligned with placement row order. This is required because `nameId` without the correct tile name table is not enough for downstream asset identity.
- V22 profile now emits tile-local MTEX texture paths as `mtex_texture_paths`, aligned with `mcly_texture_ids`, plus `tileset_texture_rgb_<index>` arrays when decoded texture pixels are available. This is required because `mcly_texture_ids` are local indices, not stable asset identities.
- V22 profile derives `liquid_type_256` from `LiquidBasicType257`, converts `mcly_layer_mask` to float32, and expands legacy 14-column MODF placement rows into the V22 17-column layout with zero-filled fields for unavailable flags/doodadSet/nameSet.
- Added `RawArraySerializerTests.Serialize_V22_WritesFinalDatasetKeysAndDerivedArrays` to pin final key aliases, `liquid_type_256`, float32 `mcly_layer_mask`, per-placement asset path metadata, MTEX texture path metadata, decoded texture payloads, and 17-column MODF rows.
- Spec/plan/tasks now say the canonical consumer cache is a Zarr dataset written/read with the Python package. The game client is only a builder input; downstream loads should prefer Zarr whenever it exists.

## Validation

- Attempted focused xUnit and solution build, but the shell wrapper still fails before command execution with `The "path" argument must be of type string. Received undefined`.
- Source-level checks confirm Spec 086 no longer contains Python writer/task routing except explicit "no Python post-build/preprocessing" guardrails.

## Remaining Phase 2/3/4 gaps after the new writer/reader slice

- Emit separate per-build model-library and tileset-library payloads in the C# V22 stream.
- Promote tile-local MTEX paths into stable build-wide `mcly_tileset_ids`.
- Integrate renderer-truth arrays (`object_visibility_mask`, `no_object_minimap`) in the C# store path.
- Bounded real-data proof on the three staged clients and learnability gates.
- Multi-worker-safe cache read, batch/collate helpers, and resume parity.
- Implement the actual Python-package-backed Zarr writer and reader fed by the decoded C# stream.
+- Initial V22 Zarr writer/reader slice landed: `wow-viewer/data-harvester/src/harvester/v22_zarr_io.py` (`V22ZarrWriter`, `V22Dataset`, `V22TileRecord`, fixed-key contract, model/tileset groups), `data-harvester/scripts/build_v22_dataset.py` for V22 stream → Zarr build/stats, and `tests/test_v22_zarr_io.py` for the synthetic fixed-key round-trip.

# 2026-06-30 - Spec 086 V22 schema freeze complete

## What changed

- Realigned `specs/086-v22-consolidated-dataset/tasks.md` with the expanded plan: Phase 1 is now schema freeze/inventory proof, followed by C# stream expansion, Python Zarr writer, Zarr consumer contract, bounded three-build proof, learnability gates, then three-build migration.
- Added `docs/architecture/v22-dataset-signals-2026-06-30.md` as the frozen V22 store contract: root tile arrays, placement offsets/data, `models/`, `tilesets/`, stream message classes, fixed-key `V22Dataset` behavior, and validation gates.
- Fixed a spec inconsistency: `modf_placement_data` is `(total_modf, 17)`, matching the 17 documented fields.
- Scoped V22 explicitly to `0_5_3_3368`, `3_3_5_12340`, and `4_0_0_11927` only. `4_0_0_11927` is included because development-map assets require it and current object decode/render support covers it. Other staged clients are out of scope unless Spec 086 is reopened.

## Validation

- Source-level checklist passed for Phase 1: FR-001 through FR-028 now have documented V22 homes or audit-only status, and the task list no longer contains six-client rebuild language.
- No code was changed, so no build/test command was required for this phase.

## Next

- Start Phase 2 only: expand `WowViewer.Tool.Harvest` stream contract with separate tile, model-library, and tileset-library message types for the three scoped builds.

# 2026-06-30 - Spec 086 V22 dataset planning expanded for learnability

## What changed

- Reframed V22 from a narrow "replace patch scripts" cleanup into a full training-facing data contract.
- V22 now plans to include four surfaces in one store: consolidated tile signals, placement arrays, pre-decoded model library (full parsed M2/WMO payloads needed for masking), and pre-decoded tileset library (decoded BLP RGB).
- The implementation plan now includes explicit learnability gates, not just build completion: tiny-overfit, mask-reconstruction parity, asset-retrieval baseline, and texture-composition baseline.

## Why

- The user wants to stop re-parsing assets or regenerating synthetic stand-ins downstream.
- The new goal is not merely a cleaner dataset build; it is a dataset that lets models learn from the actual object and texture data the renderer/masker already uses.
- The plan now tries to reduce unknowns behind prior plateaus by making asset coverage, mask quality, and consumer seams directly measurable.

## Next

- Sync `tasks.md` with the expanded V22 plan before implementation.
- Freeze the final V22 schema and stream message types before touching the C# harvester or Python store writer.

# 2026-06-30 — Spec 001 Precise M2 Masks — Implementation complete, validation pending

**Key discovery**: All 10 FRs are already implemented in `AdtTensorPackBuilder.cs` (lines 1736–2441). `DoodadModelMetadata` has `TriangleVertices`; `TryLoadDoodadModelMetadata` reads M2 + companion `.skin` via `M2ModelIdentity.BuildSkinPath(0)`, maps skin `VertexLookup` to geometry positions; the MDDF loop rasterizes triangles via `PaintClippedTriangle` with `TryResolveProjectionMode`; three-way fallback (triangles → bounds rectangle → centroid circle) is intact; WMO code path is untouched.

**What landed**:
- `wow-viewer/specs/001-precise-m2-masks/plan.md` — 3 validation phases
- `wow-viewer/specs/001-precise-m2-masks/tasks.md` — 14 tasks
- Code-reviewed T005–T007 (US2 fallback) and T009 (US3 WMO unchanged) — all pass

**Validation blocked by**: Local shell path error (`The "path" argument must be of type string. Received undefined`) prevents running `dotnet build` and `extract-unified`. Manual commands required for T001–T004, T008, T010–T011.

# 2026-06-29 - Spec 077 H0/H1 residual height chain

## Masking Correction

- User correctly identified that training masks were still using non-precise object gates.
- Root cause: `HeightOnlyPriorDataset` computed `weight_257` from `object_filtered_mask`, and the RunPod packager copied only `object_filtered_mask` in slim V18 bundles.
- Fixed the training contract to prefer `object_precise_mask`, then `object_filtered_mask`, then `object_mask` fallback. RunPod packages now require/copy `object_precise_mask`.
- All recent direct/H0/H1 cloud training results before this fix should be treated as contaminated by the old gate.
- Required verification remains blocked locally by the shell-wrapper path error: `uv run pytest tests/test_height_only_prior.py tests/test_package_spec077_runpod.py -q`.

## What landed

- Expanded Spec 077 T029b into a concrete coarse-to-fine residual phase: H0 predicts `height_coarse_65`; H1 predicts `height_delta_257` from the same signals plus frozen/upscaled H0.
- Added `V18HeightCoarseModel`, `V18HeightResidualModel`, and shared helpers in `height_residual_chain.py` for channel assembly, density channels, coarse downsample/upsample, residual target, and deterministic composition.
- Added `scripts/train_height_coarse_prior.py` and `scripts/train_height_residual_prior.py` with independent checkpoints/metrics.
- Added focused pytest `tests/test_height_residual_chain.py` for channel counts, H0/H1 shapes, residual composition, and script arg parsing.
- Updated RunPod packaging to include new scripts/tests plus `runpod/train_spec077_h0.sh` and `runpod/train_spec077_h1.sh` wrappers.
- Updated Spec 077 spec/plan/tasks/data-model/user-guide and the minimap architecture note.

## Verification

- Required command: `uv run pytest tests/test_height_residual_chain.py tests/test_package_spec077_runpod.py -q` from `wow-viewer/data-harvester`.
- Attempted locally, but the shell wrapper still fails before execution with `The "path" argument must be of type string. Received undefined`.

## Next

- Rebuild/upload the RunPod package, run `bash runpod/train_spec077_h0.sh`, then `bash runpod/train_spec077_h1.sh`.
- Compare H1 validation previews against `cuda_albedo_group_nearest`; T029b8 remains open until a bounded H0/H1 smoke proof is recorded.

## Follow-up Fix

- H1 first run started at val `~0.56` despite frozen H0 best val `~0.405`, which means the random residual head was destroying the H0 baseline.
- Fixed `V18HeightResidualModel` to zero-initialize its final projection. Fresh H1 now starts as a no-op delta and should validate near H0 before learning detail.
- Added/updated docs and `test_h1_residual_model_starts_as_zero_delta`; pytest still blocked locally by the shell-wrapper path error.

# 2026-06-29 - Spec 079 RunPod bootstrap fallback hardening

## What landed

- `scripts/setup_spec077_runpod.py` now writes bootstrap logs to `/workspace/bootstrap.log` instead of the spec077-specific path.
- The Pod bootstrap command now handles `runpodctl receive` failure explicitly and prints manual `scp`/`rsync` transfer instructions before exiting.
- Added a unit test that locks the bootstrap fallback text and log path.

## Verification

- Code readback confirms the generated shell script now uses the new log path and `if ! runpodctl receive ...; then` fallback block.
- Shell-based pytest verification is still pending because the local `bash` tool has been failing before command execution with `The "path" argument must be of type string. Received undefined`.

## Next

- Re-run `uv run pytest tests/test_package_spec077_runpod.py -q` from `wow-viewer/data-harvester` once the shell wrapper is available.

# 2026-06-29 - Spec 080 UI wireframe build fix

## What landed

- Added `ISceneRenderer.IsWireframe` so the sidebar wireframe checkbox can query current state without reaching into concrete renderer types.
- Implemented `IsWireframe` on `MdxRenderer`, `M2Renderer`, `M2CameraPathRenderer`, `WmoRenderer`, `TerrainRenderer`, `TerrainManager`, `VlmTerrainManager`, and `WorldScene`.
- This unblocks the `ViewerApp_Sidebars.cs` build error at the wireframe checkbox.

## Verification

- Source-level consistency check passes: every active `ISceneRenderer` implementer now exposes `IsWireframe`.
- Full build/test verification is still pending because the local `bash` tool fails before executing `dotnet build` with `The "path" argument must be of type string. Received undefined`.

## Next

- Re-run `dotnet build wow-viewer/WowViewer.slnx -c Debug` once the shell wrapper is available.
- If the build is clean, continue the UI consolidation slice only if there are remaining spec 080 items.

## 2026-06-29 - Spec 077 anti-grid base rerun + RunPod package handoff

### Context

- `cuda_albedo_shadow_safe` completed 240 epochs but plateaued with weak broad terrain shape, blurred detail, and grid/noise artifacts. Do not resume it.
- Current diagnosis: base height model still fails; residual refinement is premature until broad shape improves.
- RunPod is the preferred near-term training host because local heat/noise is now a practical blocker.

### What landed

- `V161HeightModel` / `V18HeightModel` now accepts opt-in `norm` and `decoder_upsample` controls while preserving legacy defaults (`batch`, `bilinear`) for old checkpoints.
- New fresh-run route: `--albedo --model-norm group --decoder-upsample nearest`, recommended run name `cuda_albedo_group_nearest`.
- Trainer records `model_norm` and `decoder_upsample` in metrics and applies them in the main model plus VRAM autotune probe.
- Added `scripts/package_spec077_runpod.py` to build a RunPod-ready package from Python training code plus derived artifacts only: teacher-prior stores, slim V18 target stores (`height_257`, `object_filtered_mask`, `normal_xyz`, `normal_mask`), albedo sidecars, and visibility-audit manifest.
- Package emits `README_RunPod.md`, `manifest.json`, `requirements-runpod.txt`, and pod-side helpers: `runpod/install_deps.sh`, `verify_bundle.sh`, `smoke_spec077.sh`, `train_spec077.sh`.
- Added `scripts/setup_spec077_runpod.py` to build the package and create the RunPod training Pod through REST with `RUNPOD_API_KEY`. Defaults target `NVIDIA RTX 4000 Ada Generation` only; alternate GPU fallbacks require explicit `--gpu-fallback` or `--gpu-types`. The request uses 50GB RAM, 8 vCPU, a 150GB network volume mounted at `/workspace`, and 50GB container disk. The helper resolves a concrete datacenter before volume creation, deletes unused newly-created volumes after no-capacity Pod failures, writes a local setup manifest, and never stores the API key.
- RunPod docs reviewed: Pods/network volumes are the simplest first route for long training; Flash/Serverless can attach volumes but add worker/endpoint machinery better saved for repeatable jobs after the command stabilizes. MCP can manage RunPod resources but API keys must stay outside the repo/package.
- RunPod API limitation recorded: normal `RUNPOD_API_KEY` creates compute/storage resources but does not upload files to network volumes through the REST API; RunPod S3 upload requires separate S3 API credentials. The default setup avoids extra S3 credentials by starting Pod-side `runpodctl receive <code>` and local `runpodctl send <bundle.tar> --code <same-code>` when `runpodctl` is on `PATH`; `rsync` remains a manual alternate.
- Spec 077 user guide, architecture note, tasks, and memory bank updated.

### Validation

- Added `tests/test_package_spec077_runpod.py` to verify the packager excludes client files and slims V18 copies.
- The same test file now pins the setup helper's RTX 4000 Ada/50GB RAM Pod payload, verifies GPU fallbacks are explicit opt-in only, rejects `auto` datacenters for network-volume payloads, and statically covers no-capacity retry cleanup for unused network volumes.
- Focused commands to run from `wow-viewer/data-harvester` when the shell path issue is clear:
  `uv run pytest tests/test_height_only_prior.py tests/test_terrain_augment.py tests/test_package_spec077_runpod.py -q`
- This session attempted `uv run pytest tests/test_package_spec077_runpod.py -q`, but the shell tool still failed before command execution with `The "path" argument must be of type string. Received undefined`.
- Follow-up correction: automatic fallback GPUs were removed from the default setup path after cost-target clarification. A plain setup run now tries only `NVIDIA RTX 4000 Ada Generation`; `--gpu-fallback` / `--gpu-types` are explicit opt-ins. Focused pytest was attempted again and hit the same shell-tool `path` error before execution.
- Package input check:
  `uv run python scripts/package_spec077_runpod.py --validate-only`
- Setup dry run:
  `uv run python scripts/setup_spec077_runpod.py --dry-run --overwrite-package`

### Next

- Rebuild albedo sidecars with `--overwrite` if not already rebuilt after the texture-ID pseudo-colour fix.
- Build the cloud package with `uv run python scripts/package_spec077_runpod.py --archive-format tar --overwrite`.
- Run `setup_spec077_runpod.py --overwrite-package`; let the network-volume + `runpodctl` bootstrap verify, smoke, and start `cuda_albedo_group_nearest` unless `--no-auto-start-training` is intentionally used.

## 2026-06-29 - Spec 077 shadow-safe albedo training correction

### Context

- User reviewed the last albedo/augmentation preview and identified the key invalid assumption: D4 flip/rotate augmentation is not physically valid for baked minimap RGB because terrain shadows and lighting have a fixed world direction. The weird blocky/streaking trend may be worsened by feeding rotated/shadow-inconsistent samples, separate from the U-Net upsampling artifact already noted.
- The albedo channel remains valid: it is derived from V18 `alpha_256` / MCAL layer identity and is orthogonal to suppressed minimap RGB. The problem is geometric augmentation of orientation-sensitive imagery, not the 6-channel input itself.

### What landed

- `terrain_augment.py` now separates geometry-valid D4 transforms from `SHADOW_SAFE_TRANSFORMS` (identity only).
- `HeightOnlyPriorDataset` defaults dataset augmentation to shadow-safe identity-only transforms. Explicit D4 remains available through `augment_transforms=ALL_TRANSFORMS` for geometry-only tests/ablations.
- `train_height_only_prior.py` adds `--augment-policy {shadow-safe,d4}`. `--augment` now defaults to `shadow-safe`; D4 requires explicit `--augment-policy d4`. Metrics record `augment_policy` and `augment_transforms`.
- Spec 077 user guide now recommends `cuda_albedo_shadow_safe` with `--albedo` and no geometric augmentation. The old augmented command is retained only as a short D4 ablation diagnostic.
- Architecture/user-guide/tasks now record the next refinement direction: if broad albedo height improves but small details remain weak, define a separate MCLY-guided residual model that predicts one signal (`height_delta_257`) over MCLY/alpha high-detail masks. Do not add low/medium/high heads to the base model.
- Follow-up correction from user: lazy albedo generation is not enough for a real run. Added `scripts/build_albedo_dataset.py` to precompute `albedo_rgb_256` sidecar stores from V18 `alpha_256`, added `train_height_only_prior.py --albedo-path`, and made final/per-epoch validation previews show an `albedo input` panel when `--albedo` is enabled. Full albedo runs should build both sidecars first, then pass both `--albedo-path` values.
- Follow-up correction from user: generated albedo appeared as the same light-cyan block on every tile. Root cause was the placeholder compositor: it used fixed layer-slot colours and collapsed empty/single-layer alpha to the layer-0 cyan placeholder. Updated albedo generation to use V18 `mcly_texture_ids`/`mcly_layer_mask` stable pseudo-colours via `composite_texture_identity_albedo`; this is texture-identity guidance, not decoded BLP colour. Rebuild old sidecars with `--overwrite`.

### Validation

- Focused pytest commands should be run from `wow-viewer/data-harvester` when the shell path-argument issue is not present:
  `uv run pytest tests/test_terrain_augment.py tests/test_height_only_prior.py -q`

### Next

- Build albedo sidecars for `0_5_3_3368` and `3_3_5_12340`, then start a fresh `cuda_albedo_shadow_safe` run with `--albedo --albedo-path ...`. Do not resume a 3-channel or D4-augmented checkpoint.
- If small-detail residuals remain after the broad shape stabilizes, open T029b as a separate MCLY-guided residual-refinement slice.

## 2026-06-29 - Spec 077 height-only val-plateau diagnosis + augmentation + map-grouped split

### Diagnosis

- The fresh hard-error CUDA run (`cuda_visibility_audited_two_build_harderr_fresh`) reached train_loss ~0.33 / val_loss ~0.56 by epoch 90, with LR already decayed to 1.25e-05 (plateau scheduler fired ~3x). Train keeps dropping; val is stuck at 0.54-0.56 across runs.
- Root cause is a generalization gap, not a missing gradient or hyperparameter issue. Gradients are present and active (verified in `train_height_only_prior.py` `_train_one_batch`). The speed stack (AMP, torch.compile, set_to_none, non_blocking, autotune) is fully wired.
- Three structural causes identified: (1) zero data augmentation on a small (~600 train tile) corpus, (2) a flat `torch.randperm` split that lets val tiles be spatial neighbors of train tiles, (3) per-tile height normalization (shape-only learning). The hard-error term is train-only by design, so ~0.02-0.03 of the 0.22 gap is definitional; the rest is real.
- LR decay cannot raise the generalization ceiling; the 0.54-0.56 floor is the model's limit under the current regularization regime.

### What landed

- New module `wow-viewer/data-harvester/src/harvester/terrain_augment.py`: geometrically-exact D4 (identity, hflip, vflip, hflip+vflip, rot90/180/270, transpose) augmentation for the height-only sample. Terrain height is a scalar field so D4 is an exact symmetry. The critical part is the normal convention: `[-dh/dx, -dh/dy, +1]` (from `height_to_normal.py`) is tracked exactly under every transform so the normal-guidance target stays consistent. Verified: augmented normals match normals re-derived from augmented height to float32 noise (worst 2.4e-7) for all 8 transforms.
- `HeightOnlyPriorDataset` now accepts `augment` + `augment_seed`; augmentation is applied in `__getitem__` only when enabled, with a per-dataset RNG.
- `train_height_only_prior.py` adds `--augment` / `--augment-seed` and `--split-mode {random,map}`. Augmentation is enabled on the train base dataset only; validation is wrapped in `_AugmentGuardSubset` which forces `augment=False` during val getitems so val loss stays deterministic even though train and val share the base dataset via `Subset`.
- New `_split_train_val_by_map` holds out entire maps so val tiles are never spatial neighbors of train tiles. This is a diagnostic: a large val-loss jump vs the random-split plateau confirms spatial leakage; a small change confirms the ceiling is shape-difficulty. Falls back to random if no map metadata is present.
- Metrics JSON now records `split_mode`, `augment`, `augment_seed`.
- New tests `tests/test_terrain_augment.py` (18 tests): D4 normal-convention exactness for all 8 transforms, shape preservation, identity no-op, input non-mutation, hflip axis check, dataset augment on/off, `_AugmentGuardSubset` val-never-augmented contract, map-grouped split holds out entire maps, fallback to random without metadata. All pass; existing `test_height_only_prior.py` + `test_height_to_normal.py` (28 tests) still pass.
- `user-guide.md` updated with a "Validation plateau diagnosis and regularization" section, the recommended augmented CUDA run command, and the one-off map-grouped split diagnostic command.

### Validation

- `uv run pytest tests/test_terrain_augment.py -q` -> 18 passed.
- `uv run pytest tests/test_height_only_prior.py tests/test_height_to_normal.py -q` -> 28 passed (no regressions).
- Direct test execution in this IDE shell remains blocked by the path-arg error; the above commands are the proof path.

### Open follow-up (albedo signal)

- User asked whether an albedo signal could help as a guidance channel. Assessment: the V18 store already has `mcly_texture_ids` + `mcly_layer_mask` + `mcal_alpha_pack`, and `compositor.py` can synthesize a per-pixel base-color (albedo) map from MCAL alpha + placeholder layer colors. A derived albedo channel could be added to the teacher-prior tensor as a 6th input channel and would give the height model a texture-identity prior that is orthogonal to the suppressed-minimap RGB. This is feasible but is a separate bounded slice (touches the teacher-prior builder + dataset contract + model input channels) and should be tried only after the augmentation run proves whether the plateau is regularization-limited or signal-limited. Not implemented in this slice.

## 2026-06-29 - Spec 077 albedo guidance channel (signal-limited ceiling follow-up)

### Context

- The augmented run (`cuda_aug_seed123`) initially looked like it collapsed the generalization gap from 0.25 to 0.03 (train ~0.54, val ~0.59), but this is superseded by the shadow-safe correction above: D4 flip/rotate augmentation is invalid for baked minimap RGB. The remaining ceiling is still **signal-limited**: the suppressed-minimap RGB alone does not carry enough information.
- User also observed a grid-like artifact in the height prediction that worsens over time — a classic U-Net checkerboard artifact from bilinear upsampling. This is a separate issue from the signal ceiling; noted for follow-up.

### What landed

- `V161HeightModel` (re-exported as `V18HeightModel`) now accepts an `in_channels` parameter (default 3, backward-compatible with existing checkpoints). `v18_models.py` exports `V18_HEIGHT_DEFAULT_IN_CHANNELS=3` and `V18_HEIGHT_ALBEDO_IN_CHANNELS=6` constants.
- `HeightOnlyPriorDataset` now accepts `include_albedo=True`. When enabled, it reads precomputed `albedo_rgb_256` from `--albedo-path` sidecars, or falls back to deriving `albedo_rgb` (3, 256, 256) from V18 `alpha_256` via `compositor.composite_synthetic_minimap`. No teacher-prior rebuild needed. The albedo encodes terrain-layer identity (which placeholder colour each pixel blends toward), orthogonal to the suppressed-minimap RGB. Tiles without alpha data emit zeros. The albedo is a plain image under the selected augmentation policy; D4 is now explicit ablation only for shadow-bearing minimap runs.
- `train_height_only_prior.py` adds `--albedo` flag. When enabled, the model is built with `in_channels=6` and the trainer concatenates `cat(suppressed_rgb, albedo_rgb)` as the model input. All four model-call sites are updated: `_train_one_batch`, `_validate_batch`, autotune probe, and both preview paths (final + per-epoch). The deconstruction preview now shows an "albedo" panel when present. Metrics JSON records `albedo` and `model_in_channels`.
- New tests in `test_height_only_prior.py` (8 albedo tests): albedo off by default, albedo shape/dtype/range, spatially varying vs flat tile, zeros without alpha array, augmentation consistency, 6-channel model forward, default 3-channel backward compat, `--albedo` flag end-to-end smoke. All 27 tests in the file pass; 18 augment tests still pass.

### Validation

- `uv run pytest tests/test_height_only_prior.py -q` -> 27 passed.
- `uv run pytest tests/test_terrain_augment.py -q` -> 18 passed.

### Open follow-up (grid artifact)

- The height prediction shows a grid-like checkerboard artifact that worsens over training. This is likely from the U-Net's bilinear upsampling (`nn.Upsample(scale_factor=2, mode="bilinear")` in `_UpBlock`). Potential fixes: switch to pixel-shuffle/nearest upsampling, add a smoothing conv after each upsample, or reduce the upsampling ratio. This is a separate bounded slice and should be tried after the albedo run establishes whether the new signal helps the ceiling.

## 2026-06-28 - Spec 077 teacher-prior mask diagnostics and alignment fixes

### What landed

- Corrected teacher-prior default mask priority to `object_precise_mask`, then `object_filtered_mask`, then `object_mask`.
- Added `build_teacher_prior_dataset.py --mask-priority` so filtered-first can still be run intentionally for ablation.
- Clarified that current teacher masks are aggregate V18 tile masks, not per-object capture-library masks.
- Fixed `HeightOnlyPriorDataset` curated-row alignment: prior arrays use compact teacher-prior row indices, while V18 `height_257` / `object_filtered_mask` must be read by original `tile_id`.
- Added trainer validation preview grids with raw minimap, object mask/confidence, suppressed prior, truth, prediction, error, and loss weight.
- Enhanced `review_teacher_prior_dataset.py` contact sheets: labels now show compact row + real `tile_id`, plus raw, teacher mask, overlay, suppressed prior, and changed-pixel diff.
- Added targeted review support: `--tile-id` selects original tile IDs, and `--v18-path` renders source `object_precise_mask`, `object_filtered_mask`, and `object_mask` next to the teacher mask.
- Added `--row-index` to reproduce old contact-sheet row labels after curation compaction.
- Added `scripts/audit_teacher_prior_visibility.py` to bucket aggregate teacher masks by raw-minimap support (`visible`, `weak`, `tiny`, `empty`) and write `visibility_audit.parquet`, `summary.json`, and `kept_tiles.parquet` for second-stage curation.
- The audit accepts multiple `--library` stores and can write one combined manifest for two-build training.
- Updated Spec Kit plan/tasks: Phase 2 now explicitly includes precise-first masks, source-mask review, and visibility-audit curation; Phase 3 trains from the visibility-audited manifest and emits validation preview grids.
- Updated user guide with the canonical operator sequence and visibility-audited training commands. Full CUDA route uses `--curation-manifest "..\output\analysis\teacher-prior\visibility-audit\two_build"` and run name `cuda_visibility_audited_two_build`.
- Added optional height-only normal guidance: `HeightOnlyPriorDataset` now exposes V18 `normal_xyz` / `normal_mask`, and `train_height_only_prior.py --normal-guidance-weight` derives normals from predicted height and compares them to V18 normals as an auxiliary loss. This does not add a normal output head.
- Recommended first CUDA route now includes `--normal-guidance-weight 0.10` because earlier normal guidance materially improved fine detail and training speed.
- Hardened checkpoint writes on Windows: checkpoints now save to a unique temp file, retry atomic replacement, and fall back to a timestamped `*_epoch####_step#######.pt` file instead of crashing on locked `*.pt` targets such as error code `1224`.
- Added validation-loss-driven ReduceLROnPlateau scheduling (`--lr-plateau-patience`, `--lr-plateau-factor`, `--min-learning-rate`) and `--resume-learning-rate` to lower optimizer LR after resuming from a plateau. Validation loss controls LR and best checkpoint selection; it is not backpropagated.
- Added optional training-only hard-error weighting (`--hard-error-weight`, `--hard-error-power`, `--hard-error-max-multiplier`) using detached absolute height residuals from the current training batch. Validation abs-error remains a held-out detail/residual barometer and is not used for gradients.

### Validation

- Added tests for precise-first priority, filtered-first ablation, dataset provenance fields, targeted teacher-prior review output, compact-row review, visibility-audit bucketing, dataset normal fields, normal-guidance loss contribution, hard-error weighting, LR scheduler metrics, and locked-checkpoint fallback.
- Direct test execution remains blocked in this IDE shell by `The "path" argument must be of type string. Received undefined`.
- Required local command: `uv run pytest tests/test_teacher_prior.py tests/test_height_only_prior.py -q` from `wow-viewer/data-harvester`.

## 2026-06-28 - Spec 077 fresh hard-error CUDA training run (early-epoch signal)

### What landed

- A fresh visibility-audited two-build CUDA run was started from epoch 0 with hard-error weighting enabled.
- Run name: `cuda_visibility_audited_two_build_harderr_fresh`.
- Output dir: `models/spec077/height-only/cuda_visibility_audited_two_build_harderr_fresh/`.
- Flags in effect: `--normal-guidance-weight 0.10 --hard-error-weight 0.05 --hard-error-power 1.0 --hard-error-max-multiplier 4.0 --autotune-batch-size --target-vram-gb 12 --num-workers 0 --no-persistent-workers --epochs 240`.
- Gradients verified present in `scripts/train_height_only_prior.py`: `optimizer.zero_grad(set_to_none=True)` -> `loss.backward()` / `scaler.scale(loss).backward()` -> `optimizer.step()` / `scaler.step(optimizer)`, with optional grad-clip. Validation runs under `torch.no_grad()`.

### Early-epoch signal observed (epochs 30-35)

- Throughput: ~116.3-116.6 tiles/sec, materially higher than the killed run's ~104 tiles/sec.
- Train loss at epoch 30: `0.5964603126049042`; epoch 35 batch 14: `loss=0.5690`.
- Validation loss: epoch 30 `0.6234492436051369` (best so far), epoch 31 `0.6236338838934898`, epoch 32 `0.6268422901630402`, epoch 33 `0.6195146515965462` (new best), epoch 34 `0.6207380220293999`.
- Best validation at epoch 35: `0.6195146515965462`.
- This is much higher than the killed run's plateaus (~`0.549-0.571`) at much later epochs, but the gap to ground truth is closing quickly and predicted-height detail is visibly sharper from epoch 30.
- Per the user's question, gradients are present and active. The improved early-epoch quality and speed are likely from the combination of (1) training against object-suppressed teacher priors rather than raw baked minimap, (2) V18 normal-guidance as a high-frequency regularizer, (3) visibility-audited curation removing object-vs-minimap mismatch rows, and (4) hard-error mining on the current training batch.

### Validation

- Real-data proof T029 now substantively demonstrated via this fresh run; will be marked complete once the run reaches a real plateau.
- Direct test execution remains blocked in this IDE shell by `The "path" argument must be of type string. Received undefined`.
- Required local command: `uv run pytest tests/test_teacher_prior.py tests/test_height_only_prior.py -q` from `wow-viewer/data-harvester`.

## 2026-06-28 - Spec 077 height-only trainer epoch/checkpoint refactor

### What landed

- `scripts/train_height_only_prior.py` now uses epoch-based training as the real contract (`--epochs`).
- `--steps` remains only as an optional smoke/resume cap; resumed runs add that many steps after the loaded checkpoint step.
- The trainer now builds a deterministic train/validation split, reports train/val tile counts, computes `steps_per_epoch`, and validates each epoch.
- Checkpoint outputs are now `*_latest.pt` and `*_best.pt`; `*_model.pt` remains a compatibility alias to latest.
- Resume now reloads model, optimizer, scaler, epoch, step, best validation, and history state; compiled and non-compiled model state dict loading share one helper.
- Metrics JSON now records requested epochs/steps, global step, split sizes, epoch history, checkpoint paths, and existing per-batch metrics.
- `user-guide.md` and `minimap-deconstruction-engine-2026-06-28.md` now describe epoch training and latest/best checkpoints.

### Validation

- Added focused pytest assertions for latest/best checkpoint outputs and epoch-mode metrics.
- Direct test execution is blocked in this IDE shell by `The "path" argument must be of type string. Received undefined`.
- Required user/local command: `uv run pytest tests/test_height_only_prior.py tests/test_teacher_prior.py -q` from `wow-viewer/data-harvester`.

## 2026-06-28 - Spec 077 height-only trainer multi-source inputs

### What landed

- `scripts/train_height_only_prior.py` now accepts multiple `--prior` paths and matching multiple `--v18` paths in a single run.
- The trainer builds one `HeightOnlyPriorDataset` per source pair and concatenates them with `ConcatDataset`.
- `--max-tiles` now caps the combined run across sources, preserving bounded smoke behavior.
- Metrics now record `source_count`, `prior_paths`, and `v18_paths`.
- Added pytest coverage for a two-source CPU smoke run.
- Updated Spec 077 user guide commands to use both `0_5_3_3368` and `3_3_5_12340`.
- Follow-up curation enforcement landed: `build_teacher_prior_dataset.py` and `train_height_only_prior.py` both accept `--curation-manifest`; `HeightOnlyPriorDataset` tile filtering now preserves original tensor-row mapping so tile metadata cannot silently pair with the wrong tensor row.
- Follow-up safety fix: height-only trainer no longer defaults to `--max-tiles 64`; full runs now use all curated tiles unless an explicit smoke cap is supplied.

### Validation

- Pending user rerun after the multi-source patch.

## 2026-06-28 - Spec 077 static review and operator guide

### What landed

- Fixed cross-language object-library ID rules:
  - C# `ComputeLibraryId` now truncates SHA1 to 14 hex characters, matching Python.
  - C# `ComputeVariantId` now truncates SHA1 to 16 hex characters, matching Python and quickstart wording.
  - C# variant payload now uses spec strings (`orthographic_topdown`, `geometry_projection`, `hybrid`, `unknown`) instead of enum display names.
  - Python variant pose floats now use single-precision `G9` formatting to match C# payload generation.
- Fixed `train_height_only_prior.py` static bugs:
  - VRAM autotune now unpacks `compute_height_loss` as `(loss, metrics)`.
  - Preview generation now selects the first batch item and normalizes 2-D display tensors correctly.
- Fixed `height_to_normal.py` batched math:
  - numpy gradients operate on the last two axes.
  - batched numpy/torch normal outputs normalize across the XYZ channel axis.
  - angular difference now reduces along the correct channel axis for HWC and BCHW normals.
- Added `wow-viewer/specs/077-minimap-deconstruction-engine/user-guide.md` with PowerShell-first validation and operator commands.

### Validation status

- Static review completed for the touched bug patterns.
- Direct local execution remains blocked in this IDE shell; user-run commands in `user-guide.md` are the proof path.

### Files touched

- `wow-viewer/src/core/WowViewer.Core/Maps/{ObjectLibraryEntry,ObjectCaptureVariant}.cs`
- `wow-viewer/tests/WowViewer.Core.Tests/ObjectLibraryContractsTests.cs`
- `wow-viewer/data-harvester/src/harvester/{object_library,height_to_normal}.py`
- `wow-viewer/data-harvester/scripts/train_height_only_prior.py`
- `wow-viewer/data-harvester/tests/{test_object_library,test_height_to_normal}.py`
- `wow-viewer/specs/077-minimap-deconstruction-engine/{quickstart,user-guide}.md`
- `wow-viewer/docs/architecture/minimap-deconstruction-engine-2026-06-28.md`
- `wow-viewer/memory-bank/{activeContext,progress}.md`

## 2026-06-28 - Spec 077 Phase 6 (US5 Normal Follow-On) - decision landed (no model work)

### What landed (Python)

- Library module `wow-viewer/data-harvester/src/harvester/height_to_normal.py`:
  - `analytic_normals_from_height` derives per-vertex normals via the
    cross product of central-difference height gradients. Supports
    numpy and torch, 2-D / 3-D batched / 4-D ``(B, 1, H, W)`` inputs.
  - `analytic_normal_difference` reports the mean angular error in
    radians between normals derived from two height fields. Useful as
    a sanity check.
  - Every output normal is unit-length and points "up" out of the
    surface; the function is deterministic and runs in O(HW).
- pytest tests `wow-viewer/data-harvester/tests/test_height_to_normal.py`:
  - Constant height → unit-z normals.
  - Slope along x → normals tilt in -x.
  - Too-small height (1×1) → unit-z fallback.
  - Torch parity with numpy; 4-D ``(B, 1, H, W)`` batched input.
  - Small height delta → small angular error; mirrored height → large error.

### Decision (T042)

- **Analytic normals are sufficient for the first spec 077 pass.** No
  normal model is trained in the MVP. The decision is recorded in this
  progress entry and mirrored in
  `wow-viewer/docs/architecture/minimap-deconstruction-engine-2026-06-28.md`.
- The normal lane is explicitly deferred. If a later surface needs a
  refinement model, it MUST be a separate, independent checkpoint
  (FR-023: one model one signal; no shared weights) and its own
  bounded proof. Spec 077 T043/T044 remain open as the trigger for
  that work.

### Tasks touched

- T039, T040, T041, T042 marked complete in `wow-viewer/specs/077-minimap-deconstruction-engine/tasks.md`.
- T043, T044 deferred per the T042 decision.

### Status

- Phase 6 is code-complete on the analytic baseline. The decision gate
  (T042) closed without spinning up a new model. Spec 077's MVP is
  fully covered by Phases 1–6; the remaining open tasks are
  real-data proofs (T021, T029, T034, T038, T043, T044) that need
  staged clients and bounded wall-clock time.

## 2026-06-28 - Spec 077 Phase 5 (US4 ADT-Free Object Explanation) - first pass landed

### What landed (Python)

- Contracts module `wow-viewer/data-harvester/src/harvester/inference_object.py`:
  - `ObjectMaskPrediction` (tile_id + serialized mask + confidence + model provenance).
  - `AssetCandidate` (asset_path + library_id + score + pose_xy + pose_yaw + bbox).
  - `InferenceObjectHypothesis` (data-model.md §4.1; `top_candidate()`, `ranked_candidates()`).
  - `RecoveredObjectPlacement` (data-model.md §4.2; pitch/roll/scale explicitly None per FR-018).
  - `hypothesis_to_recovered()` lifts a hypothesis into a placement using a terrain-Z arg.
  - `collect_hypotheses()` stable-sorts by top score then tile_id.
- Matcher module `wow-viewer/data-harvester/src/harvester/asset_matcher.py`:
  - Library I/O: `load_library_assets`, `load_library_index` (Parquet over the spec 077 Phase 2 store).
  - `LibraryEntryThumbnail` (image + mask + 16-bit pHash) loaded from the flat capture directory.
  - `score_candidates` (deterministic 0.5·pHash similarity + 0.5·masked correlation, bounded in [0, 1]).
  - `build_hypothesis_from_bbox` (one-shot helper: score + emit a hypothesis with XY = bbox center, yaw = 0).
  - Mask threshold = 128 (uint8), top_k default 5; both configurable.
- ADT-free prior builder `wow-viewer/data-harvester/scripts/build_adt_free_prior.py`:
  - `build_adt_free_prior_tensor` produces the same 5-channel tensor as `teacher_prior` but with a *predicted* object mask (no ADT input).
  - CLI reads V18 `minimap_rgb` + an NPZ/Zarr predicted-mask array, writes `<build>.zarr` with `raw_minimap_rgb_256`, `predicted_object_mask_256`, `processed_minimap_prior_256` + `tiles.parquet` + group-level `metadata.json`.
- pytest tests `wow-viewer/data-harvester/tests/test_inference_object.py`:
  - pHash stability + Hamming; masked-correlation edge cases (empty masks, identical, disjoint).
  - Resize nearest; `top_candidate()` / `ranked_candidates()` ordering; `hypothesis_to_recovered` lifts XY, yaw, terrain Z; FR-018 deferred fields remain None; `collect_hypotheses` sort order.
  - `score_candidates` returns the blue library entry as the top match for a blue minimap crop.
  - `build_hypothesis_from_bbox` honors top_k and emits XY = bbox center, yaw = 0.
  - ADT-free prior: empty-mask pass-through; object-pixels suppressed; CLI end-to-end writes Zarr + tiles.parquet.

### Tasks touched

- T030, T031, T032, T033, T035, T036, T037 marked complete in `wow-viewer/specs/077-minimap-deconstruction-engine/tasks.md`.
- T034 still open — needs a trained object-mask lane (this phase ships the consumer side; the producer side is the next bounded slice).
- T038 still open — needs a real development-map proof; gated on T034 + a development-map V18 store.

### Status

- Phase 5 contracts + matcher + ADT-free prior builder are code-complete and unit-testable. The first pass is deliberately deterministic (pHash + masked correlation) so it runs without GPUs and so the ranker behavior is auditable; a learned embedding lane (DINOv2 etc.) can be slotted in without changing the public surface.
- The pipeline is wired end-to-end: predicted mask → ADT-free prior → (downstream) height model from Phase 4.

## 2026-06-28 - Spec 077 Phase 4 (US3 Height-Only Terrain Reboot) - first pass with V18 perf stack

### What landed (Python)

- Dataset `wow-viewer/data-harvester/src/harvester/height_only_prior_dataset.py`:
  - `HeightOnlyPriorDataset` reads the teacher-prior Zarr as the model input and the source V18 Zarr for the authoritative `height_257` target plus `object_filtered_mask` (the spec 077 FR-027 weight signal).
  - Default `height_norm=True` for per-tile mean/std normalization; pass `False` to keep raw world units.
  - Returns the documented `HeightOnlyTrainingSample` contract (`input_prior (5, 256, 256)`, `height_257 (1, 257, 257)`, `weight_257 (1, 257, 257)`, plus `meta_build/map/tile_id`).
  - No normal / liquid / object head; FR-012/FR-013/FR-023 enforced.
- Training script `wow-viewer/data-harvester/scripts/train_height_only_prior.py`:
  - **V18 perf stack ported**: AMP (`torch.amp.GradScaler` + `autocast`), `torch.compile` with graceful fallback, gradient clipping at `max_norm=1.0`, AdamW with `weight_decay`, `optimizer.zero_grad(set_to_none=True)`, `non_blocking=True` on every `.to(device, ...)`, multi-scale L1 loss at 257/128/64/32/16, optional Sobel gradient + normal-consistency losses, early stopping (patience + min improvement), resume from checkpoint (model + optimizer + scaler + step), labeled 4-panel preview (prior RGB / height truth / height pred / loss weight) with text strips, `DataLoader` with `num_workers` / `prefetch_factor` / `persistent_workers`, optional VRAM autotune with evidence JSON, deterministic seeding, per-step throughput reporting.
  - First 3 channels of the prior (`suppressed_rgb_r/g/b`) feed `V18HeightModel`; mask + confidence are applied as the loss weight instead of as input channels (matches the V21 design intent — height-only lane with filtered-mask loss, not a multi-channel multi-task model).
- pytest tests `wow-viewer/data-harvester/tests/test_height_only_prior.py`:
  - Dataset: documented sample contract, weight zeros out filtered pixels, height normalization zeros mean, missing V18 inference-mode fallback, dataset summary.
  - Loss: multi-scale L1 returns zero on perfect prediction, falls back to single-scale when disabled, auxiliary terms aggregate, gradient magnitude is zero on constant input.
  - Training: CPU smoke run writes metrics / checkpoint / preview, resume from checkpoint continues past `start_step`.

### Tasks touched

- T022, T023, T024, T025, T026, T027, T028 marked complete in `wow-viewer/specs/077-minimap-deconstruction-engine/tasks.md`.
- T029 still open — needs a real V18 Zarr store + teacher-prior run to record a bounded proof.

### Status

- Phase 4 is code-complete and unit-testable on CPU. The V18 perf stack is in place; the first real run is gated on having a teacher-prior store built from real placements (Phase 3 T021).
- All perf knobs are CLI flags with safe defaults; the smoke tests pass on CPU with `--no-amp --no-compile` and on CUDA with the default stack.

## 2026-06-28 - Spec 077 Phase 3 (US2 Teacher Deconstruction Priors) - first pass landed

### What landed (Python)

- Library module `wow-viewer/data-harvester/src/harvester/teacher_prior.py`:
  - `MaskSource` enum (ObjectFiltered / ObjectPrecise / ObjectMask / None).
  - `pick_object_mask()` honors the spec 077 FR-009 preference chain (filtered > precise > object).
  - `suppress_object_pixels()` fills object pixels with the per-tile median of non-object pixels (deterministic, no inpainting dependency).
  - `build_prior_tensor()` returns the 5-channel `processed_minimap_prior_256` (3 suppressed RGB + 1 mask + 1 confidence) plus the mask/confidence arrays and chosen source.
  - `TeacherPriorTileRecord` carries build/map/tile_id/tile_x/tile_y + keys + coverage + mask source.
  - `PRIOR_CHANNELS` tuple documents the phase-1 channel layout.
- CLI builder `wow-viewer/data-harvester/scripts/build_teacher_prior_dataset.py`: reads V18 `minimap_rgb` + `object_filtered_mask` / `object_precise_mask` / `object_mask` + `index.parquet`, writes a `<build>.zarr` store with the four phase-1 arrays + `tiles.parquet` + group-level `metadata.json` (records schema, build, source path, mask preference chain, fill strategy).
- CLI reviewer `wow-viewer/data-harvester/scripts/review_teacher_prior_dataset.py`: renders a 3-row contact sheet (raw / mask / suppressed) and `index.html` with per-tile coverage table.
- pytest tests `wow-viewer/data-harvester/tests/test_teacher_prior.py`: cover preference chain (filtered / precise / object / none), no-object passthrough, all-object neutral fallback, channel shape and dtype, end-to-end CLI with a synthetic V18 Zarr store (verifies metadata fields, prior layout, mask band, and pass-through equality).

### Tasks touched

- T014, T015, T016, T017, T018, T019, T020 marked complete in `wow-viewer/specs/077-minimap-deconstruction-engine/tasks.md`.
- T021 still open — needs a real V18 Zarr store (e.g. `3_3_5_12340`) to record a bounded proof on a real object-rich map.

### Status

- Phase 3 is code-complete and unit-testable without real data; T021 is the validation gate against real V18 placements.
- The next bounded run must use a staged client (per RULE 9) and target one object-rich anchor map.

## 2026-06-28 - Spec 077 Phase 2 (US1 Per-Object Capture Library) - first pass landed

### What landed (C# + Python)

- C# shared data contracts under `wow-viewer/src/core/WowViewer.Core/Maps/`:
  - `ObjectLibraryEntry.cs` (record + 4 enums: asset_type, capture_status, visibility_class, review_state) + `ComputeLibraryId` (SHA1-14 hex, prefix `objlib_`).
  - `ObjectCaptureVariant.cs` (record + `ObjectLibraryBoundingBox` + `ComputeVariantId` SHA1-16 hex, prefix `objvar_`).
  - Both default to `NotAttempted`/`Unknown`/`Unreviewed` per FR-026.
- xUnit tests `wow-viewer/tests/WowViewer.Core.Tests/ObjectLibraryContractsTests.cs` (9 tests: ID stability, distinct paths, blank handling, defaults, pose sensitivity, capture-mode sensitivity, bounding-box invariants).
- Python module `wow-viewer/data-harvester/src/harvester/object_library.py` mirrors the C# contract: same enums, same ID rules, same field names, `normalize_asset_path`/`detect_asset_type`/`is_clutter_asset` helpers.
- pytest tests `wow-viewer/data-harvester/tests/test_object_library.py` (parametrized enum coverage, validation raises, bbox helpers).
- Capture-job enumerator `wow-viewer/data-harvester/scripts/enumerate_object_capture_jobs.py`: reads V18 `placements.parquet` + `index.parquet`, collapses to one job per (instance_type, normalized asset path), writes JSONL. Honors `--skip-clutter` and `--include-mddf/--include-modf`.
- Library builder `wow-viewer/data-harvester/scripts/build_object_library.py`: reads enumerator JSONL + flat capture directory, writes Zarr v3 store (`capture_rgb/`, `capture_mask/`, `capture_alpha/`, `assets.parquet`, `index.parquet`, group-level `metadata.json`). Missing capture artifacts produce `not_attempted` entries, not silent drops.
- Review script `wow-viewer/data-harvester/scripts/review_object_library.py`: reads built store, renders per-family contact sheets under `<out>/families/<library_id>.png`, writes a top-level `index.html` and `assets.json` snapshot.
- pytest e2e test `wow-viewer/data-harvester/tests/test_object_library_e2e.py` runs the builder + reviewer against a synthetic capture directory (no client data needed).
- Quickstart `wow-viewer/specs/077-minimap-deconstruction-engine/quickstart.md` documents how to run the proof.

### Tasks touched

- T006, T008, T009 (Python side), T011, T012, T013 marked complete in `wow-viewer/specs/077-minimap-deconstruction-engine/tasks.md`.
- T007, T009 (C# writer), T010 still open — T010 is the C# one-object-at-a-time capture-lane extension in `WowViewer.Tool.ValidationCapture`; T007 depends on the C# writer that T009 needs.

### Status

- First pass is Python-first by design: enumerator + builder + review run end-to-end without the C# capture lane extension. The C# capture tool can later write directly into the flat capture directory the builder consumes, but the first proof can also stage a small mock capture directory for end-to-end validation.
- No real-data proof run yet; the next bounded pass must run `enumerate_object_capture_jobs.py` against a V18 Zarr store (e.g. `3_3_5_12340`) and feed the output through `build_object_library.py` to confirm the contract holds on real placements.

### Next slice (still Phase 2)

- Run the enumerator + builder against a bounded V18 store and capture real-data stats.
- Continue with the Python-only Zarr path before the capture tool extension (T010). Do not add a C# Zarr writer; C# only handles stream/preprocessing.

## 2026-06-28 - Spec 077 minimap deconstruction engine planned

## 2026-06-26 - WMO doodad-group selection and panel details

### What landed

- Added `TryGetDoodadDef`, `GetDoodadDefName`, `GetRenderGroupsForDoodadDef`, `GetDoodadCountForRenderGroup`, `DoodadDefCount` to `WmoRenderer` for public access to raw DoodadDef data and group-doodad linkage.
- Enhanced `DrawWmoDoodadInspector` to show detailed doodad info: MODN name, full model path, position, scale, Euler rotation, hex color (BGRA), and list of referencing groups.
- Added group filter combo to the doodad inspector to filter doodads by WMO group membership.
- Added `_standaloneWmoDoodadGroupFilter` / `_worldWmoDoodadGroupFilter` fields.
- Added "Show Doodads" button to group controls linking group selection to doodad filter.
- Added `QuaternionToEulerDegrees` helper.
- Group controls now show doodad count per selected group and total DoodadDef count.

### Files changed

- `wow-viewer/src/viewer/WoWViewer/Rendering/WmoRenderer.cs` — 5 new public methods
- `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs` — 2 new fields
- `wow-viewer/src/viewer/WoWViewer/ViewerApp_Sidebars.cs` — enhanced doodad inspector, added helper
- `wow-viewer/src/viewer/WoWViewer/ViewerApp_WmoGroups.cs` — doodad count in group details, Show Doodads button

## 2026-06-24 — Spec 076 macro paste/scar grouping correction

### What landed

- Added `segment_macro_pastes()` in `src/harvester/fractal_segments.py` for macro authored-region grouping. It streams/max-pools alpha, dilates on the coarse grid to merge nearby strokes, reprojects bboxes to full map coordinates, and filters by original alpha area.
- Added analyzer flags: `--macro-pastes`, `--macro-close-radius`, `--macro-min-area`, `--macro-min-footprint`, `--macro-max-aspect`, and `--macro-downsample-factor`.
- Added analyzer visual review flags: `--visualize-macro` writes macro alpha overview, crop contact sheets, summary, and HTML index; `--visualize-composite-signal` writes V18-style hard-region overview under the same macro boxes.
- Added `scripts/sweep_macro_paste_visuals.py` to compare close-radius/min-area settings and produce a linked sweep `index.html`.
- Added `segment_blocky_pastes()` plus analyzer `--blocky-pastes` for dense middle-scale child chunks inside giant macro parent zones. Supports `--block-size`, `--block-min-coverage`, `--block-close-radius`, and `--block-max-footprint`.
- Added macro segmentation tests for merging nearby strokes and filtering tiny strokes.
- Fixed WIP family-catalog Zarr output to store family IDs as numeric UTF-8 byte arrays plus lengths instead of unstable object/string arrays.

### Validation

- `uv run ruff check src/harvester/fractal_segments.py src/harvester/fractal_family_catalog.py scripts/analyze_fractal_raw_components.py tests/test_fractal_segments.py` -> passed.
- `uv run pytest tests/test_fractal_canvas.py tests/test_fractal_segments.py tests/test_fractal_segments_rectangle.py tests/test_fractal_library.py tests/test_fractal_raw_analysis.py tests/test_analyze_fractal_raw_components.py tests/test_fractal_near_dedupe.py tests/test_fractal_family_catalog.py -q` -> 38 passed.
- Bounded real-data smoke: `0_5_3_3368` Azeroth, `--tile-limit 16`, `--macro-pastes` -> 7 macro regions under `wow-viewer/output/analysis/full-map-fractal-brush-library/macro_smoke_tile16/`.
- Small full-map visual proof: `0_7_0_3694` `PVPZone02`, `--tile-limit 0`, close-radius 8/min-area 1024, `--macro-pastes --visualize-macro --visualize-composite-signal` -> 4 macro regions under `wow-viewer/output/analysis/full-map-fractal-brush-library/macro_visual_composite_pvpzone02_close8_area1024/`.
- Sweep proof: `macro_sweep_pvpzone02_r8_16_32_area1024_4096/index.html` compares close radius 8/16/32 and min-area 1024/4096; all settings produced 3-4 macro regions.
- Blocky proof: `0_7_0_3694` `PVPZone02`, `--blocky-pastes --block-size 16 --block-min-coverage 0.45 --block-close-radius 0 --block-max-footprint 160` -> 10 child regions under `wow-viewer/output/analysis/full-map-fractal-brush-library/blocky_visual_pvpzone02_b16_cov045_close0_max160/`.

### Status

- Raw connected alpha components and near-dedupe contact sheets are diagnostic evidence only, not the primary brush/paste target.
- Visual finding: broad macro boxes are parent canvases/context; `blocky_paste` regions are now closer to the desired internal authored chunks.
- Next route: tune blocky child segmentation against composite hard-region overview on more maps, then run canonical `--maps all` validation.
- Training remains blocked until macro paste/scar outputs are visually validated and Phase 5 one-signal model targets are approved.

## 2026-06-23 — Spec 076 replaces 074/075 brush-model direction

### Phase 1 implementation

- Added `src/harvester/fractal_canvas.py` for tile-local to map-canvas transforms, compact tile-window selection, dense bounded canvas assembly, Zarr/Parquet output, and seam overlay rendering.
- Added `scripts/build_full_map_fractal_canvas.py` CLI.
- Added `tests/test_fractal_canvas.py`; `uv run pytest tests/test_fractal_canvas.py` -> `4 passed`.
- Lint passed: `uv run ruff check src/harvester/fractal_canvas.py tests/test_fractal_canvas.py scripts/build_full_map_fractal_canvas.py`.
- Real-data smoke passed on V18 Zarr `0_5_3_3368`/`Azeroth`, tile-limit 4.
- Smoke output: `wow-viewer/output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact/`.
- Output shapes: alpha `(256,1024,4)`, height `(257,1025)`, MCLY `(16,64,4)`.

### Phase 1 status

- Phase 1 is implemented and validated for bounded compact tile windows and full-map strip processing.
- Full-continent chunk streaming is now implemented via tile-chunked Zarr writes and horizontal strip segmentation.
- Next route: review full-map strip artifacts, then improve dedupe/clustering before Phase 5 model target selection.

### Phase 2 implementation

- Added `src/harvester/fractal_segments.py` for full-map alpha region extraction, region stats, curation labels, optional 074 catalog linkage, Parquet/JSONL output, and overlay rendering.
- Added `scripts/segment_full_map_fractals.py` CLI.
- Added `tests/test_fractal_segments.py`; `uv run pytest tests/test_fractal_segments.py` -> `3 passed`.
- Lint passed: `uv run ruff check src/harvester/fractal_segments.py tests/test_fractal_segments.py scripts/segment_full_map_fractals.py`.
- Real-data strict-footprint smoke passed on the Phase 1 tile16 compact canvas: 961 regions; 11 accepted candidates, 24 fractal members, 1 composite chonker, 2 one-off details, 923 too-small rows.
- Segment output: `wow-viewer/output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/segments/`.
- Curation correction: `composite_chonker` is preserved as a composite-canvas harvest target, while default atomic samples require an `8x8` alpha-pixel minimum footprint.

### Phase 2 status

- Phase 2 is implemented and validated for bounded compact canvases and full-map strip views.
- Next route: inspect `fractal_regions_overlay.png` and metadata; add near-duplicate clustering because exact dedupe is too brittle.

### Phase 3 implementation

- Added `src/harvester/fractal_library.py` for terrain-art sample schema, stable IDs, deterministic splits, accepted/rejected filtering, Zarr tensor output, Parquet metadata output, and a smoke loader.
- Added `scripts/build_fractal_brush_library.py` CLI.
- Added `tests/test_fractal_library.py`; `uv run pytest tests/test_fractal_library.py` -> `3 passed`.
- Lint passed: `uv run ruff check src/harvester/fractal_library.py tests/test_fractal_library.py scripts/build_fractal_brush_library.py`.
- Real-data smoke passed on the Phase 2 tile16 strict-footprint segments: 35 default trainable atomic samples, 926 review/rejected rows, split counts `train=26`, `val=8`, `test=1`.
- Smoke loader read 32 samples and returned only `accepted_candidate`/`fractal_member` labels.
- Library output: `wow-viewer/output/datasets/fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/`.

### Phase 3 status

- Phase 3 is implemented and validated for the bounded compact canvas.
- Next route: Phase 4 texture/variant/BLP source-evidence inventory and joins. Do not start model training yet.

### Phase 4 initial inventory

- Added `specs/076-full-map-fractal-brush-library/research.md` documenting reusable texture evidence and gaps.
- Confirmed reusable-now evidence is MCLY texture IDs/layer masks already present in V18 Zarr, Phase 1 canvas, Phase 2 region metadata, and Phase 3 sample tensors.
- Added per-sample Phase 4 fields in `fractal_library.py`: `mcly_texture_id_counts`, `dominant_mcly_texture_id`, and `mcly_active_layer_coverage`.
- No canonical decoded terrain tileset/BLP fingerprint artifact was found in `data-harvester` or `wow-viewer/output`; object-roof fingerprints are useful prior art only.
- Next route: add a bounded decoded terrain texture/BLP evidence extractor or join a canonical fingerprint artifact when one exists.

### Raw two-build analysis runner

- Added `src/harvester/fractal_raw_analysis.py`, `src/harvester/fractal_near_dedupe.py`, `scripts/analyze_fractal_raw_components.py`, and `scripts/visualize_fractal_raw_patterns.py`.
- The runner processes each build/map sequentially, writes per-target `canvas/` and `segments_raw/`, then writes cross-target exact dedupe under `dedupe/`. It supports `--maps all`.
- Validated two-build Azeroth run: 7,317 raw components, 3,957 exact patterns, 233 duplicate patterns under `two_build_test1`.
- Footprint correction to 8x8 alpha pixels: 2,025 raw components, 2,002 exact patterns, 17 duplicates under `two_build_test2`.
- Full-map strip processing: `--tile-limit 0` loads every tile for each selected map, writes tile-chunked Zarr canvases, segments horizontal strips, offsets bboxes to global coords, dedupes strip overlaps. Full Azeroth 0.5.3 (622 tiles) produced 12,906 raw components, 12,163 exact patterns, 566 exact duplicates under `full_map_Azeroth_0_5_3_3368`. Canonical validation runs use `--maps all`.
- Near-duplicate clustering: groups raw components by translation/mirror/rotation-invariant normalized binary thumbnails. Full Azeroth 0.5.3 collapsed to 11,976 clusters (668 duplicate clusters, max size 40) with a 16x16 thumbnail and radius 0.
- Rectangle-page detection: `detect_rectangle_pages()` finds solid axis-aligned rectangular alpha pages (extent >= 0.85). Full Azeroth 0.5.3 produced 72 rectangle_page regions; with rectangles included near-duplicates became 688 clusters, max size 76.
- Added `tests/test_analyze_fractal_raw_components.py`, `tests/test_fractal_near_dedupe.py`, and `tests/test_fractal_segments_rectangle.py`; pytest passes.
- Contact-sheet visualizer renders repeated exact-pattern pages from the dedupe catalog (200 patterns / 5 pages proven).
- Near-duplicate cluster contact-sheet visualizer added: `scripts/visualize_fractal_near_patterns.py`; rendered 100 repeated clusters across 10 pages for full Azeroth 0.5.3.
- Analyzer writes a top-level `index.html` for every run, linking per-map canvases, overlays, and cross-map dedupe/near catalogs.
- Use these for broad inspection; they still detect connected alpha/fractal components, not obvious rectangular paste/canvas-page boundaries.

### What landed

- Added `specs/076-full-map-fractal-brush-library/{spec,plan,tasks}.md`.
- Added `docs/architecture/full-map-fractal-brush-library-2026-06-23.md`.
- Marked 074 as deprecated for primary training labels; its connected components are evidence rows only.
- Marked 075 as diagnostic only; whole-tile scar-mask segmentation is not the brush/fractal/paste target.
- Marked V18 paste and fractal-height-loss docs as historical/paused for current route.

### Corrected direction

- Assemble full-map alpha/MCLY/height/normal canvases before segmentation.
- Segment fractal/virtual-canvas structures in map coordinates, not ADT-tile-local coordinates.
- Treat mesh, alpha masks, MCLY texture/layer assignments, and possible source BLP/decal/effect stamps as one coupled ZBrush-like terrain-art primitive.
- Preserve chonkers as composite-canvas harvest targets; exclude one-off roads/details, tiny unique strokes, and low-repeatability noise from default atomic training manifests.
- Preserve provenance and build a Zarr/Parquet trainable library before any new model training.
- Phase 4 should also investigate likely transparent/effect BLP source assets (`textures\BloodSplats`, FX/environment/weather/decal/particle-style textures) as possible original brush sources for alpha/fractal motifs.

### Next

- Run canonical `--maps all` validation across both builds and review the resulting cross-map contact sheets/overlays.
- Tune near-dedupe thumbnail size and Hamming radius against cross-map contact-sheet review.
- Tune rectangle-page thresholds against cross-map overlays; some detected rectangles may be roads/rivers rather than authored paste pages.

## 2026-06-23 — Spec 075 V21 scar-mask segmentation Phase 1 complete

### What landed

- Created spec/plan/tasks under `specs/075-scar-mask-segmentation/`.
- Corrected naming: spec number is 075, model lane is V21 (`v21_scar_*`) because V18 is only the patched Zarr substrate.
- Added `src/harvester/v21_scar_dataset.py`: minimap input + binary scar target from `alpha_256` layers L1-L3 at threshold `0.05`.
- Added `src/harvester/v21_scar_model.py`: single-output scar-mask logits model `(B,1,256,256)`.
- Added `scripts/train_v21_scar_mask.py`: standalone trainer with BCE+Dice loss, checkpoints, metrics, and preview.
- Added `src/harvester/test_v21_scar_mask.py` and architecture doc `docs/architecture/v21-scar-mask-segmentation-2026-06-23.md`.

### Validation

- `uv run ruff check src/harvester/v21_scar_dataset.py src/harvester/v21_scar_model.py src/harvester/test_v21_scar_mask.py scripts/train_v21_scar_mask.py`
- `uv run pytest src/harvester/test_v21_scar_mask.py` -> 3 passed.
- `uv run python -m py_compile src/harvester/v21_scar_dataset.py src/harvester/v21_scar_model.py scripts/train_v21_scar_mask.py`
- Smoke: `uv run python scripts/train_v21_scar_mask.py --builds 0_5_3_3368 3_3_5_12340 --max-steps 2 --val-max-steps 1 --batch-size 2 --max-tiles 64 --base-channels 8 --run-name smoke`.

### Status

Phase 1 is mechanically complete. Smoke outputs exist at `models/v21/scar-mask/runs/smoke/`. The smoke proves the model lane runs, not that the model is useful yet. Next: choose a real training schedule/subset, then add inference + connected-component extraction for predicted scar masks.

## 2026-06-23 — Spec 074 contact-sheet visualizer added

### What landed

- Added `scripts/visualize_alpha_brush_catalog.py` to render contact-sheet PNGs from `components.jsonl`/`clusters.jsonl`/`catalog.jsonl` by reopening V18 `alpha_256` crops.
- Rendered top-100 sheets under `wow-viewer/output/analysis/alpha-brush-library/two-build-full/montages/`.
- Rendered full 1000-cluster library under `wow-viewer/output/analysis/alpha-brush-library/two-build-full/montages_all/` with 20 paginated PNG sheets and `index.html`.
- Added explicit legend: gray=L0 base/fill, blue=L1 primary brush, green=L2 transition/detail, orange=L3 highlight/detail.
- Captured human review notes in `specs/074-alpha-brush-library/visualization_notes.md`: current clusters are atomic strokes, while useful authored units are likely multi-component/multi-tile sprites/prefabs/pastes; C35 looks like low-resolution legacy heightmap-like stamps, plausibly Warcraft 3 editor-era reuse.

### Status

Next useful 074 slice is not more component clustering. It is a grouping pass that reconstructs larger multi-tile sprite/paste candidates from co-occurring component clusters, then renders those as the actual prefab library.

### Exact scar dedupe follow-up

- Added `scripts/dedupe_alpha_brush_patterns.py`: hashes exact binary alpha crops, writes `exact_patterns.jsonl`, and ranks non-exact variants in `pattern_neighbors.jsonl` by embedding similarity.
- Full run on `two-build-full`: 320,368 components -> 263,188 exact binary scars; largest exact pattern has 715 members; 2,105,504 near-neighbor rows.
- Added `scripts/visualize_alpha_brush_pattern_neighbors.py`: renders exact canonical scar + nearest non-exact neighbors per row.
- Rendered top-200 exact scars to `two-build-full/dedupe/neighbor_montages/`.

## 2026-06-23 — Spec 074 Phase 1 complete, Phase 2 implemented

### What landed

- Added `wow-viewer/data-harvester/src/harvester/alpha_brush.py` with component/cluster/catalog dataclasses, extraction, patch rendering, DINOv2 embedding, clustering, catalog builders, and JSONL serializers.
- Added `wow-viewer/data-harvester/tests/test_alpha_brush.py`; targeted pytest passes.
- Added `wow-viewer/data-harvester/scripts/extract_alpha_brush_catalog.py` for V18 Zarr bulk extraction and catalog output.
- Phase 2 two-build smoke passed: `0_5_3_3368` + `3_3_5_12340`, `--tile-limit 2`, 179 components, 16 clusters, 16 non-singleton clusters.

### Validation

- `uv run ruff check src/harvester/alpha_brush.py scripts/extract_alpha_brush_catalog.py tests/test_alpha_brush.py`
- `uv run pytest tests/test_alpha_brush.py`
- `uv run python -m py_compile src/harvester/alpha_brush.py scripts/extract_alpha_brush_catalog.py tests/test_alpha_brush.py`

### Status

074 cannot move to Phase 3 yet. T022 full two-build validation is still open because it requires DINOv2 over 1,629 + 5,134 V18 alpha tiles and should not be marked complete from a smoke run.

### Documentation follow-up

- Added `specs/074-alpha-brush-library/data-model.md` with the exact current schemas and output files.
- Added `specs/074-alpha-brush-library/quickstart.md` as the operator/user guide for setup, smoke runs, full T022 extraction, result inspection, and troubleshooting.
- Linked the 074 quickstart from `data-harvester/README.md`.
- T030 remains open for final visualization command coverage because Phase 3 visualization is not implemented yet.

## 2026-06-23 — Spec 074 Phase 0 research complete

### What landed

- Added `wow-viewer/data-harvester/scripts/_research_alpha_components.py` for one-off alpha component research.
- Ran against `wow-viewer/output/datasets/v18/0_5_3_3368.zarr` on 12 alpha-bearing Azeroth tiles.
- Threshold counts: `0.03` -> 215 components, `0.05` -> 247, `0.10` -> 333.
- DINOv2 `facebook/dinov2-small` loaded through `transformers`; 96 component patches embedded.
- Outputs written under `wow-viewer/output/analysis/alpha-brush-library/research/`, including `projection.png`, `[CLS]`/mean projections, embeddings, patch examples, and `summary.json`.
- `research.md` records the Phase 0 decision: threshold `0.05`; mean-pooled patch-token embeddings by default.

### Status

074 Phase 0 complete. Next is Phase 1 shared library `alpha_brush.py` plus synthetic-shape smoke tests.

## 2026-06-22 — Pivot from V21 height regression to 074 Alpha Brush Library

### What happened

- V21/V21c height training could not reproduce the earlier 0.3126 baseline. Runs restored to commit `d0929e2` still stalled at ~0.83 height L1 after 35 epochs.
- Decided the end-to-end minimap→height approach skips the actual terrain construction process.
- New direction: treat the ADT as a layered Photoshop canvas and reverse-engineer the artists' fractal brush library from MCAL alpha masks.

### What landed

- Spec `074-alpha-brush-library` created at `wow-viewer/specs/074-alpha-brush-library/`.
- Plan written: 5 phases (research → shared library → bulk extraction → visualization → docs/handoff).
- Tasks broken down in `tasks.md`.
- DINOv2 (`transformers`) confirmed available and loadable in the data-harvester environment.
- Memory bank updated.

### Status

074 ready to start with Phase 0 research. 071 in user testing.

---

## 2026-06-22 — 073a: Toolbar / left sidebar dedup and alignment (complete)

### What landed

- Removed `DrawWorkspaceToolbarControls`, "Open Game Folder", and "Open File" from `DrawToolbar`.
- Toolbar now shows only scene status + centered terrain controls.
- Source/workspace controls remain in the left sidebar (`DrawWorkspaceBarsPanelContent`).
- Legacy mode preserved.
- Build: 0 errors, 284 pre-existing warnings.
- Commit `b11dd518` pushed to `071-left-right-sidebar-split`.

### Status

073a complete. 073b (Tools tab converter integration) spec'd and ready for implementation in fresh chat.

## 2026-06-22 — 072: Sidebar resize + toolbar layout hotfix (complete)

### What landed

- Removed `DrawFixedSidebarWidthControl` sliders from inside tab-mode left/right sidebars.
- `DrawFixedSidebarSplitters` now draws left/right edge splitters in tab mode.
- `DrawToolbar` spans only the scene viewport width (`viewportX`..`viewportWidth`).
- `DrawToolbar` is called after sidebars in `DrawUI` so it stays on top if edges overlap.
- Build: 0 errors, 284 pre-existing warnings.
- Commit `bcdcb752` pushed to `071-left-right-sidebar-split`.

### Status

072 hotfix complete.

## 2026-06-22 — Spec 071 Phase H: Memory bank + spec sync + final build (complete)

### What landed

- Updated `specs/071-left-right-sidebar-split/spec.md` to match final implementation.
- Updated memory bank with full 8-phase history.
- Final build: 0 errors, 286 pre-existing warnings.
- Commit `8190fb65` pushed to `071-left-right-sidebar-split`.

### Status

Spec 071 complete.

## 2026-06-22 — Spec 071 Phases A-G (complete)

Summary of earlier 071 phases:
- **A**: Viewport subtracts left/right sidebars.
- **B**: Left sidebar with workspace bars, file browser, world maps.
- **C**: Right sidebar = workbench anchored to right edge.
- **D**: 3 top tabs (Model/World/Tools) with `WorkbenchNavigator` and typed `OpenWorkbenchTab` helpers.
- **E**: Model > Info sub-tab with path line.
- **F**: Model > Animations sub-tab with Play/Pause/Stop, loop, speed buttons, timeline slider; added `PlaybackSpeed`/`Loop` to `IAnimationController`.
- **G**: Model > Actions + LOD sub-tabs; selected world object auto-switches to Model > Info.

All phases built clean and pushed to `071-left-right-sidebar-split`.

## 2026-06-21 — Spec 071 drafted

- Two-side layout + Model Viewer mode, 8 phases, branch cut from `069-viewer-ui-overhaul`.

## 2026-06-21 — Spec 069: Viewer UI overhaul (tab system → workbench)

- Cells overlay, tab data model, archeology playback, sticky settings, headless content variants.
- Learned: top/bottom tab bars failed (debug overlay look), per-sub-tab popouts failed (window sprawl), single Workbench panel succeeded.
- 14 phases committed to `069-viewer-ui-overhaul`.

## Previous work
- Spec 068: fractal-aware height loss + curation hardening (V21c)
- Spec 067: V20 multi-modal terrain intent
- Spec 066: V19 minimal-signal height regressor
- PM4 surface correlation, PM4 simplification reverse-engineering

## Branch summary

- `071-left-right-sidebar-split` — 071 + 072, active, user testing.
- `069-viewer-ui-overhaul` — legacy tab UI work, salvageable concepts extracted into 071.
- `074-alpha-brush-library` — implemented candidate/evidence extraction, deprecated as primary brush truth.
- `075-scar-mask-segmentation` — diagnostic baseline only, deprecated as primary model route.
- `076-full-map-fractal-brush-library` — active dataset-truth plan; Phase 1-3 and full-map strip processing are implemented.
- `077-minimap-deconstruction-engine` — active. Phases 1-6 code-complete. `cuda_albedo_group_nearest` running on RunPod (RTX 4000 Ada, network volume, cost-target default).
- `079-runpod-integration-guide` — new. Spec written with all RunPod lessons learned.
- `071-left-right-sidebar-split` — active branch, user testing viewer UI.

## 2026-06-29 - Spec 077 cloud training on RunPod: `cuda_albedo_group_nearest` running successfully

### Context

- Fresh `cuda_albedo_group_nearest` run with `--albedo --model-norm group --decoder-upsample nearest` started on RunPod via network-volume-backed RTX 4000 Ada Pod.
- Cost-target defaults ($1/hr, 12GB VRAM, 24GB RAM, COMMUNITY cloud) replaced fixed GPU-type targeting after learning `runpodctl gpu list` has no pricing and REST API fails with stale payload fields.
- RunPod integration spec (079) captures all lessons learned for future projects.

### What landed (this session)

- Cost-target mode: `--max-cost-per-hour 1.00`, `--min-gpu-vram-gb 12`, `--min-ram-gb 24`, `--cloud-type COMMUNITY` (default).
- Minimal REST payload: `gpuTypeIds`, `gpuCount`, `imageName`, `containerDiskInGb`, `volumeInGb`, `volumeMountPath`, `ports`, `supportPublicIp`. Removed `gpuTypePriority`, `dataCenterPriority`, `minRAMPerGPU`, `minVCPUPerGPU`, `computeType`, `interruptible`.
- Excluded datacenter GPUs (A100, H100, H200, B200, L4, L40, A40, Tesla, RTX PRO, AMD) with `EXCLUDED_GPU_PREFIXES` + `_is_excluded_gpu()`.
- Error handling: `_is_no_instances_error` expanded to `_is_retryable_error` (500-level, "not found", "does not support").
- Network-volume datacenter filtering: `VALID_NETWORK_VOLUME_DATACENTERS` constant wired into `_requested_data_centers` and `_availability_candidates`.
- Hardcoded `FALLBACK_GPU_INFOS` with approximate prices because `runpodctl gpu list` has no price fields.
- `_gpu_full_info()` parses actual `runpodctl gpu list` fields (`gpuId`, `memoryInGb`, `available`, `communityCloud`) and merges with fallback pricing.
- `_requested_gpu_types(use_cost_target=True)` filters by VRAM, excludes datacenter GPUs, sorts cheapest first.
- Network volume defaults to ON with `--use-network-volume` (True by default).
- Pod image updated to `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` (Python 3.11 for zarr>=3.0).
- Real RunPod Pod created successfully with minimal payload (Pod deleted, new one with network volume running).
- Switch from fixed GPU-type targeting to cost-target mode after troubleshooting stale datacenter lists, wrong image, overloaded payload, missing SSH keys.
- `setup_spec077_runpod.py` tests updated: cost-target defaults, GPU exclusion, payload field assertions.
- Created `wow-viewer/specs/079-runpod-integration-guide/spec.md` with all RunPod lessons learned.
- Workflow decisions: `runpodctl send/receive` for bundle transfer, web terminal as SSH fallback, network volume for one-time upload persistence.

### Validation

- Real RunPod Pod created from minimal REST payload (confirmed working).
- Tests pass (when shell available): `uv run pytest tests/test_package_spec077_runpod.py -q`.
- `cuda_albedo_group_nearest` training started on RunPod — monitoring.

### Next

- Monitor `cuda_albedo_group_nearest` on RunPod for convergence and detail quality.
- If broad shape improves but small details remain, open T029b (MCLY-guided `height_delta_257` residual-refinement lane).
- Update spec 079 with plan.md and tasks.md for future project reuse.

## Out-of-Phase Work

- 070: Per-map workbench windows (deferred, large rewrite).
- 073b: Tools tab converter integration (spec'd, implementation deferred to fresh chat).
- V21/V21c height regression and fractal-aware height loss: paused pending 076 curated library validation.
# 2026-06-29 - Viewer animation controls resurfaced

## What landed

- The active viewer now exposes model animation playback controls in the default Model > Info sub-tab, not only in the Animations sub-tab.
- Added missing Stop buttons to the SQL gameobject animation panel and the world-MDX animation panel.
- Added a save-dialog-backed animation state JSON export action from the viewer UI.

## Verification

- Source-level review confirms the controls are wired into `ViewerApp.cs` / `ViewerApp_Sidebars.cs` and reuse the existing animation controller state.
- Shell build verification remains pending because the local `bash` tool still fails before executing commands with `The "path" argument must be of type string. Received undefined`.

# 2026-06-29 - Spec 080 UI wireframe build fix
