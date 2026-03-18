# Active Context — MdxViewer / AlphaWoW Viewer

## Current Focus: Recovery On v0.4.0 Baseline (Mar 17, 2026)

MdxViewer work has been reset to a v0.4.0-based branch in the main workspace tree.

- Branch: recovery/v0.4.0-surgical-main-tree
- Base commit: 343dadf (tag v0.4.0)
- .github instructions/skills/prompts restored from main and committed (845748b)

### Terrain Decode Direction (Current)

- Priority is profile-correct alpha decode behavior before broader feature intake.
- FormatProfileRegistry now carries terrain alpha decode mode per ADT profile.
- StandardTerrainAdapter alpha extraction routes by profile mode:
   - 3.x strict path
   - 0.x legacy sequential path
- Keep terrain renderer topology/shader rewrites out until decode stability is verified.

### Next Steps

1. Validate runtime terrain alpha output with real data on Alpha-era and LK 3.3.5.
2. Continue surgical intake from v0.4.0..main with SAFE-first triage.
3. Keep UI evolution incremental (no drastic layout churn).
4. Bring import/export enhancements in small, build-gated batches after decode path stabilization.

### Current Intake Decision

- Commit queue triage for the current recovery pass:
   - `177f961`: RISKY, skip
   - `37f669c`: RISKY, skip
   - `d50cfe7`, `326e6f8`, `4e2f681`, `39799bf`, `62ecf64`: MIXED, extract only isolated safe slices
- First SAFE batch is limited to the corrected alpha-atlas helper from `62ecf64`.
- Do not pull the earlier `d50cfe7` atlas helper version; it bakes in the old 63->62 edge remap during import/export.
- Do not pull ViewerApp, TerrainRenderer, terrain decode heuristic, or test-project changes in this first batch.
- First SAFE batch has now been applied and the MdxViewer solution build passed.
- Runtime real-data validation is still required before treating the helper as terrain-safe in practice.

### Rendering Recovery Follow-up (Mar 18)

- Main-branch renderer residency fix is now applied in `WorldAssetManager`:
   - do not evict live MDX/WMO renderers by default
   - keep only raw file bytes under LRU pressure
   - retry failed cached model loads instead of pinning permanent nulls
- Minimal skybox support is now present:
   - `WorldScene` classifies skybox-like MDX/M2 placements separately
   - nearest skybox renders as a camera-anchored backdrop before terrain
   - `ModelRenderer.RenderBackdrop(...)` forces no depth test/write for all layers
- Reflective M2 bugfixes were already present on this branch before this batch:
   - no inferred `NoDepthTest` / `NoDepthSet` from unstable Warcraft.NET render flags
   - guarded env-map backface handling in the model shader path
- Build passed again after the rendering batch.
- Runtime verification still required for doodad reload/culling, skybox behavior, and LK MH2O liquids.

### MCCV + MPQ Follow-up (Mar 18)

- Active chunk-based terrain rendering now includes MCCV vertex colors again.
- Implementation path is intentionally minimal:
   - `StandardTerrainAdapter` extracts `MccvData`
   - `TerrainChunkData` stores per-vertex MCCV bytes
   - `TerrainMeshBuilder` uploads RGBA as a new vertex attribute
   - `TerrainRenderer` applies the tint in shader
- Runtime follow-up corrected the semantics further:
   - MCCV bytes are now interpreted as BGRA, not RGBA
   - neutral/no-tint values are treated as mid-gray (`127`) instead of white
   - terrain tint is now derived from RGB remapped around mid-gray; MCCV alpha is preserved but not used as terrain tint strength
- `NativeMpqService` also now carries the isolated patch-reader recovery slice needed for 1.x+ patched clients and later encrypted entries.
- `NativeMpqService.LoadArchives(...)` now also scans recursively so map content in nested/custom `patch-[A-Z].mpq` archives is not skipped during archive discovery.
- Both the converter core project and the MdxViewer solution build passed after this batch.
- Real-data validation is still pending for MCCV appearance and patched MPQ chains.

### 3.x Terrain Alpha Follow-up (Mar 18)

- The incorrect offset-0 LK alpha fallback experiment was reverted after runtime validation showed it was wrong.
- Current terrain recovery direction is now explicitly profile-driven instead of heuristic-driven:
   - 3.0.1 / 3.3.5 ADT profiles treat MPHD `0x4 | 0x80` as the big-alpha mask
   - `Mcal` decode now distinguishes compressed alpha, 8-bit big alpha, and legacy 4-bit alpha while respecting the MCNK do-not-fix-alpha bit
- Build validation passed after this batch, including the alternate-output MdxViewer build used while the live viewer holds `bin/Debug` locks.
- Runtime validation follow-up is now positive on the user's real data:
   - the tested 3.0.1 alpha-build terrain now renders correctly on this path
   - the same recovery line also preserves Alpha 0.5.3 terrain after restoring the legacy edge fix in `AlphaTerrainAdapter`
- Keep broader claims narrow: this is strong evidence that the profile split is correct for the tested samples, not blanket proof for every later-era terrain dataset.

### 3.x Terrain Guardrail Update (Mar 18)

- User direction is now explicit: do not use `*_tex0.adt` split terrain sourcing in the active viewer path for current 3.x alpha recovery work.
- Active viewer profiles for `3.0.1`, `3.3.5`, and unknown `3.0.x` no longer opt into `_tex0` terrain layer/alpha sourcing.
- `StandardTerrainAdapter` also now avoids opening `_tex0` files unless a future profile explicitly re-enables that path.
- The temporary rollback of `MCNK.SizeMcal` / `SizeMcsh` trust caused a major runtime regression and was reverted immediately; the active viewer path still uses the prior 3.x header-size behavior.
- Follow-up parser guardrail: `Mcnk.ScanSubchunks(...)` now treats `MCNK.SizeMcal` / `SizeMcsh` as an optional extension of the declared MCAL/MCSH payload, never a reason to advance less than the declared subchunk size. This avoids landing the FourCC scan inside MCAL/MCSH payload bytes when header sizes are smaller than the chunk-declared span.
- Build validation passed after this parser fix:
   - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`
   - `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
- This is a guardrail rollback only. Runtime validation is still required for the remaining chunk-skip / decode-loss issue on 3.x terrain.

### 4.x / 5.x Terrain Profile Direction (Mar 18)

- Keep `_tex0.adt` and `_obj0.adt` parsing as a separate 4.x+/5.x concern, not part of the active 3.x recovery path.
- `FormatProfileRegistry` now has separate provisional `4.x` and `5.x` ADT profiles that opt into split texture and placement sourcing.
- `StandardTerrainAdapter` now routes placement parsing through `_obj0.adt` only when the resolved terrain profile explicitly requests it; 3.x remains on root-ADT placement parsing.
- This is profile scaffolding, not full Cataclysm/MoP correctness. The user requirement is broader MPQ-era support through `5.3.x`; later CASC support is a separate future track.

### ModelRenderer Follow-up From 39799bf (Mar 18)

- The commit message for `39799bf` bundled terrain and model notes together, but the only remaining model-renderer hunk on top of the already-applied MPQ fix was particle suppression on the world-scene instanced path.
- That hunk is now applied:
   - batched placed-model rendering skips particles
   - standalone model preview/rendering still allows particles
- Keep this split until particle simulation becomes instance-aware.

### World Wireframe Reveal Follow-up (Mar 18)

- World-scene wireframe toggle is now hover-driven instead of a blanket terrain-only toggle:
   - `WorldScene.ToggleWireframe()` now keeps terrain wireframe in sync while also enabling a hover reveal mode for placed WMOs and MDX/M2 doodads
   - ViewerApp refreshes the reveal set every frame from the current scene-viewport mouse position
   - hovered objects render an extra wireframe overlay pass without changing standalone model-viewer wireframe behavior
- Current validation status:
   - alternate-OutDir `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"` passed after restoring terrain wireframe and switching the hover test from a loose ray/AABB heuristic to a screen-space brush
   - `WorldAssetManager` world-model loading now resolves the canonical model path before M2 skin lookup so `.mdx` aliases that actually resolve to `MD20` roots can search for skins relative to the real asset path
   - runtime visual validation is still pending for reveal radius feel and for confirming the remaining world-scene M2 load failures are actually cleared on user data

### M2 Adapter Follow-up (Mar 18)

- `WarcraftNetM2Adapter` now treats raw `MD20` as the primary parse path instead of only using direct `MD21` parsing as a fallback after the Warcraft.NET `Model(...)` wrapper fails.
- Current rationale:
   - the user's active client data is dominated by raw `MD20` roots, not chunked `MD21` containers
   - relying on the wrapper first made the effective parse path sporadic across assets
- Build-only validation passed again on the alternate-OutDir MdxViewer solution build.
- Runtime confirmation is still required for the remaining sporadic world-scene M2 failures.

### World Load Performance Follow-up (Mar 18)

- Northrend load-time investigation confirmed AOI terrain streaming was already the default; the bigger stall was world-object asset loading on tile arrival and first render.
- `WorldScene` no longer eagerly calls blocking `EnsureMdxLoaded` / `EnsureWmoLoaded` for streamed tiles or external spawns.
- `WorldAssetManager` now has deferred MDX/WMO load queues plus a bounded per-frame `ProcessPendingLoads(...)` path.
- `WorldScene.Render(...)` now processes a small per-frame asset budget and only uses loaded renderers in render paths, queueing missing assets instead of force-loading them on the render thread.
- Instance bounds are refreshed after queued model loads complete so culling can converge from temporary fallback bounds to real model bounds.
- Follow-up asset-read recovery after runtime queue investigation:
   - the UI queue counter now reports unique pending assets instead of raw queue-node count
   - repeated `PrioritizeMdxLoad` / `PrioritizeWmoLoad` calls no longer flood the priority queues with duplicate entries every frame
   - `MpqDataSource` now builds file-path and extension indexes once at startup instead of re-filtering the full file list for repeated model/skin lookups
   - `MpqDataSource.ReadFile(...)` now has a bounded global raw-byte LRU cache so repeated model and texture reads reuse already-read archive data instead of hitting MPQ/loose-file resolution again
   - `WorldAssetManager` skin selection now caches best `.skin` matches per resolved model path instead of rescanning the `.skin` file list on retries
   - `MpqDataSource` now also has a bounded background prefetch path with separate read-only `NativeMpqService` workers so queued model bytes can be warmed into the shared raw-byte cache without sharing the primary archive reader across threads
   - `WorldAssetManager` now triggers that prefetch when new MDX/WMO assets are queued, including common extension aliases and M2 skin candidates
- Build validation passed after this change using the alternate output path:
   - `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
- No runtime real-data validation has been performed yet for the new loading behavior. Do not claim the Northrend load regression is fixed until startup responsiveness and in-world streaming are checked on real data.
- Parallel MPQ archive reads are now limited to background raw-byte warmup only:
   - GL renderer/material creation remains main-thread work in the current pipeline
   - the primary `MpqDataSource` reader is still not shared across threads; worker threads use separate `NativeMpqService` instances
   - runtime profiling is still required before increasing worker count or pushing texture/material construction off the main thread

### World-Scene M2 Render Follow-up (Mar 18)

- User runtime feedback after the deferred-load change: world M2 doodads appeared to load but remained invisible.
- Current mitigation is targeted, not a full rollback:
   - `MdxRenderer` now tracks whether it was built through the Warcraft.NET M2 adapter
   - `WorldScene` keeps the lighter batched `RenderInstance(...)` path for classic MDX models
   - M2-adapted world doodads now use the proven per-instance `RenderWithTransform(...)` path instead of the batched path
- Rationale:
   - standalone model viewing and WMO doodad rendering already rely on `RenderWithTransform(...)`
   - the invisible-M2 symptom is therefore more likely a world-scene batch-path issue than an asset-read failure
- Build validation passed after this mitigation using the alternate output path.
- Runtime real-data validation is still required to confirm M2 doodads are visible again and to measure whether the selective fallback has an acceptable frame-time cost.

### World-Scene M2 Conversion Follow-up (Mar 18)

- Historical diff review showed the stronger world-side M2 recovery path lives in `main` / `4e9237a`, not in `177f961` alone.
- `WorldAssetManager` now prefers `M2ToMdxConverter` for raw `MD20` world doodads before falling back to `WarcraftNetM2Adapter`.
- `ModelRenderer` also now disables the classic layer-0 `Transparent` hard alpha-cutout heuristic for M2-derived models so their materials follow the blended path used by the working mainline M2 support.
- Latest parity correction versus final `main` commit `62ecf64`:
   - old `main` branch world M2 behavior was simpler than this recovery branch briefly became:
      - direct `M2 + .skin` adaptation was the first-choice world load path
      - world doodads then rendered through the normal `RenderInstance(...)` path with no M2-specific world-scene split
   - recovery branch is now back on that shape:
      - direct Warcraft.NET adaptation is tried first for world M2s
      - byte-level `M2ToMdxConverter` conversion is now only a fallback after adapter failure
      - world-scene rendering no longer special-cases M2-adapted doodads into `RenderWithTransform(...)`; all loaded world doodads use the normal instanced world path again
- Deferred world-model loading now preserves the older retry semantics for failed entries:
   - queued MDX/WMO loads only short-circuit when a non-null renderer is already cached
   - queued `null` entries are allowed back through `ProcessPendingLoads(...)` for retry instead of becoming permanent invisible instances
   - `.mdx` and `.m2` aliases are now both considered during direct reads and file-set resolution so LK-era model-extension mismatches have an exact-path fallback before basename heuristics
- Build-only validation passed after these changes using the alternate output path:
   - `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
- No automated tests were added or run for this slice.
- Runtime real-data validation is still the blocker:
   - confirm Northrend or NorthrendBG now shows nonzero MDX/M2 world-object load/render stats
   - confirm the converted M2 path does not regress frame time or material appearance

### WMO Doodad M2 Loader Follow-up (Mar 18)

- Remaining parity gap after the world-scene fixes: `WmoRenderer` doodad-set loading was still on an older MDX-only path.
- Concrete issue:
   - `GetOrLoadDoodadModel(...)` only did raw `MdxFile.Load(...)` after a direct file read
   - it never attempted direct `.m2` / `MD20` / `MD21` adaptation with companion `.skin`
   - it also round-tripped raw bytes through a shared cache filename, which could collide on duplicate doodad basenames across different directories
- Current fix now mirrors the shared world/standalone behavior more closely:
   - `WmoRenderer` resolves canonical doodad paths through the file set before loading
   - WMO doodad M2s now try Warcraft.NET adapter + `.skin` first
   - raw `MD20` doodads then fall back to `M2ToMdxConverter` only after adapter failure
   - non-M2 doodads now load from in-memory streams instead of cache-file writes
   - adapted and converted M2 renderers are explicitly marked as M2-derived so `ModelRenderer` keeps them on the non-cutout transparent-material path
- Same M2-derived renderer flag is now also applied in `WorldAssetManager` and standalone `ViewerApp.LoadM2FromBytes(...)`.
- Build validation passed after this change using the alternate output path:
   - `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
- Runtime real-data validation is still required:
   - confirm WMO doodad sets now populate visible M2s instead of just the WMO shell
   - confirm world doodads also recover with the restored shared M2 load path

### MPQ Listfile Recovery Follow-up (Mar 18)

- Root-cause follow-up for the latest standalone M2 `.skin` failure:
   - `ViewerApp` UI text already claimed the community listfile was auto-downloaded
   - actual `Open Game Folder` flow still passed `null` into `LoadMpqDataSource(...)`, so `MpqDataSource` never received any external listfile unless one was supplied manually
- Current fix:
   - `ViewerApp.LoadMpqDataSource(...)` now resolves the listfile path before constructing `MpqDataSource`
   - resolution order is: explicit path, bundled repo/runtime `community-listfile-withcapitals.csv`, then cached/downloaded `ListfileDownloader` path
   - if none are available, viewer now logs that it is falling back to archive-internal names only
- Why this matters:
   - many MPQ internal listfiles do not expose `.skin` entries even when `.m2` entries are present
   - without the external listfile, companion `.skin` discovery can fail and surface as `Missing companion .skin for M2`
- Build-only validation passed after this fix using the alternate output path:
   - `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
- Runtime real-data validation is still required to confirm standalone M2 loading and world/WMO M2 recovery on the user's client data.

## Current Focus

**v0.4.0 Release — 0.5.3 Rendering Improvements + Initial 3.3.5 Groundwork** — Major rendering improvements for Alpha 0.5.3 (lighting, particles, geoset animations). Initial 3.3.5 WotLK support scaffolding added but **NOT ready for use** — MH2O liquid and terrain texturing are broken. Only client versions 0.5.3 through 0.12 are currently usable.

## 3.3.5 WotLK Status: IN PROGRESS (NOT USABLE)

**Known broken:**
- MH2O liquid rendering — parsing exists but rendering is broken
- Terrain texturing — alpha map decode not working correctly for LK format
- These must be fixed before 3.3.5 data can be used

## Immediate Next Steps

1. **Fix 3.3.5 MH2O liquid rendering** — Parsing exists but output is broken
2. **Fix 3.3.5 terrain texturing** — Alpha map decode for LK format not working
3. **3.3.5 terrain alpha maps** — Current LK path uses basic Mcal decode; needs full `AlphaMapService` integration without breaking 0.5.3
4. **Light.dbc / LightData.dbc integration** — Replace hardcoded TerrainLighting values with real game lighting data per zone
5. **Skybox rendering** — Minimal backdrop routing is now implemented; real-data runtime verification is still pending
6. **Ribbon emitters (RIBB)** — Parsed but no rendering code yet
7. **M2 particle emitters** — WarcraftNetM2Adapter doesn't map PRE2/particles to MdxFile format yet

## Session 2026-02-13 Summary — WDL/WL/WMO Fixes

### Completed

1. **WDL parser correctness**
   - Strict chunk parsing (`MVER`/`MAOF`/`MARE`) with version `0x12` validation
   - Proper `MARE` chunk header handling before height reads

2. **WDL terrain scale + overlay behavior improvements**
   - WDL cell size corrected to `WoWConstants.TileSize` (8533.3333), not chunk size
   - Existing ADT-loaded tiles hidden from WDL at load-time
   - Polygon offset added to reduce z-fighting with real terrain
   - UI toggle added to fully disable WDL rendering for testing

3. **WDL preview reliability**
   - `.wdl.mpq` fallback path and error propagation (`LastError`)
   - Preview dialog now displays failure reason instead of closing silently

4. **WMO intermittent non-rendering fix**
   - Converted WMO main + liquid shader programs to shared static programs with ref-counted lifetime
   - Prevents per-instance shader deletion race (same class of bug previously fixed in MDX renderer)

5. **WL liquids transform tooling**
   - Replaced hardcoded axis swap with configurable matrix transform (rotation + translation)
   - Added `WL Transform Tuning` controls in UI and `Apply + Reload WL`
   - Added `WorldScene.ReloadWlLiquids()` for fast iteration

## MDX Particle System — IMPLEMENTED (2026-02-15)

Previously deferred issue now resolved. ParticleRenderer rewritten with per-particle uniforms, texture atlas support, and per-emitter blend modes. Wired into MdxRenderer — emitters created from PRE2 data, updated each frame with bone-following transforms, rendered during transparent pass. Fire, glow, and spell effects now visible.

## Session 2026-02-15 Summary — Multi-Version Support + Lighting/Particle Overhaul

### Completed

1. **Partial WotLK 3.3.5 terrain scaffolding** (StandardTerrainAdapter) — **NOT USABLE**
   - Split ADT file loading via MPQ data source
   - MPHD flags detection for `bigAlpha` (0x4)
   - MH2O liquid chunk parsing — **BROKEN, not rendering correctly**
   - LK alpha maps via `hasLkFlags` detection — **texturing BROKEN**
   - Surgical revert of shared renderer code was needed to restore 0.5.3

2. **M2 (MD20) model loading** (WarcraftNetM2Adapter)
   - Converts MD20 format models to MdxFile runtime format
   - Maps render flags (Unshaded, Unfogged, TwoSided), blend modes
   - Texture loading from M2 texture definitions
   - Bone/animation data mapping

3. **Terrain regression fix** (surgical revert)
   - Commit e172907 broke 0.5.3 terrain rendering (grid pattern artifacts)
   - Root cause: `AlphaTextures.ContainsKey` guard skipping overlay layers + edge fix removal in TerrainRenderer.cs
   - Plus StandardTerrainAdapter ExtractAlphaMaps rewrite with broken `spanSuggestsPacked` logic
   - Surgical revert restored 0.5.3 terrain while preserving M2/WMO improvements

4. **Lighting improvements** (TerrainLighting, ModelRenderer, WmoRenderer)
   - Raised ambient values: day (0.4→0.55), night (0.08→0.25) — no more pitch black
   - Half-Lambert diffuse shading: `dot * 0.5 + 0.5` squared — wraps light around surfaces
   - WMO shader: replaced lossy scalar lighting `(r+g+b)/3.0` with proper `vec3` lighting
   - MDX shader: half-Lambert + reduced specular (0.3→0.15)
   - Moderated day directional light (1.0→0.8) to avoid blow-out with higher ambient

5. **Particle system wired into pipeline** (ParticleRenderer, ModelRenderer)
   - Rewrote ParticleRenderer: per-particle uniforms, texture atlas (rows×columns), per-emitter blend mode
   - MdxRenderer creates ParticleEmitter instances from MdxFile.ParticleEmitters2
   - Emitter transforms follow parent bone matrices when animated
   - Particles rendered during transparent pass after geosets
   - Supports Additive, Blend, Modulate, AlphaKey filter modes

6. **Geoset animation alpha** (ModelRenderer)
   - `UpdateGeosetAnimationAlpha()` evaluates ATSQ alpha keyframe tracks per frame
   - Alpha multiplied into layer alpha during RenderGeosets
   - Geosets with alpha ≈ 0 skipped entirely
   - Supports global sequences and linear interpolation

7. **WMO fixes from 3.3.5 work** (preserved)
   - Multi-MOTV/MOCV chunk handling for ICC-style WMOs
   - Strict WMO validation preventing Northrend loading hangs
   - WMO liquid rotation fixes

### Files Modified
- `TerrainRenderer.cs` — Reverted edge fix + ContainsKey guard
- `StandardTerrainAdapter.cs` — Reverted ExtractAlphaMaps to clean hasLkFlags path
- `TerrainLighting.cs` — Raised ambient/light values, better night visibility
- `ModelRenderer.cs` — Half-Lambert shader, particle wiring, geoset animation alpha
- `WmoRenderer.cs` — vec3 lighting instead of scalar, half-Lambert diffuse
- `ParticleRenderer.cs` — Complete rewrite with working per-particle rendering
- `WarcraftNetM2Adapter.cs` — MD20→MdxFile adapter (from e172907, preserved)
- `WorldAssetManager.cs` — MD20 detection + adapter routing (from e172907, preserved)

## Session 2026-02-13 Summary — MDX Animation System Complete

### Three Bugs Fixed

1. **KGRT Compressed Quaternion Parsing** (`MdxFile.cs`, `MdxTypes.cs`)
   - Rotation keys use `C4QuaternionCompressed` (8 bytes packed), not float4 (16 bytes)
   - Ghidra-verified decompression: 21-bit signed components, W reconstructed from unit norm
   - Added `C4QuaternionCompressed` struct with `Decompress()` method

2. **Animation Never Updated** (`ModelRenderer.cs`, `ViewerApp.cs`)
   - `ViewerApp` called `RenderWithTransform()` directly, bypassing `Render()` which was the only place `_animator.Update()` was called
   - Fix: Extracted `UpdateAnimation()` as public method, called from ViewerApp before render

3. **PIVT Chunk Order — All Pivots Were (0,0,0)** (`MdxFile.cs`)
   - PIVT chunk comes AFTER BONE in MDX files. Inline pivot assignment during `ReadBone()` found 0 pivots
   - Fix: Deferred pivot assignment in `MdxFile.Load()` after all chunks are parsed
   - This caused "horror movie" deformation — bones rotating around world origin instead of joints

### Terrain Animation Added (`WorldScene.cs`)
- Added `UpdateAnimation()` calls for all unique MDX renderers before opaque/transparent render passes
- Uses `HashSet<string>` to ensure each renderer is updated exactly once per frame

### Other Improvements
- `MdxAnimator`: `_objectIdToListIndex` dictionary replaces O(n) `IndexOf` calls
- `GNDX`/`MTGC` chunks now stored in `MdlGeoset` for vertex-to-bone skinning
- MATS values remapped from ObjectIds to bone list indices via dictionary lookup

### Key Architecture (MDX Animation)
- `MdxAnimator` — Evaluates bone hierarchy per-frame, stores matrices in `_boneMatrices[]` by list position
- `ModelRenderer.UpdateAnimation()` — Public method to advance animation clock
- `BuildBoneWeights()` — Converts GNDX/MTGC/MATS to 4-bone skinning format
- Bone transform: `T(-pivot) * S * R * T(pivot) * T(translation) * parentWorld`
- Shader: `uBones[128]` uniform array, vertex attributes for bone indices + weights

### Files Modified
- `MdxTypes.cs` — Added `C4QuaternionCompressed` struct
- `MdxFile.cs` — Fixed `ReadQuatTrack`, stored GNDX/MTGC, deferred pivot assignment
- `MdxAnimator.cs` — `_objectIdToListIndex` dict, cleaned diagnostics
- `ModelRenderer.cs` — Extracted `UpdateAnimation()`, ObjectId→listIndex remapping in `BuildBoneWeights`
- `ViewerApp.cs` — Added `mdxR.UpdateAnimation()` before standalone MDX render
- `WorldScene.cs` — Added per-frame animation update for unique MDX doodad renderers

## Session 2026-02-09 Summary

### WMO v16 Root File Loading Investigation
- **Symptom**: WMO v16 root files (e.g., `Big_Keep.wmo`) fail to load with "Failed to read" — group files load but without textures/lighting
- **Root cause chain**: `MpqDataSource.ReadFile` → `NativeMpqService.ReadFile` → `FindFileInArchive` succeeds → `ReadFileFromArchive` returns null
- **Block info**: offset=435912, size=318 (compressed), fileSize=472 (decompressed), flags=0x80000200 (EXISTS|COMPRESSED)
- **Decompression failure**: Compression type byte = `0x08` (PKWARE DCL), but remaining data has dictShift=0 (expected 4/5/6)
- **0.6.0 MPQ structure**: All files in standard MPQ archives (`wmo.MPQ`, `terrain.MPQ`, etc.) — NOT loose files, NOT per-asset `.ext.MPQ` wrappers

### Key Findings About 0.6.0 MPQs
- 11 MPQ archives: base, dbc, fonts, interface, misc, model, sound, speech, terrain(2331), texture(33520), wmo(4603)
- All have internal listfiles (56573 total files extracted)
- Zlib (0x02) works fine for large files (groups extract correctly)
- PKWARE DCL (0x08) fails for small files (root WMOs, possibly some ADTs)
- `FLAG_COMPRESSED (0x200)` = per-sector compression with type byte prefix
- `FLAG_IMPLODED (0x100)` = whole-file PKWARE without type byte (not seen in these archives)

### StormLib Reference Code Available
- `lib/StormLib/src/pklib/explode.c` — Complete PKWARE DCL explode implementation
- `lib/StormLib/src/pklib/pklib.h` — Data structures (`TDcmpStruct`, lookup tables)
- `lib/StormLib/src/SCompression.cpp` — Decompression dispatch (`Decompress_PKLIB`, `SCompDecompress`)
- Key: `explode()` reads bytes 0,1 as ctype/dsize_bits, byte 2 as initial bit buffer, position starts at 3

### WMO Liquid Rendering Added
- MLIQ chunk now parsed in `ParseMogp` sub-chunk switch
- `WmoRenderer` has liquid mesh building + semi-transparent water surface rendering
- Diagnostic logging added for failed material textures

### Ghidra RE Prompts Written
- `specifications/ghidra/prompt-053-mpq.md` — 0.5.3 MPQ implementation (HAS PDB — best starting point)
- `specifications/ghidra/prompt-060-mpq.md` — 0.6.0 MPQ decompression (no PDB, use string refs)

### Files Modified This Session
- `NativeMpqService.cs` — Added diagnostic logging throughout ReadFile/ReadFileFromArchive/ReadFileData/DecompressData
- `MpqDataSource.cs` — Added diagnostic logging to ReadFile and TryResolveLoosePath
- `WmoV14ToV17Converter.cs` — Added diagnostic logging to ParseWmoV14Internal
- `WmoRenderer.cs` — Added WMO liquid rendering, material texture diagnostics
- `PkwareExplode.cs` — New file, PKWARE DCL decompression (needs fixing — current impl fails)
- `AlphaMpqReader.cs` — Wired up PkwareExplode for 0x08 compression
- `StandardTerrainAdapter.cs` — Added ADT loading diagnostics

## Session 2026-02-08 (Late Evening) Summary

### Standard WDT+ADT Support
- **ITerrainAdapter interface** — New common contract for all terrain adapters
- **StandardTerrainAdapter** — Reads LK/Cata WDT (MAIN/MPHD) + split ADT files from MPQ via IDataSource
- **TerrainManager refactored** — Accepts `ITerrainAdapter` (was hardcoded to `AlphaTerrainAdapter`)
- **WorldScene refactored** — New constructor accepts pre-built `TerrainManager`
- **ViewerApp detection** — File size ≥64KB → Alpha WDT, <64KB → Standard WDT (requires MPQ data source)

### Format Specifications Written
- `specifications/alpha-053-terrain.md` — Definitive WDT/ADT/MCNK/MCVT/MCNR/MCLY/MCAL/MCSH/MDDF/MODF spec
- `specifications/alpha-053-coordinates.md` — Complete coordinate system documentation
- `specifications/unknowns.md` — 13 prioritized format unknowns needing Ghidra investigation

### Ghidra LLM Prompts Created
- `specifications/ghidra/prompt-053.md` — 0.5.3 (HAS PDB! Best starting point)
- `specifications/ghidra/prompt-055.md` — 0.5.5 (diff against 0.5.3)
- `specifications/ghidra/prompt-060.md` — 0.6.0 (transitional format detection)
- `specifications/ghidra/prompt-335.md` — 3.3.5 LK (reference build, well-documented)
- `specifications/ghidra/prompt-400.md` — 4.0.0 Cata (split ADT introduction)

### Converter Master Plan
- `memory-bank/converter_plan.md` — 4-phase plan: LK model reading → format conversion → PM4 tiles → unified project

## Session 2026-02-08 (Evening) Summary

### What Was Fixed

#### MCSH Shadow Blending (TerrainRenderer.cs)
- **Problem**: Shadow map (MCSH) was only applied on the base terrain layer. Alpha-blended overlay texture layers drawn on top would cover/wash out the shadows.
- **Root cause**: Both the C# render code and GLSL shader had `isBaseLayer` guards on shadow binding/application.
- **Fix**: Removed `isBaseLayer` condition from both:
  - C# `RenderChunkPass()`: Changed `bool hasShadow = isBaseLayer && chunk.ShadowTexture != 0` → `bool hasShadow = chunk.ShadowTexture != 0`
  - GLSL fragment shader: Changed `if (uShowShadowMap == 1 && uIsBaseLayer == 1 && uHasShadowMap == 1)` → `if (uShowShadowMap == 1 && uHasShadowMap == 1)`
- **Result**: Shadows now darken all texture layers consistently.

#### MDX Bounding Box Pivot Offset (WorldScene.cs, WorldAssetManager.cs)
- **Problem**: MDX model geometry is offset from origin (0,0,0). The MODL bounding box describes where geometry actually sits. MDDF placement position targets origin, but geometry center is elsewhere, causing models to appear displaced.
- **Fix**: Pre-translate geometry by negative bounding box center before scale/rotation/translation:
  - Added `WorldAssetManager.TryGetMdxPivotOffset()` — returns `(BoundsMin + BoundsMax) * 0.5f`
  - Transform chain: `pivotCorrection * mirrorX * scale * rotX * rotY * rotZ * translation`
  - `pivotCorrection = Matrix4x4.CreateTranslation(-pivot)`
  - Applied in both `BuildInstances()` and `OnTileLoaded()` in WorldScene.cs
- **WMO models**: Do NOT need pivot correction — their geometry is already correctly positioned relative to origin.

#### VLM Terrain Rendering (Previous session, 2026-02-08 afternoon)
- **GLSL shader em-dash**: Replaced unicode em-dash with ASCII hyphen in shader comment.
- **NullReferenceException**: Fixed null-conditional access in `DrawTerrainControls`.
- **VLM coordinate conversion**: Fixed `WorldPosition` in `VlmProjectLoader.cs` — swapped posX/posY, removed MapOrigin subtraction.
- **Minimap for VLM projects**: Refactored `DrawMinimap()` to work with either `_terrainManager` or `_vlmTerrainManager`. Added `IsTileLoaded()` to `VlmTerrainManager`.

#### Async Tile Streaming (TerrainManager.cs, VlmTerrainManager.cs)
- Both terrain managers now queue tile parsing to `ThreadPool` background threads.
- Parsed `TileLoadResult` objects enqueued to `ConcurrentQueue`.
- `SubmitPendingTiles()` runs on render thread each frame, uploading max 2 tiles/frame to avoid GPU stalls.
- `_disposed` flag prevents background threads from accessing disposed resources.

#### Thread Safety (VlmProjectLoader.cs, AlphaTerrainAdapter.cs, TerrainRenderer.cs)
- `TileTextures` → `ConcurrentDictionary` in both adapters.
- `_placementLock` protects dedup sets (`_seenMddfIds`, `_seenModfIds`) and placement lists in both adapters.
- `TerrainRenderer.AddChunks()` parameter widened from `Dictionary` to `IDictionary` to accept both.

#### VLM Dataset Generator (ViewerApp.cs)
- New menu item: `File > Generate VLM Dataset...`
- Dialog UI: client path (folder picker), map name, output dir, tile limit, progress log.
- Runs `VlmDatasetExporter.ExportMapAsync()` on `ThreadPool` with `IProgress<string>` feeding real-time log.
- "Open in Viewer" button after export completes.

### Key Technical Decisions
- **Coordinate system**: Renderer X = WoW Y, Renderer Y = WoW X, Z = height. MapOrigin = 17066.66666f, ChunkSize = 533.33333f.
- **MDX pivot**: Bounding box center, NOT PIVT chunk (PIVT is for per-bone skeletal animation pivots).
- **Shadow blending**: Apply to ALL layers, not just base. Overlay layers must also be darkened.
- **Thread safety**: `ConcurrentDictionary` for shared tile data, `lock` for placement dedup sets.

## What Works

| Feature | Status |
|---------|--------|
| Alpha WDT terrain rendering + AOI | ✅ |
| **Standard WDT+ADT terrain (WotLK 3.3.5)** | ✅ Partial — terrain + M2 models + WMO loading |
| Terrain MCSH shadow maps | ✅ (all layers, not just base) |
| Terrain alpha map debug view | ✅ (Show Alpha Masks toggle) |
| Async tile streaming | ✅ (background parse, render-thread GPU upload) |
| Standalone MDX rendering | ✅ (MirrorX, front-facing) |
| MDX skeletal animation | ✅ (standalone + terrain, compressed quats, GPU skinning) |
| MDX pivot offset correction | ✅ (bounding box center pre-translation) |
| MDX doodads in WorldScene | ✅ Position + animation + particles working |
| WMO v14 rendering + textures | ✅ (BLP per-batch) |
| WMO v17 rendering | ✅ Partial (groups + textures, multi-MOTV/MOCV) |
| M2 model rendering | ✅ MD20→MdxFile adapter (WarcraftNetM2Adapter) |
| Particle effects (PRE2) | ✅ Billboard quads, texture atlas, bone-following |
| Geoset animation alpha (ATSQ) | ✅ Per-frame keyframe evaluation |
| WMO rotation/facing in WorldScene | ✅ |
| WMO doodad sets | ✅ |
| MDDF/MODF placements | ✅ (position + pivot correct) |
| Bounding boxes | ✅ (actual MODF extents) |
| VLM terrain loading | ✅ (JSON dataset → renderer) |
| VLM minimap | ✅ |
| VLM dataset generator | ✅ (File > Generate VLM Dataset) |
| Live minimap + click-to-teleport | ✅ (WDT + VLM) |
| AreaPOI system | ✅ |
| GLB export (Z-up → Y-up) | ✅ |
| Object picking/selection | ✅ |
| Format specifications | ✅ (specifications/ folder) |
| WMO liquid rendering (MLIQ) | ✅ (semi-transparent water surfaces) |
| Object picking/selection | ✅ (ray-AABB, highlight, info) |
| Camera world coordinates | ✅ (WoW coords in status bar) |
| Left/right sidebar layout | ✅ (docked panels) |
| Ghidra RE prompts (5+2 versions) | ✅ (specifications/ghidra/) |
| 0.6.0 MPQ file extraction | ❌ PKWARE DCL (0x08) decompression fails |
| Half-Lambert lighting | ✅ Softer shading on MDX + WMO models |
| Improved ambient lighting | ✅ Day/night cycle with WoW-like brightness |

## Key Files

- `Terrain/WorldScene.cs` — Object instance building, pivot offset, rotation transforms, rendering loop
- `Terrain/WorldAssetManager.cs` — Model loading, bounding box/pivot queries
- `Terrain/AlphaTerrainAdapter.cs` — MDDF/MODF parsing, coordinate conversion, thread-safe placement dedup
- `Terrain/VlmProjectLoader.cs` — VLM JSON tile loading, thread-safe TileTextures/placements
- `Terrain/VlmTerrainManager.cs` — VLM terrain AOI, async streaming
- `Terrain/TerrainManager.cs` — WDT terrain AOI, async streaming
- `Terrain/TerrainRenderer.cs` — Terrain shader, shadow maps on all layers, alpha maps, debug views
- `Rendering/WmoRenderer.cs` — WMO geometry, textures, doodad sets
- `Rendering/ModelRenderer.cs` — MDX rendering, MirrorX, blend modes, textures
- `ViewerApp.cs` — Main app, UI, DBC loading, minimap, VLM export dialog
- `Export/GlbExporter.cs` — GLB export with Z-up → Y-up conversion

## Dependencies (all already integrated)

- `MdxLTool` — MDX file parser
- `WoWMapConverter.Core` → `gillijimproject-csharp` — Alpha WDT/ADT/MCNK parsers, WMO v14 parser, VLM dataset export
- `SereniaBLPLib` — BLP texture loading
- `Silk.NET` — OpenGL + windowing + input
- `ImGuiNET` — UI overlay
- `DBCD` — DBC database access
