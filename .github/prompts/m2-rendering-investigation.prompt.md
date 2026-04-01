---
description: "Systematic investigation to fix M2 model rendering in MdxViewer. Combines live Ghidra reverse-engineering of a running 3.3.5.12340 client with adapter-level validation and renderer parity work to make M2 doodads and game objects visible in the world scene."
name: "M2 Rendering Investigation — Ghidra + Adapter + Renderer"
argument-hint: "Optional: specific model path, symptom, or investigation phase to start from"
agent: "agent"
---

Fix invisible M2 models in `gillijimproject_refactor/src/MdxViewer` through a three-phase investigation: diagnostic triage, native-client reverse-engineering via Ghidra on a live 3.3.5.12340 sandbox, and targeted renderer fixes.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/memory-bank/data-paths.md`
4. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
5. `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
6. `.github/prompts/m2-material-parity-implementation-plan.prompt.md`
7. `.github/prompts/wow-viewer-m2-runtime-plan-set.prompt.md`

## The Problem

M2 models load successfully (MDX ok/fail shows 0 failures) but render as invisible geometry in the world scene. Only bounding box wireframes appear. This affects ALL M2 models systematically — the viewer displays terrain, WMOs render, but every M2 doodad and game object is invisible.

The models are wrapped through `WarcraftNetM2Adapter` (Warcraft.NET M2+skin → `MdxFile` runtime format) and rendered via `ModelRenderer`. The adapter produces `MdxFile` objects with geosets, materials, textures, and vertex data. The renderer uploads those to OpenGL VAOs and draws them through a shared MDX/M2 shader.

## Why Previous Fixes Failed

Multiple targeted fixes have been applied and none restored M2 visibility:

1. **Bone skinning suppression** — gated bone matrix upload with `!_isM2AdapterModel` so the shader's bone-skinning path (`boneTransform = mat4(0.0) + sum of uBones[idx] * weight`) uses identity instead of unvalidated matrices. M2 models remained invisible.

2. **Animator creation suppression** — gated `new MdxAnimator(mdx)` with `!isM2AdapterModel` so animation-track-driven alpha evaluation (geoset animation alpha, layer transparency, layer color) falls back to static defaults instead of producing zero values from newly-parsed keyframe data. M2 models remained invisible.

3. **The `GetArrayReferenceElements` IEnumerable fix** (prior session) caused `HasAnimationData()` to return true for M2 models for the first time, triggering the animator and bone-skinning issues above. With both now suppressed, M2 models should theoretically return to the pre-fix visibility baseline — but they don't.

This pattern suggests the rendering failure is upstream of animation and bone skinning entirely: either the adapter is producing bad vertex/index data, or the renderer's material/texture/blend pipeline is misconfigured for M2-sourced geosets, or both.

## Known Candidate Blockers

From systematic code audit:

### 1. Index Validation Deleting Geosets (HIGH — likely silent killer)

`ModelRenderer.InitBuffers` validates that `maxIndex < vertCount` for every geoset and **deletes the entire VAO** if any index is out of range. The adapter uses `geoset.Indices.Add(mappedIndex)` where `mappedIndex` is the adapter-local index from `TryGetVertex`. If the adapter's vertex-to-index mapping is producing indices that exceed the actual vertex count in the geoset, every geoset is silently destroyed.

The log message `[ModelRenderer] Geoset {i} skipped: index out of range` would appear in the console but could be missed in the log volume from 3000+ models loading.

**File**: `ModelRenderer.cs` ~line 1400
**File**: `WarcraftNetM2Adapter.cs` ~line 560 (`BuildGeosets` index mapping)

### 2. Empty Geosets from Adapter Processing Failures

If `TryGetVertex` returns false for all vertices in a skin submesh, the geoset ends up with 0 vertices and is filtered out in `InitBuffers` before VAO creation. No geometry, no rendering, no log message.

**File**: `WarcraftNetM2Adapter.cs` ~line 580 (`TryGetVertex`)

### 3. Texture Resolution Failures with Fallback Suppression

For non-301-profile M2 models (confirmed: 3.3.5 models are NOT 301 profile), missing textures should still produce at minimum magenta fallback geometry on layer 0, or white fallback geometry when no layers render. If ALL geosets have `MaterialId = -1` (possible when `materialCount = 0`), the fallback path renders white geometry — which should be visible.

This blocker alone cannot explain complete invisibility unless it combines with #1 or #2.

### 4. Warcraft.NET Limitations with Pre-Legion Assets

Warcraft.NET was designed primarily for modern (Legion+) WoW formats. Its M2/skin parsing for 3.3.5 (Wrath) assets may have subtle structural mismatches:
- Vertex format differences between Wrath and modern M2
- Skin section/submesh layout differences
- Bone combo and bone lookup table interpretation
- Texture type/flag mapping
- Material/render flag semantics

These mismatches could cause the adapter to produce structurally valid but semantically wrong runtime data — geosets that pass validation but render wrong geometry, or correct geometry with wrong winding, or geometry at wrong positions.

## Investigation Phases

### Phase 1: Diagnostic Triage (no Ghidra needed)

**Goal**: Determine EXACTLY which blocker is killing M2 visibility right now.

1. **Add diagnostic counters to `ModelRenderer` constructor** after `InitBuffers` completes:
   - Count total geosets vs geosets with valid VAOs
   - Count geosets rejected by index validation
   - Count geosets skipped as empty
   - Log these counts for every M2 model: `[M2-DIAG] {modelPath}: {totalGeosets} geosets, {validVaos} valid, {indexRejects} index-rejected, {emptySkips} empty-skipped`

2. **Add diagnostic counters to `WarcraftNetM2Adapter.BuildGeosets`**:
   - Count vertices produced per geoset
   - Count indices produced per geoset
   - Count `TryGetVertex` failures per geoset
   - Count index-out-of-bounds skips per geoset
   - Log: `[M2-ADAPT] {modelPath} geoset {i}: {verts} verts, {indices} indices, {vertFails} vert-fails, {indexSkips} index-skips`

3. **Validate one model end-to-end**: Pick a specific model visible in the tooltip (e.g., `AZJOL_ROOF84.M2`), trace it from MPQ load through adapter through renderer, and verify every number.

4. **Check the console/log output** for existing `[ModelRenderer] Geoset skipped` messages that may already be present but buried.

**Deliverables**: Exact counts showing where M2 geometry is being lost.

### Phase 2: Ghidra Native-Client Investigation

**Goal**: Understand exactly how the native 3.3.5.12340 client renders M2 models, using a live sandbox server with the development map loaded.

#### Setup

- Run a 3.3.5.12340 sandbox server with the target map
- Launch the 3.3.5.12340 game client connected to the sandbox
- Navigate to an area with visible M2 doodads/game objects
- Attach Ghidra to the running client process

#### Investigation Targets

**2a. Vertex Buffer Layout**

- Find the native M2 vertex upload path (the code that builds GPU vertex buffers from parsed M2/skin data)
- Document the exact vertex stride, attribute layout, and data types
- Compare against what `WarcraftNetM2Adapter.AddVertexToGeoset` produces
- Verify: position components, normal components, UV components, bone index encoding, bone weight encoding
- Key question: does the native client use the same vertex format for all M2 models, or does it vary by model version/flags?

**2b. Index Buffer Construction**

- Find how the native client builds index buffers from skin submesh data
- Document the exact index remapping: skin vertex indices → global vertex indices → GPU index values
- Compare against `WarcraftNetM2Adapter.BuildGeosets` index mapping
- Key question: is our `mappedIndex` calculation correct? Do we handle the skin vertex offset correctly?

**2c. Render State Per Material**

- Find the M2 material/render-flag application path
- Document what GL/D3D state is set for each blend mode, render flag combination
- Confirm the native combiner/effect system documented in `m2-native-client-research-2026-03-31.md` findings #8
- Map native render states to our `ModelRenderer.RenderGeosets` blend/depth/cull configuration
- Key question: are there render states we're not setting that would cause geometry to be invisible (e.g., wrong face culling, wrong depth function)?

**2d. Texture Binding**

- Find how the native client resolves M2 texture references (inline path, replaceable ID, skin texture lookup)
- Document the texture unit assignment for multi-layer materials
- Key question: are there texture type codes or flags in 3.3.5 M2 that Warcraft.NET doesn't expose or misinterprets?

**2e. Skin Section → Submesh → Draw Call Mapping**

- Find how the native client walks skin sections to produce draw calls
- Document: which sections are drawn, which are skipped, how section sorting works
- Verify our adapter's section-to-geoset mapping matches native behavior
- Key question: does the native client use bone combo indices differently than our adapter?

**2f. Geoset Visibility Determination**

- Find the native geoset visibility logic (equivalent of our `UpdateGeosetAnimationAlpha` and geoset ID filtering)
- Document: what determines whether a geoset is drawn at frame 0 with no animation running
- Key question: do M2 geosets at animation frame 0 default to visible or to invisible in the native client? (Our suppressed animator was producing alpha=0, suggesting the keyframe data starts at transparent)

#### Ghidra Entry Points

The existing research in `m2-native-client-research-2026-03-31.md` provides these confirmed native functions as starting points:

| Function | Purpose | Relevance |
|----------|---------|-----------|
| `FUN_00957ca0` | Model cache entry, .mdl/.mdx/.m2 normalization | Model identity |
| `FUN_00984b00` | Model bootstrap, MD20 validation | Parse entry point |
| `FUN_00983970` | Skin file loading (%02d.skin) | Skin loading |
| `FUN_00983e40` | Skin parse completion, instance rebuild | Skin → render state |
| `FUN_009833d0` | Skin-profile init, section remap | Section → draw call mapping |
| `FUN_00a03be8` (PPC) | Effect/combiner synthesis | Material → shader routing |
| `FUN_009e7d08` (PPC) | Model-local lighting/diffuse/emissive | Per-model render state |
| `FUN_009fe2c4` (PPC) | Doodad batch submission | Scene render path |
| `FUN_009ff5a4` (PPC) | Scene submission coordinator | Render family dispatch |

For the Win32 3.3.5.12340 binary specifically, the OS X and PPC function addresses above won't directly translate, but the same code structures, strings, and patterns should be findable through:
- String references: `%02d.skin`, `MD20`, `CM2Model`, `CM2Shared`, `M2UseZFill`, `Diffuse_T1`, `Combiners_Opaque`
- The model cache hash function pattern from FUN_00957ca0
- The vertex upload path near the GX/D3D buffer creation calls

### Phase 3: Adapter Truthing

**Goal**: Validate or rebuild the `WarcraftNetM2Adapter` output against native-client ground truth from Phase 2.

Based on Phase 2 findings, verify or fix these adapter paths:

1. **Vertex format correctness**
   - Position: coordinate system, component order, scale
   - Normal: normalization, coordinate system
   - UV: component order, V-flip convention, UV set assignment
   - Bone indices: encoding (float cast of byte index), range validation
   - Bone weights: encoding (byte/255.0 normalization), sum validation

2. **Index buffer correctness**
   - Skin vertex index → model vertex index mapping
   - Submesh start/count offset handling
   - Index range vs vertex count consistency (the suspected silent killer)

3. **Material/texture mapping**
   - M2 render flags → MdlMaterial flag mapping
   - M2 blend mode → MdlLayerFilterMode mapping
   - Texture type → texture path resolution
   - Texture unit assignment for multi-texture materials

4. **Section/submesh layout**
   - Skin section → geoset mapping completeness
   - Bone combo / bone lookup usage
   - Section filtering (which sections should produce draw calls)

### Phase 4: Renderer Parity

**Goal**: Make `ModelRenderer` render M2-sourced geosets correctly.

Based on Phase 2 and 3 findings, fix the renderer:

1. **Render state corrections** — face culling, depth test, depth write, blend function per M2 material type
2. **Texture pipeline** — correct texture unit binding, sampler state, mip filtering for M2 textures
3. **Alpha and transparency** — correct default alpha for M2 geosets at animation frame 0 (not zero, not one — whatever the native client uses)
4. **Effect/combiner routing** — implement at least the major native combiner families (`Combiners_Opaque`, `Combiners_Mod`, `Combiners_Add`, `Combiners_Mod2x`) instead of treating all M2 materials as one flat path

### Phase 5: Validation

**Goal**: Confirm M2 models render in the viewer matching native-client appearance.

1. Load the development map in both the native client (via sandbox) and MdxViewer
2. Navigate to the same camera position
3. Compare:
   - M2 model visibility (are the same models visible?)
   - Approximate geometry shape (right mesh, not collapsed/inverted?)
   - Approximate material appearance (textured vs magenta vs invisible)
   - Transparency and blend behavior
4. Document any remaining differences as follow-up work items

## Out of Scope

- Full animation parity (bone skinning, keyframe evaluation, particle emitters). Animation is a separate slice once static M2 rendering is working.
- wow-viewer library extraction. This investigation targets `MdxViewer` directly. Findings should be recorded in `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md` for future library work.
- WMO rendering issues (WMOs appear to render already).
- Terrain rendering (already working).
- Pre-3.0.1 M2 format support (that's a different prompt).
- Performance optimization (batching, culling). Get it visible first.

## Files Likely To Change

| File | Purpose |
|------|---------|
| `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs` | M2+skin → MdxFile adaptation |
| `src/MdxViewer/Rendering/ModelRenderer.cs` | OpenGL rendering, shader, material state |
| `src/MdxViewer/Terrain/WorldAssetManager.cs` | M2 loading, skin resolution |
| `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md` | Native findings |

## Files For Reference Only

| File | Purpose |
|------|---------|
| `_recovery_343dadf/gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs` | Pre-regression baseline |
| `_recovery_343dadf/gillijimproject_refactor/src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs` | Pre-regression adapter |

## Warcraft.NET Limitation Context

Warcraft.NET was designed for Legion+ (7.x+) WoW assets. Pre-Legion formats differ in:
- M2 header version and field layout
- Skin file section structure
- Vertex format (no second UV set in early versions, different bone weight encoding)
- Texture type codes and flag meanings
- Material/render flag semantics
- Animation track encoding (fixed-point vs float, different compression)

The adapter currently papers over these differences through Warcraft.NET's generic M2 parser, which may produce structurally valid but semantically wrong data for 3.3.5 assets. Phase 2 (Ghidra) will establish ground truth; Phase 3 will validate or fix the adapter against that truth.

The goal is not to replace Warcraft.NET but to understand exactly where its 3.3.5 output diverges from native-client expectations and compensate in the adapter layer.

## Non-Negotiable Constraints

- Do not claim a fix based on build success. The fix is confirmed only when M2 models render visible geometry in the world scene.
- Do not suppress rendering failures with broader fallbacks. Find and fix the actual data or pipeline issue.
- Do not treat Warcraft.NET's parse output as ground truth for 3.3.5 semantics. Treat it as one input to be validated against native behavior.
- Record all Ghidra findings in `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md` so they survive across sessions.
- Keep the investigation focused on visibility first. Material fidelity, animation, and performance are follow-up work.

## Success Criteria

M2 doodads and game objects render visible, textured geometry in the MdxViewer world scene when viewing a 3.3.5 map. The geometry should be approximately correct (right shape, right position, right texture) but does not need to be pixel-perfect against the native client.

## First Output

Start with:

1. Which investigation phase you are executing
2. What specific diagnostic or Ghidra target you are attacking first
3. What you expect to find based on the known candidate blockers
4. What you are explicitly not claiming yet
