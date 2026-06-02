# Research: M2 3.0.1 Renderer Performance

**Branch**: `038-m2-301-renderer-perf-research` | **Spec**: [spec.md](./spec.md)
**Ghidra binary**: `WoW.exe 3.0.1.8303` (32-bit, base `0x00401000`, .text `0x00401000..0x0091bbff`, .rdata `0x0091c000..0x009d01ff`, .data `0x009d1000..0x01045573`)
**Validation clients**: `I:\parp\parp-tools\output\tmp\wowarchive-clients\3_0_1_8303\`, `3_3_5_12340\`
**Companion notes**: `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md` (3.3.5/4.0.0 baseline)

## 1. The 25-Cvar Graphics Options Registry (3.0.1)

Recovered from `FUN_006ee8e0` (master graphics-options registration). Each cvar is registered with a name, default value, callback function, and storage global.

| # | Cvar | Default | Range | Callback | Storage | Notes |
|---|------|---------|-------|----------|---------|-------|
| 1 | `Toggle Lod` | 1 | bool | `FUN_006eda90` | `DAT_00edfae0 & 4` | Bit `0x4` in master render flag word; logs "Terrain LOD enabled/disabled" |
| 2 | `mapShadows` | 1 | bool | `FUN_006edae0` | (via `mapShadows` global) | Map shadow toggle |
| 3 | `SmallCull` | 1 | 0..1.0 | `FUN_006edb30` | computed via `FUN_006f2a00` | 3-tier value ladder; see §4 |
| 4 | `DistCull` | 1 | `[1.0, _DAT_009858e8]` | `FUN_006ee4a0` | `_DAT_00c75e3c` | Distance cull; see §5 |
| 5 | `MaxLights` | 1 | `[1, 4]` | `FUN_006edba0` | `_DAT_00edfad4` | Per-model max light count |
| 6 | `shadowLevel` | 1 | 0..n | `FUN_006edbe0` | (via `shadowLevel` global) | Terrain shadow map mip level |
| 7 | `texLodBias` | 1 | float | `FUN_006edc20` | (via `texLodBias` global) | Texture LOD bias |
| 8 | `farclip` | 1 | float | `FUN_006edc80` | `_DAT_00985e08` | Far clip plane distance |
| 9 | `nearclip` | 1 | float | `FUN_006edd70` | `_DAT_00961318` | Near clip plane distance |
| 10 | `specular` | 1 | bool | `FUN_006ee500` | `DAT_00edfae0 & 0x8000000` | Specular; requires pixel shaders; bit `0x8000000` |
| 11 | `mapObjLightLOD` | 1 | `[0, 2]` | `FUN_006ee570` | `_DAT_00f8ae9c` | Map object light LOD; logs "MapObjLightLOD must be 0-2" |
| 12 | `particleDensity` | 1 | float | `FUN_006edf50` | (via `particleDensity` global) | Particle density |
| 13 | `waterLOD` | 1 | 0 | `FUN_006edfb0` | `DAT_00f38ec0` | **FROZEN at 0 in 3.0.1**; callback sets `DAT_00f38ec0 = 0` and emits "waterLOD fixed to 0" |
| 14 | `baseMip` | 1 | 0..n | `FUN_006ee620` | (via `baseMip` global) | Base mipmap level |
| 15 | `horizonfarclip` | 1 | float | `FUN_006ede60` | `_DAT_00985d1c` | Horizon far clip plane |
| 16 | `showfootprints` | 1 | bool | `FUN_006edff0` | (via `showfootprints` global) | Footprint particles |
| 17 | `bspcache` | 1 | bool | `FUN_006ee6c0` | (via `bspcache` global) | BSP node caching |
| 18 | `footstepBias` | `"0.125"` | float | `FUN_006ee040` | (via `footstepBias` global) | Unit footstep depth bias |
| 19 | `occlusion` | 1 | bool | `FUN_006ee0f0` | (via `occlusion` global) | Hardware occlusion test |
| 20 | `spellEffectLevel` | 1 | 0..n | `FUN_006ee130` | (via `spellEffectLevel` global) | Spell effects |
| 21 | `worldPoolUsage` | `"Dynamic"` | enum | `FUN_006ee1e0` | (via `worldPoolUsage` global) | CGxPool static/dynamic |
| 22 | `terrainAlphaBitDepth` | 1 | enum | `FUN_006ee250` | `_DAT_00985bd8` | Terrain alpha map bit depth |
| 23 | `groundEffectDensity` | 1 | float | `FUN_006ee2d0` | `_DAT_0097b93c` | Ground effect density |
| 24 | `groundEffectDist` | 1 | float | `FUN_006ee350` | `_DAT_00985ae4` | Ground effect dist |
| 25 | `vertexShaders` | 1 | bool | `FUN_006ee5a0` | (via `vertexShaders` global) | Use vertex shaders |
| 26 | `objectFade` | 1 | bool | `FUN_006ee400` | (via `objectFade` global) | Fade objects into view |
| 27 | `objectFadeZFill` | 1 | bool | `FUN_006ee450` | (via `objectFadeZFill` global) | Fade objects using ZFill pass |

(Counts to 27 when including all distinct registration calls; 25 unique cvar names + 2 `horizonfarclip` duplicate at the end of the function.)

**Source path**: `FUN_006ee8e0` is called once from `FUN_006e2180` (the boot-time runtime control registration function, see §2). `FUN_006ee810` is the master applier that loads 12 cvar values via `FUN_00680a70` and applies them via per-option appliers `FUN_006ed980`/`FUN_006ed9e0`/`FUN_006eda30`.

**wow-viewer gap**: None of these 25 cvars are registered as runtime controls in `wow-viewer/src/core/WowViewer.Core.Runtime/M2/`. The `M2RuntimeOptions` enum in `M2SceneSubmissionCoordinator.cs:4-14` has 7 flags (ZFill, ClipPlanes, Threads, Faster, BatchDoodads, BatchParticles, ForceAdditiveParticleSort) but no `SmallCull`, `DistCull`, `MaxLights`, `mapObjLightLOD`, `waterLOD`, `terrainAlphaBitDepth`, `objectFade`, `groundEffectDensity`, etc.

**Recommended slice**: First implementation slice takes the 4 most performance-impactful: `SmallCull`, `DistCull`, `MaxLights`, `mapObjLightLOD`. Other cvars are follow-on slices.

## 2. The Master Render Flag Word (3.0.1)

Recovered from `DAT_00edfae0` (initialized to `0x7104b73` in `FUN_006e2180` at offset `0x006e2185` via `OR dword ptr [0x00edfae0], 0x7104b73`).

| Bit | Name | Set by | Effect |
|-----|------|--------|--------|
| `0x4` | `terrainLOD` | `FUN_006eda90` (Toggle Lod cvar) | Terrain LOD enabled |
| `0x8000000` | `specular` | `FUN_006ee500` (specular cvar) | Specular rendering enabled (requires pixel shaders) |

The init value `0x7104b73` is a 32-bit constant. Decoded as a 32-bit number, this means 22 bits are unknown and represent the boot-time default state. The two confirmed bits (`0x4`, `0x8000000`) are the only ones that have been mapped to cvars via decompilation.

**Source path**: `FUN_006e2180` is the master cvar registration function (not the graphics options one, but the runtime controls one); it sets `DAT_00edfae0 | 0x7104b73` and registers ~50 cvar strings (at addresses `0x984a98..0x984dc0`) via `FUN_00685550`. The cvar names from this block are not yet fully recovered; the 25 graphics options from §1 are a separate registration in `FUN_006ee8e0`.

**wow-viewer gap**: wow-viewer has no equivalent single-source-of-truth render flag word. Each runtime flag in `M2RuntimeOptions` is consulted independently.

**Recommended slice**: Not the first slice. Follow-on. The first slice takes only the per-batch cull and the 3-tier SmallCull/DistCull.

## 3. The Per-Batch Alpha-Cull Algorithm (3.0.1)

Recovered from `FUN_00788fb0` (GetNumBatches) and `FUN_00789440` (GetNumPrimitives).

**Per-batch formula**:
```
for each batch B in the model:
    alpha = model.m_alpha_scalar           // at +0x1cc
    if B.transparency_index < data.transparencyLookup.count:
        alpha *= transparencyLookup[B.transparency_index]    // at +0xcc + 0x1c
    if B.color_index != 0:
        alpha *= colorLookup[B.color_index]                  // at +0xd4 + 0x8 + ... + 0xc
    if alpha < _DAT_009455a8:                 // cull threshold (double compare with NAN-safe macro)
        count++
```

**Key field offsets** (in 0x18-stride batch records):
- `+0x04` = `materialId` (used to look up at `+0xc4` table; if non-zero, batch is "active")
- `+0x08` = `transparencyIndex` (used at `+0xcc + 0x1c` for transparency weight)
- `+0x0e` = `colorIndex` (used at `+0xd4 + 0x8 + ... + 0xc` for color weight)
- `+0x14` = `colorAnimationIndex` (used at `+0xd4 + 0x8 + (data.colorAnimLookup+colorIndex*2)*0xc` for animated color weight)

**Hand-unrolled 4-iteration loop** at the top of the function: this is likely a manual unroll for SSE-style batch processing (processes 4 batches per loop iteration). The unroll is visible as the `local_1c = (uVar7 - 4 >> 2) + 1; uVar6 = local_1c * 4;` setup followed by a `do { ... 4 batches ... local_1c = local_1c + -1; } while (local_1c != 0);` loop.

**Cull threshold constant**: `_DAT_009455a8` is a double-precision value at address `0x009455a8` (in .rdata). Its actual value is not yet recovered from Ghidra; for the wow-viewer port, it should be exposed as a tunable cvar and default to a sensible value (recommend 0.01 based on typical alpha-cutoff behavior).

**Source paths**: `FUN_00788fb0` and `FUN_00789440` are both called from `FUN_00705230` (the scene-side draw-list builder, see §6). They are also called by per-model render prep code.

**wow-viewer gap**: `M2StaticRenderModelBuilder` builds sections unconditionally with no per-batch cull. `M2StaticRenderModelBuilder.cs:28-104` iterates `activeSection` and `passes` with no alpha-weight check. The per-batch alpha cull is the single biggest missing piece in wow-viewer's M2 renderer for dense scenes.

**Recommended slice**: **First slice. Highest priority.** Maps directly to a new `M2BuildCullPolicy.ShouldCullBatch(M2ActiveSkinBatch, M2GeometryDocument, M2ActiveSkinProfile)` predicate.

## 4. The 3-Tier SmallCull Value Ladder (3.0.1)

Recovered from `FUN_006edb30` (SmallCull callback) and `FUN_006f2a00` (squared-distance precompute applier).

**The ladder** (3.0.1's `FUN_006edb30`):
```
read cvar as float param_1
default: local_8 = 0x3f800000  // 1.0
if param_1 < _DAT_00985318:
    local_8 = _DAT_00937f88    // tier-1 small-cull value
elif param_1 < _DAT_00985310:
    local_8 = _DAT_00976c04    // tier-2 small-cull value
elif param_1 < _DAT_00985308:
    local_8 = 0x3f800000       // tier-3 (1.0)
apply via FUN_006f2a00(local_8)
```

**The precompute** (`FUN_006f2a00`):
```
_DAT_00c75e80 = param_1 * _DAT_00c75e6c    // small-cull × lod-multiplier[0]
_DAT_00c75e84 = param_1 * _DAT_00c75e70    // small-cull × lod-multiplier[1]
_DAT_00c75e88 = param_1 * _DAT_00c75e74    // small-cull × lod-multiplier[2]
_DAT_00c75e8c = param_1 * _DAT_00c75e78    // small-cull × lod-multiplier[3]
... 12 globals total in groups of 4:
    _DAT_00c75e80, _DAT_00c75e84, _DAT_00c75e88, _DAT_00c75e8c
    _DAT_00c75e94, _DAT_00c75e98, _DAT_00c75e9c, _DAT_00c75ea0  (squared versions)
    _DAT_00c75ea8, _DAT_00c75eac, _DAT_00c75eb0, _DAT_00c75eb4  (minus versions)
    _DAT_00c75ebc, _DAT_00c75ec0, _DAT_00c75ec4, _DAT_00c75ec8  (squared-minus versions)
```

This is a **single-call precompute**: when the cvar changes, the 12 squared-distance values are precomputed once, then the per-frame cull test is a single multiply + compare against a precomputed squared constant. Brilliant optimization for the hot path.

**Source path**: `FUN_006f2a00` is called from `FUN_006edb30` (the SmallCull callback). The 4-tuple `_DAT_00c75e6c/0x70/0x74/0x78` are the per-tier lod multipliers; the 3 threshold constants `_DAT_00985318/0x985310/0x985308` are the cvar-value tier boundaries; the 3 result values `_DAT_00937f88/0x976c04/0x3f800000` are the per-tier small-cull values.

**wow-viewer gap**: wow-viewer has no SmallCull cvar and no 3-tier value ladder. The wow-viewer `M2RuntimeOptions` enum has no equivalent.

**Recommended slice**: First slice, alongside the per-batch alpha cull. The new `M2BuildCullPolicy` owns the tier ladder and the precompute.

## 5. The DistCull Clamp (3.0.1)

Recovered from `FUN_006ee4a0` (DistCull callback).

```
read cvar as float param_1
default _DAT_009858e8 = max distance (value not yet recovered)
if 1.0 <= param_1 <= _DAT_009858e8:
    _DAT_00c75e3c = param_1
    return 1
else:
    emit "DistCull must be in range 1.0 - %f." with _DAT_009858e8
    return 0
```

This is the **per-frame distance-cull threshold**: when rendering a model, compute distance from camera; if `distance > _DAT_00c75e3c`, skip the model. The clamp `[1.0, _DAT_009858e8]` ensures the threshold is always positive and bounded.

**Source path**: `FUN_006ee4a0` is the DistCull callback registered in `FUN_006ee8e0`.

**wow-viewer gap**: wow-viewer has no distance cull. The `M2SceneSubmissionCoordinator` (lines 231-308) sorts by family/modelKey/effectKey but does not skip models by distance.

**Recommended slice**: First slice, alongside the per-batch alpha cull and SmallCull. The new `M2BuildCullPolicy` owns the clamp and the runtime threshold.

## 6. The Scene-Side Draw-List Builder (3.0.1)

Recovered from `FUN_00705230`.

**Per-entry layout** (40 bytes / 10 ints):
```
offset  field
+0x00   GetFileName() result    // unique id from the model
+0x04   0                       // reserved
+0x08   world_x
+0x0c   world_y
+0x10   world_z
+0x14   length×scale            // distance-squared × small-cull multiplier
+0x18   param_2 (model index)
+0x1c   GetNumBatches result    // visible batch count after alpha cull
+0x20   GetNumPrimitives result // visible triangle count after alpha cull
+0x24   boneCount               // from data->boneCount (+0x134 + 0x34)
```

**Behavior**:
1. For each model in the composite tree (`+0x68` children, `+0x70` siblings), allocate a 10-int entry in the flat array.
2. Compute world position via `FUN_00467350` (likely a transform/unproject from a view matrix at `+0xe4` or `param_3+0x60`).
3. Compute `length = sqrt(local_4c² + local_54² + local_50²)`, then `local_8 = length × scale` — this is the per-model distance × cull threshold, used downstream for LOD selection.
4. Call `GetNumBatches` (FUN_00788fb0) to get the visible batch count after per-batch alpha cull.
5. Call `GetNumPrimitives` (FUN_00789440) to get the visible triangle count.
6. Recurse into `+0x68` (child) and `+0x70` (sibling).

**Composite model tree**: `+0x68` and `+0x70` are the per-model child and sibling pointers. Composite models (e.g. character + weapon + shield) are linked this way. The draw-list builder walks the entire tree.

**Source path**: `FUN_00705230` is called from `FUN_00705230` (recursive, for the tree walk) and from the scene-side render code that consumes the flat list. The actual sort/draw consumer is upstream of this function.

**wow-viewer gap**: `M2SceneSubmissionCoordinator.cs` sorts entries by family/modelKey/effectKey but does not have a per-model distance × cost metric, and does not walk composite model chains.

**Recommended slice**: Not the first slice. Follow-on slice. The first slice is per-batch cull only. The scene-side draw-list builder is a larger architectural change.

## 7. The Per-Method Timing Gate (3.0.1)

Recovered from `FUN_00786b20`.

```
FUN_00786b20(param_1, param_2):  // param_1 = model, param_2 = method-name (or 0)
    iVar1 = *(int *)(*(int *)(param_1 + 0x30) + 0xc)
    if (iVar1 != 0) {
        FUN_0045f2d0(iVar1)  // debug assertion
    }
    if (*(int *)(param_1 + 0x38) != 0 && ...) {
        FUN_0045f2d0(...)    // debug assertion
    }
    if ((*(byte *)(param_1 + 0x10) & 1) == 0) {
        FUN_0068cf10(0x85100000, ".\\M2Model.cpp", 0x5a9, "m_loaded", 0, 1)  // assertion failure
    }
    if (param_2 != 0) {
        FUN_00697ec0("Model2: CM2Model::%s stalled: %s\n", param_2, *(int *)(param_1 + 0x30) + 0x20)
        // emits: Model2: CM2Model::GetBoundingBox stalled: filename.mdx
    }
```

**31+ callers** (method names recovered from string anchors):
- `FUN_00787600` = `MakeLoaded` (sets bit `0x100`, recurses via `+0x68`)
- `FUN_00787680` = `MakeSkinned` (sets bit `0x2`, calls `FUN_0045b3b0` per texture, sets bit `0x200`)
- `FUN_00787bb0` = `GetBoundingBox` (recursive AABB via `+0x68`)
- `FUN_00788420` = `GetBoneSequenceInfo` (looks up `data->boneSequences`, then `data->bones` at 0xb4-stride runtime bone)
- `FUN_00788fb0` = `GetNumBatches` (per-batch alpha cull, see §3)
- `FUN_00789440` = `GetNumPrimitives` (per-batch alpha cull, see §3)
- `FUN_007889e0` / `FUN_00788a80` / `FUN_00788b20` / `FUN_0078cbf0` / `FUN_00794b10` / `FUN_00794c80` / `FUN_007950a0` / `FUN_00795130` — other CM2Model methods (names not yet recovered)

**Behavior**: Every expensive CM2Model method calls `FUN_00786b20` with its own name string. The function asserts the model is loaded (bit `0x10 & 1`), then if a method name is passed, emits a "stalled" warning to the log. This is the native engine's per-method timing assertion framework.

**wow-viewer gap**: wow-viewer has no per-method timing assertion framework for M2 operations. The closest is `M2RenderConsumerFrameState` which records frame-level metrics, not per-method.

**Recommended slice**: Not the first slice. The per-method timing gate is a debug/telemetry concern; the first slice is per-batch cull.

## 8. The Detail-Doodad Subsystem (3.0.1)

Recovered from string anchors and the `DetailDoodad.cpp` source path string.

| Anchor | String | Notes |
|--------|--------|-------|
| `0x00987ab8` | `.\DetailDoodad.cpp` | Source file path |
| `0x00987b0c` | `WDETAILDOODADINST` | World detail doodad instance struct |
| `0x00987b44` | `CDetailDoodad_idx` | Detail doodad index buffer class |
| `0x00987b58` | `CDetailDoodad_vtx` | Detail doodad vertex buffer class |
| `0x00984ad0` | `detailDoodadAlpha` | Cvar for detail doodad alpha cutoff |
| `0x00984b80` | `showDetailDoodads` | Cvar for show/hide |
| `0x00984c08` | `detailDoodadTest` | Cvar for test mode |
| `0x0098639c` | `visDetailDoodadList.Head() == 0` | Per-frame visibility list assertion |
| `0x00988688` | `chunk->detailDoodadInst == 0` | Per-chunk detail doodad instance gate |
| `0x00988a5c` | `mapDetailDoodadUpdateList.Head() == 0` | Map-wide update list assertion |

The detail-doodad subsystem is a separate path from M2 models. It handles the per-chunk grass/pebbles/small-props population. Per-chunk visibility and update lists are managed separately from the main M2 model list.

**wow-viewer gap**: wow-viewer has no detail-doodad subsystem. The terrain renderer (MdxViewer-side) renders terrain but not detail doodads. The wow-viewer Core.IO Maps readers (`M2ModelReader` etc.) don't touch detail-doodad data.

**Recommended slice**: Not the first slice. Detail-doodad is a major subsystem; first slice is per-batch cull.

## 9. The Projected-Texture Render Path (3.0.1)

Recovered from `FUN_0088ff30` (`RenderModelBatchesForProjectedTexture`) and the 4 combiner family strings.

| Anchor | String | Notes |
|--------|--------|-------|
| `0x00984570` | `Projected_FadeAdd` | Combiner: fade-add |
| `0x00984584` | `Projected_FadeOpaque` | Combiner: fade-opaque |
| `0x0098459c` | `Projected_ModAdd` | Combiner: mod-add |
| `0x009845b0` | `Projected_ModMod` | Combiner: mod-mod |
| `0x0088ff30` | (in `RenderModelBatchesForProjectedTexture`) | Dedicated draw call for projected textures |

The projected-texture render path is a **separate draw call from the main M2 model draw**: when a model has projected textures (e.g. character selection circle, ground-targeted effects), `FUN_0088ff30` is called with the model's batches and uses one of the 4 `Projected_*` combiners. The 4 combiners are a separate code path from the main `Combiners_*` family.

**wow-viewer gap**: `M2EffectRecipe` in `M2StaticRenderModelBuilder.cs:292-342` does not handle projected textures; it only handles the regular `Diffuse_T1` / `T1T2` / etc. families.

**Recommended slice**: Not the first slice. Projected-texture is a renderer-level concern; first slice is per-batch cull.

## 10. Cross-Build Evidence Map

| Finding | 3.0.1 (this binary) | 3.3.5.12340 (research note) | 4.0.0.11927 (research note) |
|---------|---------------------|------------------------------|------------------------------|
| Per-batch alpha cull | `FUN_00788fb0` + `FUN_00789440` | Inferred from `M2_BuildCombinerEffectName` and runtime flag bit `0x40`; not explicitly decompiled | Inferred from `M2Faster` callback; not explicitly decompiled |
| 3-tier SmallCull | `FUN_006edb30` + `FUN_006f2a00` | Listed in spec 036 inventory as "object size culling" (3.3.5 cvar name confirmed as `SmallCull`); not decompiled | Inferred; not in research note |
| DistCull | `FUN_006ee4a0` | Listed in spec 036 inventory as "object distance culling" (3.3.5 cvar name confirmed as `DistCull`); not decompiled | Inferred; not in research note |
| MaxLights | `FUN_006edba0` (clamp `[1, 4]`) | Listed in spec 036 inventory; not decompiled | Inferred; not in research note |
| mapObjLightLOD | `FUN_006ee570` (clamp `[0, 2]`) | Listed in spec 036 inventory; not decompiled | Inferred; not in research note |
| specular cvar | `FUN_006ee500` (toggles bit `0x8000000`, requires pixel shaders) | Inferred from `specular not enabled. Requires pixel shaders` string | Confirmed in 4.0.0.11927 strings |
| Scene-side draw-list builder | `FUN_00705230` (per-model 40-byte entry, composite walk) | Inferred; not decompiled | Inferred; not in research note |
| Per-method timing gate | `FUN_00786b20` (31+ callers) | Inferred; not in research note | Inferred; not in research note |
| Detail-doodad subsystem | `DetailDoodad.cpp` + `CDetailDoodad_idx`/`vtx` + per-chunk gate | Strings exist in 3.3.5 (per spec 036 inventory `doodadAnimAlways`); subsystem not decompiled | Inferred; not in research note |
| Projected-texture render | `FUN_0088ff30` + 4 `Projected_*` combiners | Confirmed in 3.3.5 strings (`Projected_ModMod_Unlit`, etc. in `m2-native-client-research-2026-03-31.md` particle effect list) | Confirmed in 4.0.0 (continuity with 3.3.5) |
| waterLOD (frozen at 0) | `FUN_006edfb0` | Listed in spec 036 inventory; 3.3.5 allows non-zero | Inferred; not in research note |
| terrainAlphaBitDepth | `FUN_006ee250` | Listed in spec 036 inventory; not decompiled | Inferred; not in research note |
| groundEffectDensity/Dist | `FUN_006ee2d0` + `FUN_006ee350` | **3.0.1 only.** No equivalent cvars in 3.3.5 | **3.0.1 only.** No equivalent cvars in 4.0.0 |
| objectFade / objectFadeZFill | `FUN_006ee400` + `FUN_006ee450` | **3.0.1 only.** No equivalent cvars in 3.3.5 | Inferred; not in research note |
| footstepBias | `FUN_006ee040` | **3.0.1 only.** No equivalent cvars in 3.3.5 | Inferred; not in research note |

## 11. wow-viewer Gap Inventory (Per Finding)

### Finding 1: Per-batch alpha cull (FUN_00788fb0)
- **Current wow-viewer behavior**: `M2StaticRenderModelBuilder` builds all sections unconditionally; no per-batch alpha weight check.
- **File**: `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2StaticRenderModelBuilder.cs:28-104`
- **Absence**: no equivalent in `M2SkinnedRenderModelBuilder` either.
- **Recommended slice**: First. Maps to `M2BuildCullPolicy.ShouldCullBatch(...)`.

### Finding 2: 3-tier SmallCull (FUN_006edb30 + FUN_006f2a00)
- **Current wow-viewer behavior**: no SmallCull cvar; no tier ladder.
- **File**: absence in `M2RuntimeOptions` enum at `M2SceneSubmissionCoordinator.cs:4-14`.
- **Recommended slice**: First. Maps to `M2BuildCullPolicy.SmallCullTier1/2/3` and precompute.

### Finding 3: DistCull (FUN_006ee4a0)
- **Current wow-viewer behavior**: no DistCull cvar; no distance cull.
- **File**: `M2SceneSubmissionCoordinator.cs:241-249` sorts by family/modelKey/effectKey but does not skip by distance.
- **Recommended slice**: First. Maps to `M2BuildCullPolicy.DistCullMin/Max/Threshold`.

### Finding 4: MaxLights (FUN_006edba0)
- **Current wow-viewer behavior**: no MaxLights cvar; no per-model light count limit.
- **File**: absence in `M2RuntimeOptions` enum.
- **Recommended slice**: First. Maps to `M2BuildCullPolicy.MaxLights` with `[1, 4]` clamp.

### Finding 5: mapObjLightLOD (FUN_006ee570)
- **Current wow-viewer behavior**: no mapObjLightLOD cvar; no per-object light LOD.
- **File**: absence in `M2RuntimeOptions` enum.
- **Recommended slice**: First. Maps to `M2BuildCullPolicy.MapObjLightLOD` with `[0, 2]` clamp.

### Finding 6: specular cvar (FUN_006ee500)
- **Current wow-viewer behavior**: no specular cvar; specular is always-on if the renderer supports it.
- **File**: absence in `M2RuntimeOptions` enum.
- **Recommended slice**: First. Maps to `M2BuildCullPolicy.Specular` with pixel-shader check.

### Finding 7: Master render flag word (DAT_00edfae0)
- **Current wow-viewer behavior**: per-flag checks scattered across `M2RuntimeOptions`.
- **File**: `M2RuntimeOptions` enum at `M2SceneSubmissionCoordinator.cs:4-14`.
- **Recommended slice**: Follow-on. The flag word is a clean state model but doesn't directly improve performance.

### Finding 8: Scene-side draw-list builder (FUN_00705230)
- **Current wow-viewer behavior**: `M2SceneSubmissionCoordinator.BuildPlan` sorts entries but does not compute per-model distance × cost.
- **File**: `M2SceneSubmissionCoordinator.cs:233-308`.
- **Recommended slice**: Follow-on. Larger architectural change; first slice is per-batch cull.

### Finding 9: Per-method timing gate (FUN_00786b20)
- **Current wow-viewer behavior**: no per-method timing assertion; closest is `M2RenderConsumerFrameState`.
- **File**: absence.
- **Recommended slice**: Follow-on. Debug/telemetry concern.

### Finding 10: Detail-doodad subsystem
- **Current wow-viewer behavior**: no detail-doodad subsystem.
- **File**: absence across the entire `wow-viewer/` tree.
- **Recommended slice**: Follow-on. Major subsystem.

### Finding 11: Projected-texture render path (FUN_0088ff30)
- **Current wow-viewer behavior**: `M2EffectRecipe` handles `Diffuse_T1` / `T1T2` / `Projected` but not the 4 specific `Projected_*` combiners.
- **File**: `M2StaticRenderModelBuilder.cs:292-342`.
- **Recommended slice**: Follow-on. Renderer-level concern.

## 12. The First Recommended Implementation Slice

**Title**: Per-Batch Alpha-Cull + 3-Tier SmallCull + DistCull + MaxLights + mapObjLightLOD + Specular

**Scope** (intentionally narrow):
1. New `M2BuildCullPolicy` service in `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2BuildCullPolicy.cs`
2. New `M2BuildProfile`-gated `M2BuildCullPolicyFactory` for 3.0.1 vs 3.3.5 vs 4.0.0
3. Modify `M2StaticRenderModelBuilder` and `M2SkinnedRenderModelBuilder` to consult the policy
4. New per-batch cull telemetry counter
5. New cvar registry hooks: `SmallCull`, `DistCull`, `MaxLights`, `mapObjLightLOD`, `specular`

**Out of scope for this slice**:
- Scene-side draw-list builder (FUN_00705230) — follow-on slice
- Per-method timing gate (FUN_00786b20) — follow-on slice
- Detail-doodad subsystem — follow-on slice
- Projected-texture render path — follow-on slice
- Master render flag word (DAT_00edfae0) — follow-on slice
- 3.0.1-specific cvars (`groundEffectDensity`, `objectFade`, `footstepBias`, etc.) — out of scope; these are 3.0.1-only and don't apply to the build-agnostic 3.3.5 staging target

**Validation plan**:
- Stage `3_0_1_8303` client: load `Creature\Wolf\Wolf.mdx` and confirm per-batch cull counter decrements for invisible batches.
- Stage `3_3_5_12340` client: load `Spells\ErrorCube.mdx` and confirm per-batch cull counter decrements for invisible batches.
- Compare visible batch count before/after in dense outdoor 3.3.5 route (e.g. Stormwind); target ≥30% reduction in culled-frame batch count per spec 036 SC-009.
- Run on same machine, compare to pre-slice baseline.

**Risks**:
- Wrong cull threshold value (0xb4 constant recovered but threshold constant value not recovered)
- Wrong tier-ladder values (3 threshold constants and 3 result values are unknown)
- Wrong MaxLights clamp (4 is the max from FUN_006edba0, but the default is 1)
- Wrong mapObjLightLOD clamp (2 is the max from FUN_006ee570, but the default is 1)
- Per-batch cull changes visible scene; must be validated against the actual 3.3.5 staged client before declaring done

## 13. Companion Notes for spec 036

The 036 convergence plan owner should be aware of:
- **3.0.1 has 25 graphics options**, larger than the 3.3.5 list in the current 036 inventory. The 3.0.1-specific cvars (`groundEffectDensity`, `objectFade`, `footstepBias`, `objectFadeZFill`, `horizonfarclip`) are not in 3.3.5 and should be gated behind `M2BuildProfile == Build301` (or similar).
- **3.0.1 has a frozen `waterLOD` cvar** (forced to 0). Any wow-viewer implementation must not assume `waterLOD != 0` works in 3.0.1 staging.
- **The 3.3.5 research note** is consistent with the 3.0.1 findings for: `mapObjLightLOD` (clamp `[0, 2]`), `MaxLights` (clamp `[1, 4]`), `SmallCull` (exists in 3.3.5 but not decompiled there), `DistCull` (exists in 3.3.5 but not decompiled there), `terrainAlphaBitDepth` (exists in 3.3.5 but not decompiled there).
- **The 3.0.1 0xb4-stride runtime bone** is owned by spec 037; this research does not duplicate that.
- **The 3.3.5 deferred "scene submission / final draw submission path" gap** from `m2-native-client-research-2026-03-31.md` (line 1183) is partially resolved by `FUN_00705230` in 3.0.1; the 3.3.5 equivalent is still pending.

## 14. Failure Modes

This research is invalidated if:
- The loaded Ghidra binary is not actually `WoW.exe 3.0.1.8303` (the function addresses assume this is the case).
- The 25-cvar list changes between 3.0.1 sub-versions (no other 3.0.1 builds are staged; the list is a 3.0.1.8303 snapshot).
- The per-batch alpha cull formula was decompiled incorrectly (the formula was recovered from `FUN_00788fb0` decompilation; the hand-unrolled loop makes this error-prone).
- The 3-tier SmallCull tier-ladder values are wrong (3 threshold constants and 3 result values are unknown; the formula structure is correct but the actual numbers are not yet recovered).

For each of these failure modes, the recommended first slice's `M2BuildCullPolicy` should expose all values as tunable cvars (not hard-coded), so the implementation can be tuned at runtime.
