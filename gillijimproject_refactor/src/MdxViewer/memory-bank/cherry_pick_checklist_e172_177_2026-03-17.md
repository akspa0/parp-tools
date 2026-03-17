# Cherry-Pick Checklist: e172907b and 177f9613

Scope: surgical extraction planning from post-baseline commits while avoiding terrain alpha-mask regressions.

Baseline: `343dadfa27df08d384614737b6c5921efe6409c8`

## Commit e172907b03838f56e4f4b3ed61835d5f74b4971a

### SAFE
- `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs` (new file)
  - Extract full file.
  - Reason: isolated M2 runtime adaptation logic; no terrain alpha decode/render path.

- `src/MdxViewer/Terrain/WorldAssetManager.cs`
  - Extract only M2 loading adaptation hunks in `LoadMdxModel` path:
    - switch from `ConvertM2ToMdx(...)` to `WarcraftNetM2Adapter.IsMd20(...)`
    - skin candidate scoring/selection loop
    - remove private `ConvertM2ToMdx` helper
  - Reason: model loading behavior only.

- `src/MdxViewer/ViewerApp.cs`
  - Extract only M2 disk-loading path migration:
    - `LoadM2FromDisk` / `LoadM2FromBytes` using `WarcraftNetM2Adapter`
  - Reason: standalone model loading path only.

### MIXED
- `src/MdxViewer/Rendering/WmoRenderer.cs`
  - Safe hunks to extract:
    - `ResolveBatchMaterialId(...)`, `LogMaterialFallback(...)`, `ResolveMaterialTextureName(...)`
    - material fallback logging limits and dedupe fields
  - Mixed/risky hunks to skip unless explicitly requested:
    - MLIQ orientation auto-fit (`SelectBestLiquidOrientation`, `MapLiquidVertex`)
    - global override `MliqRotationQuarterTurns` and mesh rebuild path (`EnsureLiquidMeshesUpToDate`)
  - Reason: WMO liquid orientation is behavior-changing rendering logic.

- `src/MdxViewer/ViewerApp.cs`
  - Safe hunks:
    - viewer settings persistence plumbing not tied to terrain
  - Mixed/risky hunks to skip by default:
    - WMO liquid rotation UI (`DrawWmoLiquidRotationControls`, tuning section)
  - Reason: coupled to WMO liquid orientation behavior.

### RISKY (DO NOT EXTRACT)
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
  - Skip all alpha decode changes, including:
    - `_useBigAlpha = (_mphdFlags & (0x4u | 0x80u)) != 0`
    - `ExtractAlphaMaps(...)` rewrite to `AlphaMapService.ReadAlpha(...)`
    - per-layer span inference (`ResolveNextAlphaOffset`, 2048/4096 heuristics)
  - Reason: direct alpha decode semantics change.

- `src/MdxViewer/Terrain/TerrainRenderer.cs`
  - Skip all alpha upload/render behavior changes, including:
    - overlay guard `if (!chunk.AlphaTextures.ContainsKey(layer)) continue;`
    - removal of renderer-side edge-fix duplication in `UploadAlphaTextures(...)`
  - Reason: direct alpha blend/visual behavior changes.

## Commit 177f96135fc443556af91e11f487fcf4e2dd43f6

### SAFE
- `src/MdxViewer/Rendering/ModelRenderer.cs`
  - Extract backface normal flip guard in fragment shader:
    - from unconditional `if (!gl_FrontFacing)`
    - to `if (uSphereEnvMap == 1 && !gl_FrontFacing)`
  - Reason: isolated model shading correction; no terrain path.

- `src/MdxViewer/ViewerApp.cs`
  - Extract perf/debug UI only:
    - `_showPerfWindow`, `DrawPerfWindow()` window and menu item
    - `AlphaMaskChannel` radio buttons (A1/A2/A3) control wiring
  - Reason: UI diagnostics; does not alter terrain decode.

### MIXED
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
  - Small additive hunk only:
    - populate `AreaId` and `McnkFlags` in `TerrainChunkData`
  - Keep only if required by downstream chunk-info consumers.
  - Reason: low-risk data plumbing but in terrain adapter.

- `src/MdxViewer/Terrain/TerrainManager.cs`
  - Mixed due tile-batching integration.
  - Potentially safe micro-hunks (only if cherry-picked with all dependencies satisfied):
    - setting `UseWorldUvForDiffuse` per adapter type
  - Skip by default:
    - `TerrainTileMeshBuilder` migration, `AddTile/RemoveTile` flow, loaded tile storage change
  - Reason: ties into new renderer mesh path.

- `src/MdxViewer/Terrain/VlmTerrainManager.cs`
  - Skip by default unless doing full tile-batching rollout:
    - migration from chunk meshes to tile meshes and `AddTile/RemoveTile`
  - Reason: same dependency chain as `TerrainManager`.

### RISKY (DO NOT EXTRACT)
- `src/WoWMapConverter/WoWMapConverter.Core/Formats/LichKing/Mcal.cs`
  - Skip decoder rewrite:
    - `GetAlphaMapForLayer(...)` contract change to nullable and new decode path
    - `ReadCompressedAlpha(...)`, `ReadBigAlpha(...)`, `ReadUncompressedAlpha4Bit(...)`
    - `ApplyEdgeFix(...)` semantics
    - `MclyFlags` reinterpretation cleanup
  - Reason: core MCAL behavior in alpha regression hotspot.

- `src/MdxViewer/Terrain/TerrainRenderer.cs`
  - Skip tile-shader/tile-mesh pipeline and alpha path coupling:
    - `_tileShader`, tile maps/chunk info maps, `AddTile`, `RemoveTile`, `RenderTiles`
    - chunk lookup path rewrites (`TryGetTileKey`, `TryFindChunkInfoByBounds`, etc.)
    - alpha debug/channel and texture-array path coupling in tile renderer
  - Reason: broad terrain rendering pipeline replacement.

- `src/MdxViewer/Terrain/TerrainTileMesh.cs` (new)
- `src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs` (new)
  - Skip by default.
  - Reason: introduces fused per-tile topology and packed alpha/shadow array behavior.

## Recommended Extraction Order
1. e172 SAFE: `WarcraftNetM2Adapter.cs` + `WorldAssetManager.cs` M2 hunks + `ViewerApp.cs` M2 hunks.
2. 177 SAFE: `ModelRenderer.cs` shader guard + `ViewerApp.cs` perf/debug-only hunks.
3. Rebuild after each step; stop on first regression.
4. Do not extract any item listed as RISKY without explicit approval.

## Verification Gate (required after each extraction step)
- Build:
  - `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Manual terrain check on both data families:
  - Alpha-era terrain (0.5.3 style)
  - LK 3.3.5 terrain
- Validation language:
  - report as "build passed" and "manual spot-check completed" only; do not claim full regression safety without real-data coverage.

## Fresh-Chat Prompt Plan (Copilot)

Use this prompt in a new Copilot chat when you want implementation work (SAFE first, then selected MIXED) executed end-to-end.

```md
You are doing surgical integration in `I:/parp/parp-tools/gillijimproject_refactor`.

Goal:
- Implement SAFE items from commits `e172907b03838f56e4f4b3ed61835d5f74b4971a` and `177f96135fc443556af91e11f487fcf4e2dd43f6`
- Then implement MIXED items only if they pass the gates below
- Do NOT apply RISKY items

Primary spec:
- `gillijimproject_refactor/src/MdxViewer/memory-bank/cherry_pick_checklist_e172_177_2026-03-17.md`

Hard safety constraints:
1. Baseline for behavior comparison is `343dadfa27df08d384614737b6c5921efe6409c8`.
2. Never run `git cherry-pick` for entire commits `e172907b` or `177f9613`.
3. Do not apply any hunk touching MCAL decode, alpha packing, shadow-mask packing, terrain tile shader blending, or fused tile topology.
4. If a hunk is ambiguous SAFE vs RISKY, classify it as RISKY and skip.

Execution order:

Phase 0: Prep + evidence
- Read:
  - `gillijimproject_refactor/src/MdxViewer/memory-bank/cherry_pick_checklist_e172_177_2026-03-17.md`
  - `gillijimproject_refactor/memory-bank/activeContext.md`
  - `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
  - `.github/instructions/terrain-alpha.instructions.md`
- Reconfirm SAFE/MIXED/RISKY file/hunk boundaries with targeted `git diff`.

Phase 1: Implement SAFE (mandatory)
- e172 SAFE:
  - Extract full file: `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`
  - Apply only SAFE M2 hunks in:
    - `src/MdxViewer/Terrain/WorldAssetManager.cs`
    - `src/MdxViewer/ViewerApp.cs`
- 177 SAFE:
  - Apply ModelRenderer shader guard hunk in:
    - `src/MdxViewer/Rendering/ModelRenderer.cs`
  - Apply SAFE perf/debug UI hunks in:
    - `src/MdxViewer/ViewerApp.cs`

SAFE gate:
- Build after SAFE phase:
  - `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- If build fails, fix only SAFE-related integration breakage and rerun.

Phase 2: Implement MIXED (conditional)
- Start only if SAFE gate passes.
- Allowed MIXED targets (hunk-by-hunk, no broad copy):
  - `src/MdxViewer/Rendering/WmoRenderer.cs`
    - allow material fallback and texture-name resolution hunks only
    - skip all MLIQ orientation/rotation override behavior hunks
  - `src/MdxViewer/ViewerApp.cs`
    - allow settings plumbing not coupled to WMO liquid rotation behavior
  - `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
    - only additive `AreaId`/`McnkFlags` chunk metadata hunk
  - `src/MdxViewer/Terrain/TerrainManager.cs`
    - skip tile-mesh migration; only trivial non-topology plumbing if dependency-safe
  - `src/MdxViewer/Terrain/VlmTerrainManager.cs`
    - skip tile-mesh migration by default

MIXED gate:
- Rebuild after each mixed file batch.
- Stop immediately on any terrain visual regression indicator and report the exact hunk.

Validation + reporting:
1. Build result.
2. Files/hunks extracted.
3. Files/hunks skipped and why.
4. Residual risks and what still needs runtime real-data validation.
5. Do not claim full safety; state exactly what was and was not validated.

Required output format:
- `Extracted`
- `Skipped`
- `Build`
- `Runtime Validation Needed`
- `Next Surgical Step`
```

### Optional Follow-up Prompt (MIXED only pass)

Use this only after SAFE is already merged and stable:

```md
Continue from current branch and execute MIXED-only integration from
`cherry_pick_checklist_e172_177_2026-03-17.md`.

Rules:
- No terrain alpha decode/render pipeline changes.
- No MLIQ orientation override behavior changes.
- Build after each file.
- If a mixed hunk causes any compile or runtime risk signal, revert that hunk only and continue.

Report in the same `Extracted / Skipped / Build / Runtime Validation Needed / Next Surgical Step` format.
```

### Ultra-Short Single-Shot Prompt (Minimal Tokens)

```md
Surgical integrate SAFE then MIXED from:
`gillijimproject_refactor/src/MdxViewer/memory-bank/cherry_pick_checklist_e172_177_2026-03-17.md`

Repo: `I:/parp/parp-tools/gillijimproject_refactor`
Baseline: `343dadfa27df08d384614737b6c5921efe6409c8`

Rules:
- No full `git cherry-pick` of `e172907b` or `177f9613`
- Never apply RISKY items
- If unsure SAFE vs RISKY, treat as RISKY
- Never apply MCAL decode, alpha packing, terrain tile shader blending, shadow-mask packing, or fused tile-topology hunks

Do now:
1. Reconfirm SAFE/MIXED boundaries via targeted `git diff`
2. Apply all SAFE hunks/files only
3. Build: `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
4. If SAFE build passes, apply MIXED hunks allowed by checklist (hunk-by-hunk)
5. Rebuild after each mixed file batch; stop at first regression signal

Output exactly:
- Extracted
- Skipped
- Build
- Runtime Validation Needed
- Next Surgical Step
```
