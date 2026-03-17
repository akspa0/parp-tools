# Recovery Ledger (v0.4.1-dev)

Branch target: v0.4.1-dev
Baseline: 343dadfa27df08d384614737b6c5921efe6409c8

## Objective

Restore known-good terrain behavior while retaining useful post-baseline improvements.

## Candidate Commits Since Baseline (Initial Bucket)

| Commit | Bucket | Status | Notes |
|---|---|---|---|
| 177f961 | C mixed | SPLIT_REQUIRED_HIGHEST | confirmed introduction of fused alpha+shadow tile pass symbols in TerrainRenderer |
| d50cfe7 | D/C mixed | SPLIT_REQUIRED_HIGH | mixed UI and terrain follow-up touching alpha/shadow texture plumbing |
| 39799bf | C (terrain high risk) | SPLIT_REQUIRED_MEDIUM | follow-up terrain refactor across renderer, builder, adapter, MCAL |
| 37f669c | C/D mixed | HOLD_AFTER_WAVE_1 | mixed terrain decode + ViewerApp + UI config, defer until topology stable |
| 4e2f681 | D mixed | HOLD_AFTER_WAVE_1 | mostly tools/export/UI; verify TerrainRenderer deltas before replay |
| 326e6f8 | D | KEEP_CANDIDATE | docs + import/export UI, low terrain-risk if adapter untouched |

## Confirmed Evidence (Wave 0)

- `177f961` is the first post-baseline commit that introduces the fused pass symbols `uAlphaShadowArray`, `AlphaShadowArrayTexture`, and `uShadowSampler` in `TerrainRenderer`.
- In that same commit, tile shader path reads alpha and shadow data from a shared array flow, making this the primary rollback/split target.
- `177f961` is a large mixed commit touching 9 files:
  - `src/MdxViewer/Terrain/TerrainRenderer.cs`
  - `src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs`
  - `src/MdxViewer/Terrain/TerrainTileMesh.cs`
  - `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
  - `src/WoWMapConverter/WoWMapConverter.Core/Formats/LichKing/Mcal.cs`
  - `src/MdxViewer/Terrain/TerrainManager.cs`
  - `src/MdxViewer/Terrain/VlmTerrainManager.cs`
  - `src/MdxViewer/ViewerApp.cs`
  - `src/MdxViewer/Rendering/ModelRenderer.cs`

## Terrain Surgical Replay Order (Locked)

1. Wave 1 (topology rollback first)
	- Drop fused alpha+shadow tile-pass behavior from `TerrainRenderer`, `TerrainTileMeshBuilder`, and `TerrainTileMesh`.
	- Preserve baseline-style independent alpha and shadow sampling behavior.
	- Keep Alpha era path isolation strict (no later-client decode/render fallback logic in Alpha route).
2. Wave 2 (safe feature replay)
	- Replay non-fused, low-risk improvements from mixed commits in small file-scoped slices.
	- Explicitly exclude UI churn and profile-guess behavior from this wave.
3. Wave 3 (deferred refactors)
	- Re-evaluate `39799bf`/`37f669c` only after Wave 1 parity passes on Alpha and later-client targets.

## Current Track State

- Active baseline worktree branch: `recovery/terrain-surgical-343dadf`.
- Principle: no broad rollback; split mixed commits and keep only behaviorally safe slices.
- Version policy target: explicit version-family selection in app flow, no silent terrain profile guessing.

## Per-File Split Map (Wave 1 Start)

### Commit 177f961

| File | Action | Reason |
|---|---|---|
| `src/MdxViewer/Terrain/TerrainRenderer.cs` | DROP_FOR_WAVE_1 | Contains fused alpha+shadow array tile pass introduction (`uAlphaShadowArray`, array alpha used for shadow) |
| `src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs` | DROP_FOR_WAVE_1 | Built around array-packed alpha/shadow tile data flow |
| `src/MdxViewer/Terrain/TerrainTileMesh.cs` | DROP_FOR_WAVE_1 | Topology companion to fused tile-array path |
| `src/WoWMapConverter/WoWMapConverter.Core/Formats/LichKing/Mcal.cs` | DEFER_TO_WAVE_2 | MCAL changes may be good but must be replayed after topology parity |
| `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` | DEFER_TO_WAVE_2 | Adapter changes coupled to decode/render behavior; replay only after Wave 1 gate pass |
| `src/MdxViewer/Terrain/TerrainManager.cs` | KEEP_CANDIDATE | Likely non-fused management improvements; replay with compile gate |
| `src/MdxViewer/Terrain/VlmTerrainManager.cs` | KEEP_CANDIDATE | Similar to TerrainManager, potentially safe |
| `src/MdxViewer/Rendering/ModelRenderer.cs` | KEEP_CANDIDATE | M2/rendering support not directly tied to fused alpha-shadow topology |
| `src/MdxViewer/ViewerApp.cs` | SPLIT_REQUIRED | Mixed UI/app flow changes; keep only terrain-debug controls that do not alter profile routing |

### Commit d50cfe7

| File | Action | Reason |
|---|---|---|
| `src/MdxViewer/Terrain/TerrainRenderer.cs` | DROP_FOR_WAVE_1 | Reinforces alpha-shadow array plumbing; keep out until parity established |
| `src/MdxViewer/Export/TerrainImageIo.cs` | HOLD_AFTER_WAVE_1 | Potentially useful import/export features, but not required for topology rollback |
| `src/MdxViewer/ViewerApp.cs` | SPLIT_REQUIRED | Very large mixed UI change set |

### Commit 39799bf

| File | Action | Reason |
|---|---|---|
| `src/MdxViewer/Terrain/TerrainRenderer.cs` | HOLD_AFTER_WAVE_1 | Refactor over terrain path while fused behavior still unresolved |
| `src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs` | HOLD_AFTER_WAVE_1 | Directly intersects tile alpha handling |
| `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` | HOLD_AFTER_WAVE_1 | Decode pathway risk area |
| `src/WoWMapConverter/WoWMapConverter.Core/Formats/LichKing/Mcal.cs` | HOLD_AFTER_WAVE_1 | High MCAL regression risk; evaluate only after parity |

## UI-Only Rollback Track

Worktree branch: recovery/ui-only-rollback

Immediate rollback candidates:
- ViewerApp.cs broad panel/menu/window changes
- imgui.ini layout churn
- optional ViewerApp partials that expanded workflow complexity without terrain-debug value

Keep candidates (UI):
- targeted controls that directly help terrain diagnosis
- map load / terrain controls that do not alter decode or blend behavior

## Terrain-First Surgical Track

Worktree: _recovery_343dadf (baseline)

First pass:
1. Identify exact alpha+shadow merged-pass change set.
2. Revert/split that topology first.
3. Validate on alpha and later-client target areas.
4. Re-introduce non-regressing features by small replay waves.

## Required Runtime Gates Per Replay Step

1. Alpha visual parity gate
2. Later-client target terrain gate
3. Alpha-debug overlay visibility gate
4. Missing-texture fallback behavior gate
