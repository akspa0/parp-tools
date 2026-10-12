# Receipt — T060 & T092 File Splits: WorldAssetManager & WmoRenderer (2026-09-30)

Splits required by SpecKit governance (`AGENTS.md` §10 God-Class Line Budget: target < 2,000 lines per file):
- **T060**: Split `Terrain/WorldAssetManager.cs` (2,128 lines before -> **1,576 lines**).
- **T092**: Split `Rendering/WmoRenderer.cs` (3,824 lines before -> **1,937 lines**).

## Files changed and created

### T060 (`Terrain/WorldAssetManager.cs`)
- `src/viewer/WoWViewer/Terrain/WorldAssetReadStats.cs` (new): read timing and size stats records.
- `src/viewer/WoWViewer/Terrain/WmoMeshSummary.cs` (new): WMO collision mesh summary record.
- `src/viewer/WoWViewer/Terrain/MdxCollisionMeshSummary.cs` (new): MDX collision mesh summary record.
- `src/viewer/WoWViewer/Terrain/AssetManifest.cs` (new): asset manifest record.
- `src/viewer/WoWViewer/Terrain/WmoMeshSummaryBuilder.cs` (new): encapsulates WMO group collision geometry extraction and batch summary construction.
- `src/viewer/WoWViewer/Terrain/WorldAssetPathResolver.cs` (new): encapsulates client asset path canonicalization, MPQ/CASC resolution, and extension normalization.
- `src/viewer/WoWViewer/Terrain/WorldAssetManager.cs`: delegates path resolution, collision mesh summary building, and data records. File reduced from 2,128 to **1,576 lines**.

### T092 (`Rendering/WmoRenderer.cs`)
- `src/viewer/WoWViewer/Rendering/WmoDoodadInfo.cs` (new): doodad inspection info model.
- `src/viewer/WoWViewer/Rendering/WmoOpaqueDoodadBatchItem.cs` (new): instanced batch item structure.
- `src/viewer/WoWViewer/Rendering/WmoRenderPass.cs` (new): render pass enum.
- `src/viewer/WoWViewer/Rendering/WmoRenderStats.cs` (new): render pass diagnostics and draw count stats.
- `src/viewer/WoWViewer/Rendering/WmoGeometryHelper.cs` (new): vertex lighting calculation, MONR/parsed normal computation, AABB transformation, EGx blend mode resolution, and packed BGRA unpacking.
- `src/viewer/WoWViewer/Rendering/WmoLiquidRenderer.cs` (new): MLIQ shader compilation and caching, liquid vertex mesh building, orientation scoring and layout quarter-turn transforms, pass-3 liquid rendering, and mesh disposal.
- `src/viewer/WoWViewer/Rendering/WmoMaterialManager.cs` (new): material texture cache, deferred texture upload queue and per-frame budget dispatcher, batch material fallback resolution, and GPU texture cleanup.
- `src/viewer/WoWViewer/Rendering/WmoDoodadController.cs` (new): doodad sets/defs, MODN name resolution, instance lifecycle, M2/MDX loading, skin discovery, embedded root-profile fallbacks, ray picking, bounds, animations, instanced batching, and shared cache integration.
- `src/viewer/WoWViewer/Rendering/WmoRenderer.cs`: delegates doodad control, material management, liquid rendering, and geometry calculations to owned service classes. File reduced from 3,824 to **1,937 lines**.

## Verification

| Command | Exit | Result |
|---|---|---|
| `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` | 0 | 0 errors; clean build across all projects |
| `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` | 1 | Failed: 10, Passed: 1,608, Skipped: 1, Total: 1,619 (exact parity with baseline pre-existing failures) |

## Criterion → evidence

| Criterion | Evidence |
|---|---|
| T060: `WorldAssetManager.cs` file budget (< 2,000 lines) | 1,576 lines measured |
| T092: `WmoRenderer.cs` file budget (< 2,000 lines) | 1,937 lines measured |
| Zero behavioral regressions | Solution builds with 0 errors; `WowViewer.Core.Tests` runs with exact pre-existing 10 failures |
