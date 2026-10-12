# Receipt — T090 amendment D: WMO doodad loads (2026-09-27)

Operator: "Fix building doodad loads" (answer to the D/P3 question after `evidence/capture-t073-2026-09-27.md`).

## Files changed
- `src/viewer/WoWViewer/Rendering/WmoDoodadModelShare.cs` (new): reference-counted doodad model cache shared by
  every world WMO of one `WorldAssetManager`; owns a `SkinPathIndex`.
- `src/viewer/WoWViewer/Rendering/WmoRenderer.cs`: `DoodadModelShare` init property; `GetOrLoadDoodadModel`
  acquires from / adds to the share; `Dispose` releases instead of disposing when shared; cache hits do not
  count against the per-frame doodad load allowance (D4); `ResolveBestSkinPath` uses the share's
  `SkinPathIndex` (D1); adapter parse skipped when the native renderer is preferred (D2); native doodad
  renderers created with `deferInitialTextureLoads: _deferInitialDoodadLoads` (D3; true only for world WMOs).
- `src/viewer/WoWViewer/Terrain/WorldAssetManager.cs`: owns the share, passes it to world WMOs, times
  `ProcessDeferredDoodadLoads` calls that loaded something as `ModelLoadPhase.WmoDoodadLoad` (D5).
- `src/viewer/WoWViewer/Terrain/ModelLoadPhaseStats.cs`: `WmoDoodadLoad = 9`, `PhaseCount = 10`.
- `src/viewer/WoWViewer/Workbench/Services/Chrome/ViewerChromeService.Sidebars.cs`: Runtime Stats line
  "WMO doodad load avg/max (count), shared doodad models, hits".
- Unchanged: standalone viewer and screenshot WMO renderers (no share: own cache and whole-list skin scan).

## Verification
| Command | Exit | Result |
|---|---|---|
| `dotnet build WowViewer.slnx -c Debug -p:EnableWindowsTargeting=true` | 0 | 0 errors; compiler warnings identical to base (NU1903 excluded) |
| `dotnet test WowViewer.slnx -c Debug` | 1 | 26 failures (Core.Tests 18, PM4 8), identical to the pre-existing E4 environmental set |

## Criterion → evidence
| Criterion | Evidence |
|---|---|
| D1 indexed skin lookup | Code; `SkinPathIndex` equivalence already shown for P1 (0 mismatches / 1,854 queries) |
| D2–D4 | Code only |
| DeferredAssetLoads while streaming drops | **Not proven.** Operator capture owed (T091) |
