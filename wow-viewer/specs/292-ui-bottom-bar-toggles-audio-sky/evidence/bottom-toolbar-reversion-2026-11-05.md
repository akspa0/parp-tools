# Spec 292 Amendment — Corrected Bottom Toolbar Receipt

**Date:** 2026-11-05  
**Scope:** Apply the operator's latest toolbar direction: per-liquid-family toggles, resident-terrain layer/alpha counts, no Sky/Fog, one Mute control, and muted-by-default audio. Texture Masks remains absent from the bottom toolbar.

## Files changed

| File | Change |
|---|---|
| [`BottomBarService.cs`](../../src/viewer/WoWViewer/Workbench/Services/Chrome/BottomBarService.cs) | Renders Water/Ocean/Magma/Slime, Doodads/WMOs/Terrain/PM4/Grid/Mute, L0–Ln sized to `TerrainRenderer.MaxResidentLayerCount`, one A1–An selector per non-base layer, and both opacity sliders. A<n> selects its channel and enables the overlay; clicking the selected channel again disables it. |
| [`BottomBarControlPolicy.cs`](../../src/viewer/WoWViewer/Workbench/Services/UiManager/BottomBarControlPolicy.cs) | New pure policy for resident layer/alpha counts and the four liquid-family toggle definitions. |
| [`LiquidRenderer.cs`](../../src/viewer/WoWViewer/Terrain/LiquidRenderer.cs) | Adds per-family visibility for MCLQ/MH2O and WL* meshes; the render loop skips hidden families. |
| [`WmoRenderer.cs`](../../src/viewer/WoWViewer/Rendering/WmoRenderer.cs), [`WmoLiquidRenderer.cs`](../../src/viewer/WoWViewer/Rendering/WmoLiquidRenderer.cs) | Adds shared WMO MLIQ family visibility and filters liquid meshes by basic type. |
| [`ViewerSettingsService.cs`](../../src/viewer/WoWViewer/Workbench/Services/Settings/ViewerSettingsService.cs) | Persists the four liquid-family toggles and alpha overlay/channel; migrates the legacy single `BottomToggleWater`; defaults audio mute to true. |
| [`WorldScene.cs`](../../src/viewer/WoWViewer/Terrain/WorldScene.cs) | Audio starts muted by default. |
| [`MinimapAndStatusService.MinimapAndStatus.cs`](../../src/viewer/WoWViewer/Workbench/Services/MinimapStatus/MinimapAndStatusService.MinimapAndStatus.cs) | Removes the duplicate status-bar Mute button; the bottom toolbar is the single quick Mute control. |
| [`SkyControlsService.cs`](../../src/viewer/WoWViewer/Workbench/Services/Sky/SkyControlsService.cs) | Removes the obsolete bottom-bar Sky state write. |
| [`ViewerFooterLayoutTests.cs`](../../tests/WoWViewer.Core.Tests/ViewerFooterLayoutTests.cs) | Covers the corrected item inventory, resident layer/alpha counts, and the no-terrain case. |
| [`ui-controls.json`](../../eng/ui/ui-controls.json), [`UI-REFERENCE.md`](../../docs/WoWViewer/UI-REFERENCE.md) | Registers the four liquid-family toggles and regenerates the reference. |
| [`spec.md`](../spec.md), [`tasks.md`](../tasks.md), [`STATUS.md`](../../STATUS.md), [`activeContext.md`](../../../memory-bank/activeContext.md), [`progress.md`](../../../memory-bank/progress.md) | Records the operator-directed correction, verification results, and the remaining real-client proof gate. |

## Verification

| Command | Exit | Result |
|---|---:|---|
| `pwsh -NoProfile -File wow-viewer/eng/ui/Update-UiReference.ps1 -Write -RepositoryRoot i:/parp/parp-tools` | 0 | Generated `docs/WoWViewer/UI-REFERENCE.md`. |
| `pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1 -RepositoryRoot i:/parp/parp-tools` | 0 | Inventory 178 rows / 166 distinct labels; 166 registry entries; 0 errors, 0 warnings. |
| `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.CrossPlatform.csproj -c Debug --no-restore --nologo -v minimal -t:Compile` | 0 | Viewer compile succeeded, 0 errors. |
| `dotnet test wow-viewer/tests/WoWViewer.Core.Tests/WoWViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~ViewerFooterLayoutTests" --nologo -p:BaseOutputPath=i:/parp/parp-tools/wow-viewer/artifacts/verify-bin/` | 0 | 30 passed, 0 failed, 0 skipped. |
| `dotnet build wow-viewer/WoWViewer.slnx -c Debug --no-restore --nologo -v minimal -p:BaseOutputPath=i:/parp/parp-tools/wow-viewer/artifacts/verify-bin/` | 0 | Full solution build succeeded, 0 errors, 266 warnings. |

The normal `bin` outputs are held by the operator's running viewer (`ParpToolsWoWViewer`, PID 12260), so the full build and tests were redirected to `wow-viewer/artifacts/verify-bin/` inside the repository. The source is identical; only the output path differs.

## Requirement → evidence

| Requirement | Evidence |
|---|---|
| Exclude Sky/Fog from the bottom toolbar | `BottomBarService.DrawBottom` builds only the requested scene/liquid toggle IDs; registry places Sky/Fog only in Workspace. |
| Auto-size L/A controls to the loaded terrain | `BottomBarControlPolicy.ResolveLayerCount` uses `TerrainRenderer.MaxResidentLayerCount`; `ResolveAlphaChannelCount` is `layers - 1`; tests cover 0, 4, and 8 layers. |
| Separate liquid-family controls | Water/Ocean/Magma/Slime toggles drive `LiquidRenderer` per-family flags for MCLQ/MH2O and WL* meshes and `WmoRenderer` MLIQ family flags. |
| A1–An select one channel and enable the overlay | Seven selected-state `Selectable` controls apply one `AlphaMaskChannel` at a time and the shared `ShowAlphaMask` flag; clicking the selected channel disables the overlay. |
| Retain the other scene toggles and both opacity sliders | Draw code retains Doodads, WMOs, Terrain, PM4, separate Tile/Chunk/Cell grid toggles, Mute, M2/MDX opacity, and WMO opacity. |
| Layer/alpha controls scale beyond eight | The toolbar now has one **Layers** button; its popup lists the resident layers, opens a per-layer detail popup, and lists alpha channels. No fixed L0–L7/A1–A7 row remains. |
| Reconstruction Workbench routing | The right action bar's **Reconstruction** button opens `ViewerPanel.Reconstruction`, which draws the map-layer stack; the old Archaeology panel remains available through its own destination. |
| Skybox models are selectable | `SceneAtmosphere.AvailableSkyboxModels` merges LightSkybox DBC names with every `.mdx`/`.m2` under `World\Skybox`; the Sky panel combo uses that list. |
| One Mute control, muted by default | Status-bar audio button removed; `WorldScene._audioMuted` and the settings default are true. |
| Persist state | Viewer settings load/save include the four liquid families and alpha overlay/channel; the legacy single Water value migrates. |
| Registry/reference and layout checks | UI verifier passes with zero warnings; 30 focused footer tests pass; viewer compile has zero errors. |

## Operator gate owed

Run the viewer on the target real client and visually confirm the complete toolbar is present and usable, L/A counts follow the loaded terrain, each liquid family toggles independently (including magma/slime/ocean), A1–An change the intended alpha overlay one channel at a time, the selected channel toggles off on a second click, opacity sliders affect their models, Mute appears once and starts muted, and state survives restart. Record the client build/root/fingerprint and screenshots. Build and unit-test results do not prove these runtime behaviors.
