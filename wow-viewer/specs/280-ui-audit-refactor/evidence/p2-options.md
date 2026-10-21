# P2 Options content provider — bounded implementation receipt

Date: 2026-10-21 UTC. Scope: Chrome service partials only, plus this receipt.
Provider work is implemented; **P2 integration and visual acceptance are not complete**.
No task, memory-bank, status, host-interface, renderer, core, shell or app edits; no commits.

## Changed files

All source links below are relative to this receipt.

| File | Change |
|---|---|
| [ViewerChromeService.cs](../../../src/viewer/WoWViewer/Workbench/Services/Chrome/ViewerChromeService.cs) | Content-only ownership contract; removed unused splitter constant and imports. |
| [ViewerChromeService.Host.cs](../../../src/viewer/WoWViewer/Workbench/Services/Chrome/ViewerChromeService.Host.cs) | Removed obsolete layout/navigation/toolbar bridges; existing host contract unchanged. |
| [ViewerChromeService.Sidebars.cs](../../../src/viewer/WoWViewer/Workbench/Services/Chrome/ViewerChromeService.Sidebars.cs) | Deleted both bars, splitter windows, three standalone popup paths, toolbar measurements and duplicate workspace render controls. Retained workspace content, width helpers, runtime diagnostics and time-of-day controls. |
| [ViewerChromeService.Options.cs](../../../src/viewer/WoWViewer/Workbench/Services/Chrome/ViewerChromeService.Options.cs) | Added grouped Options content and a compact FPS/frame-time content provider. |
| [ViewerChromeService.Opacity.cs](../../../src/viewer/WoWViewer/Workbench/Services/Chrome/ViewerChromeService.Opacity.cs) | Extracted opacity/texture content without a popup; retained presets, clamping, M2/MDX synchronization and settings persistence. |

## Parent integration contract

- Call [DrawOptionsContent()](../../../src/viewer/WoWViewer/Workbench/Services/Chrome/ViewerChromeService.Options.cs:11) inside the manager-owned Options region. It does not create, position, size or open a window/popup, change destinations, or reference the new manager.
- Call [DrawStatsContent()](../../../src/viewer/WoWViewer/Workbench/Services/Chrome/ViewerChromeService.Options.cs:44) in the persistent bottom bar if desired. It displays ImGui's measured framerate and reciprocal milliseconds, not a new performance sampler. The parent may instead own the readout.
- Seven shared-widget groups: Display, Overlays, Layers, Grids, Objects, Audio, Simulation. Display delegates to the existing Render Quality, Lighting and Theme content APIs; Audio delegates to the existing Audio content API.
- Window rectangles, Options/chat anchoring, destination buttons, splitters, popup lifetime and design tokens remain parent-owned.
- [DrawRuntimeStatsPanelContent()](../../../src/viewer/WoWViewer/Workbench/Services/Chrome/ViewerChromeService.Sidebars.cs) and [DrawTimeOfDayControl()](../../../src/viewer/WoWViewer/Workbench/Services/Chrome/ViewerChromeService.Sidebars.cs) are unchanged. The workspace content and fixed-sidebar width/clamp helpers remain callable while their callers migrate.

### Required caller fixes outside this assignment

The final build reports exactly these four missing-member errors:

| Caller | Removed member / parent action |
|---|---|
| [ViewerApp.cs:1404](../../../src/viewer/WoWViewer/ViewerApp.cs:1404) | Remove the call to DrawFixedSidebarSplitters; manager owns splitters. |
| [ViewerApp.cs:1407](../../../src/viewer/WoWViewer/ViewerApp.cs:1407) | Remove the call to DrawToolbar; no replacement top-toolbar window here. |
| [ViewerApp.cs:1409](../../../src/viewer/WoWViewer/ViewerApp.cs:1409) | Replace the DrawBottomBar call with manager-owned bottom-bar navigation and stats. |
| [ViewerApp_Host.cs:163](../../../src/viewer/WoWViewer/ViewerApp_Host.cs:163) | Rehome or remove the DrawToolbarPopupButton delegation; it no longer belongs to this provider. |

The popup bridge still has downstream users: [ModelInspectorPanelService.Sidebars.cs:212](../../../src/viewer/WoWViewer/Workbench/Services/ModelInspector/ModelInspectorPanelService.Sidebars.cs:212), [SqlSpawnStreamingService.cs:744](../../../src/viewer/WoWViewer/Workbench/Services/SqlSpawnStreaming/SqlSpawnStreamingService.cs:744), and [WorldObjectsPanelService.cs](../../../src/viewer/WoWViewer/Workbench/Services/WorldObjects/WorldObjectsPanelService.cs) (PM4, POI, WL body/transform and LIT actions). Move their popup requests to the parent-owned popup mechanism; do not restore window ownership here.

Top-toolbar investigation mode buttons were duplicates of the existing [investigation toolbox](../../../src/viewer/WoWViewer/Workbench/Services/Investigation/InvestigationService.Investigation.cs:65), which remains untouched. The old PM4 Inspect navigation shortcut is intentionally not reproduced in Options: manager owns destination switching. Settings remains a global menu concern.

### Auto-Stream access gap

[Auto-Stream state](../../../src/viewer/WoWViewer/Workbench/Services/SqlSpawnStreaming/SqlSpawnStreamingService.cs:100) and its [chat content method](../../../src/viewer/WoWViewer/Workbench/Services/SqlSpawnStreaming/SqlSpawnStreamingService.cs:448) are private. No existing host property/content API exposes the toggle. This bounded implementation does not use reflection, duplicate state, or modify that service/host.

Simulation exposes the existing world's Spawn BBs Only state and states that Auto-Stream remains available in chat. **An actual Auto-Stream checkbox in Options is blocked on parent-owned exposure/delegation of that existing state.** The chat checkbox is preserved, not replaced with a disconnected control.

## Verification commands and actual results

Working directory: repository root. These commands are PowerShell 7 compatible.

```powershell
dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug --no-restore -v minimal
git diff --check -- wow-viewer/src/viewer/WoWViewer/Workbench/Services/Chrome
```

- Build run three times: initial exit 1 / 4 parent errors; import cleanup run exit 1 / 6 errors (two local missing rendering imports, fixed); final exit 1 / **4 parent errors, 315 warnings**, elapsed 4.47 s. Final errors are exclusively the caller sites above; no final errors in Chrome content files. This is **not a passing build**.
- Whitespace check: exit **0**, no whitespace defects (Git emitted a line-ending conversion warning).
- Tests/runtime launch not run: app integration intentionally remains broken until the parent migrates the old calls.

The following inline PowerShell assertions were also executed (exit **0**; line breaks below are formatting only):

```powershell
$ErrorActionPreference = 'Stop'; $root = 'wow-viewer/src/viewer/WoWViewer/Workbench/Services/Chrome/';
$all = (Get-ChildItem $root -Filter '*.cs' | ForEach-Object { Get-Content $_.FullName -Raw }) -join [Environment]::NewLine;
if ($all -match 'ImGui\.(Begin|BeginPopup|OpenPopup|SetNextWindowPos|SetNextWindowSize|PushStyleColor)\(') { throw 'Provider owns chrome' };
Write-Output 'PASS: zero chrome window/popup/style ownership calls';
$options = Get-Content ($root + 'ViewerChromeService.Options.cs') -Raw;
foreach ($group in @('Display','Overlays','Layers','Grids','Objects','Audio','Simulation')) { if ($options -notmatch ('DrawSectionHeader\(' + [char]34 + $group + [char]34)) { throw ('Missing group: ' + $group) } };
Write-Output 'PASS: all seven Options groups';
foreach ($api in @('DrawOptionsContent','DrawStatsContent','DrawRenderQualityContent','DrawLightingContent','DrawUiThemeSettingsContent','DrawAudioContent')) { if ($options -notmatch ($api + '\(')) { throw ('Missing content API: ' + $api) } };
Write-Output 'PASS: content entry points and existing Display/Audio/Theme APIs';
$old = (git show HEAD:wow-viewer/src/viewer/WoWViewer/Workbench/Services/Chrome/ViewerChromeService.Sidebars.cs) -join [Environment]::NewLine;
$current = Get-Content ($root + 'ViewerChromeService.Sidebars.cs') -Raw;
$anchor = '    internal void DrawRuntimeStatsPanelContent()';
if ($old.Substring($old.IndexOf($anchor)).Trim() -cne $current.Substring($current.IndexOf($anchor)).Trim()) { throw 'Runtime/time-of-day helpers changed' };
Write-Output 'PASS: runtime stats and time-of-day helpers unchanged';
$start = $old.IndexOf('    private void DrawObjectTransparencyPopupContent');
$end = $old.IndexOf('    /// <summary>', $start);
$originalOpacity = $old.Substring($start, $end - $start);
$opacity = Get-Content ($root + 'ViewerChromeService.Opacity.cs') -Raw;
$pattern = '(?:WmoRenderer|M2Renderer|MdxRenderer)\.\w+\s*=(?!=)[^;]+;';
$before = @([regex]::Matches($originalOpacity, $pattern) | ForEach-Object Value);
$after = @([regex]::Matches($opacity, $pattern) | ForEach-Object Value);
if (($before -join '|') -cne ($after -join '|')) { throw 'Opacity assignments changed' };
if ([regex]::Matches($originalOpacity, '_settings.SaveViewerSettings\(\)').Count -ne [regex]::Matches($opacity, '_settings.SaveViewerSettings\(\)').Count) { throw 'Opacity persistence changed' };
Write-Output ('PASS: all ' + $after.Count + ' opacity/texture assignments and persistence calls preserved');
foreach ($state in @('ShowAlphaMask','ShowShadowMap','UseMccv','ShowContours','ShowLiquid','ShowWlLiquids','ShowBoundingBoxes','ShowPm4Overlay','ShowWdlTerrain','ShowTileGrid','ShowChunkGrid','ShowCellGrid','SetShowLayer','SetIgnoreTerrainHolesGlobally','SetTerrainWireframeEnabled','SetObjectWireframeEnabled','WorldDoodadAnimationEnabled','ToggleWireframe','ColorSpawnBoundingBoxes','ShowSpawnBoundingBoxesOnly','_standaloneWmoGroupOverlayEnabled','_standaloneWmoGroupLabelsAllEnabled','AuthenticWoWGauntlet')) { if ($options -notmatch [regex]::Escape($state)) { throw ('Missing preserved control: ' + $state) } };
Write-Output 'PASS: all 23 render/object/simulation state paths retained';
git diff --check -- wow-viewer/src/viewer/WoWViewer/Workbench/Services/Chrome;
if ($LASTEXITCODE -ne 0) { throw 'Diff check failed' };
Write-Output 'PASS: diff whitespace check'
```

Actual output: all seven PASS lines above, with **26** opacity/texture assignments preserved. These assertions inspect real source files, not runtime controls; they do not establish interaction or rendering correctness.

## Criterion → evidence / remaining proof

| Criterion | Evidence / result |
|---|---|
| T022/T027: move render toggles into groups | Real Options source has all 23 asserted state paths; includes Base through resident L7, All/Base/Hide presets, terrain holes, WDL, grids, overlays, standalone WMO bounds/names, wireframes, animations and cursor. Structural pass; interaction unverified. |
| T023/T028: delete three popup paths | Zero window/popup/style ownership calls across Chrome source; opacity and layer controls now draw directly in Options. Structural pass. |
| Preserve opacity and persistence | Original-vs-current source check matches all 26 renderer assignments and save-call counts, including M2/MDX synchronization and reset presets. Runtime appearance unverified. |
| T025/T026: Render Quality/Lighting/Audio | Existing content API delegates present in Display/Audio, plus existing Theme API; no format/render implementations changed. Visual/audio proof unverified. |
| T024: keep stats | FPS/frame-time content API exists; parent must draw it in every destination. Persistent visibility not yet proven. |
| Simulation / IA map | Spawn BBs Only uses existing live world state; Auto-Stream remains chat-owned pending access gap above. Partial, explicitly blocked. |
| U1/U4: content-only provider | Direct Chrome window creation/positioning/style/popup calls removed. Existing content services remain their current implementations; full manager migration is parent-owned. |
| P2 full gate | Not claimed: destination navigation, Options anchoring, stats persistence, Utilities deletion and real-client visual/interaction proof require parent integration/operator runs. |
