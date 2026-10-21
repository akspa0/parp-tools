# P5 bounded archaeology receipt — 2026-10-21

## Result and limits

Implemented the operator-bounded P5 source changes. **Integration is pending parent-owned
navigation/caller migrations; this is not a passing P5 runtime gate.** The IA and operator's
five-page instruction supersede the older “6-page” wording in the plan/tasks. No task, status,
memory, shell, app, host-interface, navigator, or panel files were edited by this slice. No commit.

## Changed files

All source paths below are under `wow-viewer/src/viewer/WoWViewer/`:

- `Workbench/Services/Archaeology/ArchaeologyPanelService.Sidebars.cs`: deleted the floating
  UniqueId window and its private content bridge, legacy five-tab dispatcher/nested combo, and
  Quick launchers. Kept the canonical dispatcher with cases 0–4: Weak Signal & Stratigraphy,
  UniqueId Timeline, Layers & Provenance, Playback & Capture, Cartography. PM4 is an explicit
  cross-destination button, not a sixth page.
- `Workbench/Services/Archaeology/ArchaeologyPanelService.cs`: removed the legacy tab-index state.
- `Workbench/Services/Archaeology/ArchaeologyPanelService.Host.cs`: removed unused legacy-only
  window/export flags, temporal-stratigraphy/visual-investigation/launcher bridges, editor-host
  initialization bridge, and compatibility navigation overloads. Shared analysis/capture bridges remain.
- `Workbench/Services/Archaeology/ArchaeologyPanelService.PhaseLayers.cs`: removed the bespoke
  Cartography sub-tab buttons. Layer Stack, DBC Phase Discovery, Add Phase Map, Donor Tile Grid
  Picker, Synthesized Minimap, and Save Merged Output are sibling shared-widget sections. The
  donor picker no longer has another nested collapsing header. Existing project persistence,
  layer locks/reordering/channels/transforms, refresh, discovery, donor placement, and export
  actions remain. DBC prerequisites now have an explicit empty state.
- `Workbench/Services/Archaeology/ArchaeologyPanelService.Editor.cs`: removed PM4 reconciliation
  UI and its private preview/apply workflow per T066/AD-9. Kept the existing output-materialization
  operation for editor undo/redo; see [P6](p6-editor.md).
- This receipt and [the P6 receipt](p6-editor.md) are the only documentation writes.

## Parent integration — exact remaining callsites

Paths are relative to `wow-viewer/src/viewer/WoWViewer/`; line numbers describe this handoff.

| Parent-owned location | Required migration |
|---|---|
| `ViewerApp.cs:1453` | Remove the `DrawUniqueIdArchaeologyWindow()` call and its enclosing obsolete window-only condition. Do not recreate the floating window. |
| `Workbench/Services/WorkbenchPanels/WorkbenchPanelsService.cs:746` | Remove `DrawArcheologySubTabContent()` and its legacy inspector route; use the canonical archaeology destination, not a second content renderer. |
| `Workbench/Services/WorkbenchPanels/WorkbenchPanelsService.cs:1122` | Remove `DrawArchaeologyQuickSection()` and its Quick-only section. |
| `Workbench/WorkbenchNavigator.cs:145–146` | `GetArchaeologyWorkbenchLabels()` must return exactly the five labels listed above. Delete “PM4 Analysis”; Cartography becomes index **4**, not 5. |
| `Workbench/Services/WorkbenchPanels/WorkbenchPanelsService.cs:892` | This existing `OpenWorkbenchTab(WorkbenchTab.Archaeology, 4)` is an old **PM4** route. Replace it with Viewer PM4 navigation; leaving it unchanged now opens Cartography. |
| Persisted navigation/settings | Reset old archaeology page indices along with the parent's layout reset; old 4 meant PM4, old 5 meant Cartography. Do not blindly retain those values. |
| `Pm4WorkbenchService.OpenPm4Workbench()` | The shortcut now invokes the existing canonical `WorkbenchTab.Inspect` route, then existing PM4 inspector focus. Parent P4 must coordinate its Viewer destination name and the current `FocusInspectorTab(4)` index with the new inspector tab order. No new host API was introduced. |

The removed host bridges are private to this service, not declarations on the shared host
interface. Parent may remove their shared declarations only after checking other consumers.
The Layer Stack's existing Save/Load project actions were preserved; any later IA relocation to
the parent's left-sidebar Layer Projects surface must not duplicate them.

## Verification commands and receipts

Run from the repository root with PowerShell 7. The following source-only check is reproducible
without building the blocked shell. It parses C# with the installed SDK's Roslyn parser, checks
the bounded structure, and compares preserved method tokens against HEAD (not runtime behavior).

### Source verification script

```powershell
$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()
$base = 'wow-viewer/src/viewer/WoWViewer/'
$arch = $base + 'Workbench/Services/Archaeology/'
$pm4 = $base + 'Workbench/Services/Pm4Workbench/'
$pages = $base + 'Workbench/Pages/'
$sdk = Get-ChildItem ((Split-Path (Get-Command dotnet).Source) + '/sdk') -Directory |
    Sort-Object Name -Descending | Select-Object -First 1
Add-Type -Path ($sdk.FullName + '/Roslyn/bincore/Microsoft.CodeAnalysis.dll')
Add-Type -Path ($sdk.FullName + '/Roslyn/bincore/Microsoft.CodeAnalysis.CSharp.dll')
function Check([bool]$ok, [string]$message) {
    if (-not $ok) { throw $message }
}
function Parse([string]$text) {
    [Microsoft.CodeAnalysis.CSharp.CSharpSyntaxTree]::ParseText($text).GetRoot()
}
function MethodTokens($root, [string]$name) {
    $method = @($root.DescendantNodes() | Where-Object {
        $_ -is [Microsoft.CodeAnalysis.CSharp.Syntax.MethodDeclarationSyntax] -and
        $_.Identifier.ValueText -eq $name
    })
    Check ($method.Count -eq 1) ('Expected one method: ' + $name)
    ($method[0].DescendantTokens() | ForEach-Object { $_.ToString() }) -join ' '
}
$files = @(Get-ChildItem $arch -Filter '*.cs'; Get-ChildItem $pm4 -Filter '*.cs';
    Get-Item ($pages + 'EditorPlacementObjectsPage.cs'), ($pages + 'EditorDataIoPage.cs'),
        ($pages + 'EditorWorkbenchPages.cs'))
foreach ($file in $files) {
    $tree = [Microsoft.CodeAnalysis.CSharp.CSharpSyntaxTree]::ParseText(
        [IO.File]::ReadAllText($file.FullName))
    $errors = @($tree.GetDiagnostics() | Where-Object Severity -eq Error)
    Check ($errors.Count -eq 0) ($file.Name + ': ' + ($errors -join '; '))
}
Write-Output ('PASS: ' + $files.Count + ' scoped C# files; zero syntax errors.')
$archText = (Get-ChildItem $arch -Filter '*.cs' | Get-Content -Raw) -join "`n"
$pm4Text = (Get-ChildItem $pm4 -Filter '*.cs' | Get-Content -Raw) -join "`n"
Check ($archText -notmatch 'DrawArcheologySubTabContent|DrawUniqueIdArchaeologyWindow|DrawUniqueIdArchaeologyContent|DrawArchaeologyQuickSection|DrawNestedSubTabStrip|ArcheologyBottomTab|_cartographySubTab') 'Legacy archaeology surface remains'
$dispatch = MethodTokens (Parse (Get-Content ($arch + 'ArchaeologyPanelService.Sidebars.cs') -Raw)) 'DrawArchaeologyWorkbenchSubTabContent'
Check (([regex]::Matches($dispatch, 'case [0-4] :')).Count -eq 5) 'Expected five archaeology cases'
Check ($dispatch -notmatch 'case 5') 'Sixth archaeology page remains'
$layers = Get-Content ($arch + 'ArchaeologyPanelService.PhaseLayers.cs') -Raw
foreach ($label in @('Layer Stack', 'DBC Phase Discovery', 'Add Phase Map', 'Donor Tile Grid Picker', 'Synthesized Minimap', 'Save Merged Output')) {
    Check ($layers.Contains('SectionHeader("' + $label + '"')) ('Missing shared section: ' + $label)
}
Check ($pm4Text -notmatch 'Pm4BottomTab|DrawPm4AlignmentContent|SaveCurrentPm4Alignment|DrawPm4WmoCorrelationContent|DrawPm4MatchCuration|TrySavePm4ObjectMatch|Pm4ObjectMatchStore|Pm4ObjectMatchEntry|BuildSavedPm4ObjectMatchKey|ExportPm4WmoCorrelationJson') 'Offline-only PM4 UI or store consumer remains'
Check ($archText -notmatch 'DrawPm4ReconciliationContent|DrawReconciliationPanel|RunReconciliationPreview|ApplyAcceptedReconciliation') 'Reconciliation UI remains'
$main = Get-Content ($pm4 + 'Pm4WorkbenchService.cs') -Raw
Check ($main.Contains('["Overlay", "Selection", "Info", "Outliner", "Object Groups"]')) 'Incorrect PM4 inspection pages'
Check (([regex]::Matches($pm4Text, 'DrawPm4ExportCommandSet\(\)')).Count -eq 2) 'Expected one PM4 export draw caller plus declaration'
Check (-not (Test-Path ($base + 'Pm4ObjectMatchStore.cs'))) 'Manual store still exists'
$placement = Get-Content ($pages + 'EditorPlacementObjectsPage.cs') -Raw
$io = Get-Content ($pages + 'EditorDataIoPage.cs') -Raw
Check ($placement.Contains('"Archaeology Tasks"') -and $placement.Contains('"Rosetta Object Library"') -and $placement.Contains('"Population"')) 'Editor section owner missing'
Check ($placement -notmatch 'PM4 Placement Tools|Tasks & Workspace|3D Object Library') 'Legacy editor section remains'
Check ($io.Contains('"Archaeology Imports"') -and $io.Contains('"New Map Creator"')) 'Data I/O owner missing'
Check ($io -notmatch 'DrawPm4Exports|"PM4 Exports"|"Imports & Exports"') 'Duplicate editor export home remains'
Write-Output 'PASS: bounded archaeology/editor/PM4 structure and single export home.'
$groups = @(
    @(($arch + 'ArchaeologyPanelService.Sidebars.cs'), ($arch + 'ArchaeologyPanelService.Sidebars.cs'), @('DrawArcheologyRangeSubTab', 'DrawArcheologyLayersSubTab', 'DrawArcheologyPlaybackSubTab', 'DrawArcheologyPlaybackTransportControls', 'StartArcheologyPlayback', 'StopArcheologyPlayback', 'DrawArcheologyCaptureSubTab')),
    @(($arch + 'ArchaeologyPanelService.PhaseLayers.cs'), ($arch + 'ArchaeologyPanelService.PhaseLayers.cs'), @('DrawDbcPhaseDiscovery', 'DrawPhaseLayerAddControls', 'DrawPhaseLayerStack', 'DrawPhaseLayerStatusBadge', 'PhaseChannelCheckbox', 'DrawCartographySaveMergedSubTab')),
    @(($arch + 'ArchaeologyPanelService.Editor.cs'), ($arch + 'ArchaeologyPanelService.Editor.cs'), @('MaterializeReconciliationOutput')),
    @(($pm4 + 'Pm4WorkbenchService.cs'), ($pm4 + 'Pm4WorkbenchService.cs'), @('DrawPm4SceneFacts', 'DrawPm4Outliner', 'DrawPm4ObjectGroups')),
    @(($pm4 + 'Pm4WorkbenchService.Alignment.cs'), ($pm4 + 'Pm4WorkbenchService.Exports.cs'), @('ExportPm4ObjectsJson', 'ExportPm4ObjectsObjSet', 'FocusCameraOnBounds', 'ApplySavedPm4AlignmentToScene'))
)
$preserved = 0
foreach ($group in $groups) {
    $beforeText = (git show ('HEAD:' + $group[0])) -join "`n"
    Check ($LASTEXITCODE -eq 0) ('Cannot read baseline ' + $group[0])
    $before = Parse $beforeText
    $after = Parse (Get-Content $group[1] -Raw)
    foreach ($name in $group[2]) {
        Check ((MethodTokens $before $name) -ceq (MethodTokens $after $name)) ('Preserved method changed: ' + $name)
        $preserved++
    }
}
git diff --quiet -- ($pm4 + 'Pm4WorkbenchService.Collections.cs')
Check ($LASTEXITCODE -eq 0) 'PM4 collection/export data logic changed'
Write-Output ('PASS: ' + $preserved + ' preserved methods token-identical to HEAD; Collections unchanged.')
```

Invocation (the first fenced PowerShell block is the script above):

```powershell
$receipt = Get-Content wow-viewer/specs/280-ui-audit-refactor/evidence/p5-archaeology.md -Raw
$script = [regex]::Match($receipt, '(?s)```powershell\r?\n(.*?)\r?\n```').Groups[1].Value
& ([scriptblock]::Create($script))
```

Result: **exit 0**, using the invocation above via `pwsh -NoProfile -Command`:

```text
PASS: 15 scoped C# files; zero syntax errors.
PASS: bounded archaeology/editor/PM4 structure and single export home.
PASS: 21 preserved methods token-identical to HEAD; Collections unchanged.
```

This check does not type-check the entire app or exercise a client. During verification-script
development, checks exited 1 for an overbroad comment-matching assertion, array-expression
grouping, and native git output encoding; the script above corrects those test issues explicitly.
No preservation discrepancy required a source-code change.

## Criterion → evidence / proof boundary

| Contract | Evidence | Runtime status |
|---|---|---|
| T050–T052 delete duplicate archaeology entry points | Source check above; deleted methods and private nested combo | Parent caller removals pending |
| Five canonical pages; PM4 shortcut only | Dispatcher cases 0–4; parent label migration listed above | Destination click proof operator-owned |
| T053 flatten Cartography Layers | Six sibling shared-widget sections, original data helpers preserved | Layer interaction proof operator-owned |
| T054 remove deleted-only private host bridges | Host bridge diff; no legacy window/Quick/compatibility navigation bridges | Shared host cleanup parent-owned |
| Preserve analysis, playback, composition behavior | Token comparison of named existing helpers, not synthetic output | Real map/playback/export proof not claimed |

Build baseline, post-change failure, editor migration gate, and PM4 integration are recorded in
[P6](p6-editor.md). No full build, test-suite, real-client, visual, or real export success is claimed.
