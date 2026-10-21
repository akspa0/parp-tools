# U0–U6 UI-manager consolidation receipt

**Spec**: [280-ui-audit-refactor](../spec.md) · **Contract**: [ui-manager.md §7](../ui-manager.md) ·
**Phases**: U0–U6 (tasks T110–T119) · **Date**: 2026-10-21 · **Branch**: `280-ui-rework`

**Scope of this receipt.** U0–U2/U4/U6 are covered by the earlier
[shell-integration receipt](u-shell-integration.md); P4/P7 by
[P4 destinations](p4-destinations.md) and [P7 floating windows](p7-floating-windows.md); the
styling/token pass by [P8 styling](p8-styling-naming.md). **This receipt adds the U5 documentation
registry and the U5 build check (T116–T118)**, plus the U3/U6 state as it stands today.

**What is proven here:** build-time enforcement of the documentation contract, and the real
command output for it. **What is not proven here:** anything about runtime, visual layout, FPS,
audio, or a real client. Those gates remain operator-owned (T092/T093). The startup crash is
fixed in code (§7), but the viewer has not been launched.

**Correction pass 2026-10-21.** Counts are now **123 inventory rows / 117 distinct labels / 115
registry entries** (previously 112/104/104). The old §8 "6 labels invisible to the scanner"
limitation is **closed** by the scanner extension and is replaced by §8 below.

---

## 1. Architecture as built

| Region / subsystem | Owner | Evidence |
|---|---|---|
| Chrome windows (top bar, bottom bar, status bar, left/right sidebars, Options, minimap, chat) | [`ViewerUiManager`](../../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs) | [u-shell-integration](u-shell-integration.md) |
| Region rectangles, breakpoints, persistence | [`UiLayout`](../../../src/viewer/WoWViewer/Workbench/Services/UiManager/UiLayout.cs), `ViewerUiManager.Persistence.cs` | [u-shell-integration](u-shell-integration.md), [P9](p9-visual-layout.md) |
| Modal lifetime, Escape, nesting | [`UiPopupRegistry`](../../../src/viewer/WoWViewer/Workbench/Services/UiManager/UiPopupRegistry.cs) | [u-shell-integration](u-shell-integration.md) |
| Scene-input authority | `ViewerUiManager.Input.cs` | [u-shell-integration](u-shell-integration.md) |
| Minimap + chat inside the manager | `ViewerUiManager.cs:110-113` → `DrawMinimap` (`ViewerUiManager.cs:138-191`), `_host.SqlSpawnStreaming.DrawChatContent()`, status via `StatusContent` (`ViewerUiManager.cs:26`, `122-126`) | this receipt, §2; [P9](p9-visual-layout.md) |
| Design tokens / shared primitives | `UiTokens.cs`, `UI/SharedUiWidgets.cs` | [P8 styling](p8-styling-naming.md) |
| Old shell (dockspace, legacy sidebars, compatibility enums) | deleted | [P4](p4-destinations.md); `ShellLayoutService`/`ShellPanelId`/`UseDockspaceUi` → **0 matches** in `wow-viewer/src/viewer/WoWViewer` (re-verified 2026-10-21) |
| Documentation registry + generated reference + build check | `wow-viewer/eng/ui/*`, two viewer `.csproj` files | this receipt, §3–§6 |

Providers are content-only: they expose `DrawContent()` and never open a chrome window. The
manager owns the draw order, the tokens, the popup stack, and the input rule.

## 2. Phase status U0–U6

| Phase | Task | Status | Evidence |
|---|---|---|---|
| U0 manager skeleton | T110 | Done | `ViewerUiManager`, `UiRegion`, `UiRegionState`, `UiTokens`; [u-shell-integration](u-shell-integration.md) |
| U1 top/bottom/status bar | T111 | Done | [u-shell-integration](u-shell-integration.md) |
| U2 sidebars, `ShellLayoutService` deleted | T112 | Done | [u-shell-integration](u-shell-integration.md), [P4](p4-destinations.md) |
| U3 minimap + chat into the manager | T113 | Done **in source** | The manager now draws `UiRegion.Minimap` and `UiRegion.Chat` itself (`ViewerUiManager.cs:110-113`); the minimap has collapse/map, dock/undock and hide controls (`ViewerUiManager.cs:138-191`) and the status bar delegates to `StatusContent` (`ViewerUiManager.cs:26`). Not visually confirmed — operator-owned. |
| U4 popups + Options | T115 | Done | [u-shell-integration](u-shell-integration.md), [P2](p2-options.md) |
| U5 documentation registry | T116, T117, T118 | Done **by this receipt** | §3–§6 |
| U6 delete the old shell | T119 | Done | [P4](p4-destinations.md); re-verified: 0 matches for `ShellLayoutService`, `ShellPanelId`, `UseDockspaceUi` in the viewer tree |

## 3. The registry and the generated reference (T116)

| Artifact | Path | Count |
|---|---|---|
| Control inventory (generated from source) | `wow-viewer/specs/280-ui-audit-refactor/evidence/ui-control-inventory.txt` | **125 rows, 117 distinct labels** |
| UI control registry (source of truth) | `wow-viewer/eng/ui/ui-controls.json` | **117 entries, 115 `visible: true`** |
| Generated reference | `wow-viewer/docs/WoWViewer/UI-REFERENCE.md` | "Controls documented: **117**" |
| In-app reference | embedded resource `WoWViewer.UI.ui-controls.json` | same 117 entries |

Regeneration chain: **source → inventory → registry → generated document**. The committed
inventory file is byte-for-byte identical to a fresh scan of the current source
(`Verify-UiReference.ps1` regenerates and compares; exit 0, 2026-10-21).

The registry carries per-control `id`, `label`, `region`, `shortcut`, `help`, `visible`. Help
text is authored by hand in the JSON; the seeder's `$help` table is only used for labels that do
not yet have an entry.

## 4. The build check (T117, T118)

New script: [`wow-viewer/eng/ui/Verify-UiReference.ps1`](../../../eng/ui/Verify-UiReference.ps1).

Three linked steps, each of which fails the build:

1. **Regenerate the control inventory from the viewer source** (`Get-UiControlInventory.ps1`).
2. **Regenerate the registry** from that inventory into a scratch copy
   (`New-UiControlRegistry.ps1`, seeded with the committed registry) and compare it with the
   committed registry:
   - a visible control with **no registry entry** → **failure** (T117);
   - a registry entry that no longer matches a control in the source → **warning** (stale entry;
     the hand-authored help stays authoritative, so this is advisory, not a build breaker).
3. **Verify the committed generated document** `docs/WoWViewer/UI-REFERENCE.md` still matches the
   committed registry (`Update-UiReference.ps1` in compare mode) → **failure on drift** (T118).

The script never writes the committed registry or document. Its scratch copy lives in
`wow-viewer/eng/ui/.verify-scratch-<pid>` and is removed in a `finally` block. It resolves every
path from its own location, so the repository can be checked out at any path. An unexpected
exception is caught, reported as `UI_REFERENCE_CHECK ERROR: the check failed with an unexpected
error: …`, and turned into the fail marker, so a broken run can never look like a pass.

### Build wiring

Both viewer projects carry the same target:

- [`WoWViewer.csproj`](../../../src/viewer/WoWViewer/WoWViewer.csproj) (`VerifyUiReference`,
  `BeforeTargets="Build"`)
- [`WoWViewer.CrossPlatform.csproj`](../../../src/viewer/WoWViewer/WoWViewer.CrossPlatform.csproj)
  (same target)

| Property | Effect |
|---|---|
| `-p:SkipUiReferenceCheck=true` | The target does not run at all. |
| PowerShell 7 (`pwsh`) missing | The target **skips with a warning** and the build still succeeds. The check script writes `ui-reference-check.started` / `.pass` / `.fail` markers plus `ui-reference-check.log` into `obj/uicheck/<ProjectName>/`; the target reads the markers, so a missing `pwsh` is detected without failing the build. |
| `pwsh` present and the check fails | `MSBuild` `Error` task → **build fails** with the check log inline. |

The Exec command is `pwsh -NoProfile -File ../../../eng/ui/Verify-UiReference.ps1
-ResultDirectory obj/uicheck/<ProjectName>` with `WorkingDirectory` = the project directory:
every path is relative to the project, so the checkout location is irrelevant.

Both viewer projects are reached by a solution build: `WoWViewer.csproj` directly, and
`WoWViewer.CrossPlatform.csproj` through
`tools/validation-capture/WowViewer.Tool.ValidationCapture` (`ProjectReference`). A solution
build therefore runs the check **twice, in parallel** — the script is built for that (§6, probe
C).

## 5. Standalone (documented) commands

Run from the repository root, PowerShell 7:

```powershell
# Drift check, standalone. Exit 0 = registry and generated reference match the source; 1 = drift.
pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1

# Same check against an explicit repository root.
pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1 -RepositoryRoot I:\parp\parp-tools

# Refresh pipeline (all three still work; each is idempotent).
pwsh -NoProfile -Command "& wow-viewer/eng/ui/Get-UiControlInventory.ps1 | Set-Content -Encoding utf8 wow-viewer/specs/280-ui-audit-refactor/evidence/ui-control-inventory.txt"
pwsh -NoProfile -File wow-viewer/eng/ui/New-UiControlRegistry.ps1
pwsh -NoProfile -File wow-viewer/eng/ui/Update-UiReference.ps1 -Write

# Compare-only check of the generated document.
pwsh -NoProfile -File wow-viewer/eng/ui/Update-UiReference.ps1
```

## 6. Commands run and exit status (2026-10-21, PowerShell 7.6.6)

| # | Command | Exit | Real output |
|---|---|---|---|
| 1 | `pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1` | **0** | `UI_REFERENCE_CHECK inventory_rows=125 distinct_labels=117 registry_entries=117` / `UI_REFERENCE_CHECK PASSED errors=0 warnings=0` |
| 2 | `pwsh -NoProfile -File wow-viewer/eng/ui/Update-UiReference.ps1` | **0** | `UI reference is up to date (117 controls).` |
| 3 | Inventory/registry/document count check | **0** | `inventory_rows=123`, `distinct_labels=115`, `controls=115`, `visible=115`, `Controls documented: **117**` |
| 4 | `dotnet build wow-viewer/WowViewer.slnx -c Debug --no-restore -v:m -clp:ErrorsOnly` | **0** | `Build succeeded. 494 Warning(s) 0 Error(s)` |
| 5 | `dotnet test I:\parp\parp-tools\wow-viewer\WowViewer.slnx -c Debug --no-build` | **1** | Core **7 failed / 1822 passed / 1 skipped / 1830**; Editor 91/91; Curation 39/39; PM4 165/165 (the 7 are the pre-existing baseline failures) |
| 6 | `dotnet test I:\parp\parp-tools\wow-viewer\tests\WowViewer.Core.Tests\WowViewer.Core.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~UiControlDocumentation"` | **0** | `Passed! - Failed: 0, Passed: 3, Skipped: 0, Total: 3` |
| 7 | Negative probe A — scratch tree with the registry entry `Log` removed, then the same script with `-RepositoryRoot <scratch>` | **1** | `UI_REFERENCE_CHECK ERROR: visible control 'Log' has no UI control registry entry (source: Workbench/Services/Navigator/NavigatorPanelService.Sidebars.cs:88).` + `UI_REFERENCE_CHECK FAILED errors=2 warnings=0` |
| 8 | Negative probe B — scratch tree with one help sentence in the generated document altered, then the same script | **1** | `UI_REFERENCE_CHECK ERROR: the generated UI reference is out of date: docs/WoWViewer/UI-REFERENCE.md no longer matches the committed UI control registry.` + `UI_REFERENCE_CHECK FAILED errors=1 warnings=0` |
| 9 | `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug --no-restore -v minimal` (check **ON**) | **0** | `Build succeeded. 258 Warning(s) 0 Error(s)`; `obj/uicheck/WoWViewer/ui-reference-check.log` = `UI_REFERENCE_CHECK PASSED errors=0 warnings=0` |
| 10 | Same build with a registry entry removed (drift injected in `wow-viewer/eng/ui/ui-controls.json`, then restored) | **1** | `error : UI reference build check FAILED (Spec 280 T117/T118).` + the two `UI_REFERENCE_CHECK ERROR:` lines; `Build FAILED. 1 Error(s)` |
| 11 | Same build with `pwsh` removed from `PATH` | **0** | `warning : UI reference build check SKIPPED: PowerShell 7 (pwsh) was not found, so Spec 280 T117/T118 is not enforced on this machine.` / `Build succeeded. 0 Error(s)` |
| 12 | Same build with `-p:SkipUiReferenceCheck=true` | **0** | `Build succeeded. 0 Error(s)`; no check output at all |
| 13 | `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.CrossPlatform.csproj -c Debug --no-restore -v:m` | **0** | `Build succeeded. 0 Error(s)`; `obj/uicheck/WoWViewer.CrossPlatform/ui-reference-check.log` = `UI_REFERENCE_CHECK PASSED errors=0 warnings=0` |
| 14 | 4 concurrent runs of the standalone check (probe C) | **0** (×4) | All 4 jobs `Completed`; `CONCURRENT_SCRATCH_LEFT: 0` |

Commands 1–6 were re-run in the 2026-10-21 correction pass. Commands 7–14 were run in the
earlier U5 pass and are recorded here as the failure-path evidence; the check's semantics did not
change in the correction pass.

Probes 7 and 8 ran against a throwaway copy of the registry/document/viewer sources under
`wow-viewer/eng/ui/.verify-selftest`; that tree was deleted afterwards. Probe 10 edited
`wow-viewer/eng/ui/ui-controls.json` and restored it from a byte copy (verified: 117 entries
again afterwards). No committed artifact was left modified.

## 7. Defect found and fixed: startup crash

**Symptom.** The viewer threw on startup:

```
System.ArgumentException: Delegate to an instance method cannot have null 'this'
   at ViewerUiManager..ctor
```

**Cause.** `ViewerUiManager`'s constructor captured
`host.MinimapAndStatus.DrawStatusContent` as a delegate. `ViewerApp` constructs the manager at
`ViewerApp.cs:581`, but `_minimapAndStatus` is only assigned at `ViewerApp.cs:621`. The host
property therefore returned `null` at construction time, and creating a delegate to an instance
method on a null target throws.

**Fix.** The provider is resolved at draw time instead of construction time:

```csharp
// Resolve the provider at draw time: the manager is constructed before the
// minimap/status service, so capturing the delegate here would bind a null target.
StatusContent = () => host.MinimapAndStatus.DrawStatusContent();
```

Source: `Workbench/Services/UiManager/ViewerUiManager.cs:24-26`.

**Verification.** `dotnet build wow-viewer/WowViewer.slnx -c Debug --no-restore` → **exit 0,
0 errors** (command 4). This is a **code-level** fix: the viewer has **not** been launched, so
startup is still operator-owned proof.

## 8. Scanner extension (the former "6 invisible labels" gap is closed)

The control inventory is produced by matching a fixed set of ImGui/shared-widget call shapes and
reading the **first argument** of each call. The scanner now accepts a plain literal **or a
conditional whose branches are literals**, so labels such as
`_minimapCollapsed ? "Map##ExpandMinimap" : "Collapse##CollapseMinimap"` are detected. The
PowerShell inventory (`Get-UiControlInventory.ps1:37-66`, `Get-FirstArgumentLabels`) and the C#
scanner (`UI/UiControlScanner.cs:30-31`, `ExtractFirstArgumentLabels`) implement the same rule.

The manager's minimap and sidebar buttons were also rewritten as literal-label calls
(`Map##ExpandMinimap`, `Collapse##CollapseMinimap`, `Dock##MinimapDock`,
`Undock##MinimapDock`, `Collapse left sidebar##CollapseLeft`,
`Expand left sidebar##ExpandLeft`, `Collapse right sidebar##CollapseRight`,
`Expand right sidebar##ExpandRight`), so the scanner sees them.

Result: the previously disclosed labels (`Collapse`, `Map`, `Dock`, `Undock`, `Collapse left
sidebar`, `Expand left sidebar`, `Collapse right sidebar`, `Expand right sidebar`) are now
scanned and registered. The inventory grew from 112 to **125 rows** and the registry from 104 to
**117 entries**; the generated reference reports **115** controls. **No known scanner-invisible
label remains.**

## 9. Files changed

### 9.1 Implementation changes referenced by this receipt (not made by this documentation pass)

| File | Change |
|---|---|
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs` | Startup-crash fix (`:24-26`); minimap/sidebar buttons rewritten as literal-label calls. |
| `wow-viewer/eng/ui/Get-UiControlInventory.ps1` | First-argument/conditional-label scanning (`:37-66`). |
| `wow-viewer/src/viewer/WoWViewer/UI/UiControlScanner.cs` | Same first-argument/conditional-label rule (`:30-31`, `:89-108`). |
| `wow-viewer/eng/ui/ui-controls.json` | Regenerated: 117 entries, 115 visible. |
| `wow-viewer/docs/WoWViewer/UI-REFERENCE.md` | Regenerated: "Controls documented: **117**". |
| `wow-viewer/specs/280-ui-audit-refactor/evidence/ui-control-inventory.txt` | Regenerated: 125 rows, 117 distinct labels. |

### 9.2 Files changed by this correction pass

| File | Change |
|---|---|
| `wow-viewer/specs/280-ui-audit-refactor/evidence/u-ui-manager.md` | This receipt — counts corrected, §8 rewritten, startup-crash defect added. |
| `wow-viewer/specs/280-ui-audit-refactor/evidence/final-receipt.md` | Whole-spec final receipt — same corrections. |
| `wow-viewer/specs/280-ui-audit-refactor/evidence/p7-floating-windows.md` | **New** — P7 receipt. |
| `wow-viewer/specs/280-ui-audit-refactor/evidence/p9-visual-layout.md` | **New** — P9 receipt. |
| `wow-viewer/specs/280-ui-audit-refactor/evidence/p8-styling-naming.md` | Stale counts/claims corrected only. |
| `wow-viewer/specs/280-ui-audit-refactor/tasks.md` | T072 and T102/T104/T105 ticked with receipt pointers; stale counts/blocker corrected. |

No viewer source, no core library, no format reader, no renderer, no `docs/` text file, no
`spec.md`/`plan.md`/`ui-manager.md`, no `STATUS.md`, no memory-bank file was touched by this
correction pass. No commit was made. The existing C# test
[`UiControlDocumentationTests`](../../../tests/WowViewer.Core.Tests/UiControlDocumentationTests.cs)
needed **no** change and still passes 3/3 (command 6).

## 10. What is NOT proven by this receipt

- **No runtime proof.** Nothing here shows the viewer running: no window was opened, no client
  was loaded, no frame was rendered. The startup crash is fixed in code, but the viewer has not
  been launched.
- **No visual proof.** The minimap/chat/Options/sidebar layout, the two-click destination rule,
  and the "Options sub-menu does not cover the chat or the status bars" rule are **not** proven.
  The receipts that claim layout behaviour claim it structurally only ([P9](p9-visual-layout.md)).
- **No FPS, audio, or performance claim.** `DrawStatsContent` reports ImGui's measured
  framerate; no frame-time measurement was taken.
- **No real-client proof.** No export, import, open, or save was executed against real client
  data by this receipt. The file-operation repairs remain operator-verified in
  [P1](p1-file-operations.md).
- **Operator-owned gates not closed by this receipt:** **T092** (operator visual confirmation of
  the bottom bar, the Options sub-menu and the destination sidebars on a real client) and
  **T093** (operator functional confirmation of open, export and import on a real client).
- The build check only proves that the **documentation** matches the **source**; it proves
  nothing about whether a documented control actually does what its help text says.
