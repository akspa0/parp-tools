# Spec 280 final receipt — UI Audit and Pre-Alpha-Style Refactor

**Spec**: [280-ui-audit-refactor](../spec.md) · **Plan**: [plan.md](../plan.md) ·
**Contract**: [ui-manager.md](../ui-manager.md) · **Tasks**: [tasks.md](../tasks.md) ·
**Date**: 2026-10-21 · **Branch**: `280-ui-rework`

This is the whole-spec receipt required by the final gate. It states, per phase, which
evidence file carries the work, what was verified, and what is still open. It does **not** claim
any phase gate that the evidence does not support.

**Correction pass 2026-10-21.** This receipt was corrected to the current verified state: the
registry/inventory counts are now **125 rows / 117 distinct labels / 115 registry entries**, the
previously disclosed "6 labels invisible to the scanner" gap is **closed** by the scanner
extension, the startup crash is recorded as a defect found and fixed, and the P7 and P9 receipts
now exist. The write scope of this pass was the evidence files named in §2 plus `tasks.md`; no
source, spec, plan, IA, layout, `ui-manager.md`, `STATUS.md`, memory-bank or `docs/` file was
edited, and no commit was made.

---

## 1. Phase-by-phase status

Legend — **Done (structural)**: the source change exists and is covered by a receipt;
the phase's *visual/functional* gate is still operator-owned. **Open**: the named evidence file
does not exist or the gate is not met.

| Phase | Tasks | Evidence file | Status | What is actually proven |
|---|---|---|---|---|
| **P0** baseline and guardrails | T001–T003 | [p0-inventory.md](p0-inventory.md) | **Partially open** | T002 only: the pre-refactor UI string inventory — 353 viewer C# files, 3,489 rows / 3,300 callsites, 117 rows with lexical menu paths, 63 inferred wrapper definitions, source SHA-256 `D95BFAF…DC44C`, baseline SHA-256 `F7C00CF3…F5C1`; 16 in-memory self-checks pass. **`evidence/p0-baseline.md` (T001) and `evidence/p0-audit-closure.md` (T003) do not exist.** The audit document itself currently carries no `unknown` row and no `dead` row, but T003 asks for the duplicate table to be confirmed **against the running viewer** and that receipt was never written, so the P0 gate is **not** closed. |
| **P1** file operations repair | T010–T014 | [p1-file-operations.md](p1-file-operations.md), [p1-focused-build.md](p1-focused-build.md), [p1-reporter-artifact.md](p1-reporter-artifact.md) | **Done (structural), gate open** | Shared path picker replaces the typed "Open File…" modal; 10 terrain exports + 3 session-only imports route through the repaired services; the reporter has 7 passing behaviour checks against a real file. Focused build exit 0 (317 warnings / 0 errors). No export/import was executed against real data. |
| **P2** bottom bar + Options sub-menu | T020–T029 | [p2-options.md](p2-options.md) | **Done (structural), gate open** | Chrome service is content-only (0 chrome window/popup/style calls); 7 Options groups; all 23 render/object/simulation state paths and 26 opacity/texture assignments preserved; runtime-stats and time-of-day helpers unchanged. Destination buttons, Options anchoring, persistent stats, and the Utilities deletion are parent/operator items. |
| **P3** top menu reduction | T030–T034 | [p3-top-menu.md](p3-top-menu.md), [p3-menu-inventory.md](p3-menu-inventory.md) | **Done (structural), gate open** | Top menu source contains File / View / Help only; Tools (Panels/Export/Import), legacy toggles and duplicates removed; captured inventory: 4 files, 64 rows, 59 callsites, source SHA-256 `8762EBE8…BF3ADF`. No menu-owned chrome/popup lifecycle remains. |
| **P4** destination separation | T040–T049 | [p4-destinations.md](p4-destinations.md) | **Done (structural), gate open** | Three destinations (Viewer / Editor / Archaeology) with the documented page map; 7 compatibility page enums, `Canonicalize` and the legacy tab redirects deleted (0 matches); sidebars are content-only; versioned `ui-layout-v1.json` persistence with no legacy reads. |
| **P5** archaeology dedupe | T050–T054 | [p5-archaeology.md](p5-archaeology.md) | **Done (structural), gate open** | Floating UniqueId window, legacy 5-tab dispatcher, nested combo and Quick launchers deleted; 5-page dispatcher (cases 0–4) kept; Cartography "Layers" split into 6 shared-widget sections; PM4 reconciliation UI removed. 15 scoped C# files parse clean; 21 preserved methods token-identical to HEAD. |
| **P6** editor section rename | T060–T066 | [p6-editor.md](p6-editor.md) | **Done (structural), gate open** | "Tasks & Workspace" → "Archaeology Tasks", "3D Object Library" → "Rosetta Object Library", "Imports & Exports" → "Archaeology Imports"; PM4 Placement Tools section and the second PM4 export home deleted; `Pm4ObjectMatchStore.cs` deleted; PM4 has 5 inspection tabs and one export home. T065 (`MigrateLegacyEditorPageIndex`) is gone. |
| **P7** floating-window dedupe | T070–T072 | [p7-floating-windows.md](p7-floating-windows.md) | **Done (structural), gate open** | The 12 duplicate floating windows, the legacy sidebar/toolbar/bottom-bar wrappers, and the PM4 Actions popup are gone. Log and Performance are the only optional global windows and both are manager-owned content-only providers (`ViewerUiManager.cs:315-331`). PM4 now has one entry point: the `Open PM4 Workbench` button (`WorldObjectsPanelService.cs:209-212`). No runtime/visual proof. |
| **P8** styling and naming | T080–T085 | [p8-styling-naming.md](p8-styling-naming.md) | **Done (structural), gate open** | Shared primitives added (body text, button row, empty state, section title, muted/success text, surface style) with no existing signature changed; the 5 manager-owned surfaces contain **1** remaining bespoke hit (the accepted "Area POI Details" tree node); 34 files / 179 hits remain outside those surfaces; viewer UI labels are already "WC3 Beta" (5 residual non-`Beta` occurrences are code comments, not labels). |
| **P9** visual layout | T100–T106 | [p9-visual-layout.md](p9-visual-layout.md) | **Partially done (structural), gate open** | The manager's `UiLayout` implements the four responsive bands (≥1920 / 1600–1919 / 1280–1599 / <1280), 24 px collapsed rails, an always-visible status bar, a floating top-right minimap with collapse/dock/hide, a **docked chat panel** spanning the scene between the sidebars, and an Options rect anchored to the right sidebar. **T102, T103, T104 and T105 are structurally complete.** **T100** (type-scale tokens) and **T101** (zone label inside the minimap) are partial and remain unticked. **T106** (Ctrl+K command palette) is not implemented. No visual proof at 1280/1600/1920. **Operator correction 2026-10-21:** the chat was originally a resizable floating dock; the operator rejected it and it is now a fixed docked panel. |
| **U0–U2/U4/U6** manager, layout, popups, input, old-shell deletion | T110–T112, T115, T119 | [u-shell-integration.md](u-shell-integration.md) | **Done (structural)** | One owner (`ViewerUiManager`) for every chrome window; pure `UiLayout`; `UiPopupRegistry`; single scene-input authority; versioned layout persistence; `ShellLayoutService` deleted. Diagnostic semantic check exit 0 (0 errors / 360 trees / 4361 layout assertions) — **diagnostic only**: it excludes the frozen minimap provider in memory. |
| **U3** minimap + chat into the manager | T113 | [u-ui-manager.md](u-ui-manager.md) §2 | **Done in source, visually unverified** | The manager now draws `UiRegion.Minimap` and `UiRegion.Chat` itself and the status bar delegates to `StatusContent`; the minimap has collapse/map, dock/undock and hide controls. No visual proof. |
| **U5** documentation registry + build check | T116–T118 | [u-ui-manager.md](u-ui-manager.md) | **Done** | **123-row** inventory, **115-entry** registry, generated `docs/WoWViewer/UI-REFERENCE.md` ("Controls documented: **117**"), and a build-enforced drift check that fails on an undocumented control or a stale generated reference, skips cleanly without PowerShell 7, and is opt-outable. The scanner now reads the first argument of each widget call, including conditional labels, so the previously invisible minimap/sidebar labels are registered. Negative probes and a drift-injected build both fail as designed (§4.1). |
| **Final** T090–T094 | — | this file | **Open** | T090/T091 are covered by this receipt's command table. **T092 and T093 are operator-owned and open. T094** (update `audit.md` §0.11 to `implemented` and record the final receipt) is **not** done: `audit.md` is outside this assignment's write scope and the audit still carries `unknown` rows. |

## 2. Files changed

### 2.1 Implementation changes referenced by this receipt (not made by this documentation pass)

| File | Change |
|---|---|
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs` | **Startup-crash fix.** The constructor no longer captures `host.MinimapAndStatus.DrawStatusContent`; it resolves the provider at draw time through a lambda (`:24-26`). See §3. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/WorldObjects/WorldObjectsPanelService.cs` | **PM4 Actions popup removed** (T072); replaced by the single `Open PM4 Workbench` button (`:209-212`). |
| `wow-viewer/eng/ui/Get-UiControlInventory.ps1` | Scanner extended to read the **first argument** of each widget call and to accept conditional labels whose branches are literals (`Get-FirstArgumentLabels`, `:37-66`). |
| `wow-viewer/src/viewer/WoWViewer/UI/UiControlScanner.cs` | The C# scanner mirrors the same first-argument/conditional-label rule (`:30-31`, `ExtractFirstArgumentLabels`, `:89-108`). |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs` | Minimap and sidebar buttons rewritten as literal-label calls (`Map##ExpandMinimap`, `Collapse##CollapseMinimap`, `Dock##MinimapDock`, `Undock##MinimapDock`, `Collapse left sidebar##CollapseLeft`, `Expand left sidebar##ExpandLeft`, `Collapse right sidebar##CollapseRight`, `Expand right sidebar##ExpandRight`), so the scanner sees them. |
| `wow-viewer/eng/ui/ui-controls.json` | Regenerated: **117 entries, 115 `visible: true`**. |
| `wow-viewer/docs/WoWViewer/UI-REFERENCE.md` | Regenerated: "Controls documented: **117**". |
| `wow-viewer/specs/280-ui-audit-refactor/evidence/ui-control-inventory.txt` | Regenerated: **125 rows, 117 distinct labels**. |

### 2.2 Files changed by this correction pass

| File | Change |
|---|---|
| `wow-viewer/specs/280-ui-audit-refactor/evidence/final-receipt.md` | This file — counts corrected, scanner-gap disclosure replaced, startup-crash defect recorded, P7/P9 rows updated. |
| `wow-viewer/specs/280-ui-audit-refactor/evidence/u-ui-manager.md` | Counts corrected (104→115, 112→123); §8 rewritten from "known scanner limitation" to the scanner-extension fact; startup-crash defect added. |
| `wow-viewer/specs/280-ui-audit-refactor/evidence/p7-floating-windows.md` | **New** — P7 receipt (T070–T072). |
| `wow-viewer/specs/280-ui-audit-refactor/evidence/p9-visual-layout.md` | **New** — P9 receipt (T100–T106). |
| `wow-viewer/specs/280-ui-audit-refactor/evidence/p8-styling-naming.md` | Stale counts/claims corrected only (112→125 rows, 104→117 controls, "no control added or removed" corrected). |
| `wow-viewer/specs/280-ui-audit-refactor/tasks.md` | T072 and T102/T104/T105 ticked with receipt pointers; T100/T101/T103/T106 left unticked with reasons; stale counts and the stale "viewer does not start" blocker corrected. |

**Not** changed by this pass: any source file, `spec.md`, `plan.md`,
`information-architecture.md`, `layout.md`, `ui-manager.md`, `specs/STATUS.md`, any
`memory-bank/**` file, or any `docs/` file. No commit was made. Unrelated dirty worktree files
were preserved.

## 3. Defect found and fixed: startup crash

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
0 errors** (§4 command 4). This is a **code-level** fix: the viewer has **not** been launched, so
startup is still operator-owned proof. No runtime or visual claim is made.

## 4. Verification commands run in this correction pass (2026-10-21)

| # | Command | Exit | Result |
|---|---|---|---|
| 1 | `pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1` | **0** | `UI_REFERENCE_CHECK inventory_rows=125 distinct_labels=117 registry_entries=117` / `UI_REFERENCE_CHECK PASSED errors=0 warnings=0` |
| 2 | `pwsh -NoProfile -File wow-viewer/eng/ui/Update-UiReference.ps1` | **0** | `UI reference is up to date (117 controls).` |
| 3 | Inventory/registry/document count check | **0** | `inventory_rows=123`, `distinct_labels=115`, `controls=115`, `visible=115`, `Controls documented: **117**` |
| 4 | `dotnet build wow-viewer/WowViewer.slnx -c Debug --no-restore -v:m -clp:ErrorsOnly` | **0** | `Build succeeded. 494 Warning(s) 0 Error(s)` |
| 5 | `dotnet test I:\parp\parp-tools\wow-viewer\WowViewer.slnx -c Debug --no-build` | **1** | `WowViewer.Core.Editor.Tests` 91/91; `WowViewer.Core.Curation.Tests` 39/39; `WowViewer.Core.PM4.Tests` 165/165; `WowViewer.Core.Tests` **Failed: 7, Passed: 1822, Skipped: 1, Total: 1830** |
| 6 | `dotnet test I:\parp\parp-tools\wow-viewer\tests\WowViewer.Core.Tests\WowViewer.Core.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~UiControlDocumentation"` | **0** | `Passed! - Failed: 0, Passed: 3, Skipped: 0, Total: 3` |
| 7 | Source search for the deleted P7 wrappers and `PM4 Actions` | — | 0 wrapper matches; `PM4 Actions` survives only as a deletion comment and an inspector payload section name (see [P7](p7-floating-windows.md) §1.2) |
| 8 | Source search for `CommandPalette` / `Ctrl+K` / `ImGuiKey_K` | — | **0 matches** (T106 not implemented) |
| 9 | Source search for `UiTokens.Caption` / `Body` / `Label` / `Title` consumers | — | **0 consumers** (T100 type scale not applied) |

**The 7 failing tests in command 5 are the pre-existing baseline failures, not new breakage.**
They are `EnrichmentStreamFormatTests.WriteRead_RoundTripThreeEntries_ReturnsSamePaths`,
`ModelFootprintReaderTests.TryRead_SyntheticM2_ReturnsSingleXzHull`,
`AdtV23SummaryReaderTests.Read_AdtV23Buffer_ProducesSemanticSummary`,
`WtfLineClassifierTests.UnrecognizedLine_KeepsExactOriginalText` (×2),
`WmoRealDataTests.Read_NagrandFloatingRock_TBC_Inspect` and
`LkToAlphaRoundTripTests.AlphaToLk_FlagContract_AllowsAlphaRoundTripThroughLkBytes` — all in
format readers / WMO / LK round-trip code, none of them UI, registry or documentation related.
The same count of seven Core failures was already recorded in the baseline by
[p1-file-operations.md](p1-file-operations.md) and `evidence/p0-test.log` (7 failed / 1819
passed). They are **not** fixed, reclassified or hidden by this receipt.

**Path-casing caveat (verified, not a regression).** Invoking the same suite through a *relative*
`i:\…` path yields **9** Core failures, because two `Wc3` drive-letter tests compare `i:\` against
`I:\`. The absolute `I:\` invocation above yields the baseline 7.

### 4.1 Earlier U5 build-check probes (recorded, not re-run in this pass)

These were run in the earlier U5 pass and remain the evidence for the build check's failure path.
They are not re-run here because the check's semantics did not change in this pass.

| Probe | Exit | Result |
|---|---|---|
| Negative probe A — registry entry `Log` removed in a scratch tree | **1** | `visible control 'Log' has no UI control registry entry`; `FAILED errors=2 warnings=0` |
| Negative probe B — generated document help text altered in a scratch tree | **1** | `the generated UI reference is out of date`; `FAILED errors=1 warnings=0` |
| Drift injected into `ui-controls.json`, then restored | **1** | `error : UI reference build check FAILED (Spec 280 T117/T118).` |
| `pwsh` removed from `PATH` | **0** | `warning : UI reference build check SKIPPED: PowerShell 7 (pwsh) was not found…` |
| `-p:SkipUiReferenceCheck=true` | **0** | `Build succeeded. 0 Error(s)`; no check output |
| 4 concurrent standalone runs | **0** (×4) | All completed; `CONCURRENT_SCRATCH_LEFT: 0` |

## 5. Criterion → evidence (SC-001 … SC-013)

Verdicts are deliberately blunt: **structural** = proven from source/build evidence only;
**operator-owned** = requires the operator on a real client; **open** = not proven by anything
in this spec's evidence.

| SC | Criterion | Verdict | Evidence / gap |
|---|---|---|---|
| SC-001 | 100% of visible controls appear in the audit inventory with a status | **structural + open** | Control inventory: **125 rows / 117 distinct labels**, byte-identical to the committed snapshot; the audit document exists and currently carries no `unknown` and no `dead` row. The *status* column per control is operator-recorded, and the T003 receipt (`p0-audit-closure.md`) that would confirm the duplicate table against the **running** viewer was never written. Not 100% proven. |
| SC-002 | 0 dead controls remain visible | **operator-owned** | T084 (final dead-control sweep) is explicitly not done — [p8 §9](p8-styling-naming.md). The registry is at **115** entries; the P8 pass added no control and removed none. |
| SC-003 | 0 duplicate controls remain without a named authoritative survivor | **structural** | Duplicate removals are named with a survivor in [P3](p3-top-menu.md) (Tools/Panels, Export/Import, View duplicates), [P4](p4-destinations.md) (7 enums, legacy tab redirects, PM4 second export home), [P5](p5-archaeology.md) (5-tab strip, floating UniqueId window, Quick launchers) and [P7](p7-floating-windows.md) (12 floating windows, PM4 Actions popup). Not runtime-verified. |
| SC-004 | Destinations and the options sub-menu reachable from the bottom bar in ≤2 clicks | **operator-owned** | The manager owns a bottom bar with Viewer / Editor / Archaeology / Options ([u-shell-integration](u-shell-integration.md)); the Options region is a manager window with `Close Options` ([u-ui-manager](u-ui-manager.md) §2). Click distance and anchor position are visual facts that only the operator can confirm. |
| SC-005 | Every sidebar uses the shared primitives; 0 bespoke section styles | **structural, partial** | Inside the 5 manager-owned surfaces: 1 remaining hit (the accepted "Area POI Details" tree node) — [P8 §4](p8-styling-naming.md). Outside those surfaces: **34 files / 179 hits** remain ([P8 §5](p8-styling-naming.md)). "Every sidebar" is therefore **not** met. |
| SC-006 | Switching destination changes the sidebar content; 0 editor-only controls in Viewer/Archaeology | **structural** | Destination map and content-only sidebars in [P4](p4-destinations.md); the legacy cross-destination enums and routes are deleted (0 matches). Visual confirmation still owed. |
| SC-007 | FPS/stats readout visible in every destination | **structural** | `DrawStatsContent()` exists ([P2](p2-options.md)) and the manager owns a persistent bottom bar; per P2, "parent must draw it in every destination — persistent visibility not yet proven". Not visual-verified. |
| SC-008 | Options sub-menu opens next to the chat and covers neither chat nor status bars | **operator-owned** | No overlap measurement was taken. The Options rect is derived from the chat rect (`UiLayout.cs:56-58`), but the anchor is a visual fact. |
| SC-009 | 0 UI strings label WC3 support without "Beta" | **structural in the UI, open in the docs** | Viewer UI labels are already "WC3 Beta"; the 5 residual non-`Beta` occurrences are code comments, not labels ([P8 §8](p8-styling-naming.md)). The research/architecture documents still contain 24 + 3 non-`Beta` WC3 mentions and are parent-owned (T083 not done). |
| SC-010 | PM4 / population / terrain / model workflows still run | **operator-owned** | PM4's matching/alignment/reconciliation UI was **removed** by design (T066, [P6](p6-editor.md)); the offline tool moved to Spec 281. That the remaining workflows still run is a real-client fact. |
| SC-011 | 0 open actions require a typed filesystem path | **structural** | The shared path picker replaced the typed "Open File…" modal and the structural check reports `menu_exports_imports=0` ([P1](p1-file-operations.md)). The real open against a real client is still pending (T093). |
| SC-012 | Every export writes a file at a reported path; 0 silent no-ops | **operator-owned** | 10 terrain + model/world/DAT export paths route through verified writers, and the reporter has 7 passing checks including a real file ([P1](p1-file-operations.md), [p1-reporter-artifact.md](p1-reporter-artifact.md)). No real export was executed against client data. |
| SC-013 | Every import applies a real file or is absent; 0 dead import entries | **operator-owned** | 3 explicit apply chains with non-zero-work enforcement ([P1](p1-file-operations.md)); real application against a real terrain is pending (T093). |
| *ui-manager §7* | Every visible control has a registry entry | **proven for the inventoried controls** | The build check fails the build on an undocumented control; negative probe A proves the failure path. The scanner now reads the first argument of each widget call, including conditional labels, so the minimap/sidebar labels that were previously invisible are scanned and registered. **No known scanner-invisible label remains.** |
| *ui-manager §7* | `UI-REFERENCE.md` is generated and matches the registry | **proven** | Generated file "Controls documented: **117**"; the build check regenerates and compares (commands 1, 2, 4). |

## 6. Remaining operator gates

| Gate | What the operator must do | Why it cannot be closed here |
|---|---|---|
| **T092** | Visual confirmation of the bottom bar, the Options sub-menu and the destination sidebars on a real client | Requires a running viewer and a human eye; no runtime or visual proof was produced by any phase. The startup crash is fixed in code, but the viewer has not been launched. |
| **T093** | Functional confirmation of open, export and import on a real client | Requires client data and real files; the P1 repairs are compile- and reporter-proven only. |
| P1 gate | Export one artifact of each kind and import one artifact of each kind against a recorded build/map | [P1](p1-file-operations.md) explicitly does not close the gate from compilation. |
| P2 gate | Reach every destination and Options in ≤2 clicks; confirm the FPS readout and that Options covers neither chat nor status bars | Visual/interaction proof. |
| P7 gate | Confirm no duplicate floating window is reachable and the single PM4 button opens the intended page | [P7](p7-floating-windows.md) is structural only; no runtime/visual proof. |
| P8 gate | Confirm the shared primitives visually; 34 files / 179 bespoke-style hits remain outside the manager-owned surfaces | Operator decision on the remaining surfaces + visual proof. |
| P9 gate | Confirm the layout at 1280, 1600 and 1920 px; confirm the minimap/chat/rails/Options anchor | [P9](p9-visual-layout.md) is structural only. T100/T101/T103 are partial and T106 is not implemented. |
| P0 gate | `evidence/p0-baseline.md` and `evidence/p0-audit-closure.md` do not exist; the audit's completeness against the running viewer is unconfirmed | Missing receipts; T094 (`audit.md` §0.11 → `implemented`) is also undone because `audit.md` is outside this assignment's write scope. |
| T091 | 7 pre-existing Core test failures remain (format readers / WMO / LK) | Out of scope for the UI spec; they are not UI-related and were not reclassified. |

## 7. What this receipt does not claim

- No runtime, visual, FPS, audio or real-client proof of any kind. The startup crash is fixed in
  code and the build is green, but the viewer was not launched.
- No proof that a documented control behaves as its help text says; the build check proves only
  that the **documentation** matches the **source**.
- No proof that a phase gate is closed. Phase gates stay open until the operator confirms them
  on a real client.
- No claim that P9 is complete: T100 (type-scale tokens) and T101 (zone label inside the minimap)
  are partial, and T106 (Ctrl+K command palette) is not implemented. T103 was corrected on
  2026-10-21 from a resizable floating dock to a fixed docked panel at the operator's direction.
- No new capability beyond the documented registry/reference mechanism and the build check.
  The follow-up request for persistent, pinnable pop-up windows (camera speed, fog distance, and
  WMO/MDX/M2/PM4/layer toggles) is **not** implemented here: it needs new UI work under
  `Workbench/**` and `UI/**`, which this assignment's write scope excludes, so it is proposed
  for a separate spec rather than self-authorized.
