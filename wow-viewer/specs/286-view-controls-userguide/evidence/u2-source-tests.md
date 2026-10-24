# Spec 286 U2: Maps Once Per Successful Activation — Source and Test Receipt

**Date:** 2026-10-23  
**Spec:** [Spec 286: View Controls, Contextual Disclosure and Screenshot Guide](../spec.md)  
**Task:** T023 — Source/test gate for Phase U2.  
**Branch:** `286-view-controls-userguide`

---

## 1. Summary of Changes

Phase U2 replaces the legacy heuristic (`_autoOpenWorldMapsPanel = _discoveredMaps.Count > 0 && previousDiscoveredMapCount == 0`) with session-owned deterministic activation tracking:
- **[MapsActivationPolicy.cs](../../../src/viewer/WoWViewer/Workbench/Services/DataSourceSession/MapsActivationPolicy.cs):** Pure deterministic state policy managing monotonic client activation generations, discovery outcome tracking, and single-consumption auto-open logic.
- **[MapsActivationCoordinator.cs](../../../src/viewer/WoWViewer/Workbench/Services/DataSourceSession/MapsActivationCoordinator.cs):** Coordinates data-source activation events and discovery notifications with UI auto-open actions.
- **[DataSourceSessionService.cs](../../../src/viewer/WoWViewer/Workbench/Services/DataSourceSession/DataSourceSessionService.cs):** Integrates `MapsActivationCoordinator` into MPQ and WC3 client load paths, ensuring `CommitActivation` stamps a new generation and `RefreshDiscoveredMaps` reports discovery outcome.
- **[CascAhdrSourceService.cs](../../../src/viewer/WoWViewer/Workbench/Services/CascAhdr/CascAhdrSourceService.cs):** Routes CASC data-source activations into `DataSourceSessionService.CommitActivation` and discovery notifications.
- **[ViewerUiManager.cs](../../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs) & [ViewerActionBarsService.cs](../../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerActionBarsService.cs):** Expose `IsMapsOpen`, `OpenMapsWindow()`, and `CloseMapsWindow()`, reusing the existing Maps window rather than duplicating windows, and providing a clean API to dismiss the Maps window.
- **[WorldLoaderService.cs](../../../src/viewer/WoWViewer/Workbench/Services/WorldLoading/WorldLoaderService.cs):** Invokes `_host.UiManager.CloseMapsWindow()` upon completion of terrain loading in both `LoadWdtTerrain()` and `LoadRosettaDatastoreTerrain()`, automatically closing the Maps popup once the map is loaded.
- **[NavigatorPanelService.Sidebars.cs](../../../src/viewer/WoWViewer/Workbench/Services/Navigator/NavigatorPanelService.Sidebars.cs):** Replaces silent return when `_discoveredMaps.Count == 0` with an honest disabled text message: `"No maps discovered in active client."`.
- **[MapsActivationPolicyTests.cs](../../../tests/WowViewer.Core.Tests/MapsActivationPolicyTests.cs):** 11 regression unit tests covering first activation, replacement activation, already-open suppression, close-then-refresh, zero-map honest empty state, failure, cancellation, discovery error, and out-of-order stale completion.
- **[ui-controls.json](../../../eng/ui/ui-controls.json) & [UI-REFERENCE.md](../../../docs/WoWViewer/UI-REFERENCE.md):** Updated `maps-action` documentation to note automatic opening once per successful client activation.

---

## 2. Verification Commands and Exit Statuses

### Command B: Full Solution Build
```powershell
dotnet build ./WowViewer.slnx -c Debug
```
- **Exit Status:** 0
- **Errors:** 0
- **Warnings:** 600 (pre-existing library / analyzer warnings)

### Command U2: Maps Activation Policy Tests
```powershell
dotnet test ./tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~MapsActivationPolicyTests"
```
- **Exit Status:** 0
- **Passed:** 11 / 11
- **Failed:** 0
- **Skipped:** 0
- **Duration:** 115 ms

### Command ALL (Cumulative Focused Policy Tests)
```powershell
dotnet test ./tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~ViewerFooterLayoutTests|FullyQualifiedName~ViewerUiLayoutTests|FullyQualifiedName~ViewPreferencePolicyTests|FullyQualifiedName~MapsActivationPolicyTests|FullyQualifiedName~UiControlDocumentationTests"
```
- **Exit Status:** 0
- **Passed:** 101 / 101
- **Failed:** 0
- **Skipped:** 0
- **Duration:** 73 ms

### Command R: UI Reference Documentation Verification
```powershell
pwsh -NoProfile -File ./eng/ui/Verify-UiReference.ps1
```
- **Exit Status:** 0
- **Output:** `UI_REFERENCE_CHECK inventory_rows=128 distinct_labels=119 registry_entries=119`
- **Errors:** 0
- **Warnings:** 0

---

## 3. Criterion to Evidence Mapping

| Criterion | Source / Test Assertion | Status | Evidence |
|---|---|---|---|
| **FR-007** (Open Maps once after successful client activation) | Verified by `FirstActivation_IncrementsGenerationMonotonically_AndConsumesOnceOnSuccess` and `ReplacementActivation_AdvancesGeneration_ClearsPriorDiscovery_AndConsumesForNewGeneration`. | PASS (Source) | `MapsActivationPolicyTests.cs` (11/11 tests pass) |
| **FR-008** (Reuse existing window, respect close) | Verified by `CloseThenOrdinaryRefresh_DoesNotReopenMaps` and `MapsAlreadyOpen_MarkConsumed_PreventsSubsequentAutoOpenInSameGeneration`. `OpenMapsWindow` sets `_windows["maps"].Open = true` on the existing window. | PASS (Source) | `MapsActivationPolicyTests.cs`, `ViewerActionBarsService.OpenMapsWindow` |
| **FR-009** (Failed/cancelled loads do not open; honest empty state) | Verified by `Failure_DoesNotAdvanceCommittedGeneration_AndDoesNotAutoOpen`, `Cancellation_DoesNotAutoOpenMaps`, `DiscoveryError_DoesNotAutoOpenMaps_AndRecordsErrorOutcome`, and `ZeroMaps_SuccessfulEmptyClient_AutoOpensOnceWithEmptyStatus`. Empty client renders `"No maps discovered in active client."`. | PASS (Source) | `MapsActivationPolicyTests.cs`, `NavigatorPanelService.Sidebars.cs:122` |
| **SC-004** (Runtime client activation opens exactly one Maps window; close respected) | Policy and coordination wired; actual runtime client launching and window observation is operator-owned. | PENDING (Operator) | Left open for operator runtime task T024. |
| **SC-005** (Runtime failed/cancelled/no-map transitions show no false success or stale list) | Policy drops stale completions and resets prior discovery; actual runtime client confirmation is operator-owned. | PENDING (Operator) | Left open for operator runtime task T024. |

---

## 4. Modeled vs Actual Source-Route Coverage

As detailed in [u2-activation-route-audit.md](u2-activation-route-audit.md):
- **Modeled State Transitions:** The 11 unit tests in `MapsActivationPolicyTests` comprehensively verify all permutations of first activation, client replacement, zero-map empty clients, failures, user cancellation, discovery errors, and out-of-order arrival of discovery results.
- **Source Seams Wired:**
  1. `LoadMpqDataSource`: Calls `CommitActivation(gamePath)` on source load; `RefreshDiscoveredMaps()` reports discovery completion.
  2. `LoadCascDataSource`: Calls `_dataSourceSession.CommitActivation(installDir)` on source load; reports discovery completion (including fallback).
  3. `LoadWc3DataSource`: Calls `CommitActivation(rootPath)` and reports `mapCount = 0` (empty client).
  4. `AttachLooseMapOverlay`: Calls `RefreshDiscoveredMaps()` without committing a new activation, preserving closed window state.
- **Operator-Owned Proof (T024):** Real client loading in the live Silk.NET / ImGui renderer with real client installations (`H:\CLIENTS\...`) remains operator-owned and is tracked in task T024.
