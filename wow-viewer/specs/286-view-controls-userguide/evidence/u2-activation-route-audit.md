# Spec 286 U2: Client Activation & Map Discovery Route Audit

**Date:** 2026-10-23  
**Spec:** [Spec 286: View Controls, Contextual Disclosure and Screenshot Guide](../spec.md)  
**Task:** T018 — Audit folder, saved-client, startup, archive/installation and CASC activation/discovery paths.

---

## 1. Executive Summary

This audit traces all active client activation, map discovery, and map window open routes in `wow-viewer`.
Under the legacy implementation, automatic Maps window opening relied on an ad-hoc heuristic in `DataSourceSessionService.RefreshDiscoveredMaps`:
```csharp
_autoOpenWorldMapsPanel = _discoveredMaps.Count > 0 && previousDiscoveredMapCount == 0;
```
This heuristic suffered from several defects:
1. **Client Replacement Defect:** When switching from an active client with maps (`count > 0`) to another client with maps, `previousDiscoveredMapCount == 0` evaluated to `false`, preventing the Maps window from opening for the newly loaded client.
2. **Empty Client Defect:** A client that successfully loaded but contained 0 maps never triggered auto-open, preventing users from seeing the honest empty state.
3. **Stale Maps Defect:** Map discovery failures or cancelled loads could leave previous client discovery lists in memory.
4. **Lack of Generation Tracking:** Out-of-order or asynchronous discovery outcomes had no monotonic generation to prevent stale results from overwriting current state.

Phase U2 replaces this heuristic with a session-owned `MapsActivationPolicy` and `MapsActivationCoordinator`.

---

## 2. Inventory of Activation & Discovery Routes

### Route 1: Folder Selection Dialog
- **Trigger Location:** `ClientDialogsService.ClientDialogs.cs` (via `OpenClientFolder` or `FolderAction`)
- **Seam / Entry Point:** `DataSourceSessionService.LoadMpqDataSource(selectedPath, null, buildVersion)`
- **Success Transition:**
  - `_dataSource = new MpqDataSource(gamePath, resolvedListfilePath)` successfully instantiated.
  - Commits monotonic activation generation via `_coordinator.CommitActivation(gamePath)`.
  - Discovered maps list cleared (`_discoveredMaps.Clear()`).
- **Discovery Seam:** `DataSourceSessionService.RefreshDiscoveredMaps()` runs `MapDiscoveryService.DiscoverMaps()`.
- **Notification Seam:** Outcome published to `_coordinator.NotifyDiscoveryCompleted(generation, _discoveredMaps.Count, isSuccess: true)`.
- **Failure Seam:** Exception in MPQ constructor or DBC init caught; status set to `"Failed to load MPQs: ..."`; activation is aborted and never committed.

### Route 2: Saved / Recent Client Selection
- **Trigger Location:** `MainMenuBarService.cs` (menu `File > Recent Clients > [path]`)
- **Seam / Entry Point:** `DataSourceSessionService.LoadMpqDataSource(savedBasePath, null, savedBuildVersion, deferWorldReload: true)`
- **Success Transition:** Same as Route 1. Monotonic generation incremented.
- **Discovery Seam:** Same as Route 1.
- **Failure Seam:** Same as Route 1.

### Route 3: Startup Automation (`--client-path`)
- **Trigger Location:** `StartupAutomationService.cs` (`ProcessStartupArguments`)
- **Seam / Entry Point:** `_dataSourceSession.LoadMpqDataSource(request.GamePath, request.ListfilePath, request.BuildVersion)`
- **Success Transition:** Same as Route 1.
- **Discovery Seam:** Same as Route 1.
- **Failure Seam:** Same as Route 1.

### Route 4: Warcraft III / Archive Source
- **Trigger Location:** `DataSourceSessionService.LoadWc3DataSource(string rootPath, string? listfilePath)`
- **Success Transition:** `_dataSource = new Wc3DataSource(rootPath, listfilePath)` instantiated.
- **Discovery Seam:** WC3 data does not contain WoW maps (`_discoveredMaps.Clear()`).
- **Notification Seam:** Outcome published with `MapCount = 0`, `DiscoveryOutcomeStatus.Empty`.
- **Failure Seam:** Exception caught; status set to `"Failed to open WC3 Beta data: ..."`.

### Route 5: CASC / Modern Client Activation
- **Trigger Location:** `CascAhdrSourceService.cs` (`LoadCascDataSource(installDir, productOrder, allowCdnFill)`)
- **Success Transition:**
  - `_dataSource = new CascDataSource(storages, listfile)` successfully instantiated.
  - Commits monotonic activation generation via `_dataSourceSession.CommitActivation(installDir)`.
  - Discovered maps list cleared (`_discoveredMaps.Clear()`).
- **Discovery Seam:** Calls `_dataSourceSession.RefreshDiscoveredMaps()`. If Map.db2 discovery fails, falls back to `MapDiscoveryService.DiscoverLooseMapsOnly(_dataSource)`.
- **Notification Seam:** Outcome published via `_dataSourceSession.NotifyDiscoveryCompleted(generation, ...)`.
- **Failure Seam:** `storages.Count == 0` or exception during CASC open sets `_statusMessage`; activation is aborted and never committed.

### Route 6: Loose Map Overlay Attach (Refresh, NOT an Activation)
- **Trigger Location:** `DataSourceSessionService.AttachLooseMapOverlay(selectedPath)`
- **Nature of Route:** Augmentation of the *existing* client session with loose development files.
- **Behavior:** Calls `RefreshDiscoveredMaps()`. Because no new client activation was committed, `ConsumedGeneration == CommittedGeneration`; Maps window is NOT auto-opened or reopened if closed.

### Route 7: User Discovery Refresh (Refresh, NOT an Activation)
- **Trigger Location:** Manual trigger or programmatic refresh without client change.
- **Behavior:** Refreshes map definitions in-place. Consumed generation matches committed generation; window visibility is untouched.

---

## 3. Maps Window Management Seam

- **Window State Owner:** `ViewerActionBarsService._windows["maps"]`
- **Draw Site:** `ViewerActionBarsService.DrawToolWindows()` calls `DrawToolWindow("maps", "Maps", ..., _host.NavigatorPanel.DrawMapsContent)`
- **Show/Toggle Affordance:**
  - Action bar icon: `ActionMaps` in `ViewerActionBarsService.DrawLeftBar()`
  - Programmatic: `ViewerUiManager.OpenMapsWindow()` -> sets `_windows["maps"].Open = true`
- **Close Affordance:** User clicks close button `X` on tool window header -> sets `_windows["maps"].Open = false`.
- **Content Rendering:** `NavigatorPanelService.Sidebars.cs`:
  - `DrawMapsContent()` calls `DrawWorldOverviewContent()`, `DrawMapExportControls()`, and `DrawMapDiscoveryContent()`.
  - In `DrawMapDiscoveryContent()`: updated to show an honest empty message `ImGui.TextDisabled("No maps discovered in active client.")` when `_discoveredMaps.Count == 0` instead of returning silently.

---

## 4. Contract Conformance Summary

| Spec Requirement | Implementation Mechanism |
|---|---|
| **FR-007 (Auto-open once per activation)** | `MapsActivationPolicy.TryConsumeAutoOpen` returns `true` exactly once per committed generation. |
| **FR-008 (Reuse existing window, respect close)** | Programmatic open sets `_windows["maps"].Open = true`. User close sets `Open = false`. Subsequent frames/refreshes in the same generation do not re-open. |
| **FR-009 (Honest empty state, no false success)** | Empty results (`MapCount = 0`) trigger open once with an honest empty UI. Errors/cancellations do not commit or trigger auto-open. |
| **Monotonic Generation** | Stamped by `CommitActivation()`, incrementing monotonically; out-of-order or stale discovery completions are dropped. |
