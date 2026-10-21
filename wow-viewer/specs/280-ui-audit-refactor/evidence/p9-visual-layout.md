# P9 visual-layout receipt — 2026-10-21

**Spec**: [280-ui-audit-refactor](../spec.md) · **Layout contract**: [layout.md](../layout.md) §8–§9 ·
**Tasks**: T100–T106 · **Branch**: `280-ui-rework`

**Scope of this receipt.** P9 is the layout phase: regions, responsive breakpoints, collapse rails,
the floating minimap, the chat dock, and the Options anchor. This receipt records the **structural**
implementation and its exact code locations. It does **not** claim a visual gate: no frame was
rendered at 1280, 1600 or 1920 px, and the viewer has not been launched since the startup crash fix
(see [final-receipt.md](final-receipt.md) §3). The visual gate is operator-owned.

---

## 1. Regions and their owners

`UiLayout` is a pure logical-pixel layout: it computes rectangles and never touches ImGui.
`ViewerUiManager` owns every window and draws the region content.

| Region | Rect source | Drawn by | Source |
|---|---|---|---|
| `Scene` | full area between top bar, bottom bar and status bar | renderer (not a chrome window) | `UiLayout.cs:61` |
| `TopBar` | full width, 28 px (clamped to display height) | `ViewerUiManager.DrawRegion` → `MainMenuBar.DrawContent()` | `UiLayout.cs:62`, `ViewerUiManager.cs:102-105` |
| `StatusBar` | full width, 22 px, **always visible** | `ViewerUiManager` → `StatusContent` | `UiLayout.cs:63`, `ViewerUiManager.cs:122-126` |
| `BottomBar` | full width, 32 px | `ViewerUiManager.DrawBottom` | `UiLayout.cs:64`, `ViewerUiManager.cs:114`, `:258-268` |
| `LeftSidebar` | left, scene height, persisted width or 24 px rail | `ViewerUiManager.DrawLeft` | `UiLayout.cs:65`, `ViewerUiManager.cs:106`, `:193-212` |
| `RightSidebar` | right, scene height (minus a docked minimap), persisted width or 24 px rail | `ViewerUiManager.DrawRight` | `UiLayout.cs:66`, `ViewerUiManager.cs:108`, `:214-256` |
| `Options` | anchored to the right edge of the chat rect + 8 px | `ViewerUiManager` → `ViewerChrome.DrawOptionsContent()` | `UiLayout.cs:56-58`, `:67`, `ViewerUiManager.cs:115-119` |
| `Minimap` | floating top-right, or docked at the top of the right sidebar | `ViewerUiManager.DrawMinimap` | `UiLayout.cs:38-53`, `:68`, `ViewerUiManager.cs:110-111`, `:138-191` |
| `Chat` | bottom-left, 420 × 160 default | `ViewerUiManager` → `SqlSpawnStreaming.DrawChatContent()` | `UiLayout.cs:55`, `:69`, `ViewerUiManager.cs:112-113` |

Every region window is drawn with `NoTitleBar | NoMove | NoResize | NoCollapse | NoSavedSettings |
NoDocking` and the shared token surface (`ViewerUiManager.cs:277-290`). The manager, not the
provider, owns position, size and lifetime.

## 2. Responsive breakpoints (T105)

`UiLayout` implements four width bands. The minimap size and the default sidebar state change at
the band boundaries:

| Display width | Minimap size | Left sidebar default | Right sidebar default | Left width | Right width | Source |
|---|---|---|---|---|---|---|
| **≥ 1920** | 180 px | visible | visible | 280 px | 360 px | `UiLayout.cs:34`, `:74-75` |
| **1600–1919** | 160 px | visible | visible | 240 px | 320 px | same |
| **1280–1599** | 140 px | hidden (24 px rail) | visible | 240 px | 320 px | same |
| **< 1280** | 120 px | hidden (24 px rail) | hidden (24 px rail) | 240 px | 320 px | same |

`UiLayout.Defaults(width)` is the single source of the default visibility/width decision
(`UiLayout.cs:74-75`); `ViewerUiManager.Persistence.ApplyDefaults` applies it on first run and on
`ResetLayout` (`ViewerUiManager.Persistence.cs:66-89`).

## 3. Collapse rails and persistent widths (T104)

- A hidden sidebar is laid out as a **24 px rail**: `Math.Min(24, w / 2)` (`UiLayout.cs:32-33`).
- The rail carries an expand button; a visible sidebar carries a collapse button:
  `Collapse left sidebar` / `Expand left sidebar` (`ViewerUiManager.cs:193-212`) and
  `Collapse right sidebar` / `Expand right sidebar` (`ViewerUiManager.cs:214-231`).
- Widths and visibility persist in the versioned layout document
  `output/settings/ui-layout-v1.json` (`ViewerUiManager.Persistence.cs:10-11`), loaded at
  `:30-64` and saved atomically through a `.pending` file at `:91-114`. Widths are clamped to
  180–1080 px on load (`:43-44`).

## 4. Floating minimap (T101)

- **Floating top-right** is the default: `new UiRect(w - rightOffset - mapSize - 8, top + 8,
  mapSize, mapSize)` (`UiLayout.cs:52`).
- **Docked** mode places it at the top of the right sidebar and shortens the sidebar below it
  (`UiLayout.cs:42-48`).
- The manager draws the chrome: `Collapse` / `Map` (collapse/expand), `Dock` / `Undock`, and
  `Hide` (`ViewerUiManager.cs:138-191`). Collapse shrinks the rect to at most 32 px
  (`UiLayout.cs:36`).
- The minimap body shows the current **tile coordinates** (`Tile: (x,y)`), zoom and loaded-tile
  count (`MinimapAndStatusService.MinimapAndStatus.cs:664-674`).
- The **zone label and coordinates** are drawn under the map by the manager:
  `_host.CurrentAreaName` (truncated with a tooltip) and `Tile (y, x)  X/Y/Z`
  (`ViewerUiManager.cs:226-233`). T101's zone-label clause is met.

## 5. Chat dock (T103)

> **Operator correction, 2026-10-21.** The original receipt described a resizable bottom-left
> floating dock. The operator rejected it: the chat covered the UI and could not be moved. The
> chat is now a fixed docked panel. The resize clause is withdrawn; the dock and collapse clauses
> stand.

- The chat is a docked panel: rect `(lw, top + sceneHeight - dockHeight, w - lw - rw, dockHeight)`
  where `dockHeight = min(180 * scale, 28% of sceneHeight)` (`UiLayout.cs:71-80`). It spans the
  scene between the sidebars and can never overlap a sidebar, the bottom bar or the status bar.
- The manager draws it with `NoMove | NoResize` (`ViewerUiManager.cs:159-161`); there is no drag or
  resize handle.
- The chat content keeps **Auto-Stream** and **Spawn BBs Only**:
  `SqlSpawnStreamingService.cs:457-466` (`Auto-Stream##ChatAutoStream`,
  `Spawn BBs Only##ChatSpawnBBs`).
- **Collapsible:** `Collapse chat` / `Expand chat` collapse the dock to a 28 px one-line bar
  (`ViewerUiManager.cs:186-211`, `UiLayout.cs:74-76`), persisted as `ChatCollapsed`
  (`ViewerUiManager.Persistence.cs:30`, `:51`, `:80`, `:109`).
- The persisted floating rect (`ChatWidth`/`ChatHeight`/`ChatX`/`ChatY`) is deleted and the layout
  document is bumped to v3 (`ViewerUiManager.Persistence.cs:10-13`), so a stale floating rect
  cannot be restored.
- T103's dock and collapse clauses are met structurally; the rendered result is operator-owned.

## 6. Options anchor (T114 / SC-008)

The Options rect anchors to the right sidebar's left edge, independent of the chat dock:

```csharp
float ox = Math.Clamp(w - rw - ow - 8 * s, 0, Math.Max(0, w - ow));
[UiRegion.Options] = new(new(ox, Math.Max(top, h - status - bottom - oh), ow, oh), !hidden && options);
```

Source: `UiLayout.cs:82-85`, `:94`. The manager draws it only while `_options` is true and offers
`Close Options` (`ViewerUiManager.cs:163-167`). Whether it visually covers neither the chat nor the
status bar is an operator visual fact (SC-008).

## 7. Task verdicts

| Task | Criterion | Verdict | Evidence / gap |
|---|---|---|---|
| **T100** | Apply the design tokens from [layout.md](../layout.md) §8: colour, type, spacing | **Partial — not ticked** | Colour, spacing and surface tokens are consumed (`SharedUiWidgets.cs:124-238`; `UiTokens.cs:15-49`). The **type scale** (`Caption`/`Body`/`Label`/`Title`, `UiTokens.cs:27-31`) is defined but has **0 consumers** in the viewer tree; no font size is applied from it. The "type" clause is not met. |
| **T101** | Move the minimap to the floating top-right instrument; add collapse, zone label, and coordinates | **Done (structural)** | Floating top-right, collapse/expand, dock/undock, hide, zone label and coordinates are implemented (§4). |
| **T102** | Delete the Utilities Minimap page; the top-right minimap is canonical | **Done (structural)** | The `Utilities` tab and its pages were deleted (T029); 0 matches for a minimap page enum/label in the viewer tree. The only minimap surface is the manager-owned `UiRegion.Minimap`. |
| **T103** | Convert the chat to a docked panel; keep Auto-Stream and Spawn BBs Only | **Done (structural)** | Docked panel spanning the scene between the sidebars, both controls, and a persisted collapse state are implemented (§5). The resize clause was withdrawn by operator correction 2026-10-21. |
| **T104** | Add sidebar collapse rails and persistent widths | **Done (structural)** | 24 px rails (`UiLayout.cs:32-33`), collapse/expand buttons (`ViewerUiManager.cs:193-231`), versioned persistence (`ViewerUiManager.Persistence.cs:10-11`, `:30-64`, `:91-114`). |
| **T105** | Add the responsive breakpoints and default collapse from [layout.md](../layout.md) §9 | **Done (structural)** | Four bands and their defaults are implemented in `UiLayout.cs:34`, `:74-75` (§2). The visual "layout holds at 1280/1600/1920" clause is operator-owned. |
| **T106** | Optional: `Ctrl+K` command palette over destinations and actions | **Not implemented** | 0 matches for `CommandPalette`, `Ctrl+K` or `ImGuiKey_K` in the viewer tree. |

## 8. Verification commands

| # | Command | Exit | Result |
|---|---|---|---|
| 1 | `dotnet build wow-viewer/WowViewer.slnx -c Debug --no-restore -v:m -clp:ErrorsOnly` | **0** | `Build succeeded. 494 Warning(s) 0 Error(s)` (2026-10-21) |
| 2 | `pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1` | **0** | `UI_REFERENCE_CHECK inventory_rows=125 distinct_labels=117 registry_entries=117` / `PASSED errors=0 warnings=0` |
| 3 | `dotnet test I:\parp\parp-tools\wow-viewer\WowViewer.slnx -c Debug --no-build` | **1** | Core 7 failed / 1822 passed / 1 skipped / 1830; Editor 91/91; Curation 39/39; PM4 165/165 — the 7 are the pre-existing baseline failures |
| 4 | `dotnet test I:\parp\parp-tools\wow-viewer\tests\WowViewer.Core.Tests\WowViewer.Core.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~UiControlDocumentation"` | **0** | `Passed! - Failed: 0, Passed: 3, Skipped: 0, Total: 3` |

## 9. What is NOT proven

- **No visual check at 1280, 1600 or 1920 px.** The breakpoint arithmetic is proven from source;
  the rendered result is not.
- **No real-client run.** The viewer was not launched. The startup crash was fixed in code
  (`ViewerUiManager` resolves the status provider at draw time), but startup remains
  operator-owned proof.
- **No proof that the Options sub-menu covers neither the chat nor the status bar** (SC-008).
- **No proof that the minimap/chat/rails look correct** or that the collapse/expand controls are
  reachable and behave as labelled.
- **T106 is not implemented**; there is no command palette.
- **T100 type scale is not applied** (see §7): the viewer uses one ImGui font, so the token values
  are the contract for a future multi-font setup rather than a per-line size today.
