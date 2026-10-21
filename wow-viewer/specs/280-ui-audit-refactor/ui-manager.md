# UI Manager — One Cohesive Shell

The operator's big ask: consolidate every old bar, panel, and popup into **one UI manager/handler**
so the whole interface is cohesive from day one instead of a pile of services that each draw their
own window in their own style.

This document is the architecture for that manager. It is the implementation contract for
[plan.md](plan.md) P2–P9.

## 1. The problem

Today at least six services own windows and bars independently:

| Service | Owns |
|---|---|
| `MainMenuBarService` | Top menu bar, About popup, open-file popup |
| `ViewerChromeService` | Bottom bar, top toolbar, overlays/layers/transparency popups |
| `MinimapAndStatusService` | Minimap window, fullscreen minimap, status bar |
| `FakeChatWindowService` | Chat window |
| `NavigatorPanelService` | Left sidebar |
| `WorkbenchPanelsService` | Right sidebar and workbench tabs |
| `ShellLayoutService` | Dockspace, panel placement, legacy layout |
| `SettingsWindowService`, `PerfPanelService`, `LightingPanelService`, `AudioPanelService`, `TaxiPanelService`, `CapturePanelService`, `AssetCatalogView` | Their own floating windows |

Each one sets its own position, size, padding, colors, and fonts. That is why the UI looks like
three rebuilds: it *is* three rebuilds, each with its own window code.

## 2. The decision

Introduce **one** `ViewerUiManager` that owns every chrome region, the design tokens, the draw
order, input routing, and the documentation registry. Existing services become **content
providers**: they expose `DrawContent()` and never call `ImGui.Begin` for chrome.

```text
ViewerUiManager
├── UiLayout          (regions, sizes, breakpoints, persistence)
├── UiTheme           (color, type, spacing tokens)
├── UiChrome          (top bar, bottom bar, status bar, chat, minimap)
├── UiSidebars        (left and right hosts; content comes from providers)
├── UiPopups          (one popup/modal registry; one draw order)
├── UiInput           (which region owns the pointer; shortcut routing)
└── UiDocumentation   (every control's label, shortcut, and help text)
```

### 2.1 What the manager owns

- The ImGui window for every chrome region.
- The position, size, padding, and visibility of every region.
- The draw order: viewport → sidebars → minimap → chat → bottom bar → status bar → popups.
- The theme tokens; no provider sets a color or font.
- The popup stack; no provider opens its own popup.
- The input ownership rule; no provider decides whether the scene gets the mouse.
- The documentation registry; no control exists without an entry.

### 2.2 What a provider owns

- Its content, its state, and its actions.
- A `DrawContent()` method that draws into the region the manager gives it.
- A `DescribeControls()` method that returns its control metadata for the documentation registry.

A provider never calls `ImGui.Begin`, `ImGui.SetNextWindowPos`, `ImGui.PushStyleColor`, or
`ImGui.OpenPopup` for chrome. It may use ImGui widgets inside its content.

## 3. Region model

```csharp
enum UiRegion
{
    TopBar,
    Minimap,
    LeftSidebar,
    Viewport,
    RightSidebar,
    Chat,
    Options,
    BottomBar,
    StatusBar,
    Popup,
}
```

Each region has a `UiRegionState`:

```csharp
sealed record UiRegionState(
    UiRegion Region,
    bool Visible,
    UiRect Rect,
    UiDockSide Dock,        // Floating, Left, Right, Top, Bottom
    float Width,
    float Height,
    bool Collapsed);
```

The manager computes the rects from the layout rules in [layout.md](layout.md) §3 and §9. Providers
receive a rect and draw inside it.

## 4. Draw order

```text
1. Viewport (the 3D scene)
2. Left sidebar
3. Right sidebar
4. Minimap (floating or docked)
5. Chat
6. Bottom bar
7. Status bar
8. Options sub-menu
9. Popups and modals (topmost)
```

The order is fixed. A provider cannot raise itself above another region.

## 5. Input routing

The manager decides which region owns the pointer before the scene sees it:

```text
if pointer is over a chrome region -> chrome owns it
else if pointer is over the viewport -> scene owns it
```

This replaces the current `CanSceneConsumeMouse` / `IsSceneMouseCaptureBlocked` checks scattered
across services. One rule, one place.

Keyboard shortcuts are registered in the manager:

| Key | Action | Owner |
|---|---|---|
| `Tab` | Hide/show chrome | Manager |
| `[` / `]` | Toggle left/right sidebar | Manager |
| `M` | Toggle minimap | Manager |
| `I` | Toggle inspector | Manager |
| `P` | Toggle PM4 workbench | Manager |
| `Ctrl+K` | Command palette | Manager |
| `Esc` | Close topmost popup | Manager |
| `F1` | Keyboard shortcuts | Manager |

A provider can register a shortcut, but the manager resolves conflicts and shows it in the UI
reference.

## 6. Theme tokens

The manager owns the only token set. Providers ask for a token, never a raw color.

```csharp
static class UiTokens
{
    // Color
    public static readonly Vector4 Surface0, Surface1, Surface2, Border;
    public static readonly Vector4 Text, TextMuted, Brass, BrassDim, Danger, Success;

    // Type
    public static readonly float Caption, Body, Label, Title;

    // Spacing
    public static readonly float Space4, Space8, Space12, Space16, Space24, Space32;
    public static readonly float ControlHeight, RowHeight, SectionHeaderHeight;
}
```

`SharedUiWidgets` is rebuilt on top of these tokens. A provider that hard-codes a color fails the
style check.

## 7. Documentation registry

The operator's second ask: documentation must not lag behind. The manager makes that structural.

Every control registers:

```csharp
sealed record UiControlDoc(
    string Id,
    string Label,
    string Region,
    string? Shortcut,
    string Help,
    bool Visible);
```

The registry feeds:

1. The in-app **Help > Keyboard Shortcuts** window.
2. The in-app **Help > UI Reference** window (searchable).
3. The generated `docs/WoWViewer/UI-REFERENCE.md`.
4. A build check that fails when a visible control has no entry.

### 7.1 The rule

> A control that is not in the registry is not allowed to ship.

This is how the minimap's hidden Shift/Ctrl behaviour gets fixed: the behaviour is either given a
visible control and a registry entry, or it is removed.

### 7.2 Generated reference

`docs/WoWViewer/UI-REFERENCE.md` is generated from the registry, not hand-written. It lists every
region, every control, every shortcut, and every modifier. A CI check regenerates it and fails if
the committed file differs.

## 8. Migration map

| Current owner | Becomes | Manager takes over |
|---|---|---|
| `MainMenuBarService` | `TopBarContentProvider` | Top bar window, About popup |
| `ViewerChromeService` | `BottomBarContentProvider` + `OptionsContentProvider` | Bottom bar, status bar, popups |
| `MinimapAndStatusService` | `MinimapContentProvider` + `StatusBarContentProvider` | Minimap window, fullscreen overlay, status bar |
| `FakeChatWindowService` | `ChatContentProvider` | Chat window, Options anchor |
| `NavigatorPanelService` | `LeftSidebarContentProvider` | Left sidebar window |
| `WorkbenchPanelsService` | `RightSidebarContentProvider` | Right sidebar window, tab strip |
| `ShellLayoutService` | Deleted | Region layout and persistence |
| `SettingsWindowService` | `SettingsContentProvider` | Settings popup |
| `PerfPanelService` | `DiagnosticsContentProvider` | Diagnostics region |
| `LightingPanelService` | `OptionsContentProvider` | Options > Display |
| `AudioPanelService` | `OptionsContentProvider` | Options > Audio |
| `TaxiPanelService` | `SceneToolsContentProvider` | Scene Tools region |
| `CapturePanelService` | `SceneToolsContentProvider` | Scene Tools region |
| `AssetCatalogView` | `AssetsContentProvider` | Assets region |

The providers keep their logic. They lose their window code.

## 9. Phases

### U0 — Manager skeleton

- Create `ViewerUiManager`, `UiRegion`, `UiRegionState`, `UiTokens`.
- Draw the empty regions with the correct rects.
- No provider migration yet.

### U1 — Top bar, bottom bar, status bar

- Move `MainMenuBarService`, `ViewerChromeService` bottom bar, and `MinimapAndStatusService` status
  bar into the manager.
- Delete their window code.

### U2 — Sidebars

- Move `NavigatorPanelService` and `WorkbenchPanelsService` into the manager's sidebar hosts.
- Delete `ShellLayoutService`.

### U3 — Minimap and chat

- Move the minimap into the manager; support floating and right-sidebar dock.
- Move the chat into the manager; anchor Options.

### U4 — Popups and Options

- One popup registry; move Settings, About, open-file, overlays, layers, transparency.
- Build the Options sub-menu.

### U5 — Documentation registry

- Add `UiControlDoc` to every provider.
- Generate `UI-REFERENCE.md`.
- Add the build check.

### U6 — Delete the old shell

- Delete the legacy floating windows and the compatibility enums.
- Delete `ShellLayoutService` and the dockspace path.

## 10. Acceptance

- One service owns every chrome window.
- No provider calls `ImGui.Begin` for chrome.
- One token set; no provider hard-codes a color or font.
- One input-ownership rule.
- Every visible control has a registry entry.
- `UI-REFERENCE.md` is generated and matches the registry.
- The minimap has no modifier-only action.
