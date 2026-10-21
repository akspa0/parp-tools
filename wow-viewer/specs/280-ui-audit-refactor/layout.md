# Visual Layout — Old-WoW Structure, 2026 Execution

Companion to [information-architecture.md](information-architecture.md). The IA decides *what
lives where*; this document decides *how it looks and sits on screen*.

## 1. Reference and intent

The reference is the **pre-alpha / early-beta WoW UI (1999–2003)**, not the 2004 release UI and not
the 2022 dragonflight-era refactor. That early UI was closer to EverQuest and Heroes of Might and
Magic: a framed viewport, a top-right minimap, a bottom-left chat, and heavy chrome. It was
replaced because it looked dated — but its *structure* is still good: the world is the centre, the
minimap is a glanceable corner instrument, and the chat is where the player already looks.

We keep the structure and replace the 2001 surface treatment with a 2026 one:

| Keep from old WoW | Modernize for 2026 |
|---|---|
| Minimap top-right, over the viewport | Rounded-square map, thin brass ring, collapse button, high-DPI |
| Chat bottom-left | Resizable dock, search, timestamps, channel tabs, no faux parchment |
| Framed viewport | Edge-to-edge viewport with floating chrome; no beveled frame |
| Bottom action bar | Slim destination bar with text + icon, not 12px bitmap buttons |
| Brass / leather / parchment palette | Dark neutral surfaces, brass only for active state and focus |
| Tooltips on everything | Same, plus keyboard focus rings and screen-reader labels |

The goal is "a 2026 tool that remembers 2001", not a 2001 skin.

## 2. Design principles

1. **The world is the centre.** Chrome floats over the viewport; it does not box it in.
2. **One accent.** Brass is the only accent colour. It marks the active destination, the focused
   control, and the selected object. Everything else is neutral.
3. **Flat, not beveled.** 1px borders, subtle elevation, no gradients on controls, no faux metal.
4. **One type scale.** 12 / 14 / 16 / 20 px. One family. No per-panel font sizes.
5. **8px grid.** All padding, gaps, and control heights are multiples of 4 or 8.
6. **Labels beat icons.** Icon + label for primary actions; icon-only only for the minimap and
   window controls, always with a tooltip.
7. **Search everywhere.** Every list longer than 10 rows has a filter field.
8. **Keyboard first.** Every destination and panel has a shortcut; every popup closes on Esc.
9. **Empty states explain.** A blank panel says what to load and offers the action.
10. **Responsive by collapse, not by shrink.** Under pressure, panels collapse to a rail; text does
    not become unreadable.

## 3. Screen regions

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ File   View   Help                                        ┌───────────────┐  │
│                                                           │   MINIMAP     │  │
│                                                           │  zone / coord │  │
│                                                           └───────────────┘  │
├───────────────┬──────────────────────────────────────────────┬───────────────┤
│ Left sidebar  │                                              │ Right sidebar │
│ (destination) │                 viewport                     │ (destination) │
│               │                                              │               │
│               │                                              │               │
├───────────────┴──────────────────────────────────────────────┴───────────────┤
│ Chat (docked panel, full scene width, not movable)     [Options ▾]           │
├──────────────────────────────────────────────────────────────────────────────┤
│ Viewer │ Editor │ Archaeology │ Options              FPS / stats            │
└──────────────────────────────────────────────────────────────────────────────┘
```

| Region | Position | Default size | Behaviour |
|---|---|---|---|
| Top bar | Top, full width | 28 px | File / View / Help; transparent over the viewport |
| Minimap | Top-right, floating | 180 × 180 px | Overlays the viewport; collapsible to 32 px |
| Left sidebar | Left, full height between bars | 280 px | Destination-aware; collapsible to 0 |
| Right sidebar | Right, full height between bars | 360 px | Destination-aware; collapsible to 0 |
| Viewport | Centre | Remaining | Edge-to-edge; no frame |
| Chat | Bottom-left, above the bottom bar | 420 × 160 px | Resizable; collapsible to a one-line bar |
| Options | Anchored to the chat's right edge | 320 px wide | Opens upward; closes on Esc or outside click |
| Bottom bar | Bottom, full width | 32 px | Destinations + FPS/stats |
| Status bar | Bottom, full width | 22 px | Area, tiles, chunks, M2/WMO, audio, pending |

## 4. Minimap specification

The minimap has **two homes**, and the operator chooses:

1. **Floating top-right** (default): the old-WoW position, over the viewport.
2. **Docked in the right sidebar**: a first-class panel at the top of the right sidebar, above the
   inspector tabs.

Both homes use the same renderer, the same cache, and the same interaction state. The choice is
persisted per user.

| Property | Value |
|---|---|
| Shape | Rounded square, 12 px corner radius (a circle is the 2004 look; a rounded square is the 2026 one) |
| Size | 180 × 180 px floating; 100% of the sidebar width when docked; 32 px collapsed |
| Ring | 2 px brass border; 1 px inner shadow |
| Zone label | Below the map, 14 px, parchment text, truncated with tooltip |
| Coordinates | Below the zone label, 12 px, muted |
| Buttons | Top-right of the map: zoom +, zoom −, track, dock/undock, collapse |
| Data | The same minimap tiles the left sidebar uses today; one renderer, one cache |
| Collapse | Collapses to a 32 px brass button with the zone initial; `M` toggles |

The left sidebar's "World Overview" keeps the **map list and map export**, but the minimap itself
moves to the corner or the right sidebar. The Utilities Minimap page is deleted (see the IA).

### 4.1 Minimap controls — documented, not hidden

The operator reports that the minimap has undocumented Shift/Ctrl behaviours for moving it around.
That is a defect: a modifier-only action with no visible control and no tooltip is undiscoverable.
The rework makes every minimap action visible and documented.

| Action | Current behaviour | Target behaviour |
|---|---|---|
| Zoom | Mouse wheel over the map | Wheel, plus visible `+` / `−` buttons |
| Pan | Left-drag on the map | Left-drag, plus a visible "Reset Pan" button |
| Teleport | Triple-click a tile | Triple-click, plus a "Teleport to tile" context action |
| Focus a LIT light | Double-click a marker | Double-click, plus a context action |
| Fullscreen | `M` | `M`, plus a visible expand button |
| Donor tool | `T` in fullscreen | `T`, plus a visible toolbar button |
| Move the window | Drag the title bar; reported Shift/Ctrl variants | Drag the title bar; a visible "Dock / Undock" button; no modifier-only move |
| Dock to sidebar | Not exposed | Visible "Dock to sidebar" button |
| Reset pan | Small button when panned | Always-visible reset in the map menu |

**Rule**: no action may exist only as a hidden modifier. If a modifier shortcut is kept, the same
action must also be reachable from a visible control and listed in the UI reference.

## 5. Chat specification

The chat is a **docked panel**, not a floating window. It spans the scene between the sidebars and
sits on the bottom edge of the scene area, so it can never cover a sidebar, the bottom bar or the
status bar, and it has no drag or resize handle to lose.

> **Operator correction, 2026-10-21:** the original draft made the chat a resizable bottom-left
> floating dock. The operator rejected that: "the floating chat window ... needs to be a fucking
> panel in the ui, not a floating ... window ... that covers the fucking ui and cannot be moved."
> The chat is now a fixed docked panel. `UiLayout` derives its rect from the sidebar widths and the
> scene height; `ViewerUiManager` draws it with `NoMove | NoResize`; the persisted floating rect
> (`ChatWidth`/`ChatHeight`/`ChatX`/`ChatY`) is deleted and the layout document is bumped to v3.

| Property | Value |
|---|---|
| Position | Docked: spans the scene between the sidebars, bottom edge of the scene area |
| Size | Full scene width × `min(180 px, 28% of scene height)`; **not resizable, not movable** |
| Collapsed | One line, 28 px, with the last message and an expand chevron |
| Tabs | General, System, NPC, Errors |
| Controls | Auto-Scroll, Clear, Filter, Options ▾ |
| Population controls | Auto-Stream and Spawn BBs Only stay in the chat bar (they belong to the population system) |
| Styling | Dark translucent surface, 1 px border, no parchment texture |
| Timestamps | Optional, 12 px muted |

## 6. Sidebar specification

### Left sidebar

| Property | Value |
|---|---|
| Width | 280 px default; 240 px at < 1600 px; 0 collapsed |
| Sections | Collapsible headers, 28 px, brass chevron |
| Order | Workspace, World Overview, File Browser, destination sections, Diagnostics, Chat anchor |
| Scroll | One scroll region for the whole sidebar; section headers stick |
| Collapse | `[` toggles; a 24 px rail with section icons remains |

### Right sidebar

| Property | Value |
|---|---|
| Width | 360 px default; 320 px at < 1600 px; 0 collapsed |
| Tabs | Destination-specific; 28 px tab strip |
| Pages | A single combo selector under the tab strip, not a second tab row |
| Scroll | One scroll region per page |
| Collapse | `]` toggles; a 24 px rail remains |

## 7. Bottom bar specification

| Property | Value |
|---|---|
| Height | 32 px |
| Left | `Viewer | Editor | Archaeology` as text buttons with a small icon |
| Right | `Options ▾` and the FPS/stats readout |
| Active state | Brass underline and brass text; inactive is muted |
| Overflow | Horizontal scroll with Shift+wheel, as today |
| Status bar | 22 px below the bottom bar; never hidden |

## 8. Design tokens

### Colour

| Token | Value | Use |
|---|---|---|
| `surface-0` | `#12100E` | Viewport surround, deepest background |
| `surface-1` | `#1B1815` | Sidebar and bar background |
| `surface-2` | `#24201C` | Raised control, popup |
| `border` | `#3A332B` | 1 px borders |
| `text` | `#E8DFD0` | Primary text |
| `text-muted` | `#9A8F7E` | Secondary text |
| `brass` | `#C9A227` | Active destination, focus, selection |
| `brass-dim` | `#8A6F1E` | Hover |
| `danger` | `#C4553B` | Errors |
| `success` | `#6E9E5A` | Success |

### Type

| Token | Size | Weight | Use |
|---|---|---|---|
| `caption` | 12 px | 400 | Coordinates, timestamps, hints |
| `body` | 14 px | 400 | Default |
| `label` | 14 px | 600 | Section headers, buttons |
| `title` | 20 px | 600 | Page titles only |

One family: the existing UI font. No per-panel overrides.

### Spacing

`4 / 8 / 12 / 16 / 24 / 32`. Control height 24 px; row height 28 px; section header 28 px.

## 9. Responsive breakpoints

| Width | Behaviour |
|---|---|
| ≥ 1920 | Full layout; both sidebars; 180 px minimap |
| 1600–1919 | Left 240 px, right 320 px, minimap 160 px |
| 1280–1599 | Left sidebar collapses to a rail by default; minimap 140 px |
| < 1280 | Both sidebars collapse to rails; minimap 120 px; bottom bar scrolls |

Collapse is a user choice that persists; the breakpoint only sets the default.

## 10. Interaction and keyboard

| Key | Action |
|---|---|
| `Tab` | Hide/show all chrome |
| `[` / `]` | Toggle left / right sidebar |
| `M` | Toggle minimap |
| `I` | Toggle inspector |
| `P` | Toggle PM4 workbench |
| `Ctrl+K` | Command palette (2026 addition) |
| `Esc` | Close the topmost popup |
| `F1` | Keyboard shortcuts |

The command palette is the modern replacement for the deep menus: it searches every destination,
page, and action by name. It does not replace the menus; it makes them optional.

## 11. What we deliberately do not copy

- The 2001 beveled metal buttons and parchment panels.
- The 2004 circular minimap with the gold dragon ring.
- The 12 px bitmap font.
- The fixed 4:3 viewport frame.
- The action-bar grid of 12 slots. Our bottom bar is navigation, not spells.

## 12. Implementation order

1. **Tokens and primitives** (P8): colour, type, spacing, section header, button, empty state.
2. **Minimap relocation** (P9): move the renderer to the top-right, add collapse and the zone
   label; delete the Utilities Minimap page.
3. **Chat dock** (P9): docked panel spanning the scene between the sidebars; not movable or
   resizable; keep the population controls.
4. **Sidebar rails** (P9): collapse behaviour and persistent width.
5. **Bottom bar** (P2): destinations, Options, FPS/stats.
6. **Command palette** (P9, optional): `Ctrl+K` search over destinations and actions.
7. **Responsive pass** (P9): breakpoints and default collapse.

## 13. Open questions for the operator

- Rounded-square minimap (proposed) or circular minimap (2004 look)?
- Chat as a resizable bottom-left dock (proposed) or a full-width bottom bar (current)?
- Command palette in scope for this refactor, or a later addition?
- Keep the brass accent (proposed) or move to a cooler 2026 palette?
