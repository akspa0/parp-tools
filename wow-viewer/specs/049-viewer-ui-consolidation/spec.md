# Spec 049: Viewer UI Consolidation

**Status**: In Progress | **Priority**: P1 | **Owner**: WoWViewer

## Problem

The viewer's right sidebar has too many separate panels (Selection, World Objects, Model Info, Runtime Stats, Terrain Controls, PM4 Workbench) that all relate to the same task: inspecting and understanding a 3D scene. Users must hunt through multiple collapsible sections and docked panels to find settings, losing context. Meanwhile, specialist tools like UniqueId Archaeology, Taxi Panel, and Weak Signal Amplifier are buried inside generic panels.

## User Stories

### US1: Consolidated Scene Inspector (P1)
As a user inspecting a scene, I want one panel that shows selection info, world object controls, model details, runtime stats, and terrain controls in a tabbed layout, so I don't have to open and rearrange multiple panels to do my work.

### US2: Specialist Tool Windows (P1)
As a user, I want specialist tools (UniqueId Archaeology, Taxi Panel, Weak Signal Amplifier, Chunk Clipboard) as separate floating windows that stay where I put them, so they're available when I need them and out of the way when I don't.

### US3: Tools Menu Organization (P2)
As a user, I want every tool window listed in the Tools menu with its current open/closed state visible, so I can discover and toggle features without memorizing keyboard shortcuts or digging through menus.

## Functional Requirements

### FR-001: Consolidated Panel
- The right-side panels (Selection, World Objects, Model Info, Runtime Stats, Terrain Controls) merge into one `Scene Inspector` docked panel
- Uses a tab bar at the top for switching between: Selection | World | Model | Stats | Terrain
- Each tab shows the EXACT same content as the original panel — no functionality removed

### FR-002: Tab Persistence
- The active tab survives panel close/reopen within a session
- Each tab uses the same internal state as the existing standalone panels

### FR-003: Existing Panels Remain
- The old individual panels (Selection, World Objects, etc.) remain in the shell panel system and can still be enabled/disabled
- The Scene Inspector panel defaults to the right lane alongside (not replacing) existing panels
- Users can close the old panels and keep only the Scene Inspector

### FR-004: Tool Windows
- UniqueId Archaeology: floating window with layer table, filter, scope
- Taxi Panel: floating window with route list, ride camera, controls
- Weak Signal Amplifier: floating window with range controls, quick buttons, scale
- Chunk Clipboard: already exists as floating window

### FR-005: Tools Menu
- All floating tool windows listed in Tools menu with checkbox toggle
- Scene Inspector listed in View menu alongside other shell panels

## Success Criteria

1. Scene Inspector panel renders all five tabs without duplicating code
2. All original controls present in each tab
3. Tool windows open/close via Tools menu
4. Tool window positions persist across restarts
5. No functionality regression in any panel

## Assumptions

- ImGui tab bars are sufficient; no custom tab widget needed
- The existing `DrawSelectionPanelContent`, `DrawWorldObjectsPanelContent`, etc. methods can be called directly from tab content
- Floating window positions are persisted by ImGui's .ini system automatically
