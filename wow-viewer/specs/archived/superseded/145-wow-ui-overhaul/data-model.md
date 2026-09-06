# Data Model: WoWViewer UI Overhaul

## ViewerKeyBinding

| Field | Meaning |
|---|---|
| `Id` | Stable action identifier used by help and future rebinding work |
| `Category` | Human-readable grouping such as Global, Capture, Model, World, or Utilities |
| `Gesture` | Display form of the current key or key combination |
| `Description` | User-facing explanation of the action |
| `Context` | Page context in which the action is eligible |
| `IsGlobal` | Whether the action remains eligible in every page context |

## ViewerKeyContext

| Field | Meaning |
|---|---|
| `TopPage` | Model, World, or Tools main page |
| `NestedPage` | Utilities, Capture, Camera Path, or the relevant nested page identity |
| `DisplayName` | Help-dialog label for the current context |

## PersistentUtilityWindow

| Field | Meaning |
|---|---|
| `Visibility` | Whether the window should be drawn this frame |
| `Title` | Title-bar label and ImGui identity |
| `Owner` | Page that can reopen the window |
| `CloseBehavior` | Title-bar X is authoritative and clears visibility |

## Relationships

- One active `ViewerKeyContext` selects zero or more eligible `ViewerKeyBinding` records.
- A `PersistentUtilityWindow` may be launched by one page but remains independent of page selection while visible.
- Help displays global bindings plus the bindings matching the active context.
