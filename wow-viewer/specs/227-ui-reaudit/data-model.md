# Audit Record Model — Spec 227

The inventory is documentation, not a runtime data model. Every inventory row has the following
required fields so an implementation task can be scoped and evidenced without guessing.

| Field | Meaning | Validation |
|---|---|---|
| Inventory ID | Stable `UIV2-*` identifier | Unique within `surface-inventory-v2.md`. |
| Workspace/profile | Viewer, Editor, Archaeology, or Legacy shell | Matches the route that makes the surface visible. |
| Surface and source host | User-visible label plus exact source symbol/file | Source path and symbol resolve in the current worktree. |
| Data/action authority | Existing owner of the shown data or action | One owner remains after consolidation. |
| Duplicate family | None, Weak-signal, Minimap, or Inspector | Required for every known duplicate. |
| Disposition | Keep, route/link, retire, or pending runtime decision | A non-Keep row names a replacement ID. |
| Evidence state | Source-audited, screenshot pending, or operator-verified | Only operator-verified can support runtime/visual acceptance. |

**Relationship rule**: a route/link or retirement row references exactly one authoritative
replacement row. A row with a pending runtime decision is not an authorization for source removal.
