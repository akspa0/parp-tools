# Spec Routing Registry

Companion to [STATUS.md](STATUS.md). After the 2026-09-23 reconciliation the registry is short:

| Class | Where | How to use |
|---|---|---|
| Active epics | `specs/248-…` to `specs/254-…` | The only live plans. Select one item from [TRIAGE.md](TRIAGE.md) once it is marked Want. |
| Archived specs | `specs/archived/<id>-<name>/` | History. Each carries a banner naming its disposition and successor epic. |
| Reconciliation receipt | [archived/reconciliation-2026-09-23/](archived/reconciliation-2026-09-23/README.md) | Per-spec disposition ledger + audit reports. |
| Earlier archive rationale | [archived/ARCHIVED.md](archived/ARCHIVED.md) | Specs archived before 2026-09-23. |
| Superseded before 2026-09-23 | [archived/superseded/](archived/superseded/README.md) | 080, 145, 195. |
| Status history | [archived/status-history/](archived/status-history/README.md) | Earlier routers, registries, epic indexes and day plans. |

## Rules

- A cold or archived record is preserved history, not implementation authority, unless an epic's
  `plan.md` adopts one of its design documents by link for a named item.
- Reviving archived scope = adding an item to an epic with operator approval (AGENTS.md §9.1).
- Spec numbers continue from 255; `.specify` auto-numbering scans top-level `specs/` directories.
