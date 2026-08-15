# Documentation Status

Updated: 2026-08-12

This is a routing page, not a project history. The owning spec is authoritative for behavior and
validation.

## Default handoff

1. [AGENTS.md](../AGENTS.md)
2. [specs/STATUS.md](../specs/STATUS.md)
3. [memory-bank/activeContext.md](../memory-bank/activeContext.md)
4. The selected spec's `spec.md`, `plan.md`, and `tasks.md`
5. Only notes explicitly linked by that spec

## Operator references

- [README.md](../README.md)
- [docs/CLI-TOOLS.md](CLI-TOOLS.md)
- [docs/WoWViewer/USERGUIDE.md](WoWViewer/USERGUIDE.md)
- [data-harvester/README.md](../data-harvester/README.md)

## Background references

- `docs/architecture/` contains durable design and research notes; read only the file linked by the
  active spec.
- `memory-bank/workstream-*.md` contains durable findings and traps; it is not default context.
- `memory-bank/archive/`, `docs/audits/`, `docs/validation/`, and `plans/` are historical unless a
  live spec links them.

When a lane changes, update `specs/STATUS.md`, the owning spec, and the short memory-bank handoff
in the same pass. Do not maintain a second active-spec inventory here.
