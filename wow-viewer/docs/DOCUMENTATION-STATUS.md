# Documentation Status

Updated: 2026-08-12

This is a routing page, not a project history. The owning spec is authoritative for behavior and
validation.

## Default handoff

1. [AGENTS.md](/I:/parp/parp-tools/wow-viewer/AGENTS.md)
2. [specs/STATUS.md](/I:/parp/parp-tools/wow-viewer/specs/STATUS.md)
3. [memory-bank/activeContext.md](/I:/parp/parp-tools/wow-viewer/memory-bank/activeContext.md)
4. The selected spec's `spec.md`, `plan.md`, and `tasks.md`
5. Only notes explicitly linked by that spec

## Operator references

- [README.md](/I:/parp/parp-tools/wow-viewer/README.md)
- [docs/CLI-TOOLS.md](/I:/parp/parp-tools/wow-viewer/docs/CLI-TOOLS.md)
- [docs/WoWViewer/USERGUIDE.md](/I:/parp/parp-tools/wow-viewer/docs/WoWViewer/USERGUIDE.md)
- [data-harvester/README.md](/I:/parp/parp-tools/wow-viewer/data-harvester/README.md)

## Background references

- `docs/architecture/` contains durable design and research notes; read only the file linked by the
  active spec.
- `memory-bank/workstream-*.md` contains durable findings and traps; it is not default context.
- `memory-bank/archive/`, `docs/audits/`, `docs/validation/`, and `plans/` are historical unless a
  live spec links them.

When a lane changes, update `specs/STATUS.md`, the owning spec, and the short memory-bank handoff
in the same pass. Do not maintain a second active-spec inventory here.
