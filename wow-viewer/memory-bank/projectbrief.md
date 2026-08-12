# Project Brief — wow-viewer

`wow-viewer` is the active, standalone development target for reading and exploring World of
Warcraft client data. The legacy `gillijimproject_refactor` tree is a read-only reference unless a
user explicitly requests a bounded compatibility fix.

## Product scope

- .NET/OpenGL viewer for terrain, WMO, M2/MDX, liquids, DBC-driven context, and PM4 overlays.
- Shared format libraries and thin inspection/conversion tools.
- Optional data-harvester workflows for reproducible terrain artifacts and model experiments.
- Capture/camera authoring and future audio playback are product work; they are governed by their
  owning specs, not by this brief.

## Architecture boundaries

- Readers and reusable contracts live in `src/core/`; viewer shell work lives in
  `src/viewer/WoWViewer/`; CLI tools live in `src/tools/`; tests live in `tests/`.
- New behavior starts in `specs/`, follows Specify -> Plan -> Tasks -> one validated phase, and
  records proof in the owning spec.
- Runtime client roots are configuration. `H:\CLIENTS` is an approved local library, but no
  machine-local client path belongs in source or portable configuration.
- The user owns training, GPU jobs, broad harvests, long captures, and real-client/runtime proof.

## Continuity contract

- `memory-bank/activeContext.md` is the short current handoff.
- `memory-bank/progress.md` is a short newest-first implementation ledger.
- `memory-bank/workstream-*.md` and selected architecture notes hold durable findings only when a
  spec links them.
- Superseded detail belongs in `memory-bank/archive/` or the owning spec history. Do not load it for
  ordinary handoff.

This file is intentionally stable. Current work, task counts, and validation state belong in
`specs/STATUS.md` and the active spec, not here.
