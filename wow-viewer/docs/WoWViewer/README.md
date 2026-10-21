# WoWViewer viewer docs

This folder is the viewer-facing guide layer for the current `wow-viewer` app.

## Guides

- [Viewer project overview](../../README.md)
- [Desktop app overview](../../src/viewer/WoWViewer/README.md)
- [User guide](USERGUIDE.md)
- [UI reference](UI-REFERENCE.md) — generated list of every documented control, its shortcut and its home surface
- [Development map restoration](development-map-restoration.md)
- [CLI tooling](../../tools/README.md)
- [Expanded CLI reference](../CLI-TOOLS.md)

## Current viewer truth

- The viewer app lives at `src/viewer/WoWViewer/`.
- Start with staged client roots only.
- Legacy `MdxViewer` is a reference/compatibility lane, not the primary app.
- Command-line tooling is documented from `tools/README.md`; this folder stays viewer-facing.
- The viewer is one shell with three destinations (Viewer, Editor, Archaeology) selected from the
  bottom bar; the [user guide](USERGUIDE.md) describes the surfaces and the
  [UI reference](UI-REFERENCE.md) lists every control.

## What this folder should contain

- End-user quickstart
- Controls
- Viewer workflows
- Troubleshooting

This folder is for current user workflows, controls, and troubleshooting.
