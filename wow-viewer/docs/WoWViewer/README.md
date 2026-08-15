# WoWViewer viewer docs

This folder is the viewer-facing guide layer for the current `wow-viewer` app.

## Read order

1. [Viewer README](../../README.md)
2. [USERGUIDE.md](USERGUIDE.md)
3. [CLI-TOOLS.md](../CLI-TOOLS.md)
4. [Release notes — v0.5.2](../releases/v0.5.2.md)
5. [Spec 080 — UI consolidation](../../specs/080-wow-ui-consolidation/spec.md)

## Current viewer truth

- The viewer app lives at `src/viewer/WoWViewer/`.
- Start with staged client roots only.
- Legacy `MdxViewer` is a reference/compatibility lane, not the primary app.
- The current UI doc lane is Spec 080; Spec 145 owns the surrounding shell work.

## What this folder should contain

- End-user quickstart
- Controls
- Viewer workflows
- Troubleshooting

It should not carry stale release notes, machine-local absolute paths, or dead links to removed
exporter docs. Release notes belong in [`docs/releases/`](../releases/).
