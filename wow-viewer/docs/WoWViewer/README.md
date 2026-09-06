# WoWViewer viewer docs

This folder is the viewer-facing guide layer for the current `wow-viewer` app.

## Read order

1. [Viewer README](../../README.md)
2. [Desktop app README](../../src/viewer/WoWViewer/README.md)
3. [USERGUIDE.md](USERGUIDE.md)
4. [CLI tooling README](../../tools/README.md)
5. [Expanded CLI reference](../CLI-TOOLS.md)
6. [Release notes — v0.5.2](../releases/v0.5.2.md)
7. [Spec 227 — UI re-audit](../../specs/227-ui-reaudit/spec.md)
8. [Spec 229 — WoW shell and keybind profiles](../../specs/229-wow-shell-keybind-profiles/spec.md)

## Current viewer truth

- The viewer app lives at `src/viewer/WoWViewer/`.
- Start with staged client roots only.
- Legacy `MdxViewer` is a reference/compatibility lane, not the primary app.
- The current UI authority audit is Spec 227; the WoW-style shell and contextual keybind plan is
  Spec 229. Specs 080 and 145 are preserved only as superseded historical context.
- Command-line tooling is documented from `tools/README.md`; this folder stays viewer-facing.

## What this folder should contain

- End-user quickstart
- Controls
- Viewer workflows
- Troubleshooting

It should not carry stale release notes, machine-local absolute paths, or dead links to removed
exporter docs. Release notes belong in [`docs/releases/`](../releases/).
