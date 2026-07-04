# WoWViewer viewer docs

This folder is viewer-facing guide layer for current `wow-viewer` app.

## Read order

1. [../../README.md](/I:/parp/parp-tools/wow-viewer/README.md)
2. [USERGUIDE.md](/I:/parp/parp-tools/wow-viewer/docs/WoWViewer/USERGUIDE.md)
3. [../../docs/CLI-TOOLS.md](/I:/parp/parp-tools/wow-viewer/docs/CLI-TOOLS.md)
4. [../../specs/080-wow-ui-consolidation/spec.md](/I:/parp/parp-tools/wow-viewer/specs/080-wow-ui-consolidation/spec.md)

## Current viewer truth

- Viewer app lives at `src/viewer/WoWViewer/`.
- Start with staged client roots only.
- Legacy `MdxViewer` is reference/compatibility lane, not primary app.
- Current UI doc lane is Spec 080.
- Current shell proof is mixed: active `wow-viewer` app is canonical, but latest landed compatibility slice for Spec 080 is still source-only in legacy `MdxViewer`.

## What this folder should contain

- End-user quickstart
- controls
- viewer workflows
- troubleshooting

It should not carry stale release notes, old fixed-path machine commands, or dead links to removed exporter docs.
