# game-viewer Micro-Plan Pack

## Status

- status: active
- date: 2026-05-14
- parent: `wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md`
- host-plan: `wow-viewer/docs/architecture/game-viewer-host-plan-2026-05-13.md`
- editor-plan: `wow-viewer/docs/architecture/wow-engine-editor-and-interop-plan-2026-05-14.md`

## Purpose

This folder is the execution skeleton for the next month of `game-viewer` work.

The goal is not one giant plan. The goal is many tiny plans that can be picked up in bounded slices over hours, with minimal ambiguity and minimal re-planning.

## Execution Rhythm

- Prefer one micro-plan per focused slice.
- Land proof before opening the next dependency.
- Keep runtime, backend, editor, compatibility, and data concerns separate unless a plan explicitly joins them.
- Treat these plans as building blocks, not promises of immediate implementation.

## Lanes

### Foundation

- [GV-00-universal-content-contracts.md](GV-00-universal-content-contracts.md)
- [GV-00A-artifact-provenance-and-preservation.md](GV-00A-artifact-provenance-and-preservation.md)
- [GV-01-core-constants-registry.md](GV-01-core-constants-registry.md)
- [GV-02-wow-alpha-constants-pack.md](GV-02-wow-alpha-constants-pack.md)
- [GV-03-wow-retail-constants-pack.md](GV-03-wow-retail-constants-pack.md)
- [GV-04-warcraft3-constants-pack.md](GV-04-warcraft3-constants-pack.md)
- [GV-05-game-build-metadata-probe.md](GV-05-game-build-metadata-probe.md)
- [GV-06-compatibility-profile-registry.md](GV-06-compatibility-profile-registry.md)

### Data Roots And Interop

- [GV-07-game-root-records-and-manager-service.md](GV-07-game-root-records-and-manager-service.md)
- [GV-08-feature-gate-capability-matrix.md](GV-08-feature-gate-capability-matrix.md)
- [GV-09-archive-and-filesystem-adapter-seam.md](GV-09-archive-and-filesystem-adapter-seam.md)
- [GV-10-asset-catalog-and-virtual-path-index.md](GV-10-asset-catalog-and-virtual-path-index.md)
- [GV-11-import-job-pipeline.md](GV-11-import-job-pipeline.md)
- [GV-12-export-job-pipeline.md](GV-12-export-job-pipeline.md)
- [GV-13-cross-root-clipboard-package.md](GV-13-cross-root-clipboard-package.md)

### Runtime And Rendering

- [GV-14-render-layer-contracts.md](GV-14-render-layer-contracts.md)
- [GV-15-terrain-and-liquid-render-packets.md](GV-15-terrain-and-liquid-render-packets.md)
- [GV-16-object-model-render-packets.md](GV-16-object-model-render-packets.md)
- [GV-17-backend-bridge-vulkan-opengl.md](GV-17-backend-bridge-vulkan-opengl.md)
- [GV-18-world-session-profile-routing.md](GV-18-world-session-profile-routing.md)

### Editor And Metadata

- [GV-19-dbc-db2-schema-catalog.md](GV-19-dbc-db2-schema-catalog.md)
- [GV-20-dbc-db2-grid-editor.md](GV-20-dbc-db2-grid-editor.md)
- [GV-21-game-manager-workspace-shell.md](GV-21-game-manager-workspace-shell.md)
- [GV-22-selection-and-copy-paste-semantics.md](GV-22-selection-and-copy-paste-semantics.md)

### Future Content And Proof

- [GV-23-generated-content-package-contract.md](GV-23-generated-content-package-contract.md)
- [GV-24-diagnostics-and-proof-matrix.md](GV-24-diagnostics-and-proof-matrix.md)

## Suggested Early Order

1. GV-00, GV-00A, GV-01
2. GV-02 through GV-06
3. GV-07 through GV-10
4. GV-14 through GV-18
5. GV-19 through GV-22
6. GV-11 through GV-13
7. GV-23 through GV-24

## Design Rule

If a future task feels too big, split it into another micro-plan in this folder rather than widening the current one.

## Foundation Rule

The foundation plans must stay engine-neutral.

WoW, Warcraft 3, and future custom/forward-native content profiles are adapters layered on top of universal engine contracts. They are not allowed to define the shape of the engine core.
