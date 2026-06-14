# game-viewer Micro-Plan Pack

## Status

- status: active
- date: 2026-05-14
- working-label: `museum-explorer`
- parent: `wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md` (replaced 2026-06-14 — viewer-first, UE bridge)
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
- [GV-00B-base-engine-repo-extraction-boundary.md](GV-00B-base-engine-repo-extraction-boundary.md)
- [GV-00C-base-solution-and-folder-topology.md](GV-00C-base-solution-and-folder-topology.md)
- [GV-00D-dotnet-first-engine-and-orchestration-boundary.md](GV-00D-dotnet-first-engine-and-orchestration-boundary.md)
- [GV-01-core-constants-registry.md](GV-01-core-constants-registry.md)
- [GV-02-wow-alpha-constants-pack.md](GV-02-wow-alpha-constants-pack.md)
- [GV-03-wow-retail-constants-pack.md](GV-03-wow-retail-constants-pack.md)
- [GV-04-warcraft3-constants-pack.md](GV-04-warcraft3-constants-pack.md)
- [GV-04A-museums-forward-native-profile.md](GV-04A-museums-forward-native-profile.md)
- [GV-04B-museums-object-package-contract.md](GV-04B-museums-object-package-contract.md)
- [GV-04C-museums-shard-and-index-store.md](GV-04C-museums-shard-and-index-store.md)
- [GV-05-game-build-metadata-probe.md](GV-05-game-build-metadata-probe.md)
- [GV-06-compatibility-profile-registry.md](GV-06-compatibility-profile-registry.md)
- [GV-06A-profile-personality-library-contract.md](GV-06A-profile-personality-library-contract.md)

### Data Roots And Interop

- [GV-07-game-root-records-and-manager-service.md](GV-07-game-root-records-and-manager-service.md)
- [GV-08-feature-gate-capability-matrix.md](GV-08-feature-gate-capability-matrix.md)
- [GV-09-archive-and-filesystem-adapter-seam.md](GV-09-archive-and-filesystem-adapter-seam.md)
- [GV-09A-raw-artifact-capture-store.md](GV-09A-raw-artifact-capture-store.md)
- [GV-10-asset-catalog-and-virtual-path-index.md](GV-10-asset-catalog-and-virtual-path-index.md)
- [GV-10A-sidecar-metadata-schema-index.md](GV-10A-sidecar-metadata-schema-index.md)
- [GV-11-import-job-pipeline.md](GV-11-import-job-pipeline.md)
- [GV-11A-glb-import-adapter.md](GV-11A-glb-import-adapter.md)
- [GV-12-export-job-pipeline.md](GV-12-export-job-pipeline.md)
- [GV-12A-forward-native-export-package.md](GV-12A-forward-native-export-package.md)
- [GV-13-cross-root-clipboard-package.md](GV-13-cross-root-clipboard-package.md)

### Runtime, Rendering, And Audio

- [GV-14-render-layer-contracts.md](GV-14-render-layer-contracts.md)
- [GV-14A-audio-system-foundation.md](GV-14A-audio-system-foundation.md)
- [GV-14B-profile-audio-resolution-contracts.md](GV-14B-profile-audio-resolution-contracts.md)
- [GV-14C-runtime-audio-scene-and-mixer.md](GV-14C-runtime-audio-scene-and-mixer.md)
- [GV-14D-audio-asset-family-support-matrix.md](GV-14D-audio-asset-family-support-matrix.md)
- [GV-15-terrain-and-liquid-render-packets.md](GV-15-terrain-and-liquid-render-packets.md)
- [GV-16-object-model-render-packets.md](GV-16-object-model-render-packets.md)
- [GV-17-backend-bridge-vulkan-opengl.md](GV-17-backend-bridge-vulkan-opengl.md)
- [GV-17A-audio-backend-bridge.md](GV-17A-audio-backend-bridge.md)
- [GV-17B-midi-synth-and-instrument-bank-bridge.md](GV-17B-midi-synth-and-instrument-bank-bridge.md)
- [GV-17C-webgl-component-and-web-delivery-surface.md](GV-17C-webgl-component-and-web-delivery-surface.md)
- [GV-18-world-session-profile-routing.md](GV-18-world-session-profile-routing.md)

### Editor And Metadata

- [GV-19-dbc-db2-schema-catalog.md](GV-19-dbc-db2-schema-catalog.md)
- [GV-19A-profile-schema-source-routing.md](GV-19A-profile-schema-source-routing.md)
- [GV-20-dbc-db2-grid-editor.md](GV-20-dbc-db2-grid-editor.md)
- [GV-21-game-manager-workspace-shell.md](GV-21-game-manager-workspace-shell.md)
- [GV-22-selection-and-copy-paste-semantics.md](GV-22-selection-and-copy-paste-semantics.md)

### Future Content And Proof

- [GV-23-generated-content-package-contract.md](GV-23-generated-content-package-contract.md)
- [GV-23A-distilled-portable-model-packages.md](GV-23A-distilled-portable-model-packages.md)
- [GV-24-diagnostics-and-proof-matrix.md](GV-24-diagnostics-and-proof-matrix.md)
- [GV-25-small-model-story-template.md](GV-25-small-model-story-template.md)
- [GV-26-plan-pack-audit-matrix.md](GV-26-plan-pack-audit-matrix.md)

## Suggested Early Order

1. GV-00, GV-00A, GV-00B, GV-00C, GV-00D, GV-01
2. GV-02, GV-03, GV-04, GV-04A, GV-04B, GV-04C, GV-05, GV-06, GV-06A
3. GV-07, GV-08, GV-09, GV-09A, GV-10, GV-10A
4. GV-14, GV-14A, GV-14B, GV-14C, GV-14D, GV-15, GV-16, GV-17, GV-17A, GV-17B, GV-17C, GV-18
5. GV-19, GV-19A, GV-20, GV-21, GV-22
6. GV-11, GV-11A, GV-12, GV-12A, GV-13
7. GV-23, GV-23A, GV-24, GV-25, GV-26

## Design Rule

If a future task feels too big, split it into another micro-plan in this folder rather than widening the current one.

## Foundation Rule

The foundation plans must stay engine-neutral.

WoW, Warcraft 3, and future custom/forward-native content profiles are adapters layered on top of universal engine contracts. They are not allowed to define the shape of the engine core.

## Museums Rule

`Museums` is a named supported data type in this project.

It should be treated as:

- a first-class forward-native profile family
- an evolving specification
- a likely shard/index-backed ecosystem
- a future bridge between artifact preservation, generated metadata, and portable distilled models

## Small-Model Rule

Every micro-plan should eventually be auditable against `GV-25` and `GV-26`.

If a plan does not clearly state touched surfaces, stop conditions, and proof, it is not yet small-model safe.

## Base Engine Rule

The future extracted engine repo should have a first-class `BASE` layer or folder family.

`BASE` owns engine-neutral runtime, rendering, audio, content-service, provenance, and diagnostics contracts.

WoW, Warcraft 3, Museums, and future content families are profile/personality libraries layered on top of `BASE`. They are not allowed to redefine the engine core.
