# Plans Overview

This file is current-state map, not running spec counter.

## Active implementation lanes

### Spec 089 — DA-V2 height predictor

- Active model lane.
- Owner: `data-harvester/`.
- Depends on Spec 088 V22 store.
- Current proof owner: local 12 GB CUDA envelope, not remote Pod creation.

### Spec 088 — V22 enrichment from V18

- Active dataset contract.
- Owner: `tools/enrich/` + `data-harvester/`.
- Canonical shape: paths-only V22 store.
- Remaining bounded gate: rerun proof for `4_0_0_11927`.

### Spec 080 — WoW UI consolidation

- Active viewer-shell doc lane.
- Current landed slice is compatibility-first and source-only in legacy `MdxViewer`.
- Treat compile proof for that slice as still open because legacy solution has broader missing refs outside touched work.

## Active support lanes

### Spec 047 — focused V18 terrain operator lane

- Keep for curation/training operator work.
- Not primary R&D lane now.

### Spec 079 — RunPod integration guide

- Shared bundle/runtime pattern.
- Use when remote packaging or Pod bootstrapping is reopened.

## Paused but reusable

### Spec 076 — full-map fractal brush library

- Useful research outputs exist.
- Not current front-of-queue implementation lane.

### Spec 077 — minimap deconstruction engine

- Useful prior surfaces exist.
- Reopen only with explicit target and proof owner.

## Deprecated or superseded

### Deprecated model lanes

- Spec 074 — evidence/catalog only.
- Spec 075 — diagnostic baseline only.
- Spec 066 / 067 / 068 — historical terrain-model detours.

### Superseded dataset lanes

- Spec 086 and Spec 087 — superseded by Spec 088.
- Keep on disk only for redirect/evidence.

## Historical / archive rules

- `specs/archived/` = closed history.
- `plans/` = planning archive unless a live spec links it directly.
- Do not quote old active-spec counts from earlier reset docs. They are stale by design.

## Where to look first

- Repo rules: [AGENTS.md](/I:/parp/parp-tools/wow-viewer/AGENTS.md)
- Doc registry: [docs/DOCUMENTATION-STATUS.md](/I:/parp/parp-tools/wow-viewer/docs/DOCUMENTATION-STATUS.md)
- Current continuity: [memory-bank/activeContext.md](/I:/parp/parp-tools/wow-viewer/memory-bank/activeContext.md)
- Current progress: [memory-bank/progress.md](/I:/parp/parp-tools/wow-viewer/memory-bank/progress.md)
