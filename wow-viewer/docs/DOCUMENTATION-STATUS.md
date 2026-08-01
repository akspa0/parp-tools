# Documentation Status

Updated: 2026-08-01

This is canonical doc map for `wow-viewer/`.

## Spec inventory (read first)

- **[docs/specs-audit-2026-08-01.md](/I:/parp/parp-tools/wow-viewer/docs/specs-audit-2026-08-01.md)** — the full
  spec inventory: DONE (archive candidates), ACTIVE (front-of-queue), and DRAFT (backlog).
  Use this before reading any individual spec.

## Read first

- [README.md](/I:/parp/parp-tools/wow-viewer/README.md)
- [AGENTS.md](/I:/parp/parp-tools/wow-viewer/AGENTS.md)
- [memory-bank/activeContext.md](/I:/parp/parp-tools/wow-viewer/memory-bank/activeContext.md)
- [memory-bank/progress.md](/I:/parp/parp-tools/wow-viewer/memory-bank/progress.md)
- [docs/architecture/wow-engine-modernization-plan-2026-05-14.md](/I:/parp/parp-tools/wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md)

## Current operator docs

- [docs/CLI-TOOLS.md](/I:/parp/parp-tools/wow-viewer/docs/CLI-TOOLS.md)
- [docs/WoWViewer/USERGUIDE.md](/I:/parp/parp-tools/wow-viewer/docs/WoWViewer/USERGUIDE.md)
- [data-harvester/README.md](/I:/parp/parp-tools/wow-viewer/data-harvester/README.md)
- [docs/runpod-integration-cookbook.md](/I:/parp/parp-tools/wow-viewer/docs/runpod-integration-cookbook.md)

## Current spec docs

- `specs/089-dav2-height-predictor/`
- `specs/088-v22-enrichment-from-v18/`
- `specs/080-wow-ui-consolidation/`
- `specs/047-v18-distill-corpus-open-source-loop/` when focused V18 operator work is reopened

## Background only

- `specs/076-full-map-fractal-brush-library/`
- `specs/077-minimap-deconstruction-engine/`
- `specs/079-runpod-integration-guide/`

## Historical only

- `specs/archived/`
- `specs/086-v22-consolidated-dataset/`
- `specs/087-v22-asset-library-payloads/`
- `plans/`
- `docs/audits/`
- `docs/validation/` older V9/V16-era notes
- `docs/MdxViewer-legacy-documentation.tar.gz`

## Repo-wide doc rules

- No `H:\CLIENTS`.
- No new ownership claims for `gillijimproject_refactor`.
- If a doc points to a missing file, fix link or remove claim in same pass.
- Do not keep stale active-spec counts. Use status categories instead.
- When code behavior changes, update matching spec, architecture doc, and memory-bank same pass.

## Audit results from this pass

- Rewrote `AGENTS.md` to current repo truth.
- Rewrote root `README.md`.
- Replaced stale `PLANS-OVERVIEW.md` active-spec counter with current lane map.
- Rewrote `data-harvester/README.md` around Spec 088 / 089 reality.
- Rewrote `docs/WoWViewer/README.md` and `USERGUIDE.md` to remove stale root paths and dead links.
