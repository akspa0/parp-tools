---
description: "Implement or plan the first wow-viewer M2 runtime foundation slice. Use when `.m2` identity, strict `MD20` validation, exact numbered `%02d.skin` ownership, or skin-profile choose/load/init behavior still live only in `MdxViewer`-side code or native notes."
name: "wow-viewer M2 Runtime 01 MD20 And Skin Runtime Foundation"
argument-hint: "Optional real asset, model family, skin-profile seam, or current reference file to prioritize"
agent: "agent"
---

Implement or plan the first shared M2-owned runtime seam in `wow-viewer`: canonical `.m2` identity plus exact numbered skin-profile ownership.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
4. `gillijimproject_refactor/plans/wow_viewer_m2_runtime_plan_2026-03-31.md`
5. `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs`
6. `gillijimproject_refactor/src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`
7. `wow-viewer/README.md`
8. `.github/copilot-instructions.md`

## Goal

Create the first wow-viewer-owned M2 parser/runtime seam for:

- canonical `.m2` identity
- strict `MD20` validation
- exact numbered `%02d.skin` ownership
- explicit choose/load/init skin-profile staging
- active skin-profile result contracts that later renderer slices can consume

## Current Concrete Problem

- native evidence says the client canonicalizes model-family paths to `.m2`
- skin ownership is exact numbered `%02d.skin`, not a fuzzy sidecar search as the design truth
- skin-profile load/init materially changes renderable state and rebuilds instances
- wow-viewer still has no M2-owned seam for any of that; the current practical ownership is trapped in compatibility code and ad hoc adapter paths

## Non-Negotiable Constraints

- Do not turn this slice into the full renderer rewrite.
- Do not let best-effort fallback skin search become the authoritative wow-viewer contract.
- Keep `MdxViewer` as a reference/compatibility input, not the design owner.
- Keep strict parse failures and proof boundaries explicit.
- Do not claim active-viewer runtime parity from a library parse or build pass.

## What The Work Must Produce

1. the exact wow-viewer contracts for model identity, skin selection, and active skin-profile staging
2. the exact files that should own parser, I/O, and runtime state in `wow-viewer`
3. the narrowest real-asset proof that the shared seam is real
4. any temporary compatibility-only fallback that must remain clearly labeled as non-authoritative
5. the follow-on hook for section/material ownership in slice 02

## Deliverables

Return all items:

1. the exact foundation seam to implement
2. why it is the first M2 slice
3. exact files to change
4. exact validation to run
5. what should stay out of scope
6. which continuity files must be updated afterward
7. which terms are native/research names versus raw file-format names

## First Output

Start with:

1. the current M2 boundary you are assuming now
2. the single first foundation seam you would land
3. the narrowest proof that would make that seam real
4. what you are explicitly not claiming yet