---
description: "Implement or plan the next narrow wow-viewer Core.PM4 slice using the library-first PM4 migration workflow. Use when extracting a PM4 reader, pm4 inspect, pm4 audit, pm4 audit-directory, pm4 linkage, pm4 mscn, pm4 unknowns, pm4 export-json, a contract, a solver seam, or a PM4 regression test without drifting into a broad viewer rewrite."
name: "wow-viewer PM4 Library Implementation"
argument-hint: "Optional PM4 seam, report family, contract, solver behavior, or test slice to prioritize"
agent: "agent"
---

Implement the next narrow `wow-viewer` PM4 library slice without losing the current source-of-truth and validation rules.

If the ask is actually a non-PM4 shared-format or shared-I/O slice, use `wow-viewer-shared-io-implementation.prompt.md` instead of this PM4-specific prompt.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/plans/wow_viewer_pm4_library_plan_2026-03-25.md`
4. `wow-viewer/README.md`
5. `.github/copilot-instructions.md`

## Goal

Move one more reusable PM4 capability into `wow-viewer/src/core/WowViewer.Core.PM4` with explicit validation and without overstating viewer-runtime proof. Typical slices include PM4 inspect or audit verbs, linkage or MSCN or unknowns reports, or narrow placement and solver extraction work.

## Current Working Rules

- `MdxViewer` is still the runtime PM4 reference implementation.
- `Pm4Research` is still the library seed and report backbone.
- The default direction is library-first `wow-viewer` work, not broader active-viewer integration.
- Shared consumer wiring into `MdxViewer` is allowed only when it is a narrow follow-up to an already extracted library seam or when the user explicitly asks for it.

## Active Library Surface To Build On

- PM4 chunk models and research document reader
- inspect verbs for `pm4 inspect`, `pm4 export-json`, `pm4 audit`, `pm4 audit-directory`, `pm4 linkage`, `pm4 mscn`, and `pm4 unknowns`
- shared placement contracts: `Pm4AxisConvention`, `Pm4CoordinateMode`, `Pm4PlanarTransform`, `Pm4PlacementContract`, and `Pm4PlacementSolution`
- shared placement math: range-based axis detection, normal-based axis scoring, planar-transform resolution, world-yaw correction, world-space centroid, pivot rotation, and corrected world-position conversion
- regression floor in `wow-viewer/tests/WowViewer.Core.PM4.Tests`

## Non-Negotiable Constraints

- Do not turn a narrow PM4 slice into a broad `MdxViewer` rewrite.
- Do not claim viewer runtime PM4 signoff from library builds, tests, or active-viewer compile success.
- Keep exploratory field semantics labeled as research.
- Prefer one concrete seam with proof over several half-validated abstractions.

## What The Work Must Produce

1. The exact seam to add or change.
2. The files that should own it in `wow-viewer`.
3. The validation required for that seam.
4. Any active-viewer touch point that must remain reference-only.
5. The continuity files that must be updated after the slice lands.

## Deliverables

Return all items:

1. the next PM4 library slice to implement
2. why that slice is the right next step
3. exact files to change
4. exact validation to run
5. what should stay out of scope for this slice
6. which memory or prompt surfaces must be updated afterward

## First Output

Start with:

1. the current PM4 library boundary you are assuming
2. the single next seam you would extract or add
3. the narrowest proof that would show the slice is real
4. what you are explicitly not claiming yet