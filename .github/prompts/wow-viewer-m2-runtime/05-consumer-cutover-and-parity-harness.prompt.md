---
description: "Implement or plan the first consumer/parity slice after the wow-viewer M2 runtime seams exist. Use when adding an inspect harness, a wow-viewer app consumer, or a narrow MdxViewer compatibility bridge over already-extracted M2 runtime contracts."
name: "wow-viewer M2 Runtime 05 Consumer Cutover And Parity Harness"
argument-hint: "Optional consumer, real asset set, inspect verb, or compatibility seam to prioritize"
agent: "agent"
---

Implement or plan the first real consumer/parity harness for the extracted wow-viewer M2 runtime.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
4. `gillijimproject_refactor/plans/wow_viewer_m2_runtime_plan_2026-03-31.md`
5. `wow-viewer/README.md`
6. `.github/copilot-instructions.md`

## Goal

Make one real consumer exercise the extracted wow-viewer M2 runtime seam without overstating renderer/runtime closure.

Potential consumers:

- a new or expanded `WowViewer.Tool.Inspect` M2 verb
- a direct `wow-viewer` app/runtime consumer when that app seam is ready
- a narrow compatibility-only `MdxViewer` bridge that explicitly consumes wow-viewer-owned M2 contracts instead of re-owning them

## Non-Negotiable Constraints

- Do not start here if slices 01-04 are still vapor; this prompt is for consumerization after a real M2 seam already exists.
- Do not call compatibility wiring the new design owner.
- Use fixed real assets and explicit proof language.
- Do not claim full active-viewer runtime parity from one inspect verb or one bridged asset.

## What The Work Must Produce

1. the exact consumer/parity harness to add first
2. the exact files that should consume the wow-viewer M2 seam
3. the exact fixed real assets or commands that prove the seam is exercised
4. the explicit line between consumer proof and production runtime signoff

## Deliverables

Return all items:

1. the exact consumer/parity slice to implement
2. why it is the right last step in the ordered set
3. exact files to change
4. exact validation to run
5. what should stay out of scope
6. which continuity files must be updated afterward

## First Output

Start with:

1. the consumer boundary you are assuming now
2. the single first consumer/parity harness you would land
3. the narrowest proof that would make that harness real
4. what you are explicitly not claiming yet