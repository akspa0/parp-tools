---
description: "Route staged wow-viewer M2 runtime and renderer recovery to the right ordered prompt. Use when planning or implementing residual section/material fidelity, animated bone/skinning work, animated material/light application, scene batching, consumer cutover, or deciding whether the request is really a broader full first-party M2 parser/renderer cutover plan."
name: "wow-viewer M2 Runtime Plan Set"
argument-hint: "Describe the M2 runtime seam, renderer problem, or migration slice you want to attack next"
agent: "agent"
---

Choose the right detailed prompt for the staged `wow-viewer` M2 runtime and renderer recovery path.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
4. `gillijimproject_refactor/plans/wow_viewer_m2_runtime_plan_2026-03-31.md`
5. `wow-viewer/README.md`
6. `.github/copilot-instructions.md`

## Goal

Route the current request to the correct ordered prompt in `.github/prompts/wow-viewer-m2-runtime/` so M2 ownership moves into `wow-viewer` as a sequence of narrow, validated slices instead of another run of mixed parser/renderer/viewer hotfixes.

## Current Validated Baseline

- slices 01 and 02 are already real in `wow-viewer`
- slice 03 is partially real already:
	- external `%04d-%02d.anim` choose/load and alias ready-state ownership exist
	- effect-recipe classification exists
	- first-party animated block parsing/evaluation for colors, texture weights, texture transforms, and lights exists
	- `WowViewer.Tool.Inspect m2 inspect --time-ms` can print evaluated animated runtime state on real assets
- the next chat should not restart from “build strict MD20 parsing” or “invent the first inspect surface” unless it is correcting a concrete regression in already-landed work

## Ordered Prompts

- `wow-viewer-m2-runtime/01-md20-and-skin-runtime-foundation.prompt.md`
- `wow-viewer-m2-runtime/02-section-classification-and-material-routing.prompt.md`
- `wow-viewer-m2-runtime/03-animation-lighting-and-effect-runtime.prompt.md`
- `wow-viewer-m2-runtime/04-scene-submission-and-batching.prompt.md`
- `wow-viewer-m2-runtime/05-consumer-cutover-and-parity-harness.prompt.md`

## Companion Prompt

- `wow-viewer-full-m2-parser-renderer-plan.prompt.md`
- `m2-cross-build-native-investigation.prompt.md`

## Routing Rules

- Use `wow-viewer-full-m2-parser-renderer-plan.prompt.md` first when the request is broader than one slice and is really about replacing the mixed M2 ownership model itself, especially when the user explicitly wants to stop relying on `Warcraft.NET`, `WarcraftNetM2Adapter`, `MdxViewer` renderer ownership, or MDX-shaped simplifications in the current M2 path.
- Use `01-md20-and-skin-runtime-foundation.prompt.md` only when the problem is correcting a concrete regression in already-landed strict root or exact skin runtime ownership, not as the default next slice.
- Use `02-section-classification-and-material-routing.prompt.md` when the remaining problem is genuinely residual section/material fidelity: active renderable section ownership gaps, bone-palette/influence remap, unresolved native flags such as `0x20` or `0x40`, or a still-proven section/batch/material-routing mismatch.
- Use `03-animation-lighting-and-effect-runtime.prompt.md` when the remaining problem is completing runtime application after the now-landed evaluator baseline: animated bone/skinning solve, applying evaluated material/light state into a render consumer, model-local diffuse/emissive semantics, or residual effect-runtime behavior beyond simple recipe classification.
- Use `04-scene-submission-and-batching.prompt.md` when the problem is render-entry family classification, doodad/particle/ribbon submission differences, batching rules, state-sort behavior, clip-plane/z-fill/additive-sort policy, or a narrow M2 runtime coordinator.
- Use `05-consumer-cutover-and-parity-harness.prompt.md` when the problem is moving beyond the already-real inspect consumer into an app consumer, a materially stronger parity harness, or a narrow compatibility-only `MdxViewer` bridge after the earlier slices are already real.
- Use `m2-cross-build-native-investigation.prompt.md` when the task is to recover and compare native M2 behavior across multiple client versions (for example 3.3.5 through 6.x) before committing implementation changes.

## Deliverables

Return all items:

1. the best next prompt to run
2. why it is the correct slice now
3. which ordered prompt or full-cutover prompt should follow after it
4. what concrete repo and file scope the next slice should include
5. what should stay out of scope for the next slice
6. what proof level is realistic for that slice
7. which M2 terms in the slice are native-client/research names versus raw format names

## First Output

Start with:

1. the exact M2 runtime or renderer problem you think the user is trying to solve
2. the single best next prompt from the ordered set
3. the narrow proof that would make that slice real
4. what you are explicitly not claiming yet