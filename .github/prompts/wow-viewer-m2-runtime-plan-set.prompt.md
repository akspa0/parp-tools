---
description: "Route staged wow-viewer M2 runtime and renderer recovery to the right ordered prompt. Use when planning or implementing M2 parser ownership, exact %02d.skin handling, active section classification, material/effect routing, animation/lighting state, scene batching, or consumer cutover."
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

## Ordered Prompts

- `wow-viewer-m2-runtime/01-md20-and-skin-runtime-foundation.prompt.md`
- `wow-viewer-m2-runtime/02-section-classification-and-material-routing.prompt.md`
- `wow-viewer-m2-runtime/03-animation-lighting-and-effect-runtime.prompt.md`
- `wow-viewer-m2-runtime/04-scene-submission-and-batching.prompt.md`
- `wow-viewer-m2-runtime/05-consumer-cutover-and-parity-harness.prompt.md`

## Routing Rules

- Use `01-md20-and-skin-runtime-foundation.prompt.md` when the problem is canonical `.m2` identity, strict `MD20` validation, exact numbered `%02d.skin` ownership, skin-profile choose/load/init behavior, or the first wow-viewer-owned M2 runtime contract.
- Use `02-section-classification-and-material-routing.prompt.md` when the problem is active renderable section ownership, bone-palette/influence remap, unresolved native flags such as `0x20` or `0x40`, material/layer routing, or effect-recipe classification.
- Use `03-animation-lighting-and-effect-runtime.prompt.md` when the problem is external `%04d-%02d.anim` loading, alias chains, animated material/texture state, model-local diffuse/emissive evaluation, or explicit combiner/effect runtime state.
- Use `04-scene-submission-and-batching.prompt.md` when the problem is render-entry family classification, doodad/particle/ribbon submission differences, batching rules, state-sort behavior, clip-plane/z-fill/additive-sort policy, or a narrow M2 runtime coordinator.
- Use `05-consumer-cutover-and-parity-harness.prompt.md` when the problem is making a real consumer exercise the extracted wow-viewer M2 seam, adding an inspect/diagnostic harness, or wiring a narrow compatibility-only `MdxViewer` bridge after the earlier slices are already real.

## Deliverables

Return all items:

1. the best next prompt to run
2. why it is the correct slice now
3. which ordered prompt should follow after it
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