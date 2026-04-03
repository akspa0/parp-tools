---
description: "Route staged wow-viewer world-runtime and WorldScene split work to the right ordered Codex prompt. Use when planning or implementing negative asset lookup suppression, repeated .skin miss suppression, world visible-set extraction, terrain or WMO or MDX or overlay pass-service extraction, WorldScene host thinning, or early wow-viewer app consumer cutover."
name: "wow-viewer World Runtime Plan Set"
argument-hint: "Describe the WorldScene split seam, runtime service slice, or performance hotspot you want to attack next"
agent: "codex"
---

Choose the right detailed prompt for the staged `WorldScene` to `wow-viewer` world-runtime split and implement one narrow slice now unless the user explicitly asks for planning-only output.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
4. `gillijimproject_refactor/plans/wow_viewer_world_runtime_service_plan_2026-03-31.md`
5. `wow-viewer/README.md`
6. `AGENTS.md`

## Goal

Route the current request to the correct ordered prompt in `.codex/prompts/wow-viewer-world-runtime/`, then execute one narrow slice so the `WorldScene` split lands as code changes instead of another planning-only cycle.

## Mandatory Execution Rule

- Unless the user explicitly asks for planning-only output, implement one narrow slice in this chat after routing.
- Run applicable validation commands and report exact changed files.
- Do not rewrite prompts/instructions/plans unless explicitly requested.

## Ordered Prompts

- `wow-viewer-world-runtime/01-negative-asset-lookup-suppression.md`
- `wow-viewer-world-runtime/02-visible-set-runtime-extraction.md`
- `wow-viewer-world-runtime/03-world-pass-service-extraction.md`
- `wow-viewer-world-runtime/04-world-scene-host-thinning.md`
- `wow-viewer-world-runtime/05-wow-viewer-app-runtime-consumer.md`

## Routing Rules

- Use `01-negative-asset-lookup-suppression.md` when the problem is repeated `.skin` lookup churn, failed MDX retry spam, noisy asset logs, negative lookup caching, or unknown performance degradation from repeated asset misses.
- Use `02-visible-set-runtime-extraction.md` when the problem is visible WMO or MDX or taxi-actor collection, render-frame scratch ownership, culling buckets, or moving visible-set logic out of `WorldScene`.
- Use `03-world-pass-service-extraction.md` when the problem is explicit terrain or WMO or MDX or overlay runtime service ownership, pass sequencing, or a world-pass coordinator inside `WowViewer.Core.Runtime`.
- Use `04-world-scene-host-thinning.md` when the problem is reducing `WorldScene` to a thin host over runtime services without yet claiming a full app cutover.
- Use `05-wow-viewer-app-runtime-consumer.md` when the problem is making `wow-viewer/src/viewer/WowViewer.App` consume the extracted world-runtime seams after the earlier slices are already real.

## Deliverables

Return all items:

1. the best next prompt to run
2. why it is the correct slice now
3. which ordered prompt should follow after it
4. what concrete repo and file scope the next slice should include
5. what should stay out of scope for the next slice
6. what proof level is realistic for that slice
7. exact files changed in this chat for the implemented slice
8. exact validation commands run in this chat

## First Output

Start with:

1. the exact world-runtime problem you think the user is trying to solve
2. the single best next prompt from the ordered set
3. the narrow proof that would make that slice real
4. what you are explicitly not claiming yet