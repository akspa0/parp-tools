---
description: "Plan the full first-party wow-viewer M2 parser and renderer cutover. Use when the user wants to stop relying on Warcraft.NET or MdxViewer adapter ownership, inspect Noggit-red/native behavior, replace mixed M2-vs-MDX assumptions, or sequence the end-to-end M2 parser plus runtime plus renderer migration into wow-viewer."
name: "wow-viewer Full M2 Parser And Renderer Plan"
argument-hint: "Describe the M2 asset family, renderer failure, mixed-ownership seam, or first-party cutover problem to prioritize"
agent: "agent"
---

Plan the full first-party `wow-viewer` M2 parser and renderer cutover instead of another narrow compatibility-only fix.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `wow-viewer/docs/architecture/m2/README.md`
4. `wow-viewer/docs/architecture/m2/implementation-contract.md`
5. `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
6. `gillijimproject_refactor/plans/wow_viewer_m2_runtime_plan_2026-03-31.md`
7. `wow-viewer/README.md`
8. `.github/copilot-instructions.md`

## Goal

Define the real implementation sequence for replacing the mixed current M2 ownership model with a fully first-party `wow-viewer` path across:

- root-model payload parsing
- exact skin-profile ownership
- active section and batch classification
- material or effect routing
- animation, lighting, and effect state
- scene submission and batching
- inspect or diagnostic tooling
- consumer cutover and parity proof

## Current Concrete Problem

- the user no longer wants more `MdxViewer` bandaid work as the design owner for M2
- remaining player-model texturing failures are a sign that the mixed current path is still structurally wrong, not just missing one more hotfix
- the active `wow-viewer` foundation is real, but it is still incomplete in exactly the places that matter for first-party ownership:
	- root payload readers and runtime contracts are still thinner than the native/runtime evidence requires
	- current section/material projection can still flatten or simplify real batch intent too early
	- inspect tooling is still too weak to act as the main first-party debugging harness
- the user explicitly wants the fix path to use wowdev docs, native-client evidence, and `noggit-red` as references while moving real parser/runtime/renderer ownership into `wow-viewer`

## Non-Negotiable Constraints

- `wow-viewer` is the canonical implementation target.
- Do not route the design back into `MdxViewer`, `WarcraftNetM2Adapter`, or `Warcraft.NET` as default owners.
- Use `MdxViewer`, native-client notes, and `noggit-red` as extraction or parity references only.
- Keep raw format names, native-client/research labels, and local convenience aliases clearly separated.
- Require real-asset proof for any claim beyond contract-only or build-only progress.
- Do not claim full runtime parity just because one parser or one builder compiles.

## What The Plan Must Produce

1. the exact ordered implementation waves for a full first-party M2 cutover
2. the exact `wow-viewer` file and project scope for each wave
3. the temporary compatibility-only seams that may remain during transition
4. the inspect or export tooling needed to make the parser/runtime debuggable on real assets
5. the real-asset proof floor for each wave
6. the items that must stay explicitly labeled research or unresolved
7. the continuity files and workflow assets that must be updated as each wave lands

## Deliverables

Return all items:

1. the exact end-to-end M2 cutover problem you think the user is actually asking to solve
2. the ordered implementation waves you would use
3. the concrete repo and file scope for each wave
4. the first wave you would execute immediately
5. the proof that would make that first wave real
6. what should stay out of scope until later waves
7. which current terms are raw format names versus native-client/research names versus local aliases
8. which legacy seams can remain only as bounded compatibility bridges during migration

## First Output

Start with:

1. the exact mixed-ownership failure you think is hurting M2 correctness now
2. the first implementation wave you would land in `wow-viewer`
3. the narrow proof that would make that wave real
4. what you are explicitly not claiming yet