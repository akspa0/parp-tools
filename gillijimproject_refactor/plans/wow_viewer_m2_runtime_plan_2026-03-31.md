# wow-viewer M2 Runtime And Renderer Plan

## Canonical Doc Surface

- The implementation-facing M2 doc set now lives under `wow-viewer/docs/architecture/m2/`.
- Use that folder as the first read for active implementation work.
- Keep this plan as staged migration history and prompt-routing context, not as the only day-to-day implementation handoff.

## Status

- status: active
- intent: move M2 parser, skin-profile ownership, section classification, material or effect routing, lighting state, and scene submission design into `wow-viewer`
- current proof floor:
  - native 3.3.5 OS X and 3.3.5 PTR OS X PowerPC behavior-recovery notes now live in `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
  - the active `MdxViewer` path has enough real regressions isolated to tell us what should not remain the long-term design owner
  - first-party M2 ownership is now landed through slice 02 and part of slice 03 in `wow-viewer` (`Core/M2`, `Core.IO/M2`, `Core.Runtime/M2`, richer `m2 inspect`, and foundation tests)
  - the main architectural gap is now after those landed seams: animated bone/skinning application, render-consumer use of evaluated material/light state, scene submission/batching, and consumer cutover beyond inspect

## Apr 15, 2026 implementation update

- landed in `wow-viewer` since the earlier reset state:
  - strict `MD20` root parse plus exact `%02d.skin` choose/load/init contracts
  - first-party geometry/material tables and structured section/pass/material routing
  - effect-recipe classification owned by `WowViewer.Core.Runtime/M2`
  - external `%04d-%02d.anim` selection/load and alias ready-state ownership
  - first-party animated block parsing for colors, texture weights, texture transforms, and lights
  - first-pass animated runtime evaluation over root or external payloads
  - `WowViewer.Tool.Inspect m2 inspect --time-ms` proof surface for evaluated animated runtime state
- real proof floor now includes:
  - `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter M2FoundationTests` passing `19/19`
  - real asset probe on fixed local client root `H:/CLIENTS/World of Warcraft Cata beta 11927` for `Creature/Wolf/Wolf.m2`, sequence `20`, with external `Wolf0096-00.anim` loading and `ANIM.RUNTIME` printed from the first-party evaluator
- remaining work for the next continuation is no longer “get parser ownership started”; it is:
  - animated bone pose solve and skinned-vertex application
  - render-consumer application of evaluated material/light state instead of inspect-only ownership
  - remaining model-local lighting/emissive semantics in the real render path
  - family-specific runtime ownership, scene submission/batching, and consumer cutover beyond inspect

## Apr 15, 2026 reset

- the user explicitly rejected more `MdxViewer`-owned bandaid work after two separate signals converged:
  - remaining player-model texturing/parity failures such as impossible back-facing facial/texturing artifacts
  - the broader training/tooling drift that still allowed live CPU training because the active `.venv` had a CPU-only torch build
- active corrective direction is now stricter:
  - keep `MdxViewer` as a compatibility proof source only when a bounded old-repo check is actually needed
  - treat the main implementation target as full first-party M2 parser plus runtime plus renderer ownership in `wow-viewer`
  - use wowdev docs, native-client research, and `noggit-red` as reference inputs for behavior recovery, not as reasons to keep the mixed current design
- concrete immediate gap after the landed slice-01 foundation is now clearer than when this plan was first written:
  - `WowViewer.Core.IO/M2/M2GeometryReader.cs` still depended on `Warcraft.NET` before the current parser-ownership recovery pass
  - active skin/material projection was still too thin and could flatten a section to its first batch, which is exactly the sort of MDX-shaped simplification the user wants removed
  - inspect tooling was still too summary-only to act as a real first-party M2 debugging surface
- a new workflow asset now exists for this reset:
  - `.github/prompts/wow-viewer-full-m2-parser-renderer-plan.prompt.md`
  - use it when the ask is broader than one staged slice and is really about replacing the mixed M2 ownership model itself

## Apr 15, 2026 Status Snapshot

- slice 01 (`MD20` and skin runtime foundation): landed
- slice 02 (section classification and material routing): landed first pass, with residual flag/bone-palette fidelity still possible as follow-on work
- slice 03 (animation, lighting, and effect runtime): partially landed through external animation ownership, animated block parsing, and first-pass evaluator state; render-consumer application and animated bone solve remain open
- slice 04 (scene submission and batching): open
- slice 05 (consumer cutover and parity harness): partially landed as an inspect consumer only; broader app/bridge/parity work remains open

## Why This Plan Exists

- the user wants a proper path to fix M2 implementation and rendering instead of continuing one-off `MdxViewer` repairs forever
- the native client evidence is now strong enough to drive a staged library-first M2 runtime plan
- the current codebase still spreads M2 ownership across:
  - `gillijimproject_refactor/src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`
  - `gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs`
  - `gillijimproject_refactor/src/MdxViewer/Rendering/MdxAnimator.cs`
  - `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`
  - `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs`
- without a staged prompt surface, fresh chats will keep mixing together parser ownership, skin-state recovery, material routing, lighting, batching, and active-viewer compatibility work

## Immediate Live Blocker

- the active `MdxViewer` path still misses a large adapted-M2 world set on the development map, especially giant root structures
- current runtime evidence says placement and bounds are often present while shaded triangle output is still wrong or missing
- those symptoms are important compatibility probes, but they should not keep dictating the long-term design surface
- the honest corrective direction is:
  - keep using `MdxViewer` as a proof source and compatibility host when needed
  - treat the actual fix path as staged M2 runtime ownership in `wow-viewer`

## Ordered Slices

### Slice 01 - MD20 And Skin Runtime Foundation

- status update:
	- landed in `wow-viewer` as a library-first slice with `WowViewer.Core/M2`, `WowViewer.Core.IO/M2`, `WowViewer.Core.Runtime/M2`, `WowViewer.Core.Tests/M2FoundationTests`, and thin `WowViewer.Tool.Inspect` `m2 inspect` wiring
	- validated with `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` and `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
	- current proof is build/test plus inspect ownership only; no real extracted asset signoff is claimed here

- target problem:
  - this seam existed as a gap when the plan was written; the first version is landed, but residual foundation ownership still matters when root-payload tables or exact skin/runtime contracts are not yet fully first-party
  - current residual examples include root payload geometry/material-table ownership and richer inspect/export surfaces for real M2 debugging
- likely destination:
  - `wow-viewer/src/core/WowViewer.Core/M2/*`
  - `wow-viewer/src/core/WowViewer.Core.IO/M2/*`
  - `wow-viewer/src/core/WowViewer.Core.Runtime/M2/*`
- proof goal:
  - next stronger proof should be one real asset opened through the shared wow-viewer-owned M2 seam that yields typed model metadata, exact numbered skin selection, and an active skin-profile result without relying on `WarcraftNetM2Adapter` as the design owner

### Slice 02 - Section Classification And Material Routing

- status update:
  - a first pass of this slice is now landed in `wow-viewer` with structured section/pass/material routing and effect-recipe classification
  - real proof exists through `M2FoundationTests` and real-asset `m2 inspect` output
  - residual work in this slice is now narrow: unresolved flag fidelity, bone-palette/influence details, and any remaining section/batch ownership mismatch that the next chat can prove concretely

- target problem:
  - the native client treats `.skin` initialization as structural render-state work, but the current runtime still tends to flatten sections/batches too early
  - unresolved native flags like `0x20` and propagated `0x40` need to remain visible instead of being erased by generic geoset assumptions
  - current practical risk is not hypothetical: player-model texturing failures and wrong layered section behavior are consistent with a still-too-thin batch/material contract
- likely destination:
  - `wow-viewer/src/core/WowViewer.Core/M2/*`
  - `wow-viewer/src/core/WowViewer.Core.Runtime/M2/*`
- proof goal:
  - wow-viewer owns a typed active-section contract with bone-palette/influence coverage, preserved unresolved flags, and explicit material/effect-routing metadata for real assets

### Slice 03 - Animation, Lighting, And Effect Runtime

- status update:
	- first-pass external animation selection/load, alias readiness, animated block parsing, and animated material/light state evaluation are now landed in `wow-viewer`
	- current proof is library/test coverage plus real `Wolf.m2` inspect output that loads `Wolf0096-00.anim` and prints evaluated `ANIM.RUNTIME`
	- the remaining gap is not “start owning external animations”; it is finishing runtime application and animated bone-driven behavior in real consumers

- target problem:
  - animated bone pose solve and skinned-vertex application are still not owned end to end in `wow-viewer`
  - evaluated material or texture or light state is still primarily an inspect/library seam rather than a real renderer-consumed runtime seam
  - model-local diffuse/emissive/lighting behavior still needs render-path ownership, not just typed evaluation output
- likely destination:
  - `wow-viewer/src/core/WowViewer.Core/M2/*`
  - `wow-viewer/src/core/WowViewer.Core.IO/M2/*`
  - `wow-viewer/src/core/WowViewer.Core.Runtime/M2/*`
- proof goal:
  - wow-viewer can load external animation state, expose ready-state/alias metadata, and evaluate a typed model-runtime lighting/effect state without hiding those decisions inside renderer globals

### Slice 04 - Scene Submission And Batching

- target problem:
  - native M2 rendering uses classified scene submission, family-specific handlers, state-aware batching, and explicit runtime knobs such as clip planes, z-fill, additive particle sorting, and doodad/particle batching
- likely destination:
  - `wow-viewer/src/core/WowViewer.Core.Runtime/M2/*`
  - optional integration touch points in `wow-viewer/src/core/WowViewer.Core.Runtime/World/*` if a narrow coordinator seam is needed
- proof goal:
  - wow-viewer owns an explicit M2 render-entry family model and a narrow submission/batching coordinator instead of burying all M2 draws in one generic renderer path

### Slice 05 - Consumer Cutover And Parity Harness

- target problem:
  - the extracted M2 seams now have an inspect consumer, but they still need a stronger app/bridge consumer and a realistic parity harness over fixed real assets
- likely destination:
  - `wow-viewer/src/viewer/WowViewer.App/*` when that consumer becomes active
  - narrow compatibility-only hooks in `gillijimproject_refactor/src/MdxViewer/*` only when needed to prove reuse of the extracted wow-viewer seam
  - `WowViewer.Tool.Inspect` if an M2 diagnostic/inspect verb is the right first consumer before app cutover
- proof goal:
  - a consumer beyond the current inspect-only path exercises the extracted wow-viewer M2 seam directly, with fixed real-asset validation and without claiming full production runtime parity

## Prompt Surface

- root router:
  - `.github/prompts/wow-viewer-m2-runtime-plan-set.prompt.md`
- full-cutover route:
  - `.github/prompts/wow-viewer-full-m2-parser-renderer-plan.prompt.md`
- ordered prompts:
  - `.github/prompts/wow-viewer-m2-runtime/01-md20-and-skin-runtime-foundation.prompt.md`
  - `.github/prompts/wow-viewer-m2-runtime/02-section-classification-and-material-routing.prompt.md`
  - `.github/prompts/wow-viewer-m2-runtime/03-animation-lighting-and-effect-runtime.prompt.md`
  - `.github/prompts/wow-viewer-m2-runtime/04-scene-submission-and-batching.prompt.md`
  - `.github/prompts/wow-viewer-m2-runtime/05-consumer-cutover-and-parity-harness.prompt.md`
- codex mirrors:
  - `.codex/prompts/wow-viewer-m2-runtime-plan-set.md`
  - `.codex/prompts/wow-viewer-m2-runtime/01-md20-and-skin-runtime-foundation.md`
  - `.codex/prompts/wow-viewer-m2-runtime/02-section-classification-and-material-routing.md`
  - `.codex/prompts/wow-viewer-m2-runtime/03-animation-lighting-and-effect-runtime.md`
  - `.codex/prompts/wow-viewer-m2-runtime/04-scene-submission-and-batching.md`
  - `.codex/prompts/wow-viewer-m2-runtime/05-consumer-cutover-and-parity-harness.md`

## Validation Rules

- default wow-viewer proof stays `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` and `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
- use real fixed assets whenever an M2 slice claims more than contract-only work
- use `MdxViewer` build/runtime only when a slice intentionally changes compatibility or needs an active-viewer proof harness
- do not describe wow-viewer build/test success as active-viewer runtime signoff

## Explicit Non-Claims

- this plan does not claim the current `MdxViewer` M2 regressions are already solved
- this plan does not claim the first M2 slice should be a full renderer rewrite
- this plan does not claim all native M2 flag semantics are closed today
- this plan does not claim the future wow-viewer M2 runtime must reuse every `MdxViewer` adapter choice verbatim