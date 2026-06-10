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
  - first-party M2 ownership is now landed through slices 01-03, has a first slice-04 submission/handler contract, and has a first slice-05 app/golden consumer in `wow-viewer` (`Core/M2`, `Core.IO/M2`, `Core.Runtime/M2`, `WowViewer.App m2-frame`, richer `m2 inspect`, and M2 tests)
  - the main architectural gap is now visual/runtime rendering rather than parser/runtime ownership: particle/ribbon parser and simulation, GPU submission, active shader backend consumption, and broader parity signoff still remain

## Apr 15, 2026 Wolf static-geometry correction

- the latest real-data failure on the active Wrath baseline `H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft`, `Creature/Wolf/Wolf.m2` was not primarily a skinning bug; the same corruption was visible in the new inspect-side static visual proof
- root cause was in static mesh assembly:
  - `M2StaticRenderModelBuilder.TryGetVertex` was applying strict skin header field `0x2C` as a blind vertex base offset before trying the direct lookup-table entry
  - on `Wolf00.skin` that field is `53`, which matches wowdev-documented `boneCountMax` values and is not a plausible LOD0 vertex base for a `557`-vertex local lookup table
- landed correction:
  - direct skin lookup now wins by default; the extra header field is only used as a fallback when the direct lookup is invalid
  - optional shadow-batch metadata is now ignored unless the advertised span is actually valid, which removes bogus values like the earlier `shadowBatches=393221`
- updated proof floor:
	- fixed local client proof on the 3.3.5 root for `Creature/Wolf/Wolf.m2`, sequence `0`, time `0`, now emits recognizable static and skinned quadruped silhouettes in `output/build-validation/wow-viewer-m2-wolf-idle-static-visual-fixed.bmp` and `output/build-validation/wow-viewer-m2-wolf-idle-skinned-visual-fixed.bmp`
  - render-frame hash for that corrected idle proof is `86048f9de460bb5e75a557d526609700f4292b61ccc0f8eae4b4bd6206f012bb`
  - software visual hash for that corrected idle proof is `71aff63b3d0fba7e1eba03bcad894f2af0f2c87448fc9d706a976506b9f17ee5`
  - the same corrected Wolf proof also reproduced on the available `4.0.0.11927` root, but that should be treated as a cross-build spot check rather than the main task baseline

## Apr 15, 2026 shared frame-pipeline and visual-proof update

- additional `wow-viewer` M2 consumer seams are now landed:
  - `M2RuntimeFramePipeline` now owns the reusable end-to-end frame build from model plus active render state through submission plan, render frame, software visual snapshot, and golden frame
  - `WowViewer.App m2-frame` now consumes that shared runtime pipeline rather than keeping a private orchestration path
  - `WowViewer.Tool.Inspect m2 inspect` now consumes the same pipeline and can emit `--render-frame-output` plus `--visual-output` as well as `--golden-output`
  - the first software visual regression surface is now shared between app and inspect, not app-only
- validation:
  - focused M2 tests passed with `31` matching core tests
  - full `wow-viewer` build passed with existing invalid `LIB` path warnings only
  - fixed local client proof on `H:/CLIENTS/World of Warcraft Cata beta 11927`, `Creature/Wolf/Wolf.m2`, sequence `20`, time `500`, produced matching app and inspect hashes for runtime golden state, render frame, and software visual output
- updated status:
  - slice 05 now has a stronger consumer cutover surface because app and inspect share one runtime-frame pipeline and one deterministic visual-proof harness
  - this is still not final GPU renderer or screenshot parity; it is stronger runtime consumer closure and regression evidence only

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

## Apr 15, 2026 follow-up implementation update

- additional `wow-viewer` runtime seams are now landed:
  - typed `MD20` bone-definition parsing
  - shared runtime track sampling, including compressed M2 quaternion values
  - sequence/time bone-pose evaluation
  - CPU-side skinned render vertex application over structured sections and skin bone lookup metadata
  - render-consumer frame state that applies evaluated material/light state into renderer-facing pass state
  - first explicit M2 scene-submission coordinator with family, state, capacity, doodad-batching, particle-batching, and additive-sort policy knobs
  - `WowViewer.Tool.Inspect m2 inspect --time-ms` now prints pose, skinned-vertex count, render-consumer state, and scene-submission plan summaries
- validation:
  - focused M2 tests passed `24/24`
  - full `wow-viewer` build passed
  - full `wow-viewer` tests passed with `260` core tests and `36` PM4 tests
  - real fixed-client proof on `H:/CLIENTS/World of Warcraft Cata beta 11927`, `Creature/Wolf/Wolf.m2`, sequence `20`, exact `Wolf0096-00.anim`
- updated status:
  - slice 03 is now materially beyond inspect-only evaluator ownership, but final app renderer consumption and shader parity remain open
  - slice 04 has a first coordinator contract, but particle/ribbon concrete family handlers and active app submission remain follow-on work
  - slice 05 still needs a stronger consumer than inspect before claiming app/runtime cutover

## Apr 15, 2026 app consumer and golden-frame implementation update

- additional `wow-viewer` M2 runtime and consumer seams are now landed:
  - `M2EffectRegistry` / `M2ResolvedEffect` exposes native-style effect object keys, native combiner family keys, depth-write, alpha-test, additive, lighting, two-sided, projected, heuristic, and state-bucket decisions
  - `M2SceneSubmissionEntryBuilder` now owns shared render-entry construction from render-consumer frame state
  - particle/ribbon submission descriptors and family policies now make handler choice explicit (`particle-dispatch`, `ribbon-direct`, `core-batch`, `projected-batch`, etc.)
  - `M2RuntimeGoldenFrameBuilder` emits deterministic runtime snapshots and hashes
  - `WowViewer.App m2-frame` now consumes the first-party M2 runtime frame and can write a golden JSON snapshot
  - `WowViewer.Tool.Inspect m2 inspect --golden-output` writes the same golden-frame shape and prints resolved effect/submission handler proof lines
- validation:
  - focused M2 test filter passed with `27` matching core tests
  - `WowViewer.App` build passed
  - fixed local client proof on `H:/CLIENTS/World of Warcraft Cata beta 11927`, `Creature/Wolf/Wolf.m2`, sequence `20`, time `500`, produced matching app and inspect golden hashes: `113f55daaad3e996476eeff4c9e6fe37aa4c4d3cc364a48e38c6a86bc6fb980e`
  - full `wow-viewer` build passed with existing invalid `LIB` path warnings only
  - full `wow-viewer` tests passed with `263` core tests and `36` PM4 tests
- updated status:
  - slice 03 has app-consumable effect/material/lighting state
  - slice 04 has first particle/ribbon handler-policy contracts, but not final particle/ribbon parser, simulation, geometry, or GPU draw behavior
  - slice 05 now has a real `WowViewer.App` consumer and deterministic golden harness, but still no active visual renderer signoff

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
- slice 03 (animation, lighting, and effect runtime): landed through external animation ownership, animated block parsing, pose/skinning, render-consumer state, and resolved effect-object state; residual shader backend parity remains open
- slice 04 (scene submission and batching): first coordinator plus particle/ribbon handler-policy contracts landed; final particle/ribbon parser/simulation/GPU behavior remains open
- slice 05 (consumer cutover and parity harness): first `WowViewer.App m2-frame` consumer and golden snapshot harness landed; active visual renderer parity remains open

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
