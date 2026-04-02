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
  - `wow-viewer` still has no first-class M2-owned library/runtime area yet, which is now the main architectural gap

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
  - this seam existed as a gap when the plan was written; it is now covered by the landed slice-01 library/runtime foundation
- likely destination:
  - `wow-viewer/src/core/WowViewer.Core/M2/*`
  - `wow-viewer/src/core/WowViewer.Core.IO/M2/*`
  - `wow-viewer/src/core/WowViewer.Core.Runtime/M2/*`
- proof goal:
  - next stronger proof should be one real asset opened through the shared wow-viewer-owned M2 seam that yields typed model metadata, exact numbered skin selection, and an active skin-profile result without relying on `WarcraftNetM2Adapter` as the design owner

### Slice 02 - Section Classification And Material Routing

- target problem:
  - the native client treats `.skin` initialization as structural render-state work, but the current runtime still tends to flatten sections/batches too early
  - unresolved native flags like `0x20` and propagated `0x40` need to remain visible instead of being erased by generic geoset assumptions
- likely destination:
  - `wow-viewer/src/core/WowViewer.Core/M2/*`
  - `wow-viewer/src/core/WowViewer.Core.Runtime/M2/*`
- proof goal:
  - wow-viewer owns a typed active-section contract with bone-palette/influence coverage, preserved unresolved flags, and explicit material/effect-routing metadata for real assets

### Slice 03 - Animation, Lighting, And Effect Runtime

- target problem:
  - external `%04d-%02d.anim` ownership, alias chains, animated material/texture state, and model-local diffuse/emissive evaluation are still not first-class runtime seams
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
  - extracted M2 seams still need a concrete consumer and a realistic parity harness over fixed real assets
- likely destination:
  - `wow-viewer/src/viewer/WowViewer.App/*` when that consumer becomes active
  - narrow compatibility-only hooks in `gillijimproject_refactor/src/MdxViewer/*` only when needed to prove reuse of the extracted wow-viewer seam
  - `WowViewer.Tool.Inspect` if an M2 diagnostic/inspect verb is the right first consumer before app cutover
- proof goal:
  - at least one consumer exercises the extracted wow-viewer M2 seam directly, with fixed real-asset validation and without claiming full production runtime parity

## Prompt Surface

- root router:
  - `.github/prompts/wow-viewer-m2-runtime-plan-set.prompt.md`
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