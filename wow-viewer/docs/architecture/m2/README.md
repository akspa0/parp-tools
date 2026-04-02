# M2 Documentation Set

## Purpose

This folder is the canonical M2 documentation surface for implementation work in:

- `wow-viewer/src/core/WowViewer.Core/M2`
- `wow-viewer/src/core/WowViewer.Core.IO/M2`
- `wow-viewer/src/core/WowViewer.Core.Runtime/M2`
- compatibility-only `gillijimproject_refactor/src/MdxViewer/*` consumers

The goal is to stop scattering M2 implementation guidance across one-off native notes, session packets, pre-release contracts, and prompt files.

## Read Order

1. `implementation-contract.md`
   - the current implementation-facing contract for parser, skin, section, animation, effect, lighting, and runtime ownership
2. `native-build-matrix.md`
   - what is actually confirmed per build, and at what proof level
3. `consumer-cutover.md`
   - how `wow-viewer` should become the design owner while `MdxViewer` remains a compatibility consumer

## What Stays Outside This Folder

These remain raw evidence or historical workflow inputs. They are still useful, but they are no longer the first documents someone should read before implementing M2 work.

- `../m2-native-client-research-2026-03-31.md`
  - canonical raw native evidence log
- `../ab-session-a-2026-04-01/*`
  - session-scoped runtime packet for the first Win32 Wrath pass
- `../../../gillijimproject_refactor/plans/wow_viewer_m2_runtime_plan_2026-03-31.md`
  - staged migration plan and prompt-routing history
- `../../../gillijimproject_refactor/documentation/pre-release-3.0.1-m2-wow-exe-guide.md`
  - pre-release `3.0.1.8303` loader and validator guide
- `../../../gillijimproject_refactor/documentation/wow-200-beta-m2-light-particle-terrain-guide.md`
  - beta `2.0.0` guide for early M2, BLS, light, particle, and terrain seams
- `../../../gillijimproject_refactor/specifications/3.0.1.8303/Contracts/M2_MDX_Implementation_Contract_3.0.1.8303.md`
  - implementation contract for the pre-release `3.0.1` parser track
- `../../../gillijimproject_refactor/reference_data/wowdev.wiki/M2.md`
  - reference structure docs used for size and field correlation

## Rules

- `wow-viewer` is the canonical implementation target for new M2 ownership.
- `MdxViewer` is a compatibility consumer or proof harness, not the design owner.
- Keep proof levels explicit: `static-only`, `runtime-only`, `static + runtime`, or `research`.
- Do not infer later-build behavior from Wrath unless the build matrix says the evidence is direct.
- Keep unresolved semantics visible instead of flattening them away.
