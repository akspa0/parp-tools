# Implementation Plan: Legacy M2 rendering (0.11–2.4.3)

**Branch**: `104-legacy-m2-rendering` | **Updated**: 2026-07-15 | **Spec**: [spec.md](spec.md)

## Summary

Legacy client assets in this feature are M2-family files. In particular, WoW 1.0.0 uses
`MD20` with version `0x100`; `.mdx` and `.mdl` are compatibility aliases, not a replacement
asset format. The immediate slice makes the viewer classify that `0x100` layout, parse it with
the dedicated era-100 M2 reader, and either render it through the native M2 runtime or report a
specific M2-reader failure. It must never silently reinterpret it as another M2 layout or direct
the user to MDX.

The 1.0.0 Ghidra trace is complete. It establishes the classic header, `M2Division` embedded
geometry, vertex lookup, sections, batches, and texture records. This slice deliberately does
not claim that the distinct 1.12.1 `0x100` layout or any TBC layout is solved.

## Technical Context

**Language/Version**: C# / .NET 10.
**Owners**: `WowViewer.Core.IO` owns detection and parsing; `WoWViewer` builds an
`M2StaticRenderModel` and submits it to `M2Renderer`.
**Tests**: focused xUnit parser/dispatcher/adapter tests, `dotnet build`, then user-driven staged-client viewer proof.
**Ground truth**: only `output/tmp/wowarchive-clients/`; record the exact staged path.
**Constraints**: no parser rewrite, no changes in `gillijimproject_refactor`, no fallback across
different `0x100` layouts, no MDX/MDL advice for an M2 source, and no WotLK+ regression.
**Out of scope**: animation correctness, particles, ribbons, attachments, bone deformation,
1.12.1 layout completion, and TBC layout work.

## Constitution Check

- **Repo independence / read-only reference**: all code and evidence remain in `wow-viewer`; PASS.
- **Library-first**: `M2ModelReaderDispatcher` and `M2Era100ModelReader` are canonical owners;
  the viewer only consumes their output; PASS.
- **Real-data validation**: unit/build proof is insufficient; the phase gate is a staged 1.0.0
  model rendering in the user-run viewer; PASS.
- **One phase at a time**: only the 1.0.0 M2 slice is active; 1.12.1/TBC remain blocked behind it; PASS.
- **Bite-sized work**: each implementation phase has at most six independently verifiable steps; PASS.

## Project Structure

```text
wow-viewer/
├── src/core/WowViewer.Core.IO/M2Chunked/M2ModelReaderDispatcher.cs
├── src/core/WowViewer.Core.IO/M2Era100/                 # 1.0.0 MD20 reader and constants
├── src/core/WowViewer.Core/M2/M2Era100Geometry.cs       # version-neutral geometry handoff
├── src/viewer/WoWViewer/Rendering/WarcraftNetM2Adapter.cs
├── src/viewer/WoWViewer/ViewerApp.cs                    # standalone-load status only
├── tests/WowViewer.Core.Tests/M2Era1121ModelReaderTests.cs
└── specs/104-legacy-m2-rendering/
    ├── research-1.0.0-ghidra-trace.md
    ├── contracts/m2-format-profile.md
    ├── quickstart.md
    └── tasks.md
```

## Phase 0 — Correct the contract and lock regression tests

1. State that 1.0.0 is `MD20`/`0x100` M2 and that MDX/MDL is not a replacement.
2. Add focused fixtures asserting that a valid classic-layout `0x100` classifies as era-100,
   while a valid 1.12.1-shaped `0x100` still classifies as era-1121.
3. Assert malformed classic-layout headers fail without reclassification or crash.
4. Replace the standalone-load MDX/MDL suggestion with an M2-specific message.

**Gate**: focused tests prove the two `0x100` layouts cannot be conflated and the user-facing
copy contains no MDX substitution advice.

## Phase 1 — Implement and harden the 1.0.0 M2 reader

1. Parse only the Ghidra-confirmed 1.0.0 header spans and validate every count/offset before use.
2. Materialize division zero into resolved render vertices, triangle indices, sections, batches,
   textures, and texture lookup data.
3. Emit that geometry through `M2ModelDocument` without making downstream consumers version-aware.
4. Route a detected era-100 document through `WarcraftNetM2Adapter` into the existing renderer.
5. Route a detected era-100 document through `M2StaticRenderModel` and `M2Renderer`; do not construct
   `MdxFile`/`MdxRenderer` or any M2-to-MDX compatibility state.
6. If the era-100 reader rejects the asset, raise its reader error; do not fall through to generic
   Warcraft.NET parsing or a different M2 layout.
7. Build the solution and run the focused test project.

**Gate**: parser/adapter test proof and a clean build. This is code proof only.

## Phase 2 — User-driven real-data signoff

1. Confirm an appropriate staged 1.0.0 root exists under `output/tmp/wowarchive-clients/`.
2. User loads a representative `.m2` (static prop or creature) in the viewer.
3. Capture the exact path, detected era, mesh/texture result, and any reader error.
4. Compare against a reference render and update the format profile with the evidence.
5. Recheck a WotLK+ M2 through the existing external-skin route.

**Gate**: visible mesh/materials from the staged 1.0.0 source plus a recorded WotLK+ no-regression
check. Until then, Plan 104 is not done.

## Deferred phases

- **1.12.1**: separate `0x100` layout, only after Phase 2.
- **TBC 0x101–0x107**: one format boundary at a time after the 1.x route is signed off.
- **0.11/0.12**: retain their working pre-`0x100` route; only trace if a real regression is found.

## Phase 3 — Native 2.x/3.0.x embedded-profile handoff

1. Reuse the existing profiled embedded-root parser and material metadata extraction; do not
   introduce a second legacy M2 reader in the viewer.
2. Convert its parsed geometry, batches, render flags, and texture lookup tables into the shared
   `M2GeometryDocument`/`M2SkinDocument` shape used by the native static runtime builder.
3. Route world placements and WMO M2 doodads with no external `.skin` through
   `M2Renderer(GL, M2StaticRenderModel, ...)`; retain M2-to-MDX only as an explicit failure fallback.
4. Record the route as `NativeEmbeddedProfile` so a real-client capture can prove whether the
   legacy model avoided the compatibility renderer.

**Gate**: focused build/test proof plus a user-run 2.x or 3.0.x client capture showing the native
route, visible geometry, and material blend counts. This is not visual shader parity signoff.

## Evidence and operator rules

- Ghidra static evidence: [research-1.0.0-ghidra-trace.md](research-1.0.0-ghidra-trace.md).
- The user runs the viewer and any debugger/heavy operation. The agent prepares code, tests, and
  exact commands only.
- A build or unit test never substitutes for staged-client render proof.
