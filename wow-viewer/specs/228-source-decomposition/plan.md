# Implementation Plan: Source Decomposition — God-Class Split

**Branch**: `223-ui-consolidation-audit` (planning occurs on the current worktree) | **Date**:
2026-09-06 | **Spec**: [spec.md](spec.md)

**Input**: operator directive recorded in [spec.md](spec.md): replace the failed partial-class
approach with owned service boundaries so feature work does not require loading tens of thousands
of lines of `WorldScene` and `ViewerApp` context.

## Summary

This is a staged, behavior-preserving C# refactor, not a rewrite. Each phase moves one cohesive
feature's state and operations to an owned class, keeps only a narrow composition/delegation seam
in the legacy class, adds characterization coverage where data can be made pure, and produces a
line-count/build/test/operator-smoke receipt. The first candidate is WorldScene hover/click
selection, but its viewer-route extraction cannot begin until Spec 227 T004 settles UI authority.

## Technical Context

**Language/Version**: C# / .NET 10 (`net10.0`)

**Primary Dependencies**: Silk.NET/OpenGL viewer shell; `WowViewer.Core.Runtime` for portable
world decision logic; xUnit test projects

**Storage**: N/A — no persisted schema or generated data is introduced

**Testing**: `dotnet test` for `WowViewer.Core.Tests` and full solution; operator smoke for moved
interactive routes

**Target Platform**: existing Windows/cross-platform desktop viewer

**Project Type**: desktop application with shared Core libraries

**Performance Goals**: preserve current interaction behavior; measure source-size reduction and
avoid allocations/renderer changes in the first selection extraction. No FPS claim without an
operator capture.

**Constraints**: no new `WorldScene` or `ViewerApp` feature members/partials; no UI, format-reader,
renderer, or data-behavior changes; each new owner remains under approximately 2,000 lines; existing
oversized files must not grow; extraction waits for relevant Spec 227 authority gates.

**Scale/Scope**: `WorldScene.cs` 16,915 lines; `ViewerApp.cs` 16,740 lines; 25 partials totaling
25,511 lines. First implementation phase is one selection service only.

## Constitution and governance check

| Gate | Plan response | Status |
|---|---|---|
| Repo independence | All paths stay under `wow-viewer/`; no external dependency or path is introduced. | Pass |
| Library-first | Deterministic selection policy and records belong in `WowViewer.Core.Runtime`; viewer adapts existing runtime objects at the edge. | Pass |
| Real-data validation | This refactor makes no format/data claim. Interactive pick proof remains an operator smoke with configured-client provenance. | Pass with operator gate |
| Frozen readers/writers | No MPQ/ADT/WMO/M2/MDX reader or `AlphaWdtWriter` modification is in scope. | Pass |
| Spec 227 authority | Selection presentation and Inspector mapping are not moved until T004. | **Blocked by T004** |
| God-class freeze | Each extraction has a named owned service and removes more legacy implementation than it adds to the god class. No partial class is created. | Pass by task gate |
| Receipt rule | Every checkbox stays open until its receipt names files, commands/status, counts, and criterion evidence. | Pass |

**Post-design check**: No unresolved technical design question remains. The only deliberate blocker is
the external Spec 227 UI-authority gate; the plan records it rather than guessing a UI home.

## Project structure

```text
specs/228-source-decomposition/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── tasks.md

src/core/WowViewer.Core.Runtime/
└── World/Selection/                 # pure selection records and service (first extraction)

src/viewer/WoWViewer/
└── Terrain/Scene/Selection/         # viewer-only snapshot/result adapter

tests/WowViewer.Core.Tests/
└── World/Selection/                 # deterministic selection service tests
```

**Structure decision**: Core Runtime owns policy that needs no UI, GL, or parent-scene state. The
viewer owns only adaptation of existing resident objects. This leaves future `Application/Capture`,
`Application/Camera`, and `Workbench/*` extractions available without pre-creating speculative
folders.

## Phased roadmap

### Phase 0 — Baseline and guardrails

Record exact line counts, candidate dependency edges, and existing over-budget ledger. Add no
feature code. The receipt establishes a reproducible before-state.

### Phase 1 — Selection boundary design (blocked by Spec 227 T004)

Once the UI inventory gate is recorded, characterize current hover/click ordering, define pure
snapshot/result records, and add tests for ranking/fall-through behavior. No presentation route is
moved and no selection source code is introduced before the authority gate.

### Phase 2 — First behavior-preserving WorldScene extraction

Move the selection algorithm into the new owned service; the viewer adapter maps resident data in
and maps the result to existing public routes. Remove the corresponding algorithm/helpers from
`WorldScene`, prove a net line-count reduction, run focused/full tests, and obtain the operator
smoke. Stop at the receipt.

### Phase 3 — Next candidate selection (not pre-authorized implementation)

Use the Phase 2 receipt and current Spec 227/223 gates to choose one next owner. Likely candidates
are capture automation or camera paths, but no code task is authorized until that selection has its
own narrow task amendment and behavior matrix.

## Complexity tracking

| Existing issue | Containment approach | Why it is not fixed in the first slice |
|---|---|---|
| Multiple source files already exceed 2,000 lines | Baseline them and prohibit growth; schedule one owner at a time. | A broad split would mix unrelated behavior and make receipts meaningless. |
| Private `WorldScene` state couples selection and presentation | Use snapshot/result values and a viewer-edge adapter. | Moving Inspector UI now would violate the Spec 227 gate. |
| `ViewerApp` partials hide a shared state space | Ban new partials and defer capture/sidebar extraction until UI/capture gates settle. | Capture has an open Spec 223 live-proof boundary. |
