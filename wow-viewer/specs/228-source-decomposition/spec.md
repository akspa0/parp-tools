# Feature Specification: Source Decomposition — God-Class Split (Spec 228)

**Feature Branch**: `228-source-decomposition`
**Created**: 2026-09-06
**Status**: Planned — 2026-09-06 design artifacts identify an evidence-gated first selection
extraction; source implementation remains blocked by Spec 227 T004.
**Input**: Operator directive 2026-09-06 (condensed verbatim): "WorldScene.cs and ViewerApp.cs need
to be refactored and split up into smaller sets of source code, so the project can be worked on
without fear of context window limits. This is an ever-evolving issue that we cannot seem to stamp
out. We tried with the split up ViewerApp_ files, but even that has failed to help us as things
progress. We need to seriously put this in writing and fix things before they continue to spiral
out of control."

## Context

`WorldScene.cs` (~16.9k lines) and `ViewerApp.cs` (~16.7k lines, plus its ~20 `ViewerApp_*.cs`
partial-class siblings) are single god-classes. The `ViewerApp_` split failed to contain growth
because **partial classes share one state space**: every new feature adds fields and methods to the
same class regardless of which file they land in, so every session must load tens of thousands of
lines of context to touch anything. This rule and spec put the fix in writing.

## Rule (binding, also added to AGENTS.md §10)

- **New code must not add members to `WorldScene` or `ViewerApp`.** New features live in their own
  owned classes/services that receive the state they need through constructors/parameters.
- **File budget**: no source file may grow past ~2,000 lines; a change that pushes a file over the
  budget must split it in the same change or open a spec task to do so.
- **Extraction pattern**: move a cohesive feature (state + methods) out of the god-class into a
  service class; the god-class keeps a single field + delegation, and the extracted class never
  reaches back into the god class.
- Existing `ViewerApp_*` partial files are legacy members of the same god class — they do NOT
  satisfy this rule.

## User Stories

### US1 — WorldScene decomposed (P1)
`WorldScene` is split into owned, testable services (e.g., hover/pick service, overlay render
services, lighting composition, transport/taxi, archeology playback, WMO-only admission) with
`WorldScene` reduced to composition + render orchestration.

**Acceptance criteria**:
- `WorldScene.cs` (the main file) shrinks below ~4,000 lines with feature services in their own
  files/classes under `Terrain/` or Core where testable.
- No extracted service references back into `WorldScene` internals.
- Full solution build + affected test suites stay green after each extraction phase.

### US2 — ViewerApp decomposed (P1)
`ViewerApp` follows the same pattern: capture automation, camera paths, converters dialogs, export
pipelines, hover/pick, and workspace shell become owned services. `ViewerApp_*.cs` partials are
gradually converted from partial-class members into service classes until the god class holds only
bootstrap, frame loop, and composition.

**Acceptance criteria**:
- Each extraction leaves behavior identical (the receipts rule applies: build + focused tests +
  operator smoke of the moved surface).
- `ViewerApp.cs` shrinks below ~4,000 lines over time; no new partial files are created.

### US3 — Context-window safety (P0)
After each phase, an agent can implement a change in the affected area without reading more than
the owning service files.

**Acceptance criteria**:
- A session working on, e.g., capture automation reads only `CaptureAutomationService.cs` and its
  tests — not 16k lines of ViewerApp.

## Constraints

- Extraction is behavior-preserving refactoring only; no feature changes ride along.
- Order of work must follow the Spec 227 UI audit where a feature's UI home is in question, so
  services are extracted to their post-audit shape, not the current one.

## Success Criteria

- **SC-1**: `wc -l` on `WorldScene.cs`/`ViewerApp.cs` trends down every phase; no file > ~2,000
  lines at phase end.
- **SC-2**: Two consecutive sessions complete features without opening either god-class file.
