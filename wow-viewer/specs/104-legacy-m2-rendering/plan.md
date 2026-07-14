# Implementation Plan: Legacy M2 model rendering (client 0.11 – 2.4.3)

**Branch**: `104-legacy-m2-rendering` (work lands on `v0.5.0-prerelease`) | **Date**: 2026-07-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/104-legacy-m2-rendering/spec.md`

## Summary

Make M2 models from client builds 0.11–2.4.3 render their mesh and materials instead of an empty
bounding box. The single confirmed root cause is that
[M2ModelReader.cs](../../src/core/WowViewer.Core.IO/M2/M2ModelReader.cs) hardcodes
`embeddedSkinProfileCount`/`embeddedSkinProfileOffset` to `0` and assumes a WotLK-era header layout.
For every in-scope version (M2 format version ≤ 263) the skin/geometry profiles — submeshes,
triangle indices, texture-unit/material bindings — are **embedded in the .m2 itself** (`nViews`/
`ofsViews`), not in external `.skin` files.

Technical approach: teach the reader to (1) branch on the M2 format version at header `0x04`, (2)
read the embedded skin profile(s) for ≤ 263, and (3) feed the extracted geometry + material bindings
into the existing render path. The investigation is phased by **format-version boundary** and moves
from documented formats (validate against wowdev.wiki + reference implementations, no debugger) to
undocumented early alphas (recover the layout via x64dbg dynamic tracing). The durable deliverable is
a **per-version format profile** research artifact plus the reader changes that consume it.

**Read-only reference:** `gillijimproject_refactor` may be *read* for M2 knowledge but never modified
(constitution: Read-Only Reference Codebase).

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: `WowViewer.Core.IO` (owns the M2 reader), `WowViewer.Core.Runtime.M2` /
the viewer's M2 render path, Silk.NET.OpenGL (rendering backend). No new external dependencies expected.

**Storage**: N/A (reads `.m2` files from staged client archives via the existing archive/MPQ readers).

**Testing**: Real-data validation against staged clients under `output/tmp/wowarchive-clients/`
(constitution III). Visual comparison against a known-good reference implementation for the
documented versions. Existing C# xUnit harness for any pure-parse assertions that don't need a client.

**Target Platform**: The viewer (win-x64 primary; the reader itself is platform-neutral `net10.0`).

**Project Type**: Desktop application + shared format library (single canonical M2 reader).

**Performance Goals**: No regression to current load times; embedded-skin reading is a bounded,
one-time-per-model parse. Not performance-critical relative to correctness.

**Constraints**:
- MUST NOT regress WotLK+ (version 264+) M2 rendering, which already works (spec FR-009).
- MUST fail safe on malformed/misidentified legacy M2 (no crash; spec FR-005).
- Ghidra is not installed → static disassembly is out of scope unless installed as a separate step;
  dynamic analysis uses the x64dbg MCP bridge (configured and responding).

**Scale/Scope**: ~10 in-scope format/version tiers (2.4.3, 1.12.1, 2.0.0 alpha, 2.1, 2.2, 2.3, 1.0.0,
0.12, 0.11), grouped into 3 priority phases. One reader file is the primary change surface.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

- **I. Repo Independence** — All changes land inside `wow-viewer/`. `gillijimproject_refactor` is read
  only for reference. No `.csproj` or source path outside `wow-viewer/` is added. **PASS.**
- **II. Library-First / one canonical format owner** — The M2 reader already exists in
  `WowViewer.Core.IO/M2/M2ModelReader.cs`; this feature **extends** it (version-aware header + embedded
  skins), never forks or duplicates it. CLI/viewer stay thin consumers. **PASS.**
- **III. Real-Data Validation** — Every version tier is signed off against real staged clients under
  `output/tmp/wowarchive-clients/`, with reproducible evidence, not mock M2s. **PASS.**
- **Read-Only Reference Codebase** — `gillijimproject_refactor` M2 code may be read as a reference; no
  writes. **PASS.**
- **Format Reader/Writer Ownership** — Do not rewrite the working WotLK path; add version branches
  around it. Check existing readers before adding anything. **PASS.**
- **Terrain Alpha Risk Area** — Not applicable (M2, not MCAL/terrain). **N/A.**
- **One Phase at a Time** — Phases P1→P2→P3 are gated: a phase is "done" only when its versions render
  and are validated against ground truth. **PASS.**
- **Spec Docs Are Source of Truth** — Spec 104 exists; this plan and its research/data-model are the
  living design docs, updated in the same commits as code. **PASS.**
- **Bite-Sized Plans** — Each phase below is ≤ 10 one-concern, independently-validatable steps. **PASS.**

No violations. Complexity Tracking table omitted (nothing to justify).

## Project Structure

### Documentation (this feature)

```text
specs/104-legacy-m2-rendering/
├── spec.md               # Feature spec (done)
├── plan.md               # This file
├── research.md           # Phase 0: per-version M2 format knowledge + open unknowns
├── data-model.md         # Phase 1: M2 header + embedded skin profile entities
├── quickstart.md         # Phase 1: how to investigate, render, and validate (incl. x64dbg flow)
├── contracts/
│   └── m2-format-profile.md   # The durable per-version format-profile schema + the reader output contract
├── checklists/
│   └── requirements.md   # Spec quality checklist (done)
└── tasks.md              # Phase 2 output (speckit-tasks — NOT created by this plan)
```

### Source Code (repository root)

```text
wow-viewer/src/core/WowViewer.Core.IO/M2/
├── M2ModelReader.cs          # PRIMARY change: version-branch header + read embedded skin profiles
├── M2ToMdxConverter.cs       # reference for existing M2 field handling (read)
└── (new, if warranted) M2SkinProfileReader.cs  # embedded-skin parsing, if M2ModelReader grows too large

wow-viewer/src/core/WowViewer.Core.IO/Models/   # M2 model document types the reader emits
wow-viewer/src/core/WowViewer.Core.Runtime/M2/  # runtime consumers of the parsed model
wow-viewer/src/viewer/WoWViewer/Rendering/M2Renderer.cs  # render path (should need little/no change if the emitted geometry contract is met)

output/tmp/wowarchive-clients/                  # staged clients per version (validation ground truth)
gillijimproject_refactor/…                       # READ-ONLY reference for M2 format knowledge
```

**Structure Decision**: Single canonical M2 reader in `WowViewer.Core.IO/M2/`. The change is
localized to the reader (header version-branching + embedded-skin extraction) so the viewer's existing
render path consumes the newly-populated geometry unchanged. A dedicated `M2SkinProfileReader.cs` is
introduced only if the embedded-skin logic makes `M2ModelReader.cs` unwieldy — decided during P1.

## Investigation & Implementation Phases

Phases are gated (constitution: One Phase at a Time). "Done" = the phase's versions render mesh +
materials and are validated against ground truth, not merely coded.

### Phase 0 — Research the format landscape (no code)

Establish, from documentation and the read-only reference, what is known per format version before
touching the reader. Output: `research.md`. Resolves: exact meaning/positions of `nViews`/`ofsViews`,
the embedded skin-profile sub-structure (submesh, index, texture-unit arrays), the header-offset
deltas between WotLK layout (current code) and ≤ 263, and which versions are documented vs. need tracing.

### Phase 1 — P1: render the documented versions (2.4.3, 1.12.1)

The MVP and the reusable mechanism. ≤ 10 steps, each independently validatable:

1. Capture the current WotLK-path behavior as the no-regression baseline (a 264+ model still renders).
2. Read the M2 format version at `0x04`; add a version classifier (embedded-skin era ≤ 263 vs external ≥ 264).
3. Confirm/correct the header field offsets for the 263/256 layout against `research.md` (bounds already
   parse — verify view/skin count+offset positions specifically).
4. Read the embedded skin-profile table (`nViews`/`ofsViews`) for ≤ 263 instead of hardcoding 0.
5. Parse one skin profile: submesh definitions + triangle index array.
6. Parse the texture-unit / material bindings and associate them with submeshes.
7. Feed extracted geometry + material bindings into the existing render path; a 2.4.3 model renders mesh.
8. Bind textures per submesh; the 2.4.3 model renders textured (matches reference).
9. Repeat validation for 1.12.1 (version 256); record any 256-vs-263 delta in the format profile.
10. Malformed/truncated-skin guard: bounds-check offsets, fail safe to bounding box, no crash.

Gate: 2.4.3 + 1.12.1 render textured mesh matching a reference implementation (SC-001, SC-002).

### Phase 2 — P2: verify the mid-range versions (2.0.0 alpha, 2.1, 2.2, 2.3)

Same format family as 2.4.3; expect reuse, verify don't assume.

1. Load representative M2s from 2.1/2.2/2.3; confirm they render via the P1 path.
2. Load 2.0.0 alpha M2s; confirm or capture the format delta (alphas drift).
3. For any version that fails, isolate the offset/structure delta and add a version-specific branch.
4. Record each version's confirmed profile (or delta) in `contracts/m2-format-profile.md`.

Gate: all mid-range versions render, or every failure is a documented, handled delta (SC-003).

### Phase 3 — P3: recover the early alphas (1.0.0, 0.12, 0.11) via dynamic tracing

Where documentation runs out and x64dbg earns its keep. Per version:

1. Confirm a staged client for the version exists; if not, mark blocked (not failed) and skip.
2. Attempt the P1/P2 reader on the version's M2s; capture exactly where parsing produces garbage.
3. Launch x64dbg on the alpha client (`start_session` → `connect`), breakpoint the M2 load path, and
   trace how the real client walks the header to reach geometry — recovering true field offsets.
4. Record the recovered header/skin layout in the format profile with the trace as evidence (FR-007).
5. Add the version's branch to the reader; the model renders, or the residual unknown is documented.

Gate: early-alpha versions with staged clients render, or their unknowns are precisely documented
with evidence (SC-004, SC-005).

### Phase 4 — Consolidate

1. Every in-scope version has a profile entry in `contracts/m2-format-profile.md` (SC-005).
2. Confirm no WotLK+ regression across a spread of 264+ models (SC-006, FR-009).
3. Update memory bank (`activeContext.md`, `progress.md`) with the format findings and reader state.

## Tooling Notes

- **x64dbg MCP**: configured and responding (`mcp__x64dbg__*`); no session attached yet. Phase 3 usage:
  `start_session` with the alpha client exe → `connect` → set breakpoints on the M2 load routine →
  `read_memory`/registers to observe header walking. Deterministic, dynamic ground truth for
  undocumented layouts.
- **Ghidra**: NOT installed. Static disassembly is out of scope unless Ghidra is installed first (a
  separate, user-approved setup step). If P3 dynamic tracing proves insufficient, installing Ghidra
  becomes a candidate follow-up, not an assumption baked into this plan.
- **Reference implementations**: wowdev.wiki M2 pages and open renderers (e.g. WoW Model Viewer) are
  the documentation baseline and visual-validation oracle for P1/P2.

## Complexity Tracking

No constitution violations; no justifications required.
