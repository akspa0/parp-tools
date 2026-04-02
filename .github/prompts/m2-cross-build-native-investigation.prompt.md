---
description: "Systematic cross-build native-client investigation for M2 behavior across WoW builds from 3.3.5 through 6.x. Use when library support is partial and you need build-specific static anchors, runtime breakpoint evidence, and a compatibility matrix before implementation."
name: "M2 Cross-Build Native Investigation - Wrath to 6.x"
argument-hint: "Optional: target build list, subsystem seam (skin, section, effects, liquids, particles, lighting), or known symptom"
agent: "agent"
---

Run a structured native-client investigation across multiple WoW builds so later-version parser and runtime work is grounded in real evidence instead of assumptions.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/memory-bank/data-paths.md`
4. `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
5. `.github/prompts/m2-rendering-investigation.prompt.md`
6. `.github/prompts/wow-viewer-m2-runtime-plan-set.prompt.md`
7. `.github/copilot-instructions.md`

## Goal

Recover and compare M2 runtime behavior across build families, especially where existing libraries have only partial support (commonly 4.x through 6.x), then produce implementation-ready guidance for `wow-viewer` with explicit confidence levels.

## Default Build Ladder

Use this ladder unless the user gives a different list:

1. `3.3.5.12340` (Wrath baseline)
2. `4.0.6a.13623` (early Cataclysm)
3. `4.3.4.15595` (late Cataclysm)
4. `5.4.8.18414` (late Mists)
5. `6.0.3.19116` (early Warlords)
6. `6.2.4.21355` (late Warlords)

If an exact build is unavailable, choose the nearest available patch in the same expansion and record that substitution.

## Required Outputs

Produce all of these:

1. Build inventory table (client exe hash, data source roots, debugger status)
2. Per-build anchor map (static addresses, recovered labels, confidence)
3. Per-build runtime capture log (breakpoint hits and argument/path evidence)
4. Cross-build behavior-diff matrix
5. Implementation backlog split into:
   - `wow-viewer` canonical ownership
   - compatibility-only `MdxViewer` stopgaps

## Phase 1: Build Matrix Setup

For each target build:

1. Confirm binary identity (`WoW.exe` hash/version)
2. Confirm data source availability (MPQ/CASC/loose)
3. Confirm tool viability:
   - offline Ghidra load works
   - x64dbg attach/pause/run works
4. Record blockers immediately (missing client files, attach failure, anti-debug behavior)

Do not continue to later phases for a build if setup is not reproducible.

## Phase 2: Static Anchor Recovery (Offline Ghidra)

For each build, recover anchors for these seams:

1. Model identity and extension gate (`.mdl/.mdx/.m2` behavior)
2. Skin profile choose/load/init (`%02d.skin` ownership)
3. Animation sidecar path formatting (`%04d-%02d.anim` or equivalent)
4. Section classification and draw-entry gating
5. Material/effect routing (combiner/effect family naming)
6. Runtime feature flags (batching, z-fill, clip planes, additive sort)

Anchor recovery method:

- Start from known string/xref families:
  - `%02d.skin`
  - `MD20`
  - `CM2Model` or equivalent class strings
  - `Combiners_` / `Diffuse_` effect names
  - `M2Use` runtime options
- Verify call graph relationships, not just isolated xrefs
- Record confidence as `high`, `medium`, or `research`

## Phase 3: Runtime Validation (x64dbg)

For each build with working attach:

1. Arm minimal breakpoint chain first (avoid noisy broad hooks)
2. Capture at least one contiguous choose/load/init/effect sequence
3. Decode stack/register arguments to recover:
   - model path
   - built skin path
   - profile selection result
   - effect family output
4. Classify path context as:
   - UI
   - portrait/auxiliary
   - world
5. Keep running until at least one world-path chain is captured, or document exactly why that failed

If repeated pauses land in system DLL frames, apply a stabilization pass before claiming runtime results.

## Phase 4: Cross-Build Diff Matrix

Build a comparison matrix for:

1. Extension-gate behavior changes
2. Skin profile strategy changes (numbered skins, fallback behavior)
3. Section and submesh ownership changes
4. Material/effect family changes
5. Runtime flag semantics and callback ownership changes
6. Any discovered format boundary shifts relevant to 4.x/5.x/6.x assets

Mark each matrix row with:

- `confirmed` (static + runtime)
- `static-only`
- `runtime-only`
- `research`

## Phase 5: Implementation Guidance

Translate evidence into concrete implementation slices:

1. `wow-viewer/src/core/WowViewer.Core.IO/M2` parser ownership updates
2. `wow-viewer/src/core/WowViewer.Core.Runtime/M2` runtime/effect/flag ownership updates
3. Inspect or diagnostic CLI additions for build-aware validation
4. Compatibility-only bridge work in `gillijimproject_refactor` (if explicitly needed)

For each slice, state proof level explicitly:

- build/test only
- static evidence only
- static + runtime evidence

## Non-Negotiable Constraints

1. Do not claim cross-build support from parser success alone.
2. Do not infer 6.x behavior from 3.3.5 assumptions without direct evidence.
3. Do not present library output as ground truth when native evidence disagrees.
4. Keep unresolved semantics labeled as research.
5. Record findings in `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md` and, when needed for large build-specific runs, a dedicated session packet under `wow-viewer/docs/architecture/`.

## First Output

Start with:

1. the exact build set you will investigate first
2. which seam you will recover first (identity, skin, section, effect, or flags)
3. what evidence you expect to collect in this pass
4. what you are explicitly not claiming yet