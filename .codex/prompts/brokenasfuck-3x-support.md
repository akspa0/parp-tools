---
description: "Run a hard recovery plan for terrain and UI regressions by resetting to baseline commit 343dadf, surgically reverting the single-pass alpha merge, and reintroducing verified-good features without breaking Alpha."
name: "Terrain Recovery And Path Hardening Plan"
argument-hint: "Optional map, tile, chunk, offending commit range, or runtime symptom to prioritize"
agent: "codex"
---
Execute a no-assumption recovery workflow for terrain alpha and UI regressions in gillijimproject_refactor.

## Goal

Restore known-good behavior first, then reintroduce changes safely.

Treat this as one recovery problem:

How do we get back to the last known-good baseline (343dadfa27df08d384614737b6c5921efe6409c8), surgically remove the single-pass alpha/shadow regression, and recover post-baseline improvements without reintroducing broken layering or UI regressions?

## Feature Retention Constraint

- Do not throw away all post-baseline improvements.
- Recovery must preserve as many valid feature gains as possible.
- If a feature conflicts with terrain correctness, isolate and rewrite only the conflicting part.

## Primary Suspect Regression

- Treat the merged alpha-layer + shadow single rendering pass as the primary regression candidate until disproven.
- Prioritize reverting or splitting this change before broader terrain rewrites.

## Non-Negotiable Baseline

- Commit 343dadfa27df08d384614737b6c5921efe6409c8 is the last known good for terrain loading and alpha UI behavior unless the user says otherwise.
- Do not argue with that baseline.
- If current behavior conflicts with baseline behavior, baseline wins until proven otherwise.

## Recovery Principles

- Stop guessing root cause. Use commit-level isolation.
- Prefer revert-and-curate over patching on top of broken state.
- Reintroduce changes in small groups only after runtime checks pass.
- If a cherry-pick reintroduces regression, drop it or split it immediately.
- Preserve working features whenever possible; remove only proven-bad behavior.

## Path Hardening Requirements

- Alpha path is the protected core path and must remain stable.
- Create separate terrain pipelines by era family:
  - Alpha prerelease path (existing working path)
  - 1.x/2.x path
  - 3.x/4.x path
- Separate both read/decode profiles and render/blend profiles.
- Do not keep two era families in one monolithic terrain shader or blend branch when semantics differ.
- Shared helpers are allowed only for neutral utilities (buffer upload, common math, logging), not era-specific decode/blend semantics.

## Version Selection Policy (No Silent Guessing)

- If client version cannot be determined with certainty, prompt the user to select version family before map load.
- Prefer explicit user choice over automatic fallback.
- Block terrain load when version family is unknown rather than silently choosing a profile.

## Scope

- In scope:
  - alpha terrain path freeze and regression guardrails
  - 1.x/2.x and 3.x/4.x terrain decode and blend separation
  - single-pass alpha/shadow merge rollback or split
  - layer texture mapping and missing-texture handling
  - alpha debug and overlay visibility behavior
  - UI rollback for terrain diagnostics and operator workflow
- Out of scope unless directly required by failing evidence:
  - unrelated world features (SQL spawns, POI, taxi, skybox, etc.)
  - broad unrelated architectural refactors
  - speculative cleanup with no runtime impact

## Validation Rules

- Signoff target is official 3.0.1-era runtime data, not fixture-only data.
- Repo fixtures are diagnostic aids, not final proof.
- "Build succeeded" does not mean "fixed".
- "Tests passed" does not mean "fixed".
- Every claimed fix requires runtime-visible evidence.

## Required Starting Files

1. gillijimproject_refactor/src/MdxViewer/Terrain/StandardTerrainAdapter.cs
2. gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Formats/LichKing/Mcal.cs
3. gillijimproject_refactor/src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs
4. gillijimproject_refactor/src/MdxViewer/Terrain/TerrainRenderer.cs
5. gillijimproject_refactor/src/MdxViewer/Terrain/FormatProfileRegistry.cs
6. gillijimproject_refactor/src/MdxViewer/ViewerApp.cs

## Required Target Files For Path Split

1. gillijimproject_refactor/src/MdxViewer/Terrain/AlphaTerrainAdapter.cs
2. gillijimproject_refactor/src/MdxViewer/Terrain/StandardTerrainAdapter.cs
3. gillijimproject_refactor/src/MdxViewer/Terrain/TerrainRenderer.cs
4. gillijimproject_refactor/src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs
5. gillijimproject_refactor/src/MdxViewer/Terrain/FormatProfileRegistry.cs
6. gillijimproject_refactor/src/MdxViewer/ViewerApp.cs

## Recovery Plan

### Phase 0: Freeze and Snapshot

1. Record current branch, HEAD, and dirty state.
2. Save a short regression snapshot:
   - alpha symptom summary
   - UI symptom summary
   - one known-bad runtime screenshot pair (normal + alpha debug)
3. Do not destroy user work. Use a dedicated recovery branch.

### Phase 1: Baseline Verification

1. Check out baseline 343dadfa27df08d384614737b6c5921efe6409c8 in a detached or throwaway branch.
2. Build and run MdxViewer.
3. Verify baseline behavior against user report:
   - terrain loads correctly
   - alpha UI behavior is sane
   - no black/white streak corruption in the previously failing area
4. If baseline is not actually good in current environment, stop and report immediately.

### Phase 2: Create Recovery Branch from Baseline

1. Create branch from baseline (example: recovery/terrain-ui-reset-343dadf).
2. Do not merge current broken branch into this recovery branch.
3. Recover forward only through curated cherry-picks.

### Phase 3: Surgical Revert of Single-Pass Alpha/Shadow Merge

1. Identify exact commit(s) that merged alpha layers and shadow into one pass.
2. Revert only that merge behavior first, before other terrain cherry-picks.
3. Validate immediately against known broken sample area.
4. If issue resolves, keep this as a hard gate and continue feature reintroduction.

### Phase 4: Commit Triage and Bucketing

Build a ledger of post-baseline commits grouped by risk:

- Bucket A (low risk): build plumbing, non-render helpers, diagnostics
- Bucket B (medium risk): tests and instrumentation only
- Bucket C (high risk): terrain decode, alpha packing, shader blend, profile routing, pass topology
- Bucket D (high risk): UI/ViewerApp panel/menu/window overhauls

For every commit, mark one status:
- KEEP_CANDIDATE
- DROP_CANDIDATE
- SPLIT_REQUIRED
- DEFER

### Phase 5: Cherry-Pick in Waves with Runtime Gates

Wave 1:
- Apply only Bucket A and B commits that cannot affect terrain output or terrain-debug UI behavior.
- Build and smoke run.

Wave 2:
- Apply minimal Bucket C commits one at a time, with alpha/shadow merge commits handled first.
- After each commit:
  - run terrain runtime check on target 3.0.1 map area
  - run alpha debug overlay visibility check
  - run alpha-prerelease safety check on known good Alpha map/tile
  - if regression appears, immediately revert that commit in recovery branch and mark as DROP or SPLIT_REQUIRED

Wave 3:
- Apply only essential UI commits from Bucket D.
- Reject broad dock/layout/menu rewrites unless they are required for terrain workflow.
- Preserve a simple, stable operator UI over feature-heavy panel sprawl.

### Phase 6: Terrain and UI Acceptance Gate

All must pass:

1. Terrain alpha layers visually correct in runtime target area
2. No black terrain with white streak artifacts in that area
3. Alpha debug mode still allows useful overlays or equivalent diagnostics
4. Layer-to-texture mapping is stable when textures are missing
5. UI remains usable for map load, terrain controls, and alpha diagnosis
6. Alpha prerelease rendering remains unchanged from baseline quality
7. Client version family is explicit or user-selected before terrain decode/render path selection

If any fails, continue triage and do not declare fixed.

## Decision Rules for Reintroducing Changes

- If a commit mixes good and bad behavior, split it; do not carry full commit blindly.
- If two commits conflict, prefer the version closest to baseline semantics unless runtime proves otherwise.
- Any commit that re-breaks alpha debug overlays is blocked until rewritten.
- Any commit that re-couples Alpha path semantics with later-client path semantics is blocked.
- Any commit that expands UI complexity without improving terrain diagnosis is a drop candidate.

## Deliverables

Return all items:

1. Recovery branch name and baseline verification result
2. Cherry-pick ledger table:
   - commit hash
   - bucket
   - status
   - reason
   - runtime result
3. Final kept commit list in replay order
4. Explicit dropped commit list with reasons
5. Final fix summary with remaining risks
6. Clear statement: fixed in runtime target, or not fixed yet
7. Decode profile matrix by version family (Alpha, 1.x/2.x, 3.x/4.x)
8. Render profile matrix by version family (separate pass/blend behavior)
9. Version-selection UX result (explicit detect vs user prompt path)

## First Output

Start with:

1. Current branch/HEAD/dirty snapshot
2. Baseline verification status for 343dadf
3. Candidate commit list for single-pass alpha/shadow merge rollback
4. Initial post-baseline commit ledger (bucketed)
5. Proposed Wave 1 cherry-pick set