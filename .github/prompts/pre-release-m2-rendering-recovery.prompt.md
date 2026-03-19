---
description: "Use when debugging pre-release 3.0.1 M2 compatibility, hybrid MDX+M2 model behavior, or neon-pink transparent surfaces in MdxViewer. Splits format-compatibility work from shared shader/material failures and requires real-data validation."
name: "Pre-release M2 Rendering Recovery"
argument-hint: "Optional model path, client build, runtime symptom, screenshot clue, or suspected shader/material path"
agent: "agent"
---

Execute a no-assumption investigation plan for pre-release `3.0.1` model compatibility and the shared neon-pink transparent-surface bug in `gillijimproject_refactor/src/MdxViewer`.

## Goal

Separate two problems that can look similar at runtime but are not the same:

1. pre-release `3.0.1` model-format compatibility
2. shared transparent-material / shader parity defects that also affect classic `MDX`

Do not treat a fix in one track as proof that the other track is solved.

## Current Working Assumptions

- Pre-release `3.0.1` model files are not safely modeled as later `3.3.5` M2 files.
- Some assets may be transitional or hybrid `MDX` + `M2` variants.
- Empty converted fallback models are already blocked by guardrails; that only removes a false-positive success state.
- Neon-pink transparent surfaces reproduce on both `MDX` and M2-family assets, so that symptom is likely in shared material, texture, blend, or shader handling.

## Scope

- In scope:
  - `WarcraftNetM2Adapter` compatibility assumptions
  - version/profile routing for model parsing
  - `ViewerApp`, `WorldAssetManager`, and `WmoRenderer` M2-family load paths
  - `ModelRenderer` texture binding, transparency routing, and shader behavior
  - replaceable-texture fallback behavior
- Out of scope unless directly required by evidence:
  - terrain alpha issues
  - unrelated UI/layout cleanup
  - broad renderer refactors with no direct impact on the failing model/material path

## Required Principles

- Do not silently broaden generic fallback heuristics when a version/profile split is the real need.
- Do not claim parser success from a syntactically valid but visually empty converted model.
- Do not treat build success or fixture-based tests as proof.
- Prefer runtime evidence from the user’s real data over synthetic examples.

## Required Starting Files

1. `gillijimproject_refactor/src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`
2. `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs`
3. `gillijimproject_refactor/src/MdxViewer/Terrain/WorldAssetManager.cs`
4. `gillijimproject_refactor/src/MdxViewer/Rendering/WmoRenderer.cs`
5. `gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs`
6. `gillijimproject_refactor/src/MdxViewer/Terrain/FormatProfileRegistry.cs`
7. `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/M2ToMdxConverter.cs`

## Investigation Plan

### Track A: Pre-release 3.0.1 Model Compatibility

1. Confirm the failing asset family and build context from runtime evidence.
2. Check whether the asset is entering:
   - direct adapter path
   - converter fallback path
   - classic MDX parser path
3. Compare pre-release container/layout assumptions against later `3.3.5` assumptions:
   - `MD20` / `MD21` root handling
   - skin/submesh expectations
   - geoset/submesh material mapping
   - any headers or chunk layouts that look transitional rather than final-WotLK
4. If behavior differs by client family, route through explicit model profile logic rather than silent heuristics.

### Track B: Shared Pink Transparency / Shader Parity

1. Reproduce or trace the neon-pink transparent surface path on both one classic `MDX` asset and one M2-family asset.
2. Audit shared failure candidates:
   - texture lookup fallback or missing binding
   - replaceable texture resolution
   - blend-mode routing
   - alpha-test vs blended-path selection
   - shader branches for transparent or env-mapped materials
3. Cross-check genuine shader/material documentation before rewriting logic.
4. Keep any renderer fix format-agnostic unless runtime evidence proves it is version-specific.

## Validation Rules

- Final signoff requires runtime-visible evidence on real user data.
- “Build passed” is not signoff.
- “The model no longer crashes” is not signoff.
- “The asset renders but transparency is still neon pink” means Track A may be improved while Track B remains open.

## Deliverables

Return all items:

1. Which track or tracks were confirmed by evidence
2. Exact failing path for the model load
3. Any profile/version split introduced or proposed
4. Any shared shader/material defect confirmed
5. Validation status with explicit runtime limitations
6. Memory-bank updates required after the investigation

## First Output

Start with:

1. active failing symptom summary
2. whether it looks like Track A, Track B, or both
3. exact files to inspect first
4. what runtime evidence is still missing