---
description: "Implement pre-release 3.0.1 M2 support in MdxViewer from the wow.exe-backed MD20 contract instead of later 3.3.5 assumptions."
name: "Pre-release 3.0.1 M2 Implementation Plan"
argument-hint: "Optional build string, failing model path, or parser area to prioritize"
agent: "agent"
---

Implement the pre-release `3.0.1` M2 parser path in `gillijimproject_refactor` using the `wow.exe` contract as the source of truth.

## Read First

1. `gillijimproject_refactor/documentation/pre-release-3.0.1-m2-wow-exe-guide.md`
2. `.github/prompts/pre-release-m2-rendering-recovery.prompt.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`

## Goal

Make real implementation progress on Track A only:

pre-release `3.0.1` model-format compatibility.

Do not let Track B scope creep into this task unless new evidence proves the remaining failure is not parser-related.

## Non-Negotiable Contract

- Root magic is `MD20`
- Accepted version range is `0x104..0x108`
- Structural split is at `0x108`
- `.mdx` / `.mdl` / `.m2` aliasing is path-level behavior, not proof that `MD21` is valid for `3.0.1`

## Required Principles

- Do not broaden generic Warcraft.NET heuristics and call that support.
- Do not treat converter fallback as parser success.
- Do not claim runtime success from build-only validation.
- Preserve the current empty-geometry guardrail.

## Suggested Starting Files

1. `gillijimproject_refactor/src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`
2. `gillijimproject_refactor/src/MdxViewer/Terrain/FormatProfileRegistry.cs`
3. `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs`
4. `gillijimproject_refactor/src/MdxViewer/Terrain/WorldAssetManager.cs`
5. `gillijimproject_refactor/src/MdxViewer/Rendering/WmoRenderer.cs`
6. `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/M2ToMdxConverter.cs`

## Required Implementation Order

1. Confirm the current failing path from logs and entry-point routing.
2. Add or refine a dedicated pre-release parser layer around the confirmed typed span validators.
3. Implement the `0x104..0x107` legacy family split separately from the `0x108` family.
4. Only then revisit skin/submesh/material mapping.
5. Keep logging specific enough that the next failure says which validator or family failed.

## Validation Rules

- Build the changed project.
- If you did not run runtime validation on real `3.0.1` data, say so explicitly.
- If automated tests were not added or run, say so explicitly.
- Do not claim Track B is fixed unless you actually verify the pink transparency symptom.

## Deliverables

Return all items:

1. Exact on-disk contract implemented
2. Files changed and why
3. Remaining unresolved record families or field mappings
4. Validation status, clearly separated into build, tests, and runtime
5. Any memory-bank or guide updates still required

## First Output

Start with:

1. current failing path summary
2. what part of the `wow.exe` contract is already implemented in code
3. what is still missing
4. the minimal code slice you will change first