---
description: "Implement the next pre-release 3.0.1 M2 support slice in MdxViewer using the wow.exe-backed MD20 contract and the current runtime evidence, without assuming the remaining failures are parser-only."
name: "Pre-release 3.0.1 M2 Implementation Plan"
argument-hint: "Optional build string, failing model path, or parser area to prioritize"
agent: "codex"
---

Implement the next pre-release `3.0.1` M2 support slice in `gillijimproject_refactor` using the `wow.exe` contract as the source of truth and the current runtime evidence as the prioritization guide.

## Read First

1. `gillijimproject_refactor/documentation/pre-release-3.0.1-m2-wow-exe-guide.md`
2. `.codex/prompts/pre-release-m2-rendering-recovery.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`

## Goal

Make real implementation progress on the active unresolved pre-release `3.0.1` path by separating three different failure classes instead of collapsing them into "the parser":

1. Track A: core container, profile, and layout compatibility
2. Track B: ancient material, effect, and shader-era feature support that Warcraft.NET does not model well for this build family
3. Track C: truly shared renderer/material parity issues that also reproduce on classic `MDX`

Do not assume Track A is the only remaining blocker. The current evidence says some `3.0.1` assets likely fail because they are effect-heavy or shader-driven objects whose required metadata and runtime behavior are still only partially extracted.

## Non-Negotiable Contract

- Root magic is `MD20`
- Accepted version range is `0x104..0x108`
- Structural split is at `0x108`
- `.mdx` / `.mdl` / `.m2` aliasing is path-level behavior, not proof that `MD21` is valid for `3.0.1`
- Warcraft.NET is not the authority for pre-release `3.0.1` material, effect, or shader semantics
- A model reaching renderable geometry is not proof that its material/effect path is correctly implemented

## Required Principles

- Do not broaden generic Warcraft.NET heuristics and call that support.
- Do not treat converter fallback as parser success.
- Do not claim runtime success from build-only validation.
- Preserve the current empty-geometry guardrail.
- Do not assume pink, opaque, or missing effect-heavy objects are caused by unread geometry.
- Treat objects with transparency, animated materials, environment mapping, ribbons, particles, or other effect-style behavior as first-class capability-gap candidates.
- Prefer comparing one plain opaque asset and one effect-heavy asset from the same client family before choosing the next code slice.

## Suggested Starting Files

1. `gillijimproject_refactor/src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`
2. `gillijimproject_refactor/src/MdxViewer/Terrain/FormatProfileRegistry.cs`
3. `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs`
4. `gillijimproject_refactor/src/MdxViewer/Terrain/WorldAssetManager.cs`
5. `gillijimproject_refactor/src/MdxViewer/Rendering/WmoRenderer.cs`
6. `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/M2ToMdxConverter.cs`
7. `gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs`

## Required Implementation Order

1. Confirm the current failing asset class from runtime evidence:
	- plain opaque object
	- cutout/translucent foliage-style object
	- effect-heavy object (texture animation, env map, shader combo, ribbon, particle, or other special material path)
2. Confirm the current failing path from logs and entry-point routing.
3. If the asset is failing before geometry/material extraction, work Track A first:
	- add or refine the dedicated pre-release parser layer around the confirmed typed span validators
	- implement the `0x104..0x107` legacy family split separately from the `0x108` family
4. If the asset reaches geometry but fails visually, work Track B first:
	- audit which pre-release material/effect fields are missing or flattened by Warcraft.NET
	- recover the minimum missing metadata needed for that asset class before broad renderer rewrites
5. Only after extraction is credible, wire or correct the renderer behavior in `ModelRenderer` for the extracted feature set.
6. Keep logging specific enough that the next failure names the stage: validator, record family, material extraction, texture binding, or shader/effect behavior.

## Track B Checklist

When runtime evidence suggests "objects with effects" or shader-driven assets are the main failures, explicitly inspect and report which of these are still unsupported, stubbed, or flattened:

- material layer stacks and per-batch material assignment
- render flags and blend/depth/cull behavior
- texture lookup/combo tables
- texture transforms or animation tracks
- material color or alpha animation tracks
- environment mapping or shader-combo style flags
- particle, ribbon, or other effect-related attachment data
- any pre-release record families that Warcraft.NET ignores because they only exist in later-era assumptions

## Validation Rules

- Build the changed project.
- If you did not run runtime validation on real `3.0.1` data, say so explicitly.
- If automated tests were not added or run, say so explicitly.
- Do not claim Track B or Track C is fixed unless you actually verify the relevant runtime symptom on real `3.0.1` data.
- If plain opaque assets improved but effect-heavy assets still fail, say that clearly instead of calling the family supported.

## Deliverables

Return all items:

1. Exact on-disk contract implemented
2. Files changed and why
3. Remaining unresolved record families, field mappings, or unsupported effect/material features
4. Validation status, clearly separated into build, tests, and runtime
5. Any memory-bank or guide updates still required

## First Output

Start with:

1. current failing path summary
2. whether the failing asset looks parser/layout-bound, effect/material-bound, or both
3. what part of the `wow.exe` contract is already implemented in code
4. what is still missing
5. the minimal code slice you will change first