---
description: "Use the 4.0.0 client's CGx/BLS/WFX evidence to tighten MdxViewer's effect-family model, shader reuse, and material-state stability."
name: "BLS Shader Parity Performance Plan"
argument-hint: "Optional material family, shader path, surface type, or effect symptom to prioritize"
agent: "codex"
---

Implement the BLS/effect parity performance slice in `gillijimproject_refactor/src/MdxViewer` with shader reuse, material-family stability, and reduced state churn as the runtime goal.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
4. `gillijimproject_refactor/documentation/wow-400-engine-performance-recovery-guide.md`
5. `gillijimproject_refactor/reference_data/wowdev.wiki/BLS.md`
6. `gillijimproject_refactor/reference_data/wowdev.wiki/WFX.md`
7. `gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs`
8. `gillijimproject_refactor/src/MdxViewer/Rendering/WmoRenderer.cs`
9. `gillijimproject_refactor/src/MdxViewer/Rendering/ShaderProgram.cs`

## Goal

Move the viewer closer to the real client's effect-family model so material choice is less flattened, shader programs are reused more deliberately, and draw-state churn becomes easier to control.

## Why This Slice Exists

- Ghidra evidence from `4.0.0.11927` shows explicit `ShaderEffectManager`, `CGxShader`, `CShaderEffect`, `CSceneMaterial`, shader directories, and `.bls` shader assets.
- The active viewer still makes many shader/material decisions locally inside specific renderers.
- This is a performance slice as much as a visual one: stable effect-family identity makes batching and program reuse easier.

## Scope

- In scope:
  - material/effect-family descriptors used by the active viewer
  - shader-program reuse and shared state setup
  - stable separation of opaque, cutout, alpha, additive, modulated, env-map, and related families
  - logging/instrumentation that proves which effect family a batch uses
- Out of scope unless direct evidence forces it:
  - full `.bls` runtime execution inside the viewer
  - terrain decode changes
  - speculative deferred-rendering rewrites

## Non-Negotiable Constraints

- Do not claim that “using BLS” means literally replacing the viewer with Blizzard shader binaries.
- Do not flatten distinct material families back into one generic transparent path.
- Do not introduce a broad shader abstraction that changes everything without proving one concrete gain first.
- Keep the change aligned with the current renderer architecture and recent parity work.
- Update memory-bank notes when shader/effect behavior or remaining gaps materially change.

## Required Implementation Order

1. Inventory the current effect-family decisions in the active renderers.
2. Compare them against the real client's visible effect families from `WFX`/`BLS` references.
3. Identify one confirmed flattening seam with runtime significance.
4. Land the smallest shared effect-family or program-reuse slice that improves that seam.
5. Keep logging specific enough to show which batches use which effect family.
6. Update memory-bank notes with the exact family behavior now supported versus still approximated.

## Investigation Checklist

- Verify where `ModelRenderer` and `WmoRenderer` duplicate or diverge in state setup.
- Verify whether current `ShaderProgram` usage already supports a shared effect layer or needs one tiny adapter first.
- Cross-check `WFX` pass types and known blend/effect families before renaming or regrouping local material families.
- Prefer one visible, engine-backed family improvement over a giant shader-system rewrite.

## Validation Rules

- Build the changed viewer solution.
- If runtime validation on real data is not run, say so explicitly.
- If automated tests are not added or run, say so explicitly.
- Do not claim BLS parity unless the delivered slice clearly states what was mirrored versus what remains local approximation.

## Deliverables

Return all items:

1. exact effect/material behavior implemented
2. files changed and why
3. what families now map more faithfully versus what is still approximate
4. build status
5. automated-test status
6. runtime-validation status
7. memory-bank updates made

## First Output

Start with:

1. the current material/effect flattening seam with the best performance value
2. what `CGx`/`BLS`/`WFX` evidence supports the chosen slice
3. what files will be changed first
4. what you are explicitly not trying to do in this pass