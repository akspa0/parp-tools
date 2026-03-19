---
description: "Triage post-change runtime failures for pre-release 3.0.1 M2 support and decide whether the remaining bug is parser/layout work or shared transparency/material work."
name: "Pre-release 3.0.1 M2 Runtime Triage"
argument-hint: "Optional failing model path, log excerpt, or screenshot clue"
agent: "agent"
---

Use this prompt after code changes when a pre-release `3.0.1` model still fails, partially loads, or renders with bad transparency.

## Read First

1. `gillijimproject_refactor/documentation/pre-release-3.0.1-m2-wow-exe-guide.md`
2. `.github/prompts/pre-release-m2-rendering-recovery.prompt.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`

## Goal

Decide which track the current runtime failure belongs to:

1. Track A: pre-release `3.0.1` parser or layout compatibility
2. Track B: shared texture, material, blend, or shader parity

## Required Inspection Files

1. `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs`
2. `gillijimproject_refactor/src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`
3. `gillijimproject_refactor/src/MdxViewer/Terrain/WorldAssetManager.cs`
4. `gillijimproject_refactor/src/MdxViewer/Rendering/WmoRenderer.cs`
5. `gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs`

## Required Triage Steps

1. Identify the exact failure stage:
   - path resolution
   - container gate
   - typed validator
   - skin/submesh mapping
   - converter fallback
   - texture lookup
   - transparency routing
   - shader behavior
2. State whether the model reached renderable geometry.
3. State whether the remaining issue reproduces on classic `MDX`, M2-family assets, or both.
4. Keep build-only evidence separate from runtime evidence.

## Deliverables

Return all items:

1. Exact failure stage
2. Whether the remaining issue is Track A, Track B, or both
3. Concrete next code slice to change
4. Validation status and missing runtime evidence
5. Any guide or memory-bank updates required

## First Output

Start with:

1. the observed runtime symptom
2. the most likely failure stage
3. the first file you will inspect