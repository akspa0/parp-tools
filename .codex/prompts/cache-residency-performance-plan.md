---
description: "Tighten cache and residency policy across terrain, models, previews, and textures so MdxViewer stops paying duplicate decode/upload cost for the same assets."
name: "Cache Residency Performance Plan"
argument-hint: "Optional cache symptom, texture family, map, or renderer to prioritize"
agent: "codex"
---

Implement the cache/residency performance slice in `gillijimproject_refactor/src/MdxViewer` with duplicate decode, upload, and eviction churn as the runtime goal.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/memory-bank/data-paths.md`
4. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
5. `gillijimproject_refactor/documentation/wow-400-engine-performance-recovery-guide.md`
6. `gillijimproject_refactor/src/MdxViewer/Terrain/WorldAssetManager.cs`
7. `gillijimproject_refactor/src/MdxViewer/Terrain/TerrainRenderer.cs`
8. `gillijimproject_refactor/src/MdxViewer/Rendering/MinimapRenderer.cs`
9. `gillijimproject_refactor/src/MdxViewer/Terrain/WdlPreviewCacheService.cs`

## Goal

Reduce duplicated in-memory and GPU residency for textures and scene assets without reintroducing the older eviction bugs that made objects disappear.

## Why This Slice Exists

- The 4.0.0 client exposes explicit texture-cache, BSP-cache, and streamed-vs-cached size behavior in the binary.
- The active viewer still has multiple local caches with different policies and limited cross-renderer coordination.
- Performance work here should improve real reuse, not just increase cache size and hope for the best.

## Scope

- In scope:
  - shared versus renderer-local texture residency
  - raw-byte versus decoded-texture cache boundaries
  - eviction policy and cache statistics
  - duplicate upload avoidance across terrain/model/minimap/preview paths
- Out of scope unless direct evidence forces it:
  - archive decompression changes
  - broad shader/effect rewrites
  - terrain-format decode rewrites

## Non-Negotiable Constraints

- Do not reintroduce bounded renderer eviction for live world placements unless runtime evidence proves it is safe.
- Do not silently keep multiple identical GL textures alive just because caches live in different renderer classes.
- Do not hide cache policy behind undocumented magic numbers.
- Keep the change minimal and explicit; this is a residency slice, not a renderer rewrite.
- Update memory-bank notes when cache behavior or known risks materially change.

## Required Implementation Order

1. Inventory which caches currently own raw bytes, decoded data, and GL handles.
2. Identify one confirmed duplication seam that affects runtime cost.
3. Land the smallest reuse or residency-policy improvement that removes that duplication.
4. Add visible counters or logging where the benefit would otherwise be impossible to verify.
5. Re-check that previously fixed disappearance/regression paths stay intact.
6. Update memory-bank notes with the exact policy now in force.

## Investigation Checklist

- Verify whether terrain, minimap, and model paths can reuse the same decoded or uploaded texture data.
- Verify whether cache keys normalize casing and path aliases consistently.
- Check whether current caches expose enough counters to prove reuse and eviction behavior.
- Check whether any cache still writes temporary files where in-memory reuse would be safer and cheaper.
- Prefer one real duplication seam over a broad speculative refactor.

## Validation Rules

- Build the changed viewer solution.
- If runtime validation on real data is not run, say so explicitly.
- If automated tests are not added or run, say so explicitly.
- Do not claim memory or GPU improvement unless you have direct counters, instrumentation, or real runtime evidence.

## Deliverables

Return all items:

1. exact cache/residency behavior changed
2. files changed and why
3. what duplication seam was reduced
4. build status
5. automated-test status
6. runtime-validation status
7. memory-bank updates made

## First Output

Start with:

1. the current duplication or residency seam with the highest value
2. which caches currently own the same data in different forms
3. what files will be changed first
4. what existing regression risk you will protect during the pass