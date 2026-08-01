# Feature Specification: Diagnose M2 Runtime Animation Failure for Converted MDX

**Feature Branch**: `078-m2-runtime-animation-diagnosis`
**Created**: 2026-06-26
**Status**: Draft
**Input**: User observation—MDX files routed through the M2 runtime conversion path (`LoadChunkedMdxFromBytes` → `MdxToM2Converter` → `M2ModelReader` → `M2StaticRenderModelBuilder` → `M2Renderer` native renderer) display the correct model at bind-pose but do not animate, even though the M2RuntimeAnimator advances frames correctly. WMO doodads (native M2 files through the same `M2Renderer` path) animate fine.

## Context

### Observed behavior

1. Loading a `.mdx` file through the standalone viewer triggers `LoadModelFromBytesWithContainerProbe` which detects MDLX container.
2. Before the fix in this session, it first attempted `LoadChunkedMdxFromBytes` → converts MDX to M2 binary → reads M2ModelDocument → builds M2StaticRenderModel → creates M2Renderer (native renderer, constructor 3: CPU skinning path).
3. The M2RuntimeAnimator advances frames (visible in the Animation Debug panel).
4. The M2BonePoseEvaluator and M2SkinnedRenderModelBuilder run every frame.
5. UploadAnimatedVertices writes new vertex data to the GPU buffer.
6. The model stays at bind-pose — no visible animation.

### What works

- **World MDX instances**: Use `MdxRenderer` (GPU skinning via bone matrix uniforms) at `WorldAssetManager.cs:1297`. Animates correctly.
- **WMO doodads (native M2 files)**: Use `M2Renderer` native renderer (CPU skinning). Animate correctly because they are genuine M2 files, not converted MDX.
- **Standalone M2 files**: Also animate correctly through the M2 native renderer.

### Suspected root cause areas

1. **`MdxToM2Converter`** (located in Core.IO): The conversion from MDX chunk data to M2 binary may produce incorrect bone data, sequence mapping, or vertex bone assignments.
2. **`M2SkinnedRenderModelBuilder.ApplyPose`**: The `ResolveBoneIndex` method at line 113 may compute wrong bone lookups for converted models, causing skinning to use identity/zero matrices.
3. **`M2BonePoseEvaluator.Evaluate`**: May produce identity or incorrect bone matrices for converted sequences.
4. **`UploadAnimatedVertices`**: The section index or vertex count guard conditions at `M2Renderer.cs:589-593` may fail silently, skipping the upload for converted sections.
5. **Vertex data mismatch**: The initial buffer data (bind-pose positions from `M2StaticRenderVertex`) may match the skinned output for converted models due to identity bone transforms.

## User Stories

### US1 — Trace the conversion pipeline
As a developer, I can load an MDX model, run it through `MdxToM2Converter`, and inspect the resulting M2 binary to verify bones, sequences, and skin data match the original MDX.

### US2 — Trace the animation evaluation pipeline
As a developer, I can hook the `M2BonePoseEvaluator` / `M2SkinnedRenderModelBuilder` pipeline for a converted model and verify that:
- `BonePoseState.Matrices` contains non-identity transforms at a known keyframe.
- `SkinnedRenderModel.Sections[i].Vertices[k].Position` differs from `StaticRenderModel.Sections[i].Vertices[k].Position`.

### US3 — Fix the root cause
As a developer, I can identify the specific step where animation breaks and land a fix that makes converted MDX files animate through the M2 native renderer.

## Non-Goals

- Rewriting `MdxToM2Converter` from scratch.
- Changing the M2 native renderer shader or vertex layout.
- Making MDX files prefer the M2 runtime over MdxRenderer (the default path is now MdxRenderer; M2 runtime for converted MDX is a future optimization).
