---
description: "Task list for diagnosing M2 runtime animation failure for converted MDX"
---

# Tasks: Diagnose M2 Runtime Animation Failure for Converted MDX

**Input**: Spec and plan from `/specs/078-m2-runtime-animation-diagnosis/`
**Prerequisites**: plan.md ✅, spec.md ✅

**Path conventions**: `wow-viewer/` is the repo root. All paths below are relative to `wow-viewer/`.

---

## Phase 1: Instrument bone pose output

**Goal**: Determine if `M2BonePoseEvaluator` produces non-identity bone transforms for a converted MDX model.

- [ ] T001 Add diagnostic logging to `M2Renderer.UpdateAnimation()` to dump the first 3 bone matrices from `bonePoseState.Matrices` (or log count and whether any differ from identity). Log at sequence start and at mid-sequence.
- [ ] T002 Load a native M2 model (e.g. a creature that animates) and verify the diagnostic shows non-identity bone matrices.
- [ ] T003 Load a converted MDX model and compare: if bone matrices are identity or all-zero, root cause is in `M2BonePoseEvaluator` or `MdxToM2Converter` sequence data.

## Phase 2: Instrument ApplyPose output

**Goal**: Determine if `M2SkinnedRenderModelBuilder.ApplyPose` produces vertex positions that differ from bind-pose.

- [ ] T004 Add diagnostic to dump position of first vertex in first section of `skinnedRenderModel` vs `sourceSection.Vertices[0].Position` after `ApplyPose` but before `UploadAnimatedVertices`.
- [ ] T005 If positions are identical for converted MDX but differ for native M2, root cause is in `ResolveBoneIndex()` or bone weight lookup for converted models.

## Phase 3: Check UploadAnimatedVertices guard conditions

**Goal**: Determine if the vertex upload is silently skipped.

- [ ] T006 Add logging to `UploadAnimatedVertices` when a section is skipped: log `section.Source.SectionIndex`, whether found in `_sectionsByIndex`, and vertex count comparison.
- [ ] T007 Check if `_staticSectionsByIndex` is populated from the same data as `_sectionsByIndex`. Verify `SectionIndex` values match between `M2StaticRenderModel.Sections` and `M2SkinnedRenderModel.Sections`.

## Phase 4: Fix

**Goal**: Land the minimal fix and verify against real data.

- [ ] T008 Implement the fix based on diagnostic findings.
- [ ] T009 Validate by loading a converted MDX model, playing an animation sequence, and observing correct skeletal animation. Compare before/after with screenshots.
- [ ] T010 Document findings in `docs/architecture/m2-mdx-converter-animation-diagnosis-2026-06-26.md`.
