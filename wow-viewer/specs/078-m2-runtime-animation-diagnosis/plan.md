# Implementation Plan: Diagnose M2 Runtime Animation Failure for Converted MDX

**Branch**: `078-m2-runtime-animation-diagnosis` | **Date**: 2026-06-26 | **Spec**: `specs/078-m2-runtime-animation-diagnosis/spec.md`

## Summary

Identify why the M2 native renderer (CPU skinning) fails to animate converted MDX models while it works correctly for native M2 files. The diagnosis is done in the app by instrumenting the pipeline at key stages, checking data integrity at each step, and comparing behavior between a working native M2 model and a broken converted MDX model.

## Diagnostic Strategy

The pipeline has these stages. Each must be verified independently:

1. **MDX → M2 conversion** (`MdxToM2Converter`): Does the resulting M2 binary have correct bones, sequences, and skin data?
2. **M2ModelDocument reading** (`M2ModelReader`): Does the round-tripped M2 document match the original MDX data?
3. **M2StaticRenderModel building** (`M2StaticRenderModelBuilder`): Are vertex bone indices/weights correct?
4. **Animation evaluation** (`M2BonePoseEvaluator`): Does `BonePoseState` contain non-identity transforms?
5. **Skinned model building** (`M2SkinnedRenderModelBuilder`): Does `ApplyPose` produce different vertex positions from bind-pose?
6. **Vertex upload** (`UploadAnimatedVertices`): Are the guard conditions met, and does `BufferSubData` actually update the positions?
7. **RenderCore**: Does the shader receive the updated vertex positions?

## Phases

### Phase 1: Compare bone poses for working vs broken model
Load one working native M2 and one broken converted MDX. Instrument `M2BonePoseEvaluator.Evaluate` to dump `BonePoseState.Matrices` (or the first 4x4 matrix) at a known keyframe. If both produce non-identity matrices, the fault is downstream.

### Phase 2: Verify ApplyPose output
If Phase 1 passes, instrument `M2SkinnedRenderModelBuilder.ApplyPose` to compare source vs skinned positions for a few vertices. If source == skinned, `ResolveBoneIndex` or the bone matrix lookup is broken for converted models.

### Phase 3: Check UploadAnimatedVertices guard conditions
If Phase 2 produces different positions, the fault is in `UploadAnimatedVertices`. Log when `section.Source.SectionIndex` is not found in `_sectionsByIndex`, or when vertex counts mismatch.

### Phase 4: Fix the root cause
Once the failing stage is identified, implement the fix. Likely candidates:
- Fix `MdxToM2Converter` bone index remapping.
- Fix `M2SkinnedRenderModelBuilder.ResolveBoneIndex` for converted model layouts.
- Fix `UploadAnimatedVertices` section index matching.

## Testing

Each phase produces a diagnostic log snippet or UI display. The final fix is validated by loading a known-animating MDX model (pre-1.12 era), playing a sequence, and observing visible skeletal movement.
