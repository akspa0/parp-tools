# Hypotheses: MSUR Height Scaling

1) Height is plane constant but stored in scaled units; needs multiplication by a scale factor S.
   - Candidates: S = 1/36 (~0.02777778) or S = 1/16 (0.0625).
2) Surfaces may be sub-divided patches with consistent N but H stepping by scaled increments.
3) Explosion artifacts arise from using H without correct scaling or normalization.

Validation: compare residuals r = dot(N, v) − H' before/after snapping and inspect geometry.
