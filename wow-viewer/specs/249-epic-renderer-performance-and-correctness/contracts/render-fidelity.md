# Renderer fidelity contracts

Local adapters/evidence only; no HTTP/GraphQL surface or new parser API is required.

## Shared boundaries
- Existing decoded documents, scene ownership, camera, loader and visibility remain authoritative.
- Policy functions receive explicit build and input metadata, return explicit outcome/fallback
  provenance, and can be tested without GL. Viewer service owns material/pass submission.
- No new app/scene god-class members; no oversized source growth. New behavior uses owned
  services/adapters; existing delegation may be replaced narrowly.
- Readers are not a default write seam. Real-file contradiction plus explicit operator approval
  is required to open a verified reader bug; renderer mismatch alone is insufficient.

## Ground effects (adopted by Spec 259)
- Keep placement density, world-distance rejection and rendered opacity separate observations.
- Fade is propagated through the actual native/compatibility route; a shared material alteration
  cannot be justified only by flora screenshots without checking unaffected model paths.
- Do not restore the reverted per-instance tint API/load regression as a side effect of fade work.

## Liquids
- WMO group/material/nibble and MH2O family/object/layout identifiers are not interchangeable.
- Preserve raw identity and table-miss reason through diagnostics; LiquidMaterial vertex format
  resolution does not assert native textures/shader parameters.
- Live and harvest decoder behavior must be compared for the same identified payload. Mixed-layer
  reduction is a distinct adapter decision; do not assume it is a format read failure.

## Sky
- Exact-build database choice and resolved asset must be observable independently of pass visibility.
- Preserve blank/simple-viewer mode and existing visual-control ownership; no permanent duplicate
  sidebar. Known-good entries require actual row/asset evidence, not schemas or guessed names.
- Reference source is cited by actual build/file/symbol/revision; unavailable decomp tools block
  source-parity claims, not ordinary repository code inspection.

## Signoff
- One correction, one bounded write set, focused tests plus real baseline/after criterion receipts.
- Operator owns client/GPU/visual/FPS runs. Planning, compilation and synthetic tests do not count
  as those outputs. Missing evidence remains visibly blocked.
