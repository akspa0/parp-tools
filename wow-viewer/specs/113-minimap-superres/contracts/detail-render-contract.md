# Detail-Render and Alignment Contract

## Detail render contract

1. The detail render MUST sample real BLP texels at the terrain texture UV (bilinear), MCAL-blended
   in file order and MCNR-lit, for the 1024 pass — it MUST NOT return a per-texture average color
   for a detail-mode pixel.
2. The detail render MUST live in `TerrainMinimapCompositor`/`TerrainTextureSampler` as an added
   mode; no second minimap renderer and no duplicate BLP reader may be introduced (constitution II).
3. The 256px minimap render MUST remain material-average (unchanged) — the detail mode applies only
   to the 1024 pass. The two are distinct render intents, not a regression of one into the other.
4. A detail-mode pixel whose texture cannot be decoded MUST fall through the existing honest paths
   (fallback material / skip), never a fabricated texel; a tile with no decodable textures is
   skipped and recorded, never emitted as a flat HR (FR-001).
5. The detail render at 1024 MUST NOT exhibit minimap-scale moire; this is a validated claim
   (SC-001 measures high-frequency energy and a sample is eyeballed), not an assumption.
6. The synthesis manifest MUST record `render_mode=detail` and the texel repeat frequency, so a
   downstream consumer can tell a detail HR from a material-average one.

## Alignment gate contract

1. Before ANY (authored LR, detail HR) pair is assembled, authored↔detail registration MUST be
   measured on a sample of tiles carrying both images (US1). Pairing MUST NOT proceed on assumed
   alignment.
2. Registration MUST search all 8 dihedral transforms plus a small translation and report the best
   transform and residual per tile and in aggregate.
3. The gate passes only as `pass_identity` (identity within tolerance) or `pass_with_transform` (one
   single transform wins for EVERY sampled tile within tolerance). A per-tile-varying best transform
   is `fail_inconsistent`: the images are not a consistent SR pair and the spec HALTS with the
   finding — it does not silently pick per-tile transforms or train on misaligned pairs (SC-002).
4. When `pass_with_transform`, the identified corrective transform MUST be applied consistently (to
   the render or the pairing) and recorded in the pair set's `corrective_transform` attr, so every
   pair is genuinely registered.

## SR training / evaluation contract

1. Training MUST use the real (authored LR, detail HR) pairs directly; it MUST NOT apply a synthetic
   degradation pipeline to fabricate LR (research Decision 4) — the real authored minimap is the
   input the model must learn to upscale.
2. The model MUST be a single-purpose SR generator (RRDBNet family), one output, no multi-task head,
   no weights shared with any terrain-signal model (constitution IV lens).
3. Training MUST be user-executed with explicit per-run go-ahead; tooling prints the exact command
   with an estimate and never launches it (standing rule; FR-007/SC-006). This includes any
   whole-map detail-render pass, which is also a heavy user-run step.
4. Every run MUST write a summary binding the checkpoint to pair-set identity, split, model/loss
   config, and a baseline comparison against at least bicubic(authored LR) on the SC-004 detail
   metric; a checkpoint without this record is not eligible for any promotion decision.
5. Evaluation metrics MUST be restricted to held-out Kalimdor and Azeroth tiles; a request naming
   any other map MUST fail closed or be clearly labeled out-of-scope (FR-009).
6. Stage 2 (GAN fine-tune) is entered only after a user reviews stage 1 (PSNR/L1) outputs — the
   smallest-signal-first discipline; a GAN run is never the first thing trained.
