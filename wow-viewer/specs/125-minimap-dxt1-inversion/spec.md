# Feature Specification: Minimap DXT1 Artifact Inversion

**Feature Branch**: `125-minimap-dxt1-inversion`

**Created**: 2026-08-02

**Status**: Draft

**Input**: User description: "Authored 0.5.3 minimaps are DXT1-compressed. Our compositor produces pristine 24-bit output, so it has been building *better than real* minimaps — every comparison scored a clean image against a lossy one and blamed the renderer. Two consequences: synthetic output must carry the same compression for any fair comparison or dataset parity, and a learned inverse of the DXT1 degradation could restore authored tiles toward their pre-compression appearance."

**Update (2026-08-02)**: Two additions from the user. First, a hypothesis: the global tile lighting may be
**normalised across all minimap tiles** — i.e. the authored tiles of a map may share a common lighting
baseline rather than each tile carrying its own independent exposure. If true, this is a second,
independent source of mismatch on top of the codec, and it must be verified and accounted for before
any per-tile comparison or calibration is trusted. Second, a concrete requirement: the synthesizer
MUST also emit a **DXT1-compressed variant** of each tile alongside the pristine render, so a synthetic
tile can be compared against a real authored tile on equal terms without a separate comparison-time
encode step. Together these sharpen the restoration avenue: if the compression method is reproduced,
restoring authored minimaps back toward their raw pre-compression form becomes a tractable layer of
terrain reconstruction.

## Background

Measured 2026-08-02 across 16 tiles of 0.5.3.3368 / Azeroth: every authored minimap is
`BLP2 / DXTC / DXT1`, 256×256, one mip level, no palette, no alpha.

DXT1 encodes each 4×4 pixel block as two RGB565 endpoint colours plus 2-bit per-pixel indices, so a
block holds **at most four colours**, all on a straight line between its endpoints. Measured unique
colours per 65,536-pixel tile: 1,196–5,269 (median 3,201) — consistent with 4,096 blocks × ≤4 colours
and far below a full-colour render. Smooth terrain gradients band to four steps per block with
visible seams: the "spray-can" texture.

RGB565 also quantises green at 6 bits against red and blue at 5 — steps of 4 versus 8. Any
channel-ratio comparison against authored therefore carries a systematic red/blue bias that belongs
to the codec, not to art direction.

Two things follow. Our synthetic minimaps are *higher fidelity than the source they are compared
against*, which biases every comparison metric and would teach any model trained on both corpora to
distinguish them by codec damage alone. And because the degradation is a known, reproducible
transform, its inverse can be learned from pairs we generate ourselves.

A second, independent source of mismatch is suspected: **global lighting normalisation**. The authored
tiles of a map may share a common lighting baseline — the client's minimap renderer may normalise
brightness/contrast across all tiles of a map rather than leaving each tile at its own raw exposure.
If that is true, then per-tile lighting differences between our synthesizer and the authored tiles are
not purely codec damage; they are a separate systematic offset that must be measured and removed
before any per-tile comparison or calibration decision is trusted. This hypothesis is unverified and
is a first-class deliverable of this feature, not an assumption to inherit.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Compare synthetic against authored on equal terms (Priority: P1)

A researcher comparing a synthesized minimap tile against its authored counterpart gets a score that
reflects how well the terrain was reconstructed, not how much the reference was damaged by its codec.
The comparison applies the authored tile's own encoding to the synthetic tile before measuring.

**Why this priority**: Every existing comparison number is confounded. Until this is fixed, no
calibration decision — water colour, ambient, shadow strength — can be trusted, and the project has
already spent several rounds tuning against a moving target. This also unblocks every later story by
establishing what "matching" means.

**Independent Test**: Score a set of authored tiles against synthetic renders with and without
encoding parity applied, and confirm the parity run reports a materially different (and better)
agreement while the relative ranking of two deliberately different render settings is preserved.

**Acceptance Scenarios**:

1. **Given** an authored tile known to be DXT1 and a pristine synthetic render of the same tile,
   **When** the comparison runs with encoding parity enabled,
   **Then** the report states the encoding applied and shows the parity-adjusted agreement alongside
   the unadjusted one.
2. **Given** the same inputs, **When** parity is enabled,
   **Then** the per-channel red/blue bias attributable to RGB565 is reduced relative to the
   unadjusted comparison.
3. **Given** an authored tile whose encoding is *not* DXT1,
   **When** the comparison runs,
   **Then** the tile is scored with its own encoding and the report names it, rather than assuming
   DXT1.
4. **Given** a tile whose authored image is a single flat colour (an unrendered tile),
   **When** the comparison runs,
   **Then** the tile is excluded from aggregate scores and reported as excluded.
5. **Given** a synthesized tile and its DXT1-compressed parity companion,
   **When** the comparison runs against an authored tile,
   **Then** it uses the parity companion directly, with no separate encode step at comparison time.
6. **Given** a set of authored tiles from the same map,
   **When** the lighting-baseline test runs,
   **Then** the report states whether a shared lighting baseline exists, and if so the comparison
   accounts for it rather than attributing the offset to codec damage.

---

### User Story 2 — Generate a corpus free of codec confound (Priority: P2)

Someone building a training corpus that mixes authored and synthesized minimap tiles can produce
both sides with matching encoding characteristics, so a model cannot separate them by compression
damage.

**Why this priority**: A model that learns "authored tiles are blocky, synthetic tiles are smooth"
will exploit that shortcut instead of learning terrain. This is a real risk for every mixed corpus
already generated, but it depends on the parity mechanism from Story 1, so it follows it.

**Independent Test**: Build a small mixed corpus with parity enabled, then train a trivial classifier
to predict which source each tile came from; confirm it performs near chance, and that the same
classifier on a non-parity corpus performs well above chance.

**Acceptance Scenarios**:

1. **Given** a corpus generation run with parity enabled,
   **When** the corpus is written,
   **Then** each row records the encoding applied to it and whether it was authored or synthesized.
2. **Given** a previously generated corpus without parity,
   **When** its manifest is inspected,
   **Then** the absence of parity is visible rather than silently assumed.

---

### User Story 3 — Restore an authored tile toward its pre-compression appearance (Priority: P3)

A restorer takes an authored minimap tile and recovers an image closer to what the client's renderer
produced before DXT1 quantised it — block banding reduced, seams removed, colour bias corrected.

**Why this priority**: This is the payoff, but it depends on the encoder from Story 1 to generate its
training pairs, and it is the only story that can fail on quality grounds rather than correctness
grounds. It also carries the highest risk of producing convincing but invented detail.

**Independent Test**: Hold out a set of pristine renders never seen in training, encode them, run the
restoration, and measure recovery against the known pre-encoding originals — ground truth that exists
without needing any authored image.

**Acceptance Scenarios**:

1. **Given** a pristine image encoded and then restored,
   **When** the result is compared to the original,
   **Then** it is measurably closer to the original than the encoded input is, on both colour error
   and block-seam metrics.
2. **Given** an authored tile with no known original,
   **When** it is restored,
   **Then** re-encoding the restored image reproduces the authored tile within a stated tolerance —
   demonstrating the restoration is consistent with the observed data rather than invented.
3. **Given** an input image that is not DXT1-damaged,
   **When** it is passed to the restoration,
   **Then** it is returned substantially unchanged rather than having artefacts removed that were
   never there.
4. **Given** a restored tile,
   **When** the hallucination gate runs,
   **Then** any region whose restored detail is not supported by the input is flagged, and the run
   reports the flagged fraction.

---

### User Story 4 — Decode terrain shadow from any authored minimap and reconstruct terrain directly (Priority: P3)

**Core hypothesis (user, 2026-08-02)**: the terrain shadow/residual is effectively the **heightmap
encoded in grayscale in every minimap tile** — the signal hiding in plain sight. Because we now know
how the minimap compositor and bake functionality work, the shadow in an authored tile is not a black
box; it is a readable, near-grayscale encoding of the terrain's shape. Once the residual tiles are
gathered (US6), comparing them against the ground-truth heightmap (MCVT) should reveal the REAL
residual, and training convergence on that signal should regenerate heightmap data from any minimap —
solely because we know how the whole thing was originally built. This is the holy grail of minimap
baking that no one has gotten quite right.

A reconstruction engineer takes *any* authored minimap tile and recovers the terrain that produced
it — going **minimap RGB → heightmap → 3D mesh** with a single model that reads the terrain shadow
and converts it into ridges, mountains, and terrain detail. This is the payoff of the whole lighting
line: because we now know how the minimap terrain shadow is created (the synthesizer's lighting
model), the shadow in an authored tile is no longer a black box — it is a readable signal that
encodes the terrain's shape.

**Why this priority**: This is the strategic goal the user stated on 2026-08-02. It depends on the
parity mechanism (Story 1) and the lighting model, and it is the only story that can fail on quality
grounds. It is deliberately the last story because it consumes everything before it. It also
reframes the restoration story: the best residual to train against is the decoded terrain shadow
itself, not just the pre-compression appearance.

**Independent Test**: Hold out authored tiles never seen in training, decode their terrain shadow,
run the reconstruction, and measure the recovered heightmap against the known ground-truth heightmap
(MCVT) for those tiles — ground truth that exists without needing any authored image.

**Acceptance Scenarios**:

1. **Given** an authored minimap tile, **When** the terrain-shadow decoder runs,
   **Then** it produces a shadow field consistent with the synthesizer's lighting model (same solar
   direction, same ambient/cast-shadow semantics).
2. **Given** a decoded shadow field, **When** the reconstruction model runs,
   **Then** it produces a heightmap whose relief correlates with the ground-truth MCVT heightmap for
   the same tile, measured on held-out tiles.
3. **Given** a reconstructed heightmap, **When** it is meshed,
   **Then** the resulting 3D mesh is a plausible terrain surface (no inverted normals, no
   disconnected spikes) and matches the authored minimap's shading when re-lit with the synthesizer's
   lighting model.
4. **Given** an authored tile whose shadow is ambiguous (flat terrain, no visible relief),
   **When** the reconstruction runs,
   **Then** it reports low confidence rather than inventing ridges.

---

### User Story 5 — Super-resolve terrain and texturing data from real low/high-res pairs (Priority: P3)

A reconstruction engineer upscales terrain and texturing data using a super-resolution model trained
on real low-res/high-res pairs. Because the synthesizer can now produce both the low-res and high-res
versions of the same terrain perfectly, from real data, without objects, the training pairs are exact
and object-free — ideal for learning to upscale terrain and texturing data specifically.

**Why this priority**: The user stated this on 2026-08-02 as another door opened by the parity and
lighting work. It depends on the synthesizer's ability to render the same terrain at multiple
resolutions with matching lighting, and it is separable from artifact removal (FR-012 keeps them
apart). It is a distinct model from restoration and reconstruction.

**Independent Test**: Hold out high-res renders never seen in training, downscale them to low-res,
run the super-resolution model, and measure recovery against the known high-res originals — ground
truth that exists without needing any authored image.

**Acceptance Scenarios**:

1. **Given** a low-res terrain/texture render, **When** the super-resolution model runs,
   **Then** it produces a high-res output measurably closer to the known high-res original than
   bicubic upscaling is, on held-out pairs.
2. **Given** a low-res render with no objects (terrain and texturing only),
   **When** the model runs,
   **Then** it upscales terrain and texturing detail without inventing object-like artefacts.
3. **Given** a super-resolved output, **When** it is compared to the high-res original,
   **Then** the improvement is reported separately from any artifact-removal or reconstruction metric
   (FR-012).

---

### User Story 6 — Export textureless terrain-shadow residuals (per-tile + stitched) (Priority: P2)

A dataset builder exports, for every map, tiles that are **just the terrain shadow residual** — the
shading signal with no objects and no textures — so a residuals-based terrain reconstruction model can
train on the cleanest possible signal. Output goes to both image files on disk and the v50 Zarr
datastore as a "textureless residuals" signal. The MCAL/MCLY/MTEX data is encoded as separate layers
so per-tileset identity is preserved.

**Why this priority**: The user stated this on 2026-08-02. Because we reverse-engineered the minimap
compositor and bake functionality, we can now emit the pure terrain-shadow residual — the exact signal
that encodes terrain shape — and use it alone to generate terrain from images. This is the cleanest
training target for the residuals-based reconstruction model (US4), and it feeds the super-resolution
model (US5) for the low-res 2001–2003 imagery used to restore the game to its pre-customer era.

**Independent Test**: Export textureless residuals for a map, confirm each tile is shading-only (no
albedo texture, no objects), and confirm the stitched whole-map output aligns with the per-tile
outputs.

**Acceptance Scenarios**:

1. **Given** a map, **When** the textureless-residual export runs,
   **Then** it emits a per-tile shading-only image (no objects, no textures) for every occupied tile.
2. **Given** the same map, **When** the export runs with stitching,
   **Then** it emits a stitched whole-map textureless-residual image aligned with the per-tile outputs.
3. **Given** the export, **When** the v50 Zarr datastore is written,
   **Then** the textureless-residual signal is stored as a named signal, and MCAL/MCLY/MTEX are
   encoded as separate per-tileset layers.
4. **Given** a textureless-residual tile, **When** it is inspected,
   **Then** it contains no albedo texture and no object pixels — only the terrain shadow.

---

### Edge Cases

- An authored tile that is a single flat colour (unrendered). Must be excluded from aggregates, not
  averaged in — this already flipped one diagnostic's verdict.
- A build or map whose minimaps are **not** DXT1. Encoding must be detected per file, never assumed
  from the 0.5.3 Azeroth measurement.
- Tiles whose authored encoding differs *within* a single map.
- A restoration model applied to an image at a resolution or encoding it was not trained on.
- DXT1's alpha-punchthrough mode, where a block encodes three colours plus transparency, behaves
  differently from the four-colour mode. Authored tiles measured so far are `alpha0`, but the encoder
  must handle both rather than assuming opaque.
- Re-encoding a restored image will not reproduce the source bit-exactly, because the encoder that
  produced the authored tiles is not ours. Tolerance must be stated, not assumed to be zero.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST detect each authored minimap tile's actual encoding (format,
  compression mode, colour depth, dimensions) from the file rather than assuming a build-wide default.
- **FR-002**: The system MUST be able to apply the same lossy encoding-and-decoding cycle to a
  synthesized minimap tile that the authored tile carries, producing an image with equivalent
  degradation characteristics.
- **FR-003**: Comparison reporting MUST offer parity-adjusted agreement alongside unadjusted
  agreement, and MUST state which encoding was applied.
- **FR-004**: Comparison reporting MUST exclude degenerate tiles (single flat colour) from aggregate
  statistics and report how many were excluded.
- **FR-005**: The system MUST record, for every generated corpus row and every comparison report,
  which encoding parity was applied — including "none".
- **FR-006**: Encoding behaviour MUST be resolved per build era, consistent with the project's
  existing era-gating, and an unrecognised build MUST be flagged rather than silently defaulted.
- **FR-007**: The system MUST generate training pairs for restoration by taking pristine images,
  applying the encoding cycle, and retaining the pre-encoding image as ground truth — requiring no
  authored reference.
- **FR-008**: The restoration MUST be evaluated on held-out pristine images whose originals are
  known, reporting improvement over the un-restored encoded input.
- **FR-009**: The restoration MUST be verifiable against authored tiles by re-encoding its output and
  measuring agreement with the authored source, within a stated tolerance.
- **FR-010**: The restoration MUST report a hallucination measure identifying detail not supported by
  its input, and MUST NOT be promoted without that measure meeting a stated gate.
- **FR-011**: The restoration MUST leave an input that carries no block-compression damage
  substantially unchanged.
- **FR-012**: Restoration at native resolution MUST be separable from any resolution increase, and
  the two MUST NOT be evaluated by a shared metric.
- **FR-013**: The system MUST report, per build and map inspected, the distribution of encodings
  found, so the DXT1 assumption is verified rather than inherited.
- **FR-014**: The encoder MUST pass a round-trip sanity check — decode an authored tile, re-encode it,
  and confirm the result closely reproduces the authored bytes. This is a correctness test on *our*
  encoder, exploiting DXT1's near-idempotency on decoded data. It is explicitly **not** an attempt to
  identify which encoder Blizzard used; close enough for the degradation to match is the bar.
- **FR-015**: The synthesizer MUST emit, for every tile it renders, a **DXT1-compressed variant**
  alongside the pristine render, so a synthetic tile can be compared against a real authored tile on
  equal terms without a separate comparison-time encode step. The pristine render remains the primary
  output; the compressed variant is a parity companion.
- **FR-016**: The system MUST test the **global lighting normalisation** hypothesis — whether the
  authored tiles of a map share a common lighting baseline — and report the result per map and build.
  If a shared baseline is found, comparison and calibration MUST account for it rather than treating
  per-tile differences as codec damage alone.
- **FR-017**: The system MUST be able to **decode the terrain shadow** from an authored minimap tile
  using the synthesizer's lighting model (solar direction, ambient, cast-shadow semantics), producing
  a shadow field that is a readable terrain-shape signal rather than a black box.
- **FR-018**: The system MUST be able to **reconstruct a heightmap from a decoded shadow field** and
  mesh it into a 3D terrain surface, going minimap RGB → heightmap → 3D mesh with a single model.
- **FR-019**: The reconstruction MUST be evaluated on held-out authored tiles against the known
  ground-truth heightmap (MCVT) for those tiles, reporting relief correlation and mesh plausibility.
- **FR-020**: The reconstruction MUST report low confidence on ambiguous tiles (flat terrain, no
  visible relief) rather than inventing ridges.
- **FR-021**: The system MUST be able to **super-resolve** terrain and texturing data from real
  low-res/high-res pairs produced by the synthesizer (same terrain, matching lighting, no objects),
  and MUST report the improvement separately from any artifact-removal or reconstruction metric.
- **FR-022**: The system MUST be able to **export textureless terrain-shadow residuals** — per-tile
  shading-only images (no objects, no textures) for every occupied tile, plus a stitched whole-map
  output — to both image files on disk and the v50 Zarr datastore as a named "textureless residuals"
  signal.
- **FR-023**: The textureless-residual export MUST encode MCAL/MCLY/MTEX data as separate per-tileset
  layers, so per-tileset identity is preserved alongside the shading signal.

### Key Entities

- **Tile Encoding Profile**: What a specific authored tile is actually encoded as — container format,
  compression mode, colour depth, alpha handling, dimensions, mip count. Detected per file.
- **Encoding Parity Pair**: A synthesized tile and the same tile after the authored encoding cycle has
  been applied, used to make comparison fair.
- **Restoration Training Pair**: A pristine image and its encoded counterpart, where the pristine
  image is ground truth. Generated locally; no authored image required.
- **Restoration Verdict**: For one restored tile — improvement over encoded input, agreement after
  re-encoding, and the hallucination measure.
- **Era Encoding Survey**: Per build and map, the distribution of encodings observed, and whether the
  build was recognised.
- **Lighting Baseline**: The common brightness/contrast normalisation shared across the authored tiles
  of a map, if one exists. Measured per map and build; used to separate lighting-baseline offset from
  codec damage in comparison and calibration.
- **Parity Companion**: The DXT1-compressed variant of a synthesized tile, emitted alongside the
  pristine render so authored and synthetic tiles can be compared on equal terms.
- **Decoded Shadow Field**: The terrain-shadow signal recovered from an authored minimap tile using
  the synthesizer's lighting model — a readable terrain-shape signal rather than a black box.
- **Reconstructed Heightmap**: The heightmap recovered from a decoded shadow field, evaluated against
  the ground-truth MCVT heightmap for the same tile.
- **Super-Resolution Pair**: A low-res and high-res render of the same terrain with matching lighting
  and no objects, produced by the synthesizer; used to train and evaluate super-resolution.
- **Textureless Residual**: A per-tile shading-only image (no objects, no textures) that encodes the
  terrain shadow; the cleanest training signal for residuals-based reconstruction. Stored per-tile and
  stitched whole-map, with MCAL/MCLY/MTEX as separate per-tileset layers.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For every tile in an evaluation set, the comparison reports agreement both with and
  without encoding parity, and states the encoding used.
- **SC-002**: With parity applied, the systematic red/blue channel bias against authored tiles is
  reduced by at least half relative to the unadjusted comparison, confirming the bias was codec-borne.
- **SC-003**: A classifier trained to distinguish authored from synthesized tiles in a parity-matched
  corpus performs within 10 percentage points of chance, against well above chance on a
  non-parity corpus.
- **SC-004**: On held-out pristine images, restoration reduces colour error against the original by at
  least 25% relative to the encoded input, and reduces block-seam discontinuity to within 15% of the
  original's.
- **SC-005**: Re-encoding a restored authored tile reproduces the authored source within a stated
  tolerance for at least 90% of evaluated tiles.
- **SC-006**: Restoration applied to undamaged input changes it by less than 2% colour error.
- **SC-007**: The encoding survey covers at least three maps across at least two build eras, and
  every tile's encoding is reported rather than assumed.
- **SC-008**: Every corpus row and comparison report generated after this feature states its parity
  status, with no row defaulting silently.
- **SC-009**: Decoding an authored tile and re-encoding it reproduces the authored bytes for at least
  95% of blocks — a correctness check on our own encoder, not an encoder-identification exercise.
- **SC-010**: Every synthesized tile has a DXT1-compressed variant available, and a parity comparison
  against an authored tile can be run using that variant without a separate encode step.
- **SC-011**: The global lighting normalisation hypothesis is tested across at least two maps and the
  result (shared baseline present or absent) is reported per map; where a baseline is found, the
  parity comparison accounts for it and the residual per-tile agreement improves.
- **SC-012**: On held-out authored tiles, the decoded terrain shadow is consistent with the
  synthesizer's lighting model (solar direction, ambient, cast-shadow semantics) for at least 90% of
  tiles.
- **SC-013**: On held-out authored tiles, the reconstructed heightmap's relief correlates with the
  ground-truth MCVT heightmap at a level materially above chance, and the meshed surface is plausible
  (no inverted normals, no disconnected spikes).
- **SC-014**: The reconstruction reports low confidence on at least 90% of ambiguous (flat, no-relief)
  tiles rather than inventing ridges.
- **SC-015**: On held-out low-res/high-res pairs, the super-resolution output is measurably closer to
  the known high-res original than bicubic upscaling is, and the improvement is reported separately
  from any artifact-removal or reconstruction metric.
- **SC-016**: The textureless-residual export produces a per-tile shading-only image and a stitched
  whole-map image for every occupied tile of a map, with no albedo texture and no object pixels, and
  stores the signal in the v50 Zarr datastore with MCAL/MCLY/MTEX as separate per-tileset layers.

## Assumptions

- The existing minimap comparison and corpus-generation paths are the integration points; this
  feature adds encoding awareness to them rather than replacing them.
- The BLP container format and DXT1 *decoding* are fully public and already solved in this codebase —
  `SereniaBLPLib` is referenced and wrapped by the existing BLP reader. DXT1 decoding is deterministic,
  so any correct decoder yields identical pixels; there is nothing to reverse-engineer on the read side.
- **The decode half exists; the encode half does not.** `SereniaBLPLib` is decode-only — its entire
  DXT surface is `DXTDecompression.DecompressImage(...)`. Supplying a DXT1 encoder is the one new
  component this feature requires.
- DXT1 *encoding*, by contrast, is a lossy fitting problem with many valid implementations that
  produce different bits for the same input. Which specific encoder Blizzard's 2003 art pipeline used
  is not determined by having a decoder. **However, this matters far less than it first appears** (see
  FR-014): DXT1 re-encoding is near-idempotent on already-decoded data, because a decoded block's four
  colours already lie on a line between two exactly-RGB565-representable endpoints, so any competent
  encoder recovers those same endpoints. Encoder choice is therefore low-risk for parity and for the
  re-encode check in FR-009, and only materially affects training-pair generation from pristine
  renders, where continuous-tone input gives encoders real freedom in endpoint fitting.
- Restoration is a learned prior over a many-to-one transform. Perfect inversion is impossible; the
  goal is a measurably better estimate, not recovery of destroyed information.
- Only 0.5.3.3368 / Azeroth has been measured. Every other build and map is unverified, and the
  survey requirement (FR-013) exists to close that gap before the assumption spreads.
- The global lighting normalisation hypothesis is **not** assumed true. It is a deliverable to test
  (FR-016, SC-011). Until tested, per-tile comparison results carry an unquantified lighting-baseline
  confound in addition to the codec confound.
- The DXT1-compressed variant (FR-015) is a parity companion to the pristine render, not a
  replacement. The pristine render remains the primary corpus output; the variant exists so authored
  and synthetic tiles can be compared on equal terms.
- "God view" whole-map imagery predating December 2003 is *presumed* to share this encoding, but that
  presumption is untested and is covered by the survey rather than asserted.
- Super-resolution beyond native 256×256 has no client-side ground truth and is out of scope for this
  feature.
- DXT1 is a standard, publicly documented texture compression format (S3TC) chosen for GPU memory and
  bandwidth reasons. It is not an obfuscation or rights-protection scheme, and no part of this work
  depends on treating it as one.

## Out of Scope

- Super-resolution above the native authored resolution.
- **Exporting BLP files. Ever.** This feature needs the DXT1 encode/decode *cycle* in memory to
  reproduce degradation. It must never write a BLP container, and no requirement here should be read
  as implying one.
- Identifying which DXT1 encoder Blizzard used. Close enough to reproduce the degradation class is the
  bar; forensic encoder matching is not wanted.
- Re-encoding or redistributing authored client assets.
- Changing the terrain compositor's rendering model; this feature only adds an encoding stage after it.
