# Feature Specification: Minimap DXT1 Artifact Inversion

**Feature Branch**: `125-minimap-dxt1-inversion`

**Created**: 2026-08-02

**Status**: Draft

**Input**: User description: "Authored 0.5.3 minimaps are DXT1-compressed. Our compositor produces pristine 24-bit output, so it has been building *better than real* minimaps — every comparison scored a clean image against a lossy one and blamed the renderer. Two consequences: synthetic output must carry the same compression for any fair comparison or dataset parity, and a learned inverse of the DXT1 degradation could restore authored tiles toward their pre-compression appearance."

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
- **FR-014**: The system MUST establish, by measurement, how much its chosen encoder differs from the
  one that produced the authored tiles — by decoding an authored tile, re-encoding it, and comparing
  against the authored bytes. A near-exact match confirms encoder choice is not a material source of
  error; a poor match makes encoder selection a blocking decision for restoration training rather
  than an assumption.
- **FR-015**: Where more than one candidate encoder is available, the system MUST be able to report
  the re-encode agreement for each, so the closest match can be chosen on evidence rather than
  convenience.

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
  95% of blocks, confirming encoder choice is not a material source of comparison error. If the
  measured figure is materially lower, encoder selection is escalated to a blocking decision before
  restoration training begins.

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
- "God view" whole-map imagery predating December 2003 is *presumed* to share this encoding, but that
  presumption is untested and is covered by the survey rather than asserted.
- Super-resolution beyond native 256×256 has no client-side ground truth and is out of scope for this
  feature.
- DXT1 is a standard, publicly documented texture compression format (S3TC) chosen for GPU memory and
  bandwidth reasons. It is not an obfuscation or rights-protection scheme, and no part of this work
  depends on treating it as one.

## Out of Scope

- Super-resolution above the native authored resolution.
- Writing BLP containers. This feature needs the encode/decode *cycle* to reproduce degradation; it
  never needs to produce a loadable BLP file.
- Re-encoding or redistributing authored client assets.
- Changing the terrain compositor's rendering model; this feature only adds an encoding stage after it.
