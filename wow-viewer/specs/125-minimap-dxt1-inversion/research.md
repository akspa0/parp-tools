# Research: Minimap DXT1 Artifact Inversion

**Phase 0 output** | **Date**: 2026-08-02 | **Spec**: [spec.md](./spec.md)

## 1. DXT1 encoder availability

**Decision**: Reuse `BCnEncoder.Net` (`CompressionFormat.Bc1`) already referenced by
`WowViewer.Core.IO`.

**Rationale**: The spec's checklist flagged the encode half as the one missing component, but
inspection shows it is already in-tree. `WowViewer.Core.IO.csproj` references `BCnEncoder.Net`, and
`AlphaBlpCompatibilityService.EncodeBlp2` already drives `BcEncoder` with
`CompressionFormat.Bc1` (DXT1) and `Bc3` (DXT3). The decode half is `SereniaBLPLib` (wrapped by
`BlpRgbReader`). No new dependency is required.

**Alternatives considered**:
- Writing a hand-rolled DXT1 block encoder — rejected. BCnEncoder is a mature, tested implementation
  already in the dependency graph; hand-rolling adds risk with no benefit.
- `BLPSharp` 0.1.0 (also referenced) — not needed; BCnEncoder covers the encode path.

## 2. DXT1 round-trip idempotency (FR-014)

**Decision**: Implement a round-trip check that decodes an authored tile and re-encodes it, then
measures block-level agreement with the authored bytes.

**Rationale**: DXT1 re-encoding is near-idempotent on already-decoded data because a decoded block's
four colours already lie on a line between two exactly-RGB565-representable endpoints, so any
competent encoder recovers those same endpoints. This makes encoder choice low-risk for parity and
for the re-encode check, and only materially affects training-pair generation from pristine renders.

**Alternatives considered**: Forensic encoder identification — explicitly out of scope per spec.
Close enough to reproduce the degradation class is the bar.

## 3. Global lighting normalisation hypothesis (FR-016)

**Decision**: Measure per-map mean/std luma across authored tiles. A shared baseline is present when
cross-tile variance is small relative to within-tile variance. When found, normalise synthetic tiles
to the authored baseline before scoring.

**Rationale**: The authored tiles of a map may share a common brightness/contrast baseline from the
client's minimap renderer. This is a second, independent confound on top of the codec. Measuring it
per map and build separates lighting-baseline offset from codec damage.

**Alternatives considered**: Assuming no baseline — rejected, because it would leave an unquantified
confound in every comparison. Treating it as an inherited assumption — rejected; the spec makes it a
first-class deliverable.

## 4. Restoration model (FR-007..FR-012)

**Decision**: A single residual network that predicts (pristine − encoded), trained on locally
generated pristine→encoded pairs. No authored reference required for training. Gated on
hallucination: re-encode agreement + unsupported-detail fraction.

**Rationale**: Matches the constitution's residual-model-chain principle (one output, own
checkpoint). Ground truth exists locally (the pristine render before encoding), so the model trains
without any authored image. The hallucination gate prevents promoting a model that invents plausible
terrain detail.

**Alternatives considered**: Multi-task or shared-weight models — rejected per constitution. Direct
full-signal prediction — rejected; residual prediction is the established pattern.

## 5. Parity companion emission (FR-015)

**Decision**: The synthesizer emits a `*_dxt1.png` parity companion per tile alongside the pristine
render, produced by the same encode/decode cycle. The pristine render remains the primary output.

**Rationale**: This lets authored vs synthetic compare on equal terms without a comparison-time
encode step, and makes the parity mechanism a first-class corpus output rather than a scoring
side-effect.

## 6. Degenerate tile handling (FR-004)

**Decision**: A tile whose authored image is a single flat colour (unrendered) is excluded from
aggregate scores and reported as excluded.

**Rationale**: Averaging a flat tile in already flipped one diagnostic's verdict. Exclusion is the
established safe behaviour.

## 7. Era gating (FR-006)

**Decision**: Encoding behaviour resolves per build era via `MinimapEraProfile`; an unrecognised
build is flagged rather than silently defaulted.

**Rationale**: Blizzard changed minimap generation across builds (0.5.3 Alpha, 0.6.0 Beta 1, 1.0.0
different again). The existing era-gating discipline already handles this; encoding awareness joins it.
