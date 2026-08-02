# Data Model: Minimap DXT1 Artifact Inversion

**Phase 1 output** | **Date**: 2026-08-02 | **Spec**: [spec.md](./spec.md)

## Entities

### Tile Encoding Profile

What a specific authored tile is actually encoded as. Detected per file (FR-001).

| Field | Type | Notes |
|-------|------|-------|
| ContainerFormat | enum (BLP2, ...) | From file header |
| CompressionMode | enum (DXT1, DXT3, ...) | From file header |
| ColourDepth | int | 0 (no alpha) or 8 (alpha) |
| HasAlpha | bool | |
| Width / Height | int | 256×256 for measured tiles |
| MipCount | int | 1 for measured tiles |
| SourcePath | string | For provenance |

### Encoding Parity Pair

A synthesized tile and the same tile after the authored encoding cycle has been applied (FR-002).

| Field | Type | Notes |
|-------|------|-------|
| Pristine | Image<Rgba32> | Primary output |
| ParityCompanion | Image<Rgba32> | After encode/decode cycle (FR-015) |
| EncodingApplied | TileEncodingProfile | Which cycle was applied |

### Restoration Training Pair

A pristine image and its encoded counterpart, where the pristine image is ground truth (FR-007).

| Field | Type | Notes |
|-------|------|-------|
| Pristine | Image<Rgba32> | Ground truth |
| Encoded | Image<Rgba32> | After encode/decode cycle |
| Residual | Image<Rgba32> | Pristine − Encoded (what the model predicts) |

### Restoration Verdict

For one restored tile (FR-008..FR-012).

| Field | Type | Notes |
|-------|------|-------|
| ImprovementOverEncoded | float | Colour error reduction vs encoded input |
| ReencodeAgreement | float | Agreement after re-encoding (FR-009) |
| HallucinationFraction | float | Unsupported-detail fraction (FR-010) |
| ChangedOnUndamaged | float | Change on undamaged input (FR-011) |

### Era Encoding Survey

Per build and map, the distribution of encodings observed (FR-013).

| Field | Type | Notes |
|-------|------|-------|
| Build | string | Build identity |
| Map | string | Map name |
| EncodingDistribution | map<Encoding, count> | |
| BuildRecognised | bool | Era-gating flag (FR-006) |

### Lighting Baseline

The common brightness/contrast normalisation shared across authored tiles of a map, if one exists
(FR-016).

| Field | Type | Notes |
|-------|------|-------|
| Map | string | |
| Build | string | |
| MeanLuma | float | Cross-tile mean |
| StdLuma | float | Cross-tile std |
| BaselinePresent | bool | True when cross-tile variance is small relative to within-tile |
| AccountedFor | bool | Whether comparison normalises synthetic to this baseline |

## Relationships

- A **Tile Encoding Profile** is detected per authored tile; the **Era Encoding Survey** aggregates
  profiles per build/map.
- An **Encoding Parity Pair** is produced per synthesized tile; its **ParityCompanion** is what
  comparison uses for fair scoring.
- A **Restoration Training Pair** is generated locally (pristine → encoded); the model learns the
  **Residual**; a **Restoration Verdict** is produced per restored tile.
- A **Lighting Baseline** is measured per map/build and, when present, is applied to synthetic tiles
  before comparison.

## Validation Rules

- Degenerate (single flat colour) tiles are excluded from aggregates (FR-004).
- Every corpus row and comparison report records its parity status, including "none" (FR-005).
- Unrecognised builds are flagged, never silently defaulted (FR-006).
- Restoration at native resolution is separable from any resolution increase (FR-012).
