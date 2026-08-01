# Feature Specification: Raw Audio Unswizzle Pattern Probe

**Feature Branch**: `091-raw-audio-unswizzle`
**Created**: 2026-07-05
**Status**: Draft
**Input**: User observed structured patterns after renaming map-derived WAV output to `.raw` and opening it as image data.

## User Scenarios and Tests

### Scenario 1 - Inspect map-derived WAV payload as image layouts

An operator can point a script at a map-derived WAV or raw byte file and generate candidate image views using likely widths, byte phases, channel deinterleaves, bitplanes, and sample interpretations.

**Acceptance**:
- Given a WAV produced by the existing map-to-audio scripts, the tool strips the WAV container and analyzes the sample payload bytes.
- Given a raw byte file, the tool analyzes from byte zero.
- The output contains candidate PNGs, a contact sheet, and a JSON summary with enough metadata to reproduce each candidate.

### Scenario 2 - Compare structure without claiming hidden-data proof

An operator can compare ranked candidates by visual and numeric structure while keeping interpretation separate from evidence.

**Acceptance**:
- Each candidate records width, layout mode, original/displayed dimensions, entropy, correlations, contrast, and a heuristic score.
- The tool labels results as layout hypotheses, not proof of steganography or concealed payloads.
- Candidate generation is bounded by a configurable maximum displayed pixel count.

## Requirements

- **REQ-001**: The tool MUST live under `wow-viewer/data-harvester/scripts/`.
- **REQ-002**: The tool MUST support `.wav` input through Python standard-library WAV parsing when possible.
- **REQ-003**: The tool MUST support raw byte input for files that are not WAV or WAV files that cannot be parsed.
- **REQ-004**: The tool MUST generate grayscale byte views across multiple widths.
- **REQ-005**: The tool MUST generate byte-phase/deinterleaved views for at least strides 2, 4, and 8.
- **REQ-006**: The tool MUST generate bitplane views for byte payloads.
- **REQ-007**: The tool MUST generate 16-bit little-endian and big-endian sample interpretations when the byte count permits.
- **REQ-008**: The tool MUST write a ranked `summary.json`.
- **REQ-009**: The tool MUST write a `contact_sheet.png` for the highest-ranked candidates.
- **REQ-010**: The tool MUST avoid adding permanent dependencies beyond existing data-harvester dependencies.
- **REQ-011**: The tool SHOULD reverse flattened `257x257` heightmap-audio samples into tile mosaics when the sample count divides cleanly.
- **REQ-012**: The tool SHOULD use dataset `index.parquet` rows to arrange tile mosaics by map coordinates when provided.

## Non-Goals

- Detecting account watermarking in screenshots.
- Proving hidden data exists.
- Training models from these projections.
- Changing existing map-to-audio extraction scripts.

## Risks

- Structured images can be caused by deterministic rasterization, sample-width aliasing, heightmap row order, or image-editor import assumptions.
- Very large WAVs can generate expensive PNGs without pixel caps.
- Width selection strongly changes visual patterns; comparison across maps should use identical candidate settings.
- A map-coordinate mosaic depends on the WAV sample order matching the same index sort used by the original extraction script.
