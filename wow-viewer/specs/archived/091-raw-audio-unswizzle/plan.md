# Implementation Plan: Raw Audio Unswizzle Pattern Probe

## Scope

Create a bounded Python analysis script that turns map-derived WAV/raw payloads into ranked image-layout hypotheses. The script is a diagnostic artifact for Spec 076-adjacent research, not a training or viewer runtime feature.

## Technical Context

- Location: `wow-viewer/data-harvester/`
- Runtime: `uv run python ...`
- Dependencies already available: `numpy`, `pillow`
- WAV parsing: Python standard library `wave`

## Phases

### Phase 1 - Single-file probe

1. Add `scripts/unswizzle_audio_raw_patterns.py`.
2. Parse WAV payload bytes or raw file bytes.
3. Generate bounded candidate PNGs for widths, byte phases, bitplanes, RGB triplets, and 16-bit interpretations.
4. Score candidates with simple structure metrics.
5. Reverse exact `257x257` sample groups into tile mosaics when the sample count permits.
6. Optionally arrange tile mosaics by `index.parquet` map coordinates.
7. Write `summary.json` and `contact_sheet.png`.

### Phase 2 - Cross-map comparison

1. Add manifest mode for multiple files.
2. Emit normalized candidate names across maps/builds.
3. Add pairwise similarity metrics for same layout modes.

### Phase 3 - Payload-specific tests

1. Add optional known-width presets for heightmap-derived audio.
2. Add checks for tile-row periodicity and ADT tile boundaries.
3. Add suspected watermark/stego detectors only if a specific signal is identified.

## Validation

- `uv run python -m py_compile scripts/unswizzle_audio_raw_patterns.py`
- `uv run python scripts/unswizzle_audio_raw_patterns.py --help`
- Optional operator run against the existing Azeroth WAV output.
