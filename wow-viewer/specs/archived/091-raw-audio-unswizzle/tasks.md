# Tasks: Raw Audio Unswizzle Pattern Probe

## Phase 1

- [x] T001 Create Spec Kit documents for the raw-audio unswizzle lane.
- [x] T002 Add a Python script under `data-harvester/scripts/`.
- [x] T003 Support WAV payload extraction and raw byte fallback.
- [x] T004 Generate grayscale byte, byte delta, bitplane, byte phase, RGB triplet, and 16-bit sample views.
- [x] T005 Score and rank candidates in `summary.json`.
- [x] T006 Write `contact_sheet.png` for top-ranked candidates.
- [x] T007 Run against the real Azeroth WAV and inspect top candidates.
- [x] T008 Add tile unswizzle output for exact `257x257` flattened heightmap samples.
- [x] T009 Add optional map-coordinate mosaic output from dataset `index.parquet`.

## Later

- [ ] T010 Add multi-file comparison mode.
- [ ] T011 Add tile-width presets from dataset metadata.
- [ ] T012 Add specific detector logic only after a repeatable signal is identified.
