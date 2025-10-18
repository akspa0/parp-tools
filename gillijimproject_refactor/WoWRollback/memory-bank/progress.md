# Progress

- **Round-trip stabilization**: Synthetic Alpha ADTs generated from LK inputs now re-parse cleanly; `AlphaDataExtractor` handles headerless/headered subchunks with bounds checks and emits warnings for anomalies.
- **Builder alignment**: `AlphaMcnkBuilder` produces Alpha-compatible payloads (headerless MCLY/MCAL/MCSH/MCSE) and corrected offset metadata, eliminating extraction overflows.
- **Diagnostics**: `roundtrip-test` executes through rebuild, exposing the current size mismatch (`1,513,746` original LK bytes vs `430,244` rebuilt) for targeted investigation.
- **Testing**: `WoWRollback.LkToAlphaModule.Tests` remain green (26/26) covering liquids and placements; regression runs now rely on CLI round-trip for manual verification.
- **Outstanding**:
  1. Instrument `LkAdtBuilder` to confirm all MCNK subchunks (liquids, placements, MFBO/MTXF) are re-emitted when sourced from Alpha data.
  2. Produce per-MCNK diff tooling/logging to compare original vs rebuilt payload sizes.
  3. Refresh CLI documentation (`ROUNDTRIP_TESTING.md`, `roundtrip-test` output) to clarify synthetic Alpha intermediates and troubleshooting steps.
  4. Address analyzer warnings (CA2022/CA2014) flagged in extraction utilities once parity is achieved.
