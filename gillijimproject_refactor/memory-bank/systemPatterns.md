# System Patterns

## Architecture
- `AlphaDataExtractor` reads Alpha ADT files, slicing raw chunk payloads into `LkMcnkSource` instances for LK reconstruction.
- `LkMcnkBuilder` consumes `LkMcnkSource` to emit LK-compatible `MCNK` chunks with serialized sub-chunks (`MCLY`, `MCAL`, etc.).
- RoundTrip CLI orchestrates Alpha → LK → Alpha parity verification by chaining extraction and rebuilding steps.

## Testing Strategy
- Existing tests in `WoWRollback.LkToAlphaModule.Tests` focus on synthetic data for `LkMcnkBuilder`.
- New parity tests must load real Alpha ADTs to validate extractor output before LK rebuilds.
