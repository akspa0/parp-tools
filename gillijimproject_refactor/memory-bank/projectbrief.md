# Project Brief

## Mission
Port Alpha World of Warcraft terrain data into Lich King (LK) formats with byte-parity fidelity by extending the WoWRollback toolset.

## Current Objective
Diagnose and fix the RoundTrip CLI so it preserves Alpha `MCLY`/`MCAL` payloads when extracting and rebuilding LK `MCNK` chunks.

## Success Criteria
- Automated tests prove `AlphaDataExtractor` captures non-zero `MCLY`/`MCAL` data from sample Alpha ADTs.
- RoundTrip CLI reproduces original Alpha chunk bytes without manual intervention.
