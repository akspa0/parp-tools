# Active Context

- **Focus**: Resolve LK→Alpha→LK round-trip size mismatch by ensuring rebuilt LK ADTs retain liquids, placements, and auxiliary chunks produced by the managed builders.
- **Current Status**: ⚠️ **Round-trip incomplete** – `roundtrip-test` now extracts all 256 MCNKs without overflow, but rebuilt LK ADT is ~430 KB vs original ~1.51 MB, indicating data loss during rebuild.
- **Completed This Session**:
  1. ✅ Updated `AlphaMcnkBuilder.BuildFromLk()` to emit headerless Alpha-compatible subchunks.
  2. ✅ Hardened `AlphaDataExtractor.ExtractFromAlphaMcnk()` with offset validation, size inference, and logging for mis-sized chunks.
  3. ✅ Captured verbose extraction warnings that highlight which chunks diverge (e.g., MCAL length mismatches).
- **What Works Now**:
  - Synthetic Alpha ADTs generated from LK sources can be re-parsed without crashing.
  - Extraction logs surface per-chunk anomalies instead of silently corrupting data.
  - Round-trip workflow reaches the LK rebuild phase, producing an output file for diffing.
- **Gaps / Next Focus**:
  - ❌ `LkAdtBuilder.Build()` likely skips or truncates data when supplied with Alpha-derived `LkMcnkSource` instances.
  - ❌ Need detailed diffing of original vs rebuilt MCNK payload sizes to pinpoint missing chunks (placements, liquids, MFBO/MTXF, etc.).
  - ❌ CLI messaging/documentation still implies “Alpha ADT saved” is final artifact; must clarify synthetic nature.
- **Next Steps**:
  1. Instrument `LkAdtBuilder` to log per-MCNK payload sizes and included subchunks during rebuild.
  2. Compare original LK `_obj0/_tex0` data with rebuilt counterparts to identify omitted sections.
  3. Update `ROUNDTRIP_TESTING.md` with troubleshooting guidance and revised workflow notes.
- **Risks / Known Issues**:
  - Data loss when shipping LK→Alpha conversions until builder parity is restored.
  - Logs may still be insufficient for large-scale maps; consider optional MCNK dump tooling.
