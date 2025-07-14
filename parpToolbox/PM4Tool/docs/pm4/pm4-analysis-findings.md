# PM4 Analysis Findings

## Key Findings
- **MSCN = Exterior Vertices:** Visual inspection confirms MSCN chunk represents exterior (boundary) vertices for each object in PM4 files.
- **MSLK = Doodad Placements:** MSLK entries (with MspiFirstIndex == -1) represent doodad placements, with group/object IDs and anchor points.
- **Chunk Relationships:** Most chunk relationships (MSUR→MSVI→MSVT, MSLK→MSVI→MSVT) are now well understood, but some (MSCN↔MSLK) remain ambiguous.
- **Mesh Extraction:** Mesh extraction logic is robust for most files, but some geometry may be missing due to incomplete chunk parsing or undocumented subchunks.
- **Type Safety Issues:** Build errors (uint vs int) in index handling have been a recurring issue.
- **Resource Management:** Test process hangs after mesh+MSCN output highlight the need for robust disposal patterns.

## Gaps & Ambiguities
- The relationship between MSCN and MSLK is not fully mapped or exposed in Core.
- Some mesh extraction logic is duplicated in analysis tools/tests rather than being fully encapsulated in Core.
- Doodad placement decoding may require additional context from MDBH or ADT files.
- Error handling and validation could be more robust and standardized.

## Opportunities for Improvement
- Move mesh extraction, boundary export, and chunk relationship logic into Core as reusable APIs.
- Standardize error handling and resource management patterns.
- Use test-driven development: codify edge cases and ambiguities as tests, then address them in Core.
- Document all chunk relationships and data flows in Core XML docs and memory bank. 