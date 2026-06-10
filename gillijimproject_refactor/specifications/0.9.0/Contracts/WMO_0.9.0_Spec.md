# WMO Container Deep Dive — 0.9.0 vs 0.8.0

## Scope
- Confirm the 0.9.0 WMO chunk ordering, sizes, and bounds guards match the already-working 0.8.0 tooling profile.
- Highlight any deviations (none found) and restate the validation rules needed for safe parsing.

## Conclusion
- Parsing contract is effectively identical to 0.8.0: same chunk order, same placement record sizes, same string tables, same bounds expectations.
- No alternate MD20-aware paths or new chunk tags observed; MDLX/MDX references are expected in doodad/model string tables.

## Required chunk order (strict)
MVER (0x11) → MOHD → MOTX → MOMT → MOGN → MOGI → MOSB → MOPV → MOPT → MOPR → MOVV → MOVB → MOLT → MODS → MODN → MODD → MFOG → (optional MCVP).
- Loader should treat any order deviation as fatal to mirror engine behavior.

## String tables and offsets
- MMDX: null-terminated model filenames table.
- MMID: u32 offsets into MMDX; every offset must land within MMDX and terminate before the chunk end.
- MWMO / MWID mirror the above for world models; same offset + null-termination guards.

## Placement records
- MODF size 0x40: nameId (→ MWMO), position (3*f32), rotation (3*f32), extents (2*f32*3), flags (u32), doodad set id (u16), name set (u16/pad).
- MODD size 0x28: nameId (→ MMDX), position (3*f32), rotation (3*f32), scale (u16), color (rgba8), flags (u8), light refs/pad (u8).
- No new fields vs 0.8.0. Bounds: indices must be in-range for MMID/MWID; coordinates are unchecked beyond basic size guards.

## Bounds/validation checklist
- Enforce strict chunk order above.
- Validate every MMID/MWID offset is within its string table and hits a null before the chunk end.
- Enforce fixed record sizes: MODF == 0x40, MODD == 0x28; reject otherwise.
- Abort on any chunk-length overrun relative to remaining file bytes.
- Optional MCVP may appear after MFOG; treat unknown extra chunks as fatal for parity with 0.8.0 tooling.

## Interop with models
- Doodads/world models are assumed to be MDLX/MDX. If an entry actually points to MD20 bytes, caller must convert MD20→MDLX before render; the WMO loader performs no translation.

## Suggested tests (parity with 0.8.0)
- Good-path: a known 0.8.0 WMO parses without warnings under the 0.9.0 profile.
- Bad order: swap MOMT/MOTX → expect fatal.
- Bad MMID: offset past MMDX length → fatal.
- Bad MODD size: mutate chunk size to 0x30 → fatal.
- Trailing bytes: add junk after last chunk → fatal (keep strict size accounting).
