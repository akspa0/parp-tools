# Research: 1.12.1 Era-Aware MD20 Reader

## Source Evidence

- `wow-viewer/docs/architecture/m2-mdx-1121-native-trace-2026-06-05.md` — the 1.12.1 Ghidra trace. Source of truth for the 1.12.1 `MD20` header layout, per-record strides, the view table offset, the per-batch render-state decisions, and the cvar option set.
- `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md` — the earlier 3.3.5 native pass. Source of the canonical `M2ModelDocument` shape and the 3.3.5 stride set.
- `gillijimproject_refactor/src/MDX-L_Tool/Formats/Mdx/MdxFile.cs` — chunked-MDX reader (NOT used for 1.12.1 MD20; 1.12.1 uses MD20 with the legacy `.mdx` extension).

## Key Findings

1. **1.12.1 `.mdx` files are `MD20` with the legacy `.mdx` extension.** Same magic as 3.0.1+ but with a different header layout. The native cache loader normalizes `.mdl`/`.mdx`/`.m2` → `.m2` and dispatches to a single `MD20` parser.
2. **The 1.12.1 `MD20` header is a flat `(count, offset)` pointer table.** Each subsequent table is a `uint32` count and a `uint32` file-relative offset. The parser adds the file base to convert offsets to pointers.
3. **The view table offset is `0x3C/0x40` in 1.12.1** (not `0x44/0x48` as in 3.3.5). The 8-byte shift is the same shift that appears throughout the 1.12.1 header.
4. **Per-record strides differ between 1.12.1 and 3.3.5:**
   - sequence: 0x6C (1.12.1) vs 0x40 (3.3.5)
   - light: 0x0C (1.12.1) vs 0x9C (3.3.5)
   - camera: 0x2C (1.12.1) vs 0x64/0x74 (3.3.5)
   - ribbon: 0x7C (1.12.1) vs 0xAC/0xB0 (3.3.5)
   - particle: 0xDC (1.12.1) vs 0x1DC/0x1EC (3.3.5)
5. **1.12.1's M2 cvar set is smaller than 3.3.5's.** `M2BatchParticles` and `M2ForceAdditiveParticleSort` do not exist. The 3.3.5 runtime flag word bits `0x80` and `0x100` are not used in 1.12.1.
6. **1.12.1's light "record" is 3 floats (position + radius).** OQ-4 confirmed: the rest of light state (color, attenuation) is runtime-default in 1.12.1.
7. **1.12.1 likely has no external `.anim` files.** The 1.12.1 binary's lack of any animation cvar supports the hypothesis that 1.12.1 either inlines animations into the sequence records or has no multithreaded animation seam at all.

## Why This Is a Sibling Reader, Not an Edit

The 3.3.5 reader in `wow-viewer/src/core/WowViewer.Core.IO/M2/M2ModelReader.cs` is shared with the 3.3.5 runtime, runtime bridge, and tensor packer. Editing it to be era-aware would break FR-018 (existing 3.3.5 tests must pass with 0 modifications). The 048 slice lands a sibling reader in `wow-viewer/src/core/WowViewer.Core.IO/M2Era1121/` and adds a version-aware dispatch branch in `M2ModelReaderDispatcher` — no existing reader is touched.

## Why Not Just Use the Chunked Reader

Spec `043-m2-chunked-mdx-classic-support` previously assumed 1.12.1 used the chunked-MDX format. The Ghidra evidence proves this is wrong: 1.12.1's `.mdx` files have `MD20` magic, not `MDLX`. The chunked reader's `ValidateChunkedMagic` correctly rejects 1.12.1 files. Spec 043 is revised to defer 1.12.1 to spec 048.

## Source of Stride Values

| Stride | Value | Source |
| --- | --- | --- |
| Sequence | 0x6C | Ghidra FUN_0071cdf0 inline parse + `M2Shared.cpp` relocator FUN_0071e8d0 (was wrong) → corrected by re-derivation of the 1.12.1 view record's 9 nested sub-tables |
| Color | 0x1C | Ghidra FUN_0071e4f0 (8-byte track, 4-byte track, 2-byte track) |
| Texture weight | 0x08 | Ghidra FUN_0071e0c0 (8-byte entries) |
| View | 0x2C | Ghidra FUN_0071e270 (44-byte view record) |
| Light | 0x0C | Ghidra FUN_0071e1b0 (12-byte entries) — matches the OQ-4 hypothesis |
| Camera | 0x2C | Ghidra FUN_0071e9d0 (44-byte camera record) |
| Ribbon | 0x7C | Ghidra FUN_0071edb0 (124-byte ribbon record) |
| Particle | 0xDC | Ghidra FUN_0071ef40 (220-byte particle record) |
| 0x101-only 0xE0 | 0x1F8 | Ghidra FUN_0071f210 (504-byte 29-sub-table record) — present in 0x101 only |

## Open Questions (Resolved by 048)

- **OQ-1**: 0x101-only 0x1F8/29-sub-table record. **Status**: deferred. The 048 MVP walks the count/offset pair with bounds checks but does not decode the per-record sub-tables (per A-006, schema changes are out of scope).
- **OQ-2**: 1.12.1 `*(byte *)(iVar10 + 4)` flag word → blend mode mapping. **Status**: deferred. The 048 reader passes the file's flags word through to `M2ModelDocument.Flags` without re-interpretation.
- **OQ-3**: 1.12.1 vertex table layout (0x0C positions + 0x0C normals). **Status**: deferred. The 048 MVP does not parse vertex or normal records.
- **OQ-4**: 1.12.1 light record = (position, radius) pair. **Status**: confirmed. The 048 reader parses 3 floats per record.
- **OQ-5**: 1.12.1 M2 cvar bit mapping. **Status**: still open. The 048 reader does not consume cvar state.
- **OQ-6**: 1.12.1 external `.anim` files. **Status**: confirmed out of scope. The 048 reader does not look for external `.anim` files.

## Follow-Ups (Out of 048 Scope)

- Spec 049 — 2.x TBC MD20 reader (currently rejected with "see spec 049" by the 048 dispatcher).
- Spec 050 — 1.12.1 vertex/normal walker (after the schema absorbs 1.12.1's separate position + normal tables).
- Spec 051 — 1.12.1 view record sub-table walker (after the schema absorbs the 9 nested sub-tables per view).
- Spec 052 — 1.12.1 cvar → runtime flag bit mapping (after Ghidra re-derivation).
