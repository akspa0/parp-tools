# Parser Profile — 0.9.0.3807 (Binary-Derived Deep Dive)

## Purpose
Build a function-level parser contract for `0.9.0.3807` from live Ghidra evidence, with enough detail to implement strict profile-gated parsing in viewer code paths.

---

## Profile IDs
- `AdtProfile_090_3807`
- `WmoProfile_090_3807`
- `MdxProfile_090_3807_Provisional`

Build range:
- Exact build: `0.9.0.3807`

Fallback policy:
- Unknown `0.9.0.x` should remain on `*_090x_Unknown` until validated.

---

## A) ADT Contract (high confidence)

## A1. Root parse chain and strictness
- `FUN_006e6000` opens ADT and dispatches into `FUN_006e6220` (sync/async wrappers).
- `FUN_006e6220` enforces root contract:
  - `MVER` required and version checked earlier in chain.
  - `MHDR` required immediately after `MVER`.
  - Root chunks resolved from `MHDR` offsets + `+8` header skip.
  - Token assertions for `MCIN`, `MTEX`, `MMDX`, `MMID`, `MWMO`, `MWID`, `MDDF`, `MODF`.

## A2. Root chunk record sizing
- `MDDF` count = `chunkSize / 0x24` (`FUN_006e6220`).
- `MODF` count = `chunkSize >> 6` (`FUN_006e6220`, i.e. `/ 0x40`).

## A3. MCIN and chunk addressing
- MCIN table is consumed as 0x10-byte entries in downstream chunk prep:
  - `FUN_006e72e0`: `entry = mcinBase + index*0x10 + 8`, first dword used as raw chunk offset.
- Chunk load path:
  - `FUN_006e72e0` -> `FUN_006d7130` -> `FUN_006d7590`.

## A4. MCNK required subchunk contract
- `FUN_006d7590` hard-asserts required subchunks from MCNK header offsets:
  - `MCVT`, `MCNR`, `MCLY`, `MCRF`, `MCSH`, `MCAL`, `MCLQ`, `MCSE`.
- Extracted MCNK header offsets used in this chain:
  - `+0x14` `MCVT`
  - `+0x18` `MCNR`
  - `+0x1C` `MCLY`
  - `+0x20` `MCRF`
  - `+0x24` `MCAL`
  - `+0x2C` `MCSH`
  - `+0x58` `MCSE`
  - `+0x60` `MCLQ`

## A5. MCLQ layout and liquid slots
- `FUN_006d7380` configures up to 4 liquid slots from MCNK flags.
- Per-slot stride in dwords is `0xC9` (`201` dwords = `0x324` bytes), matching 0.9-era layout.
- Slot pointers wired as:
  - base header fields at slot start
  - sample block pointer from `+2 dwords`
  - tile flag/mask block pointer from `+0xA4 dwords`
  - flow-related block pointer from `+0xB5 dwords`
- `FUN_006b1200` consumes 8-byte liquid samples (`+4` height lane read) across `9x9` grid.

## A6. Placement interpretation
- `FUN_006d8010` consumes doodad/mapobj refs from MCNK layer refs and mapobj refs.
- Root indirection tables (`MMID/MWID`) are wired in `FUN_006e6220` and used in ref construction chain.

## A7. Normals and bounds
- `FUN_006d7d70` converts `MCNR` packed signed bytes into normalized floats.
- `FUN_006d8650` builds tile/chunk bounds and center/radius used in culling and adjacency.

## A8. MH2O status
- No direct `MH2O` token/string evidence found in this primary ADT chain.
- Build currently behaves as strict `MCLQ` consumer in recovered path.
- Marked unresolved until constant/xref proof of any secondary MH2O path.

---

## B) WMO Contract (high confidence root/group)

## B1. WMO root parser and strict token order
- Root load path: `FUN_006e7a70` -> `FUN_006e79d0` -> `FUN_006e7bd0` -> `FUN_006e7e00`.
- `FUN_006e7cf0` enforces strict expected token sequencing.
- Required root token order recovered from `FUN_006e7e00`:
  1. `MVER` (`version == 0x11`)
  2. `MOHD`
  3. `MOTX`
  4. `MOMT`
  5. `MOGN`
  6. `MOGI`
  7. `MOSB`
  8. `MOPV`
  9. `MOPT`
  10. `MOPR`
  11. `MOVV`
  12. `MOVB`
  13. `MOLT`
  14. `MODS`
  15. `MODN`
  16. `MODD`
  17. `MFOG`
  18. optional `MCVP` via tolerant probe (`FUN_006e7dc0`).

## B2. Root record divisors (from chunk sizes)
- `MOMT`: `/ 0x40`
- `MOGI`: `/ 0x20`
- `MOPV`: `/ 0x0C`
- `MOPT`: `/ 0x14`
- `MOPR`: `/ 0x08`
- `MOVV`: `/ 0x0C`
- `MOVB`: `/ 0x04`
- `MOLT`: `/ 0x30`
- `MODS`: `/ 0x20`
- `MODD`: `/ 0x28`
- `MFOG`: `/ 0x30`
- optional `MCVP`: `/ 0x10`

## B3. Group parser and optional gates
- Group creation chain: `FUN_006e8250` -> `FUN_006e84d0` -> `FUN_006e8960`.
- `FUN_006e84d0` enforces `MVER(0x11)` then `MOGP`, maps core group fields.
- `FUN_006e8960` optional blocks gated by `MOGP` flags:
  - `0x00000200` -> `MOLR`
  - `0x00000800` -> `MODR`
  - `0x00000001` -> `MOBN` then `MOBR`
  - `0x00000400` -> `MPBV` -> `MPBP` -> `MPBI` -> `MPBG`
  - `0x00000004` -> `MOCV`
  - `0x00001000` -> `MLIQ`
  - `0x00020000` -> `MORI` then `MORB`

## B4. Group MLIQ internal layout
- On `MLIQ` gate, fields at group object offsets are assigned from `param_2[2..9]`.
- Sample pointer starts at `chunk + 0x26`.
- Sample byte extent is `widthLike * heightLike * 8` (8-byte sample stride).
- Secondary mask/flag region starts immediately after samples and uses another product term.
- This confirms 8-byte sample semantics and split sample/mask regions.

---

## C) MDX Contract (deep extraction, still provisional)

## C1. Dispatcher recovered
- Main loader dispatch function recovered: `FUN_0042a6a0`.
- Two major paths:
  - path A (flagged): richer sections (`FUN_00453930`, `FUN_00453b70`, `FUN_004541b0`, `FUN_004546a0`, `FUN_0044e660`, `FUN_00450c00`, etc.)
  - path B (alternate): reduced set (`FUN_00453a90`, `FUN_004540b0`, `FUN_00454420`, etc.)
- Global pre-read: `FUN_00453880` reads `MODL` and uses byte at `+0x174` for flags.

## C2. Confirmed section keys and constraints
- `MODL` (`0x4C444F4D`) required in loader setup.
- `TEXS` (`0x53584554`): texture section count computed as `size / 0x10C`, strict size divisibility checks.
  - Seen in `FUN_00453930`, `FUN_00453a90`, and world/detail variant `FUN_006da220`.
- `GEOS` (`0x534F4547`): geometry section consumed in `FUN_004541b0`/`FUN_00454420` and in `FUN_006da220`.
- `MTLS` (`0x534C544D`): material section in `FUN_00453b70`/`FUN_004540b0`.
- `ATCH` (`0x48435441`): attachments in `FUN_004546a0`.
- `PRE2` (`0x32455250`): emitter section in `FUN_0044e660`.
- `BONE` (`0x454E4F42`) and optional `HTST` (`0x54535448`) in `FUN_0042a7e0`.
- `SEQS` (`0x53514553`) in `FUN_0042aed0`.
- `PIVT` (`0x54564950`) in `FUN_0042b130`.
- `RIBB` (`0x42424952`) in `FUN_00450c00`.
- `LITE` (`0x4554494C`) in `FUN_0044fdb0`.
- `CAMS` (`0x534D4143`) in `FUN_0044f5a0`.
- `CLID` (`0x44494C43`) collision section in `FUN_0044c190`.
- `GEOA` (`0x414F4547`) geoset anim color/alpha in `FUN_004555a0` / `FUN_00455940`.

## C3. Animation and shape details
- `FUN_0042aed0` reads `SEQS` and builds per-sequence structures (0x28-sized logical units in array growth path).
- `FUN_0042a910` reads `HTST` shape-like data with multiple variant records (`type 0..3` switch).
- `FUN_0042b130` reads pivots (`PIVT`) at 12-byte records, with optional remap logic in one loader mode.

## C4. Detail doodad model path
- Secondary model path: `FUN_006da0d0` -> async callback `FUN_006da050` -> parser `FUN_006da220`.
- Also enforces `MDLX` magic and strict section constraints (`TEXS` size checks, `GEOS` access).

---

## D) Cross-domain stability assessment (0.9.0.3807 vs 0.9.1.3810 baseline)

- ADT: no contradictory field/record-size evidence found in recovered core chain.
- WMO: root strict order and group gates are fully recoverable and align with expected 0.9.x shape.
- MDX: section landscape and key invariants are now much better mapped, but interpolation/compression semantics are still not fully closed.

---

## E) Immediate implementation guidance

1. Add exact-build registry entries for `0.9.0.3807` using extracted contracts.
2. Keep MH2O disabled by default for this build until token-level proof appears.
3. Keep MDX profile marked provisional; wire section-presence diagnostics and strict size checks per section.
4. Gate WMO parser by recovered strict root order and group optional flag contracts.
