# ADT Binary Contract — 0.8.0.3734 (Ghidra)

## Scope
Binary-derived ADT parser contract for WoW.exe build `0.8.0.3734`, focused on root/MCNK/MCLQ behavior that affects parser correctness.

---

## 1) Function Map

- `0x006c7220` — ADT root parser (`MVER`, `MHDR`, root chunk offset wiring)
- `0x006b8f90` — MCNK required subchunk pointer resolver
- `0x006b8be0` — MCLQ layer parser/slot materialization

Confidence: **High** for these roles.

---

## 2) Root ADT Contract (A1)

From `FUN_006c7220`:

- Requires first token `MVER`.
- Requires next token `MHDR`.
- Derives root chunk pointers from MHDR offsets with fixed `+0x10` payload addressing.
- Validates chunk tokens at computed offsets:
  - `MCIN`, `MTEX`, `MMDX`, `MMID`, `MWMO`, `MWID`, `MDDF`, `MODF`.
- Placement count derivation:
  - `MDDF` count = `chunkSize / 0x24`
  - `MODF` count = `chunkSize >> 6` (`/0x40`)

### Evidence snippet
```text
if (*piVar1 != 'MVER') assert;
piVar1 = piVar1 + size + 8;
if (*piVar1 != 'MHDR') assert;
...
if (*(base + mhdr[3] + 8) != 'MCIN') assert;
...
mddfCount = *(base + mhdr[9] + 0xC) / 0x24;
modfCount = *(base + mhdr[10] + 0xC) >> 6;
```

Confidence: **High**
Contradictions: none observed.

---

## 3) MCNK Required Subchunk Contract (A2)

From `FUN_006b8f90`:

- Expects current chunk token `MCNK`.
- Uses MCNK header-relative offsets to locate and validate required subchunks:
  - `MCVT` (`+0x14` path)
  - `MCNR` (`+0x18`)
  - `MCLY` (`+0x1C`)
  - `MCRF` (`+0x20`)
  - `MCSH` (`+0x2C`)
  - `MCAL` (`+0x24`)
  - `MCLQ` (`+0x60`)
  - `MCSE` (`+0x58`)
- Stores resolved payload pointers after `+8` chunk-header skip.

### Evidence snippet
```text
if (*(mcnk + *(hdr+0x60)) != 'MCLQ') assert;
mclqPtr = mcnk + *(hdr+0x60) + 8;
if (*(mcnk + *(hdr+0x58)) != 'MCSE') assert;
```

Confidence: **High**
Contradictions: none observed.

---

## 4) MCLQ Layer Contract (A3)

From `FUN_006b8be0`:

- Consumes MCLQ payload pointer from MCNK parser (`chunk + 0xF20` object field).
- Iterates up to 4 slots gated by MCNK flag bits: `0x04`, `0x08`, `0x10`, `0x20`.
- Per enabled slot:
  - scalar0 from offset `+0x000`
  - scalar1 from offset `+0x004`
  - sample/data pointer at `+0x008`
  - tile/aux pointer at `+0x290`
  - scalar2 at `+0x2D0`
- Advances to next layer by `0xB5` dwords (`0x2D4` bytes).

### Evidence snippet
```text
record->a = *puVar4;
record->b = puVar4[1];
record->samples = puVar4 + 2;      // +0x08
record->tiles   = puVar4 + 0xA4;   // +0x290
record->c = *(puVar4 + 0xB4);      // +0x2D0
puVar4 = puVar4 + 0xB5;            // +0x2D4 stride
```

Confidence: **High** for stride/offsets; **Medium** for semantic naming of non-height fields.

---

## 5) Implementation-Ready `IAdtProfile` Seeds

```text
ProfileId: AdtProfile_080_3734
BuildRange: [0.8.0.3734, 0.8.0.3734]

RootChunkPolicy:
  RequireStrictTokenOrder: true
  UseMhdrOffsetsOnly: true

PlacementPolicy:
  MddfRecordSize: 0x24
  ModfRecordSize: 0x40

McnkPolicy:
  RequiredSubchunks: [MCVT, MCNR, MCLY, MCRF, MCSH, MCAL, MCLQ, MCSE]

MclqPolicy:
  LayerStride: 0x2D4
  LayerSlotMaskBits: [0x04, 0x08, 0x10, 0x20]
  SampleBlockOffset: 0x08
  TileBlockOffset: 0x290
  TailScalarOffset: 0x2D0
```

---

## 6) Open Unknowns

1. MCIN entry-size/count proof function for this build (not extracted in this pass).
2. Full runtime interpretation of all MCLQ sample lanes beyond the verified layout.
3. Any alternate ADT liquid path in 0.8.0.3734 that bypasses this `MCLQ` slot parser.

Impact severity:
- (1) parser break risk
- (2) visual/gameplay artifact risk
- (3) compatibility risk
