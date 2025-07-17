# PM4 Group-Key Export Plan (2025-07-14)

> This document is the SINGLE source of truth for finishing the PM4 group-key-based OBJ export.  All future work on PM4 must follow this plan and reference only the authoritative legacy code noted below.

---

## 1  Authoritative Reference Code

| Area | Legacy File (old repo) | Purpose |
|------|-----------------------|---------|
| Surface-key export logic | `Services/PM4/MsurObjectExporter.cs` | Shows grouping by `Unknown_0x1C` for diagnostics |
| MSUR←→MSLK linkage | `Services/PM4/MslkLinkScanner.cs` | Dumps linkage fields, proves the join columns |
| Group OBJ export | `Tools/Debug/MsurGroupObjExporter.cs` | Sample of correct group counts/structure |
| Analyzers | `MsurMslkAnalyzer.cs`, `MsurStatsAnalyzer.cs` | Statistics that confirm field meanings |
| CLI/Wiring | `Pm4BatchTool/Program.cs` | Demonstrates end-to-end flow of old exporter |

> ❗ No wowdev/wiki or external docs are to be used.  These files supersede all other references.

---

## 2  Chunk Layouts (Derived)

### 2.1  MSLKEntry (20 bytes)
```
00  byte  Flags               (Unknown_0x00)
01  byte  Sequence/Subtype    (Unknown_0x01)
02  uint16 ReservedPadding    (0x0000)
04  uint32 ParentIndex        (Unknown_0x04)  ← high-word of MSUR.Unknown_0x1C
08  int24  MsviFirstIndex     (-1 for doodad) (packed)
0B  byte   MsviIndexCount
0C  uint32 LinkIdRaw          (0xFFFFYYXX pattern)
10  uint16 ReferenceIndex     (groups multiple ParentIndices)
12  uint16 SystemFlag (0x8000)
```

### 2.2  MsurEntry (64 bytes)
Key offsets only (remaining fields unchanged):
```
00  byte   FlagsOrUnknown_0x00      ← matches MSLK.Flags
03  byte   IndexCount
1C  uint32 Unknown_0x1C            ← ParentIndex|Subgroup (join to MSLK.Unknown_0x04)
```

---

## 3  Implementation Roadmap

### 3.1  Phase A  – Chunk Readers
1. Rewrite `MSLKChunk`/`Entry` and `MSURChunk`/`Entry` in `Formats/P4/Chunks/Common` to match the layouts above.
2. Add unit tests to ensure `byteLength % structSize == 0` for real PM4 files.

### 3.2  Phase B  – Adapter Logic (`Pm4Adapter`)
1. Load new chunk classes.
2. For each `MsurEntry`, find owning `MSLKEntry`:
   * `Msur.Unknown_0x1C >> 16 == Mslk.ParentIndex`
   * Optional: verify `Msur.FlagsOrUnknown_0x00 == Mslk.Flags`.
3. Assign `groupKey = Mslk.ReferenceIndex` (confirmed by legacy stats) and populate `Pm4Scene.SurfaceGroup`.
4. Skip surfaces where `groupKey == 0` **when** `--skip-m2` is supplied.

### 3.3  Phase C  – Exporter (`Pm4GroupObjExporter`)
1. Group faces by `groupKey`.
2. Always flip X axis (legacy bug-fix).
3. Guardrail: warn if group count > 32.

### 3.4  Phase D  – Validation & Tests
1. Integration test using `development_00_00.pm4` expecting 8-18 groups.
2. Compare first four groups’ triangle counts with old exporter output (<±1 % difference).

### 3.5  Phase E  – Docs & CLI
1. Remove any stale `--flipx` docs (code already auto-flips).
2. Document the group-key derivation in README.

---

## 4  Milestones & Timeline
| Phase | Target Date | Deliverable |
|-------|-------------|-------------|
| A | 2025-07-15 AM | Chunk readers + unit tests green |
| B | 2025-07-15 PM | Correct `groupKey` assignment, build passes |
| C | 2025-07-16 AM | OBJ exporter parity (≤18 groups) |
| D | 2025-07-16 PM | Integration tests pass, legacy comparison script green |
| E | 2025-07-17 | Documentation updated |

---

## 5  Risks & Mitigations
* **Mis-matched join logic** – add assert: every `MsurEntry` with geometry must resolve to an `MslkEntry`; log unresolved count.
* **Off-by-one in int24 read/write** – keep unit test verifying round-trip for negative & positive indices.
* **Unexpected group count** – fail the CI if group count > 32 on sample files.

---

## 6  Glossary
* **groupKey**: `MSLK.ReferenceIndex` (0x10) – authoritative identifier for grouped geometry.
* **ParentIndex**: `MSLK.Unknown_0x04` / high-word of `MSUR.Unknown_0x1C` – hierarchical container id.
* **FlagByte**: `MSLK.Unknown_0x00` / `MSUR.FlagsOrUnknown_0x00` – corresponds to render category.

---

*Document created automatically by Cascade AI per USER request.*
