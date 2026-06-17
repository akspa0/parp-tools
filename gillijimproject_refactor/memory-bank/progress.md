# Progress

## 2026-06-16 — PM4 matcher is broken; spec 065 written

### Root cause diagnosis
- `pm4 match-report` produces "Candidate Count: 0" for ALL PM4 objects on ALL tiles because it compares PM4 raw-ADT coordinates against WoW world coordinates without any conversion. `ConvertPm4VertexToWorld` produces `(tileY*533+mappedU, tileX*533+mappedV, localUp)` but `AdtPlacementReader` produces `(17066-rawY, 17066-rawX, rawZ)`. Gap is ~24000 units — all spatial matching is dead.
- `pm4 match-assets` (the shape scorer) is architecturally correct but produces sub-threshold scores (~0.42 vs 0.45 minimum) because PM4 segments are individual surfaces, not whole-object groupings. On tile 22_18, 40543 PM4 objects become only 92 segments, each scored against 1985 WMO models.
- Tile 22_18 is NOT the oil platform (2 WMO placements in ADT). It's a "snowball fort" — multiple WMOs stacked together including Ulduar titan structures. CK24 0x0042084C type 66 is the dominant object.
- The `pm4 match-report` command is architecturally wrong for data recovery — it's placement-centered (needs existing ADT placements as anchors) instead of PM4-object-centered (match PM4 shapes against model shapes from archive).

### Fix attempts
- Attempted coordinate conversion in `Pm4MatchSupport` (RawAdtToWowWorld) — reverted. The match-report approach itself is wrong, not just the coordinates.

### Spec written
- `wow-viewer/specs/065-pm4-object-identity/spec.md` — 5-phase plan:
  - P1: Coordinate fix + known-tile correlation on 24_35
  - P1: CK24 identity table (CK24 → model path mapping)
  - P1: Synthetic PM4 signal generation from WMO collision
  - P2: CK24-grouped segment scoring (merge surfaces per CK24)
  - P2: Unknown-tile resolution using identity table + shape fallback

### Key insight
- The right approach is: generate what PM4 data WOULD look like for a given WMO/M2 model, then compare that synthetic data against real PM4 data. Not position-matching against broken ADT placements.

## 2026-06-16 — PM4 ADT writing reverted, replaced with match-report
- Deleted Pm4AdtWriter, Pm4BinaryAdtPatcher — ADT patching was corrupting output
- Replaced `pm4 write-adt` with `pm4 match-report` (human-readable markdown)
- LkAdtWriter untouched — not part of PM4 matcher work
- Checkpoint commit: 5133bfe3

## 2026-06-14 — Spec consolidation + tool fixes
- Replaced engine-program plan with viewer-first + UE bridge (509→35 lines)
- Archived 005, 020, 026, 033, 036, 059 (done/dead)
- Fixed stale status: 025→Complete, 060→Complete, 043→stale noted
- Marked research specs 030/031/032/038/040 consumed by 056
- Fixed 044 T006: removed dead MK Dataset GUI
- Added `--client-root --map` to `terrain-weak-signal-patch` for in-memory MPQ
- Ran weak signal on 0.5.3/0.5.5/1.12.1/3.0.1 maps — proven on real data
- Current focus: **046 PM4 asset matching** (C# done, Python lane needed)

## 2026-06-15 — Session polluted by hallucinations and wrong assumptions
- Implemented `pm4 dump-collision` command and WMO validation (works, 40 OIDs)
- Spent too long on tangents, wrong assumptions about M2/MD20, and coordinate systems
- Key deliverables: collision dumper tool, serialization fixes, Python scorer validation
- Memory bank updated. Needs fresh session with clear direction.
