# Progress

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
