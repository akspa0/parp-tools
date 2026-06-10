# WMO 0.9.0 Renderer Addendum Contract — Closure (2026-02-14)

## Renderer anchor set (ground-truth, Ghidra-locked)
- WMO group render entry function(s): `FUN_006dd850` (opaque) and `FUN_006ddc00` (blend/alt pass)
  - group batch render walk
  - material/state programming
  - shared draw helper dispatch
- WMO liquid build/render function(s): `FUN_006df070` (build), `FUN_006def50` (draw dispatch)
  - liquid grid vertex/index assembly
  - liquid type dispatch to draw mode variants
- Adjacent helpers in same source cluster:
  - `FUN_006cee60` (group light-link gate)
  - `FUN_006e8960` (MLIQ optional-chunk parse/copy)
  - `FUN_006dedc0` (liquid index-domain gate)

## Normative contract implications (0.9.0 profile)
1. Parser-output group stream cardinalities MUST remain coherent with renderer buffer sizes and batch domains.
2. Group light-link coherence is required; invalid untagged/non-null link-head state triggers renderer assert path.
3. Liquid index/domain layout MUST satisfy renderer index-domain bound checks against `liquidVerts.x * liquidVerts.y`.
4. Existing parser hard-fails (chunk/token/layout violations) remain hard-fails and MUST NOT be downgraded.
5. Exporter/tooling preflight MUST include group/material/light linkage and liquid layout-domain checks before handoff.

## Literal binding closure
- **`TBD_LITERAL_GROUP_COLOR_BOUND` → resolved:**
  - Literal/message: `mapObjDefGroup->lightLinkList.Head() == 0`
  - Literal address: `0x008942e8`
  - Bound function: `FUN_006cee60`
  - Disassembly proof: `0x006cef14: PUSH 0x8942e8` (mismatch path to `CALL 0x006685d0` at `0x006cef28`)

- **`TBD_LITERAL_LIQUID_LAYOUT_BOUND` → resolved:**
  - Literal/message: `(idxBase[i] - vtxSub) < (uint) (group->liquidVerts.x * group->liquidVerts.y)`
  - Literal address: `0x008958f8`
  - Bound function: `FUN_006dedc0`
  - Disassembly proof: `0x006def1c: PUSH 0x8958f8` (mismatch path to `CALL 0x006685d0` at `0x006def30`)

## Evidence linkage
- Anchor closure artifact: `WMO_0.9.0_RenderPath_Part_01_Anchors_Ghidra_2026-02-14.md`
- Value-source closure artifact: `WMO_0.9.0_RenderPath_Part_09_Ghidra_Value_Source_Map_Closed_2026-02-14.md`
- Runtime table fill remains open in run track (`Part 08` template).
