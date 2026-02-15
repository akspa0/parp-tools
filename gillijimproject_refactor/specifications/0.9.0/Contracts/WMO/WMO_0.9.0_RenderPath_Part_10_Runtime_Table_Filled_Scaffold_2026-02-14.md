# WMO 0.9.0 Render Path — Part 10 (Runtime Table Filled Scaffold)

Date: 2026-02-14  
Status: Scaffold prepared (static anchors resolved, runtime values pending)

## Scope
Runtime capture worksheet pre-bound to the closed Ghidra gate anchors and literal bindings.

## Locked runtime breakpoints

### Gate A — Group color/light linkage gate
- Function: FUN_006cee60
- Function address: 0x006cee60
- Branch window: 0x006cef02..0x006cef28
- Literal binding: 0x008942e8 (mapObjDefGroup->lightLinkList.Head() == 0)
- Mismatch call: 0x006685d0 at 0x006cef28
- Suggested breakpoints:
  - 0x006cef02 (load list head)
  - 0x006cef08 (tag test)
  - 0x006cef0c (null test)
  - 0x006cef14 (literal push)
  - 0x006cef28 (assert/log call)

### Gate B — Liquid layout/index-domain gate
- Function: FUN_006dedc0
- Function address: 0x006dedc0
- Branch window: 0x006def14..0x006def30
- Literal binding: 0x008958f8 ((idxBase[i] - vtxSub) < (uint)(group->liquidVerts.x * group->liquidVerts.y))
- Mismatch call: 0x006685d0 at 0x006def30
- Suggested breakpoints:
  - 0x006def00 (load idxBase[i])
  - 0x006def04 (subtract vtxSub)
  - 0x006def07 (load liquidVerts.x)
  - 0x006def0d (multiply by liquidVerts.y)
  - 0x006def14 (compare)
  - 0x006def1c (literal push)
  - 0x006def30 (assert/log call)

## Table A — Group color/light anomaly (runtime fill)
| Asset Case | Build/Binary | Gate Hit Count | Function | Compare Site | groupPtr | lightHeadRaw ([+0xB8]) | tagBit(AL&1) | nullCheck(EAX==0) | Branch Path (Pass/Mismatch) | Literal Addr | Outcome (Correct/Weird Color) | Notes |
|---|---|---:|---|---|---|---|---|---|---|---|---|---|
| Stormwind group (problem) | 0.9.0 / Build 3807 | TBD | FUN_006cee60 | 0x006cef02..0x006cef28 | TBD | TBD | TBD | TBD | TBD | 0x008942e8 | TBD | Primary color anomaly sample |
| Secondary problem case | 0.9.0 / Build 3807 | TBD | FUN_006cee60 | 0x006cef02..0x006cef28 | TBD | TBD | TBD | TBD | TBD | 0x008942e8 | TBD | Validation sample |
| Control (known-good group) | 0.9.0 / Build 3807 | TBD | FUN_006cee60 | 0x006cef02..0x006cef28 | TBD | TBD | TBD | TBD | TBD | 0x008942e8 | TBD | Baseline |

## Table B — Liquid arrangement anomaly (runtime fill)
| Asset Case | Build/Binary | Gate Hit Count | Function | Compare Site | idxBase[i]@0x006def00 | vtxSub | lhs=(idx-vtxSub) | liquidVerts.x | liquidVerts.y | rhs=(x*y) | Branch Path (JC Pass / Mismatch) | Literal Addr | Outcome (Correct/Misaligned) | Notes |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| Stormwind liquid (problem) | 0.9.0 / Build 3807 | TBD | FUN_006dedc0 | 0x006def14..0x006def30 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 0x008958f8 | TBD | Primary layout failure sample |
| Secondary problem case | 0.9.0 / Build 3807 | TBD | FUN_006dedc0 | 0x006def14..0x006def30 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 0x008958f8 | TBD | Validation sample |
| Control (known-good liquid) | 0.9.0 / Build 3807 | TBD | FUN_006dedc0 | 0x006def14..0x006def30 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 0x008958f8 | TBD | Baseline |

## Runtime capture checklist
1. Capture full register snapshot at each compare site and at literal push sites.
2. Record whether mismatch path reached the 0x006685d0 call.
3. Correlate each row with visual renderer outcome (screenshot/video hash if available).
4. Keep at least one failing and one control row fully populated per table.
5. Preserve raw values first; derive pass/fail interpretation second.

## Traceability
- Anchors: WMO_0.9.0_RenderPath_Part_01_Anchors_Ghidra_2026-02-14.md
- Value-source closure: WMO_0.9.0_RenderPath_Part_09_Ghidra_Value_Source_Map_Closed_2026-02-14.md
- Addendum closure: WMO_0.9.0_Renderer_Addendum_Closure_2026-02-14.md
