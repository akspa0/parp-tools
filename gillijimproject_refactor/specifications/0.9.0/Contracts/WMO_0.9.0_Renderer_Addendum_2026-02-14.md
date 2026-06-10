# WMO 0.9.0 Renderer Addendum Contract — 2026-02-14

## Purpose
Record renderer-side ground truth for WMO anomalies observed in the 2026-02-14/15 passes and state enforceable contract implications for 0.9.0 tooling.

Primary symptom classes:
- group-local color/light data appears shifted or incorrect ("weird colored groups")
- liquids are arranged incorrectly (plane orientation/placement/layout mismatch)

## Renderer anchor set (ground-truth)
- WMO group render entry function(s): `TBD_FUN_WMO_GROUP_RENDER_ENTRY`
  - group stream validation
  - group material/light/color setup
  - handoff to group surface and liquid paths
- WMO liquid build/render function(s): `TBD_FUN_WMO_LIQUID_BUILD`, `TBD_FUN_WMO_LIQUID_DRAW`
  - liquid grid decode and vertex/index assembly
  - liquid origin/axis/stride interpretation
- Adjacent helpers to lock in same source cluster:
  - `TBD_FUN_WMO_GROUP_COLOR_APPLY`
  - `TBD_FUN_WMO_LIGHT_BIND`
  - `TBD_FUN_WMO_LIQUID_COPY`

## Contract statements (normative for 0.9.0 profile)
1. Parser-output group stream cardinalities MUST remain coherent across arrays consumed by renderer copy/build paths.
2. Group color/light linkage MUST be treated as required render input and remain index-domain consistent with group materials and batches.
3. Liquid chunk interpretation MUST preserve renderer-expected axis convention, origin basis, row/column traversal, and per-vertex layout mode.
4. Any parser hard-fail condition (group chunk overrun, flagged-optional dependency mismatch, liquid payload stride violation) MUST remain hard-fail; do not degrade to warning.
5. Tooling should preflight group/material/light and liquid-grid consistency before export/render handoff.

## Literal binding closure (to be filled)
- `TBD_LITERAL_GROUP_COLOR_BOUND`: `TBD_MESSAGE`
  - bound to `TBD_FUN_GROUP_COLOR_GATE`
  - disassembly proof: `TBD_ADDR: PUSH TBD_LITERAL`
- `TBD_LITERAL_LIQUID_LAYOUT_BOUND`: `TBD_MESSAGE`
  - bound to `TBD_FUN_LIQUID_LAYOUT_GATE`
  - disassembly proof: `TBD_ADDR: PUSH TBD_LITERAL`

Status: WMO renderer subtrack opened; addendum is now structured for anchor closure and runtime evidence fill.

## Traceability
- See staged run artifacts in:
  - `specifications/0.9.0/Contracts/runs/2026-02-14-wmo/`
