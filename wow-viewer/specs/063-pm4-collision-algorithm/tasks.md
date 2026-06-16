# Tasks: 063 — PM4 Collision Algorithm Reverse Engineering

**Plan**: [plan.md](plan.md) | **Spec**: [spec.md](spec.md)

## Phase 1: WMO Group Collision Reader + CLI

- [x] P1-001: Add `pm4 dump-collision` CLI command to inspect tool
- [x] P1-002: Read PM4 chunks, filter MSUR surfaces by OID, dump per-AttrMask surface data
- [x] P1-003: Read _obj0.adt placements, match by MSCN box center (with ADT axis swap)
- [x] P1-004: Read WMO root + groups from staged client archive
- [x] P1-005: Compute face normals from WMO group MOVT/MOVI data
- [x] P1-006: Run on all 4 tiles (24_35, 24_36, 25_33, 25_34) — 40 OID-to-WMO comparisons
- [x] P1-007: Validate WMO placement matching — OIDs correctly identify WMO (BARN, MED01, SMALL01, etc.)

Results: WMO matching works. Ratios range 3x-50x (triangles per PM4 surface). Tool correctly reads WMO collision mesh and groups.

## Phase 2: M2 Support

- [ ] P2-001: Add M2 placement matching from _obj0.adt MDDF entries
- [ ] P2-002: Read M2 collision data via `MdxCollisionReader` (MDLX format)
- [ ] P2-003: Fix `MdxCollisionReader` to handle MD20 format (fallback to `MdxSummaryReader` collision summary)
- [ ] P2-004: Add kind-preference to placement matching (M2 OIDs → prefer M2 placements)
- [ ] P2-005: Run on all M2 tiles (54_22, 46_41, 47_51, 49_39, 45_42, 14_51, 47_52, etc.)

Status: MDLX M2 files work. MD20 files need `MdxCollisionReader` fix. ~80% of dev M2 files use MD20 (based on the 3 failing OIDs found in testing).

## Phase 3: Algorithm Analysis

- [ ] P3-001: Compare PM4 surface normals vs WMO triangle normals for one well-matched object
- [ ] P3-002: Compute the plane-distance transform (map WMO vertex Z to PM4 surface height)
- [ ] P3-003: Document the PM4→WMO generation algorithm
- [ ] P3-004: Do the same for M2 once Phase 2 is complete

## Completion Checklist

- [ ] WMO collision dumper: validated on 40 OIDs across 4 tiles
- [ ] M2 collision dumper: MDLX tested, MD20 in progress
- [ ] Known limitation: `MdxCollisionReader` doesn't handle MD20 format
- [ ] Known limitation: placement matching uses MSCN box center (10-20 unit error vs seg export bounds)
- [ ] No references to `H:\CLIENTS` in any new files
