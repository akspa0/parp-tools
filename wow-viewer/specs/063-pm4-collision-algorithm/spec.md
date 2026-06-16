# Feature Specification: 063 — PM4 Collision Algorithm Reverse Engineering

**Feature Branch**: `063-pm4-collision-algorithm`

**Created**: 2026-06-15

**Status**: Session polluted — deliverables exist but reasoning is contaminated. Re-validate from scratch in next session. `pm4 dump-collision` command exists and WMO comparison works. M2 comparison needs fresh analysis.

## User Scenarios & Testing

### User Story 1 - Compare PM4 per-Object surfaces against WMO collision mesh for a known placement (Priority: P1)

As a PM4 researcher, I want to dump the WMO group collision triangles (MOPY/MOVT) alongside the corresponding PM4 surfaces (MSUR/MSVT/MSCN/MPRL) for a known placement on tiles 24_35/24_36/25_33/25_34, so I can compare them side-by-side and discover the simplification algorithm.

**Why this priority**: Without this raw data comparison, I can't see how WMO collision data gets transformed into PM4 format.

**Independent Test**: Taking tile 24_35 placement DUSKWOODABANDONED_BARN (OID=9304, 122 surfaces), the tool must dump both the WMO group 0 MOPY triangle normals and the PM4 surface normals for direct visual comparison.

**Acceptance Scenarios**:

1. **Given** a tile with PM4 + _obj0.adt + WMO files in MPQ, **When** I run the comparison tool with --tile 24_35 --oid 9304, **Then** it outputs all 122 PM4 surfaces with normals/plane-distances/MSCN positions AND the WMO's group collision triangle normals from MOPY.
2. **Given** the same tile and a different OID, **When** I run the comparison, **Then** the output includes both PM4 surface data and WMO group data for the correct placement.

### User Story 2 - Statistical analysis of normal/height mapping between WMO collision and PM4 surfaces (Priority: P2)

As a PM4 researcher, I want statistical comparison between WMO MOPY triangle normals and PM4 MSUR surface normals for the same placement, to determine how collision triangles are aggregated into surfaces.

**Why this priority**: Proving the aggregation method requires showing which MOPY triangles contribute to each MSUR entry.

**Acceptance Scenarios**:

1. **Given** a WMO group with N collision triangles and a PM4 segment with M surfaces, **When** I run statistical analysis, **Then** it shows which MOPY triangles map to which MSUR entry (by normal similarity).

## Functional Requirements

- FR-001: Read WMO root file from MPQ archive and extract MOGI group entries with per-group bounds
- FR-002: Read WMO group files (.wmo with _[0-9]+ suffix) and extract MOPY collision triangles with MOVT vertices
- FR-003: Compute per-triangle normals from MOPY/MOVT data
- FR-004: Read PM4 per-Object surface data (MSUR entries filtered by Ck24ObjectId)
- FR-005: Read PM4 MSCN scene node positions associated with each surface
- FR-006: Read PM4 MSVT vertices and MSVI indices per surface
- FR-007: Read MPRL position refs to understand the link between surfaces and positions
- FR-008: Align WMO model-space data with PM4 world-space data using the MODF placement transform
- FR-009: Dump both datasets side-by-side for manual comparison
- FR-010: Optional: compute normal similarity histogram between WMO triangles and PM4 surfaces

## Success Criteria

- Can reproduce the known OID→WMO mapping for all 13 placements on tile 24_35
- WMO group triangle normals and PM4 surface normals can be visually compared
- The first pass algorithm hypothesis is documented in a research note

## Assumptions

- PM4 files use the byte-swapped FourCC format (already handled)
- WMO files are accessible from the staged 3.3.5 client
- The axis convention difference between ADT/MODF and PM4/MSCN is known (XY swap)
