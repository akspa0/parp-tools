# PM4 Database-First Architecture Plan

## 🚨 CRITICAL ARCHITECTURE BREAKTHROUGH (2025-07-27)

**GLOBAL MESH SYSTEM DISCOVERED:** The fundamental PM4 architecture has been **completely solved**. PM4 files implement a **cross-tile linkage system** requiring **multi-tile processing** for complete geometry assembly.

### **Mathematical Validation:**
- **58.4% of triangles** reference vertices from adjacent tiles (30,677 out of 52,506)
- **63,297 cross-tile vertex indices** in perfect sequential range: 63,298-126,594
- **Zero gap** between local (0-63,297) and cross-tile vertex ranges
- **Complete architectural assembly requires directory-wide PM4 processing**

### **Surface Encoding System:**
- **GroupKey 3** (1,968 surfaces): Spatial coordinates, local tile geometry
- **GroupKey 18** (8,988 surfaces): Mixed data, boundary objects spanning tiles
- **GroupKey 19** (30,468 surfaces): Encoded linkage data, cross-tile references (74% of surfaces)
- **BoundsMaxZ in encoded groups**: Hex-encoded tile/object references, NOT coordinates
- **95.5% consistency** in GroupKey 19 encoding - systematic linkage system

**IMPACT:** All previous single-tile processing approaches are fundamentally incomplete. **Multi-tile processing and cross-tile vertex resolution are mandatory.**

---

## Executive Summary
The PM4 exporter has been redesigned as a **database-first architecture** with two distinct phases:
1. **PM4 → SQLite Importer**: Comprehensive raw chunk storage for future-proofing
2. **SQLite → Export Subsystem**: Flexible SQL-driven OBJ and analysis tools

This approach treats PM4 as the complex hierarchical database it actually is, enabling robust analysis and future tooling development.

## Critical Discoveries (Updated with Database Pattern Analysis)
1. **Object Instance System**: **MPRL.Unknown4** = Object Instance ID (227 unique objects), linking to **MSLK.ParentIndex**
2. **LOD Control System**: **MPRL.Unknown14/Unknown16** encodes rendering detail levels:
   - (-1, 16383) = Full detail rendering (906 instances)
   - (0-5, 0) = LOD levels 0-5 (667 instances)
3. **Object State Management**: **MPRL** contains sophisticated encoding:
   - Unknown0 = Category ID (4630 = building type)
   - Unknown2 = State flag (-1 = active)
   - Unknown6 = Property flag (32768 = bit 15 set)
4. **Coordinate System**: Local tile coordinates require XX*533.33 + YY*533.33 world offset
5. **Cross-tile References**: ~64% vertex data loss due to missing adjacent tile vertices
6. **Legacy Parity**: Need byte-for-byte identical output to legacy exporters
7. **Database-First Validation**: SQLite pattern analysis confirmed field meanings and eliminated "outlier" misinterpretations

## Observed Implementation Drift (July 26 2025)
Despite the above discoveries being reiterated multiple times, refactors have repeatedly regressed to prior, incorrect grouping logic (e.g., SurfaceKey-based grouping).  This indicates:
- Incomplete propagation of Critical Discoveries into code changes.
- Legacy files (e.g., `NewPm4Exporter.cs`) being patched rather than replaced, re-introducing outdated assumptions.
- Lack of automated regression tests that fail when grouping deviates from **MSUR.Index_0x01 → MSLK.Unknown_0x00**.

**Action Items**
1. Freeze old exporter paths; create a clean `Pm4PerObjectExporter` that exclusively uses the validated keys.
2. Add unit test: given sample PM4, assert assembled object count equals proof-of-concept.
3. Add SHA regression test against legacy OBJ output.
4. Code reviews must cross-check every change against this plan before merge.

## Database-First Architecture Overview

### Phase 1: PM4 → SQLite Importer (Complete Data Capture)
1. **Pm4DatabaseExporter** - Comprehensive raw chunk storage with optimized batching
2. **RawChunkStorage** - Every chunk type preserved with full fidelity (MSPV, MSVT, MSVI, MSLK, MSUR, MPRL, MPRR, etc.)
3. **CrossTileVertexResolver** - Global vertex mapping and reference resolution
4. **MetadataCapture** - Parse timestamps, chunk offsets, file relationships
5. **FutureProofing** - Raw data enables new tools as PM4 understanding evolves

### Phase 2: Export Subsystem Framework (SQL-Driven Tools)
1. **SqlQueryLibrary** - Modular extraction strategies (spatial, hierarchical, surface-based)
2. **ObjExportPipeline** - Building objects, individual surfaces, collision meshes
3. **AnalysisTools** - CSV/JSON chunk relationship analysis, coordinate debugging
4. **FormatExtensibility** - OBJ/MTL, glTF, PLY, custom binary formats
5. **VisualizationHelpers** - Bounds checking, fragment analysis, cross-tile validation

### Data Flow
```
PM4File → DatabaseImporter → SQLite → [QueryStrategy] → ExportPipeline → OBJ/Analysis
```

## Implementation Strategy

### Phase 1: Complete Database Importer (In Progress ✅)
- [x] **Pm4DatabaseExporter** - Optimized batching with 50k batch sizes
- [x] **Performance Optimization** - Disabled change tracking, memory management
- [x] **Cross-tile Resolution** - Global vertex mapping and reference resolution
- [ ] **Raw Chunk Storage** - Expand to include every chunk type with full fidelity
- [ ] **Metadata Capture** - Store chunk offsets, parse timestamps, file relationships
- [ ] **Remove Building Extraction** - Focus purely on data import, not object assembly

### Phase 2: SQL Query Library Design
- [ ] **Spatial Clustering Queries** - Proximity-based object grouping
- [ ] **Hierarchical Analysis Queries** - ParentIndex-based relationships (MSLK.ParentIndex_0x04)
- [ ] **Surface Group Queries** - MSUR-based object boundaries (MSUR.Index_0x01 → MSLK.Unknown_0x00)
- [ ] **Cross-tile Reference Queries** - Global vertex mapping and validation
- [ ] **Analysis Queries** - Chunk relationship validation, bounds checking

### Phase 3: Export Subsystem Framework
- [ ] **OBJ/MTL Pipeline** - Building objects, individual surfaces, material support
- [ ] **Collision Export** - MSCN triangle export with `--collision` flag
- [ ] **Analysis Tools** - CSV/JSON chunk relationship export, coordinate debugging
- [ ] **Visualization Helpers** - Bounds checking, fragment analysis, cross-tile validation
- [ ] **Format Extensions** - glTF, PLY, custom binary formats

### Phase 4: Legacy Compatibility & Testing
- [ ] **Legacy OBJ Format** - Ensure byte-for-byte output parity with existing tools
- [ ] **Coordinate System Fixes** - X-axis inversion and transform validation
- [ ] **Regression Tests** - SHA comparison against legacy outputs
- [ ] **Performance Benchmarks** - Database query optimization and export speed

### Phase 5: Advanced Features
- [ ] **Navigation Mesh Export** - Recast/Detour integration using walkable MSCN flags
- [ ] **BVH/Spatial Acceleration** - Query optimization for large scenes
- [ ] **Multi-tile Batch Processing** - Process entire regions efficiently
- [ ] **Interactive Query Tools** - SQL-based debugging and analysis utilities

## Current Status (July 26, 2025)

### ✅ Completed Components
1. **Pm4DatabaseExporter** - Core database import with optimized performance
   - 50k batch sizes for vertices/triangles/links
   - Disabled change tracking during bulk operations
   - Memory management with entity clearing
   - Cross-tile vertex resolution working
2. **Database Schema** - EF Core models for PM4 data
   - Vertices, Triangles, Surfaces, Links, Placements tables
   - Spatial indexing and relationship tracking
3. **Performance Optimization** - MSLK links export now fast
4. **Cross-tile Loading** - Global vertex mapping resolves 64% data loss

### 🚧 In Progress
1. **Database hangs** during hierarchical building extraction
2. **Missing raw chunk storage** for future-proofing
3. **Export subsystem** not yet implemented

### 🎯 Immediate Next Steps
1. **Remove building extraction** from database importer (focus on pure data import)
2. **Add raw chunk storage tables** for every PM4 chunk type
3. **Design SQL query library** for flexible object extraction
4. **Create new export subsystem** that reads from SQLite

## Technical Specifications

### Database Schema Extensions Needed
```sql
-- Raw chunk storage for future-proofing
CREATE TABLE RawChunks (
    Id INTEGER PRIMARY KEY,
    Pm4FileId INTEGER,
    ChunkType TEXT,         -- 'MSLK', 'MSUR', 'MPRL', etc.
    ChunkOffset INTEGER,    -- Position in original file
    ChunkSize INTEGER,      -- Size in bytes
    RawData BLOB,          -- Complete raw chunk data
    ParsedAt DATETIME,     -- When this was processed
    ParserVersion TEXT     -- Version of parser used
);
```

### SQL Query Strategies
```sql
-- Spatial clustering for building extraction
SELECT * FROM Surfaces 
WHERE BoundsCenterX BETWEEN ? AND ? 
  AND BoundsCenterY BETWEEN ? AND ?
ORDER BY BoundsCenterX, BoundsCenterY;

-- Hierarchical object assembly
SELECT l.*, s.* FROM Links l
JOIN Surfaces s ON s.Id = l.SurfaceId
WHERE l.ParentIndex = ?
ORDER BY l.MspiFirstIndex;
```

### Export Pipeline Architecture
```csharp
public interface IExportStrategy {
    Task<ExportResult> ExtractAsync(int pm4FileId, ExportOptions options);
}

public class SpatialExportStrategy : IExportStrategy { }
public class HierarchicalExportStrategy : IExportStrategy { }
public class SurfaceGroupExportStrategy : IExportStrategy { }
```

## Testing Strategy

### Regression Tests
1. **SHA Comparison**: Verify byte-for-byte output parity with legacy
2. **Object Count**: Validate correct number of building objects
3. **Triangle Count**: Ensure 38K-654K triangles per building
4. **Vertex Integrity**: Check for missing/out-of-bounds indices

### Integration Tests
1. **Cross-tile Loading**: Test with multi-tile PM4 files
2. **Memory Usage**: Validate reasonable memory consumption
3. **Performance**: Benchmark against legacy export times
4. **Edge Cases**: Handle empty objects, missing data

## Migration Path

### Step 1: Clean Implementation
- Create completely new exporter class
- Implement core grouping logic
- Add basic vertex resolution

### Step 2: Legacy Integration
- Port exact legacy export format
- Add PM4File.LegacyAliases for compatibility
- Implement coordinate transformation

### Step 3: Testing & Validation
- Run comprehensive regression tests
- Compare outputs with legacy tools
- Fix any discrepancies

### Step 4: Deployment
- Replace old exporter completely
- Update CLI commands
- Update documentation

## Risk Mitigation

### Data Loss Prevention
- Comprehensive vertex index validation
- Cross-tile reference resolution
- Missing vertex detection and handling

### Compatibility Assurance
- Legacy output format preservation
- Coordinate system consistency
- Error handling for edge cases

## Success Criteria
- [ ] Build passes with zero errors
- [ ] All PM4 exports match legacy outputs (SHA256)
- [ ] Object grouping produces meaningful building-scale objects
- [ ] Cross-tile vertex references resolved correctly
- [ ] Comprehensive test coverage ≥80%
- [ ] Documentation updated with new architecture

## Timeline
- **Phase 1**: 2-3 days (clean implementation)
- **Phase 2**: 1-2 days (legacy integration)
- **Phase 3**: 2-3 days (testing & validation)
- **Phase 4**: 1 day (deployment & documentation)

Total estimated time: 5-7 days for complete refactor

## Current Status
- [x] Memory bank plan created
- [ ] Phase 1: Clean implementation pending
- [ ] Phase 2: Legacy integration pending
- [ ] Phase 3: Testing & validation pending
- [ ] Phase 4: Deployment pending

## Next Steps
1. Create new Pm4Exporter class from scratch
2. Implement reflection-based MSLK field access
3. Add MSLK.ParentIndex_0x04 grouping logic
4. Build cross-tile vertex resolution system
5. Port legacy OBJ export format
6. Run comprehensive regression tests
7. Update documentation and CLI commands
