# MSUR Export Refactor Plan - Comprehensive Strategic Approach
*Created: 2025-07-28 - Based on Memory Bank Synthesis and Proof-of-Concept Audit*

## 🚨 CRITICAL DISCOVERY SUMMARY

### Root Cause: Architectural Mismatch
**WORKING PROOF-OF-CONCEPT APPROACH:**
```csharp
var surf = pm4File.MSUR!.Entries[surfIdx];  // ✅ Direct chunk access
int first = (int)surf.MsviFirstIndex;        // ✅ Real field values
uint key = pm4File.MSUR.Entries[i].Unknown_0x1C; // ✅ Working per-object export
```

**CURRENT BROKEN APPROACH:**
```csharp
var surfaces = scene.Surfaces;  // ❌ Already processed/converted objects
var value = prop.GetValue(s);   // ❌ Reflection on processed objects = zeros
```

### The Fundamental Problem
- **Proof-of-concept used `WoWToolbox.Core.v2` with direct chunk access**
- **Current parpToolbox uses processed `Pm4Scene` objects with reflection**
- **All field values are zero because reflection targets processed objects, not raw chunks**
- **JSON export attempts failed because data never existed in processed objects**

## 🎯 STRATEGIC REFACTOR OBJECTIVES

### Primary Goals
1. **Achieve working per-object MSUR export** using direct chunk access
2. **Maintain wow.tools.local as dependency** (not fork)
3. **Eliminate tool fragmentation** (8+ overlapping exporters)
4. **Ensure byte-for-byte legacy compatibility** for validation
5. **Create maintainable, extensible architecture**

### Success Criteria
- [ ] **Per-object OBJ files** generated successfully (like proof-of-concept)
- [ ] **Real field values** in JSON exports (no more zeros)
- [ ] **SHA256 hash match** with legacy exporter output
- [ ] **Single, unified export pipeline** (no fragmented tools)
- [ ] **Clean separation** between parsing and export logic

## 📋 VALIDATED ARCHITECTURAL APPROACH

### Phase 1: Direct Chunk Access Layer
**Create intermediate layer that bridges wow.tools.local chunks to export logic**

#### 1.1 PM4 Chunk Access Service
```csharp
public interface IPm4ChunkAccessService
{
    PM4File LoadPm4File(string path);
    IReadOnlyList<MsurEntry> GetMsurEntries(PM4File pm4File);
    IReadOnlyList<MsviIndex> GetMsviIndices(PM4File pm4File);
    IReadOnlyList<MsvtVertex> GetMsvtVertices(PM4File pm4File);
}
```

**Purpose:** 
- Direct access to raw chunk data (like proof-of-concept)
- Abstracts wow.tools.local types from export logic
- Enables real field value extraction

#### 1.2 Field Mapping Service
```csharp
public interface IPm4FieldMappingService
{
    Pm4SurfaceData MapMsurEntry(MsurEntry entry, int index);
    Pm4VertexData MapMsvtVertex(MsvtVertex vertex, int index);
    // Real field mapping, not reflection
}
```

**Purpose:**
- Explicit field mapping (not reflection)
- Ensures real values are captured
- Validates against official documentation

### Phase 2: Unified Export Pipeline
**Single export service that replaces all 8+ fragmented exporters**

#### 2.1 Unified PM4 Export Service
```csharp
public interface IPm4ExportService
{
    Task<ExportResult> ExportAsync(string pm4Path, PM4ExportOptions options);
}

public class PM4ExportOptions
{
    public string OutputDirectory { get; set; }
    public ExportFormat Format { get; set; } // OBJ, JSON, CSV
    public ExportStrategy Strategy { get; set; } // PerObject, PerSurface, Unified
    public bool IncludeRawData { get; set; }
}
```

**Purpose:**
- Single entry point for all PM4 export needs
- Configurable output formats and strategies
- Replaces fragmented tool ecosystem

#### 2.2 Export Strategy Pattern
```csharp
public interface IExportStrategy
{
    Task<ExportResult> ExecuteAsync(PM4File pm4File, PM4ExportOptions options);
}

// Concrete strategies:
// - PerObjectStrategy (using Unknown_0x1C grouping)
// - PerSurfaceStrategy (individual surfaces)
// - UnifiedStrategy (complete scene)
```

**Purpose:**
- Clean separation of concerns
- Easy to test and validate
- Extensible for new export types

### Phase 3: Validation Framework
**Ensure compatibility and prevent regression**

#### 3.1 Legacy Compatibility Service
```csharp
public interface ILegacyCompatibilityService
{
    Task<bool> ValidateOutputAsync(string outputPath, string referencePath);
    Task<ValidationReport> CompareWithLegacyAsync(ExportResult result);
}
```

**Purpose:**
- SHA256 hash validation against legacy output
- Regression prevention
- Automated compatibility testing

## 🛠 IMPLEMENTATION ROADMAP

### Step 1: Foundation (Direct Chunk Access)
**Duration: 1-2 sessions**
- [ ] Create `IPm4ChunkAccessService` implementation
- [ ] Create `IPm4FieldMappingService` implementation  
- [ ] Test basic chunk access and field extraction
- [ ] Validate real field values (no more zeros)

**Validation:** JSON export with real MSUR field values

### Step 2: Unified Export Service
**Duration: 2-3 sessions**
- [ ] Create `IPm4ExportService` interface and implementation
- [ ] Implement `PerObjectStrategy` using Unknown_0x1C grouping
- [ ] Replace existing fragmented exporters with unified service
- [ ] Test per-object OBJ generation

**Validation:** Per-object OBJ files generated successfully

### Step 3: Legacy Compatibility
**Duration: 1-2 sessions**
- [ ] Create `ILegacyCompatibilityService` implementation
- [ ] Generate SHA256 hashes of outputs
- [ ] Compare with legacy exporter results
- [ ] Fix any compatibility issues

**Validation:** SHA256 hash match with legacy exporter

### Step 4: Integration and Testing
**Duration: 1 session**
- [ ] Update CLI to use unified export service
- [ ] Create integration tests
- [ ] Update documentation
- [ ] Clean up deprecated code

**Validation:** Complete working system with tests

## 🔧 KEY ARCHITECTURAL DECISIONS

### 1. Direct Chunk Access Over Scene Processing
**Decision:** Access raw chunk data directly, not processed Pm4Scene objects
**Rationale:** Proof-of-concept success, current reflection approach fundamentally broken
**Impact:** Real field values, working per-object export

### 2. Service-Based Architecture
**Decision:** Use dependency injection and service interfaces
**Rationale:** Testability, maintainability, extensibility
**Impact:** Clean separation of concerns, easy testing

### 3. Strategy Pattern for Export Types
**Decision:** Configurable export strategies (per-object, per-surface, unified)
**Rationale:** Single entry point, reduced complexity
**Impact:** Eliminates tool fragmentation

### 4. Validation-First Approach
**Decision:** Legacy compatibility testing from day one
**Rationale:** Prevent regression, ensure correctness
**Impact:** Confidence in outputs, automated validation

## 🎯 SUCCESS METRICS

### Immediate Success (Phase 1)
- [ ] JSON exports contain real field values (not zeros)
- [ ] Direct chunk access working
- [ ] Field mapping validated

### Core Success (Phase 2)
- [ ] Per-object OBJ files generated
- [ ] Unknown_0x1C grouping working
- [ ] Single export pipeline operational

### Complete Success (Phase 3-4)
- [ ] SHA256 hash match with legacy exporter
- [ ] All 8+ fragmented exporters replaced
- [ ] Integration tests passing
- [ ] Documentation complete

## 🚨 RISK MITIGATION

### Risk: Repeat of Current Issues
**Mitigation:** 
- Direct chunk access (proven approach)
- No reflection on processed objects
- Validation at each step

### Risk: wow.tools.local Compatibility
**Mitigation:**
- Service layer abstraction
- No modifications to library code
- Clean dependency usage

### Risk: Implementation Drift
**Mitigation:**
- Clear architectural decisions documented
- Service interfaces enforce contracts
- Automated tests prevent regression

## 📚 MEMORY BANK INTEGRATION

This plan addresses all major discoveries from the memory bank:
- ✅ **Tool fragmentation:** Single unified export service
- ✅ **Implementation drift:** Clear architectural decisions
- ✅ **Legacy compatibility:** SHA256 validation framework
- ✅ **Direct chunk access:** Core architectural principle
- ✅ **Field validation:** Explicit mapping, no reflection

---

## 📋 NEXT STEPS

**Phase 1 Implementation Ready**
- Foundation services defined
- Clear success criteria established
- Risk mitigation planned
- Memory bank insights integrated

**Approval Required Before Implementation**
This plan represents a comprehensive, strategic approach based on memory bank synthesis and proof-of-concept audit. Implementation should only proceed with explicit approval and commitment to this architectural direction.
