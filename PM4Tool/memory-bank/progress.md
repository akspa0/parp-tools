# Project Progress: The Great PM4FileTests.cs Refactoring

**Last Updated:** 2025-07-03

**Overall Status:** Plan approved. Execution starting now.

---

### **Phase 1: Foundational Scaffolding**

- [x] **Step 1.1: Verify Directory Structure**
- [x] **Step 1.2: Create Data Models**
    - [x] `BuildingFragment.cs`
    - [x] `WmoMatchResult.cs`
    - [x] `WmoGeometryData.cs`
    - [x] `CompleteWmoModel.cs`
    - [x] `BoundingBox3D.cs`
- [x] **Step 1.3: Create Service Contracts and Skeletons**
    - [x] `IPm4BatchProcessor.cs` / `Pm4BatchProcessor.cs`
    - [x] `IPm4ModelBuilder.cs` / `Pm4ModelBuilder.cs`
    - [x] `IWmoMatcher.cs` / `WmoMatcher.cs`

---

### **Phase 2: Logic Migration and Refactoring**

- [x] **Step 2.1: Implement `Pm4ModelBuilder`**
- [x] **Step 2.2: Implement `WmoMatcher`**
- [x] **Step 2.3: Implement `Pm4BatchProcessor`**

---

### **Phase 3: Test Suite Modernization**

- [x] **Step 3.1: Clean Slate**
- [x] **Step 3.2: Create New Test Fixtures**
    - [x] `Pm4ModelBuilderTests.cs`
    - [x] `WmoMatcherTests.cs`
    - [x] `Pm4BatchProcessorTests.cs`
- [x] **Step 3.3: Write Targeted Tests**

---

### **Phase 4: Finalization and Deprecation**

- [ ] **Step 4.1: Sunset the Legacy Test File**
- [ ] **Step 4.2: Finalize Public API**
- [ ] **Step 4.3: Document Public API**

---

### 2025-07-03 – Build Fixes In Progress
- Began cleanup of duplicate file-scoped namespaces in Core.v2 service files (`IPm4ModelBuilder`, `Pm4ModelBuilder`, etc.).
- Replacing incorrect `using Warcraft.Game.Pm4` directives with correct Warcraft.NET namespaces.
- Initial compile pass reduced error count (namespace issues resolved for first file).