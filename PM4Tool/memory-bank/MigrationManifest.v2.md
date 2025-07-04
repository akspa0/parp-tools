# PM4 Refactor: Migration Manifest

This document outlines the plan for migrating critical algorithms, data structures, and utilities from the experimental test project (`WoWToolbox.Tests`) into the core library (`WoWToolbox.Core.v2`). The goal is to consolidate all PM4 processing knowledge into a robust, maintainable, and production-ready library.

## Phase 1: Foundational Services Migration (Core Systems)

**Status: In Progress**

### Sub-Task 1.1: `Pm4CoordinateTransforms` to `CoordinateService`

- **Status:** ✅ **Completed**
- **Summary:** The static helper class `Pm4CoordinateTransforms`, which was being used extensively in test files, has been successfully migrated into a modern, dependency-injectable service within `WoWToolbox.Core.v2`.
- **Key Actions:**
    - **Investigation:** After extensive searching, the original source code for `Pm4CoordinateTransforms` could not be located. The class was likely a private or internal helper within a test file that was not accessible with the available tools.
    - **Reconstruction:** The class was reconstructed from its usage patterns observed in `PM4FileTests.cs` and `SimpleAnalysisTest.cs`.
    - **Interface (`ICoordinateService.cs`):** An interface was created to define a clear contract for coordinate transformation operations. This promotes loose coupling and testability.
    - **Implementation (`CoordinateService.cs`):** A concrete implementation was created, providing standard Z-up to Y-up coordinate transformations and a robust method for computing vertex normals.
    - **Location:** The new files have been placed in `src/WoWToolbox.Core.v2/Services/PM4/`, aligning with the project's architectural patterns.
- **Files Created:**
    - `src/WoWToolbox.Core.v2/Services/PM4/ICoordinateService.cs`
    - `src/WoWToolbox.Core.v2/Services/PM4/CoordinateService.cs`

### Sub-Task 1.2: PM4 Validation Service

- **Status:** ❌ **Blocked**
- **Blocker:** The file system tools (`grep_search`, `view_file`) are consistently failing to read or search the contents of `PM4FileTests.cs`. This prevents the original validation logic from being located and replaced with calls to the new `Pm4Validator` service. This sub-task cannot proceed until the tooling issue is resolved.
- **Goal:** To migrate the various validation checks scattered throughout `PM4FileTests.cs` into a centralized `Pm4Validator` service.
- **Source Logic:** `PM4FileTests.cs` (e.g., `ValidateChunkIndices`, `ValidateFaceCounts`, `ValidateMslkStructure`).
- **Target:** `src/WoWToolbox.Core.v2/Services/PM4/Pm4Validator.cs`.
- **Key Responsibilities:**
    - Validate chunk magic and version.
    - Verify integrity of chunk indices (e.g., MSVI indices point to valid MSVT vertices).
    - Check for degenerate triangles or invalid surface definitions.
    - Ensure consistency between related chunks (e.g., MSUR and MSVI).

## Phase 2: Geometry and Exporter Migration

This phase focuses on migrating the logic responsible for building renderable geometry and exporting it to standard formats.

### Sub-Task 2.1: Render Mesh Builder

- **Goal:** To create a `RenderMeshBuilder` service that can construct a unified, render-ready mesh from PM4 chunks.
- **Source Logic:** `PM4FileTests.cs` (e.g., `GenerateRenderMesh`, `ExportCompleteRenderMeshToObj`).
- **Target:** `src/WoWToolbox.Core.v2/Services/PM4/RenderMeshBuilder.cs`.
- **Key Responsibilities:**
    - Combine vertices from MSVT and MSPV chunks.
    - Generate faces from MSVI and MSUR chunks.
    - Compute or retrieve vertex normals (using `CoordinateService`).
    - Handle different surface types (triangles, quads, polygons).
- **Status:** ⏳ **Pending**

### Sub-Task 2.2: OBJ Exporter

- **Goal:** To create a generic `ObjExporter` service capable of writing mesh data to the OBJ file format.
- **Source Logic:** `PM4FileTests.cs` (numerous `Export...ToObj` methods).
- **Target:** `src/WoWToolbox.Core.v2/Services/Export/ObjExporter.cs`.
- **Key Responsibilities:**
    - Write vertices, normals, and texture coordinates.
    - Write face definitions.
    - Handle material library (MTL) references.
- **Status:** ⏳ **Pending**

### Sub-Task 2.3: Diagnostic Exporters (CSV/JSON)

- **Goal:** To consolidate the diagnostic export logic into dedicated services.
- **Source Logic:** `PM4FileTests.cs` (e.g., `ExportDoodadDataToCsv`, `ExportNodeDataToJson`).
- **Target:**
    - `src/WoWToolbox.Core.v2/Services/Export/CsvExporter.cs`
    - `src/WoWToolbox.Core.v2/Services/Export/JsonExporter.cs`
- **Key Responsibilities:**
    - Serialize data structures to CSV and JSON formats.
    - Provide flexible configuration for export content.
- **Status:** ⏳ **Pending**

## Phase 3: High-Level Building Extraction

This phase focuses on consolidating the complex logic for identifying and extracting complete building structures from a PM4 file.

### Sub-Task 3.1: Building Extraction Service

- **Goal:** To create a high-level `BuildingExtractionService` that encapsulates the logic for identifying and extracting building models.
- **Source Logic:** `PM4FileTests.cs` (e.g., `ExportBuildings_UsingMdsfLinks`, `ExportBuildings_UsingMslkRootNodesWithSpatialClustering`).
- **Target:** `src/WoWToolbox.Core.v2/Services/PM4/BuildingExtractionService.cs`.
- **Key Responsibilities:**
    - Implement multiple strategies for building identification (e.g., MDSF/MDOS links, MSLK root node clustering).
    - Combine structural elements (from MSLK/MSPV) and render geometry (from MSUR/MSVT).
    - Produce a list of `CompleteWMOModel` objects representing the extracted buildings.
- **Status:** ⏳ **Pending**

## Phase 4: Test Modernization and Deprecation

This phase focuses on creating a new suite of tests that validate the migrated services and deprecating the old, monolithic test files.

### Sub-Task 4.1: Unit and Integration Tests

- **Goal:** To write comprehensive tests for the new services in `WoWToolbox.Core.v2`.
- **Target:** A new test project, `WoWToolbox.Core.v2.Tests`.
- **Key Responsibilities:**
    - Write unit tests for each service, mocking dependencies where appropriate.
    - Write integration tests that validate the end-to-end functionality of the PM4 processing pipeline.
- **Status:** ⏳ **Pending**

### Sub-Task 4.2: Deprecate Legacy Test Files

- **Goal:** To remove the monolithic and experimental test files from the `WoWToolbox.Tests` project.
- **Target Files:**
    - `PM4FileTests.cs`
    - `SimpleAnalysisTest.cs`
- **Action:** Once all logic has been migrated and validated, these files will be deleted.
- **Status:** ⏳ **Pending**

This document outlines the plan for migrating critical algorithms, data structures, and utilities from the experimental test project (`WoWToolbox.Tests`) into the core library (`WoWToolbox.Core.v2`). The goal is to consolidate all PM4 processing knowledge into a robust, maintainable, and production-ready library.

## Phase 1: Foundational Services Migration (Core Systems)

**Status: In Progress**

### Sub-Task 1.1: `Pm4CoordinateTransforms` to `CoordinateService`

- **Status:** ✅ **Completed**
- **Summary:** The static helper class `Pm4CoordinateTransforms`, which was being used extensively in test files, has been successfully migrated into a modern, dependency-injectable service within `WoWToolbox.Core.v2`.
- **Key Actions:**
    - **Investigation:** After extensive searching, the original source code for `Pm4CoordinateTransforms` could not be located. The class was likely a private or internal helper within a test file that was not accessible with the available tools.
    - **Reconstruction:** The class was reconstructed from its usage patterns observed in `PM4FileTests.cs` and `SimpleAnalysisTest.cs`.
    - **Interface (`ICoordinateService.cs`):** An interface was created to define a clear contract for coordinate transformation operations. This promotes loose coupling and testability.
    - **Implementation (`CoordinateService.cs`):** A concrete implementation was created, providing standard Z-up to Y-up coordinate transformations and a robust method for computing vertex normals.
    - **Location:** The new files have been placed in `src/WoWToolbox.Core.v2/Services/PM4/`, aligning with the project's architectural patterns.
- **Files Created:**
    - `src/WoWToolbox.Core.v2/Services/PM4/ICoordinateService.cs`
    - `src/WoWToolbox.Core.v2/Services/PM4/CoordinateService.cs`

### Sub-Task 1.2: PM4 Validation Service

- **Status:** ❌ **Blocked**

- **Goal:** To migrate the various validation checks scattered throughout `PM4FileTests.cs` into a centralized `Pm4Validator` service.
- **Source Logic:** `PM4FileTests.cs` (e.g., `ValidateChunkIndices`, `ValidateFaceCounts`, `ValidateMslkStructure`).
- **Target:** `src/WoWToolbox.Core.v2/Services/PM4/Pm4Validator.cs`.
- **Key Responsibilities:**
    - Validate chunk magic and version.
    - Verify integrity of chunk indices (e.g., MSVI indices point to valid MSVT vertices).
    - Check for degenerate triangles or invalid surface definitions.
    - Ensure consistency between related chunks (e.g., MSUR and MSVI).
- **Status:** ⏳ **Pending**

## Phase 2: Geometry and Exporter Migration

This phase focuses on migrating the logic responsible for building renderable geometry and exporting it to standard formats.

### Sub-Task 2.1: Render Mesh Builder

- **Goal:** To create a `RenderMeshBuilder` service that can construct a unified, render-ready mesh from PM4 chunks.
- **Source Logic:** `PM4FileTests.cs` (e.g., `GenerateRenderMesh`, `ExportCompleteRenderMeshToObj`).
- **Target:** `src/WoWToolbox.Core.v2/Services/PM4/RenderMeshBuilder.cs`.
- **Key Responsibilities:**
    - Combine vertices from MSVT and MSPV chunks.
    - Generate faces from MSVI and MSUR chunks.
    - Compute or retrieve vertex normals (using `CoordinateService`).
    - Handle different surface types (triangles, quads, polygons).
- **Status:** ⏳ **Pending**

### Sub-Task 2.2: OBJ Exporter

- **Goal:** To create a generic `ObjExporter` service capable of writing mesh data to the OBJ file format.
- **Source Logic:** `PM4FileTests.cs` (numerous `Export...ToObj` methods).
- **Target:** `src/WoWToolbox.Core.v2/Services/Export/ObjExporter.cs`.
- **Key Responsibilities:**
    - Write vertices, normals, and texture coordinates.
    - Write face definitions.
    - Handle material library (MTL) references.
- **Status:** ⏳ **Pending**

### Sub-Task 2.3: Diagnostic Exporters (CSV/JSON)

- **Goal:** To consolidate the diagnostic export logic into dedicated services.
- **Source Logic:** `PM4FileTests.cs` (e.g., `ExportDoodadDataToCsv`, `ExportNodeDataToJson`).
- **Target:**
    - `src/WoWToolbox.Core.v2/Services/Export/CsvExporter.cs`
    - `src/WoWToolbox.Core.v2/Services/Export/JsonExporter.cs`
- **Key Responsibilities:**
    - Serialize data structures to CSV and JSON formats.
    - Provide flexible configuration for export content.
- **Status:** ⏳ **Pending**

## Phase 3: High-Level Building Extraction

This phase focuses on consolidating the complex logic for identifying and extracting complete building structures from a PM4 file.

### Sub-Task 3.1: Building Extraction Service

- **Goal:** To create a high-level `BuildingExtractionService` that encapsulates the logic for identifying and extracting building models.
- **Source Logic:** `PM4FileTests.cs` (e.g., `ExportBuildings_UsingMdsfLinks`, `ExportBuildings_UsingMslkRootNodesWithSpatialClustering`).
- **Target:** `src/WoWToolbox.Core.v2/Services/PM4/BuildingExtractionService.cs`.
- **Key Responsibilities:**
    - Implement multiple strategies for building identification (e.g., MDSF/MDOS links, MSLK root node clustering).
    - Combine structural elements (from MSLK/MSPV) and render geometry (from MSUR/MSVT).
    - Produce a list of `CompleteWMOModel` objects representing the extracted buildings.
- **Status:** ⏳ **Pending**

## Phase 4: Test Modernization and Deprecation

This phase focuses on creating a new suite of tests that validate the migrated services and deprecating the old, monolithic test files.

### Sub-Task 4.1: Unit and Integration Tests

- **Goal:** To write comprehensive tests for the new services in `WoWToolbox.Core.v2`.
- **Target:** A new test project, `WoWToolbox.Core.v2.Tests`.
- **Key Responsibilities:**
    - Write unit tests for each service, mocking dependencies where appropriate.
    - Write integration tests that validate the end-to-end functionality of the PM4 processing pipeline.
- **Status:** ⏳ **Pending**

### Sub-Task 4.2: Deprecate Legacy Test Files

- **Goal:** To remove the monolithic and experimental test files from the `WoWToolbox.Tests` project.
- **Target Files:**
    - `PM4FileTests.cs`
    - `SimpleAnalysisTest.cs`
- **Action:** Once all logic has been migrated and validated, these files will be deleted.
- **Status:** ⏳ **Pending**
