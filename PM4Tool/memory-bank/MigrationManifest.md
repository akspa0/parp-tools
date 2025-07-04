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

### 1.2. PM4 Validation Service

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

### 2.1. Render Mesh Builder

- **Goal:** To create a `RenderMeshBuilder` service that can construct a unified, render-ready mesh from PM4 chunks.
- **Source Logic:** `PM4FileTests.cs` (e.g., `GenerateRenderMesh`, `ExportCompleteRenderMeshToObj`).
- **Target:** `src/WoWToolbox.Core.v2/Services/PM4/RenderMeshBuilder.cs`.
- **Key Responsibilities:**
    - Combine vertices from MSVT and MSPV chunks.
    - Generate faces from MSVI and MSUR chunks.
    - Compute or retrieve vertex normals (using `CoordinateService`).
    - Handle different surface types (triangles, quads, polygons).
- **Status:** ⏳ **Pending**

### 2.2. OBJ Exporter

- **Goal:** To create a generic `ObjExporter` service capable of writing mesh data to the OBJ file format.
- **Source Logic:** `PM4FileTests.cs` (numerous `Export...ToObj` methods).
- **Target:** `src/WoWToolbox.Core.v2/Services/Export/ObjExporter.cs`.
- **Key Responsibilities:**
    - Write vertices, normals, and texture coordinates.
    - Write face definitions.
    - Handle material library (MTL) references.
- **Status:** ⏳ **Pending**

### 2.3. Diagnostic Exporters (CSV/JSON)

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

### 3.1. Building Extraction Service

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

### 4.1. Unit and Integration Tests

- **Goal:** To write comprehensive tests for the new services in `WoWToolbox.Core.v2`.
- **Target:** A new test project, `WoWToolbox.Core.v2.Tests`.
- **Key Responsibilities:**
    - Write unit tests for each service, mocking dependencies where appropriate.
    - Write integration tests that validate the end-to-end functionality of the PM4 processing pipeline.
- **Status:** ⏳ **Pending**

### 4.2. Deprecate Legacy Test Files

- **Goal:** To remove the monolithic and experimental test files from the `WoWToolbox.Tests` project.
- **Target Files:**
    - `PM4FileTests.cs`
    - `SimpleAnalysisTest.cs`
- **Action:** Once all logic has been migrated and validated, these files will be deleted.
- **Status:** ⏳ **Pending**

This document outlines the plan for migrating critical algorithms, data structures, and utilities from the experimental test project (`WoWToolbox.Tests`) into the core library (`WoWToolbox.Core.v2`). The goal is to consolidate all PM4 processing knowledge into a robust, maintainable, and production-ready library.

### Phase 1: Foundational Services Migration (Core Systems)

**Status: In Progress**

#### Sub-Task 1.1: `Pm4CoordinateTransforms` to `CoordinateService`

- **Status:** 
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

### 1.1. Coordinate Transformation Service

*   **Summary:** The `Pm4CoordinateTransforms` static class contains all essential logic for converting between different coordinate spaces within the PM4 file (e.g., MSVT, MSPV, MPRL) and for calculating render-ready geometry (e.g., vertex normals).
*   **Source:** `test\WoWToolbox.Tests\Navigation\PM4\PM4FileTests.cs` (as a nested static class).
*   **Target:** `src\WoWToolbox.Core.v2\PM4\Transforms\CoordinateService.cs`
*   **Refactoring Notes:**
    *   Convert the static class into a non-static service (`CoordinateService`) that can be injected or instantiated.
    *   Ensure all methods are public and well-documented with XML comments explaining the specific transformations.
    *   Create a public interface (`ICoordinateService`) to promote testability and dependency injection.

### 1.2. Data Validation Utilities

*   **Summary:** The test script includes extensive logic for validating chunk indices (e.g., ensuring MSVI indices are within the bounds of the MSVT vertex list). This is critical for handling corrupted data gracefully.
*   **Source:** `test\WoWToolbox.Tests\Navigation\PM4\PM4FileTests.cs` (scattered throughout the main test method).
*   **Target:** `src\WoWToolbox.Core.v2\PM4\Validation\Pm4Validator.cs`
*   **Refactoring Notes:**
    *   Consolidate all validation checks into a single `Pm4Validator` class.
    *   Create methods like `ValidateMsvi(PM4File file)` which return a `ValidationResult` object containing success/failure status and a list of errors.
    *   This validator can be used during file loading to provide warnings or throw exceptions based on a "strict mode" setting.

## Phase 2: Geometry and Exporter Migration

This phase focuses on migrating the logic responsible for generating geometric data and exporting it to standard formats.

### 2.1. Render Mesh Generation

*   **Summary:** The logic for generating a fully renderable mesh from MSVT, MSVI, and MSUR chunks, including normal calculation and face generation.
*   **Source:** `test\WoWToolbox.Tests\Navigation\PM4\PM4FileTests.cs` (within the main test method).
*   **Target:** `src\WoWToolbox.Core.v2\PM4\Geometry\RenderMeshBuilder.cs`
*   **Refactoring Notes:**
    *   Create a `RenderMeshBuilder` class that takes a `PM4File` object as input.
    *   The builder should produce a generic 3D model object (e.g., a custom `MeshData` class with lists of vertices, normals, and triangle indices) that is independent of any specific file format.

### 2.2. OBJ Exporter

*   **Summary:** The code responsible for writing vertices, normals, and faces to the `.obj` file format.
*   **Source:** `test\WoWToolbox.Tests\Navigation\PM4\PM4FileTests.cs` (intermingled with `StreamWriter` calls).
*   **Target:** `src\WoWToolbox.Core.v2\IO\Export\ObjExporter.cs`
*   **Refactoring Notes:**
    *   Create a generic `ObjExporter` that can take the `MeshData` object produced by `RenderMeshBuilder` and write it to a stream.
    *   This decouples the geometry generation from the file format exporting.

### 2.3. Diagnostic Exporters (CSV/JSON)

*   **Summary:** The logic for exporting detailed chunk data (e.g., `MSLK` doodad info) to CSV and JSON for analysis.
*   **Source:** `test\WoWToolbox.Tests\Navigation\PM4\PM4FileTests.cs`.
*   **Target:** `src\WoWToolbox.Core.v2\IO\Export\` (e.g., `MslkDoodadCsvExporter.cs`).
*   **Refactoring Notes:**
    *   Create dedicated exporter classes for each specific data type.
    *   These exporters will be invaluable for debugging and will form the basis of a future analysis tool.

## Phase 3: High-Level Services

This phase migrates the high-level business logic that orchestrates the entire extraction process.

### 3.1. Building Extraction Logic

*   **Summary:** The core algorithms for identifying and extracting building geometry, currently split between `PM4BuildingExtractionService.cs` (the clean, dual-strategy version) and the more experimental logic in `PM4FileTests.cs` (e.g., `ProcessHighRatioPm4File`).
*   **Source:** `PM4BuildingExtractionService.cs` and `PM4FileTests.cs`.
*   **Target:** `src\WoWToolbox.Core.v2\PM4\Buildings\BuildingExtractionService.cs` (enhancing the existing service).
*   **Refactoring Notes:**
    *   Merge the robust, dual-strategy approach from `PM4BuildingExtractionService` with any valuable adaptive logic found in the test script.
    *   The service should use the new `CoordinateService`, `RenderMeshBuilder`, and `Validators` from `Core.v2`.
    *   The final service should be the single, authoritative source for all building extraction operations.

## Phase 4: Test Modernization

Once the migration is complete, the final phase is to refactor the tests.

*   **Action:** Deprecate `PM4FileTests.cs`.
*   **New Tests:** Create a new suite of small, focused unit and integration tests in the `WoWToolbox.Tests` project.
*   **Test Focus:**
    *   Unit tests for the `CoordinateService` (e.g., assert that a specific input vertex transforms to an expected output).
    *   Unit tests for the `Pm4Validator` (e.g., test with a file that has known bad indices and assert that the validator catches it).
    *   Integration tests for the `BuildingExtractionService` that run on a small set of representative PM4 files and assert that the extracted `MeshData` has the expected number of vertices and faces.
