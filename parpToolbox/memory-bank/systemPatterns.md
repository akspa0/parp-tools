# System Patterns for parpToolbox

## Core Architecture
- **Service-Oriented Architecture:** The new `parpToolbox` project will adopt a service-oriented architecture. Logic will be encapsulated in specialized services (e.g., for parsing, conversion, exporting) to promote modularity and testability. The legacy `WoWToolbox.Core.v2` project serves as a reference for this pattern.

- **Clear Separation of Concerns:** The application logic in `parpToolbox` will be distinctly separate from the low-level file handling provided by the `wow.tools.local` dependency. This ensures that our core business logic is not tightly coupled to the internals of the library.

## Data Flow & Processing
- **Chunk-Based File Reading:** File formats like WMO, PM4, and PD4 are processed as a series of "chunks," identified by FourCC codes. On-disk FourCC values are stored little-endian; therefore a helper `FourCc.Read(BinaryReader)` reverses the bytes so the code works with canonical IDs (e.g., bytes `REVM` → string `MVER`).

- **Immutable Models:** The system will favor immutable data models. Raw file data will be parsed into strongly-typed models that are passed between services for processing and transformation without unexpected side effects.

- **Porting, Not Refactoring:** The primary effort is not to refactor the old codebase, but to selectively port correct and valuable logic from the legacy `WoWToolbox` project into the new, clean `parpToolbox` project.

- **Controlled Output:** All file generation is handled by a `ProjectOutput` utility. This service ensures that all output is directed to a timestamped subfolder within a `project_output` directory at the project root. This prevents contamination of source data and provides a predictable location for all artifacts.

- **Dependency-Light CLI:** To avoid issues with unstable or complex external packages, command-line argument parsing is handled with a simple, manual implementation directly within `Program.cs`. This provides sufficient functionality for the tool's needs while minimizing external dependencies.

## PM4 / PD4 Data Handling
- **Per-Tile Processing:** PM4 tiles are loaded into a unified `Pm4SceneLoader` that supports single-tile or multi-tile (3×3 grid) contexts, ensuring complete vertex coverage across tile boundaries.
- **Modular Exporter Pipeline:** Primary focus is the **PM4 Next Exporter** – a modular pipeline (SceneLoader → Assembler(s) → DiagnosticsService → Exporter) that preserves **all** chunk data (MSUR, MSCN, MSLK, MPRL, etc.), produces deep diagnostics (CSV/JSON), and supports per-object OBJ (legacy-parity) and future glTF outputs.

## Testing & Validation
- **Real Data Testing:** All new tests written for `parpToolbox` must use real game data to ensure the system is validated against real-world conditions.

- **Focused Test Suites:** New, focused test suites will be created for `parpToolbox` to validate its functionality. Legacy tests will serve as a reference for identifying important test cases.
