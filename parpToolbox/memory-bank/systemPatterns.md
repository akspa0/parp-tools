# System Patterns for parpToolbox

## Core Architecture
- **Service-Oriented Architecture:** The new `parpToolbox` project will adopt a service-oriented architecture. Logic will be encapsulated in specialized services (e.g., for parsing, conversion, exporting) to promote modularity and testability. The legacy `WoWToolbox.Core.v2` project serves as a reference for this pattern.

- **Clear Separation of Concerns:** The application logic in `parpToolbox` will be distinctly separate from the low-level file handling provided by the `wow.tools.local` dependency. This ensures that our core business logic is not tightly coupled to the internals of the library.

## Data Flow & Processing
- **Chunk-Based File Reading:** File formats like WMO and PM4 will be processed as a series of "chunks," identified by FourCC codes. A critical pattern to be carried over is the reversal of FourCC bytes upon reading (e.g., 'REVM' in the file is handled as 'MVER' in the code) to maintain consistency.

- **Immutable Models:** The system will favor immutable data models. Raw file data will be parsed into strongly-typed models that are passed between services for processing and transformation without unexpected side effects.

- **Porting, Not Refactoring:** The primary effort is not to refactor the old codebase, but to selectively port correct and valuable logic from the legacy `WoWToolbox` project into the new, clean `parpToolbox` project.

- **Controlled Output:** All file generation is handled by a `ProjectOutput` utility. This service ensures that all output is directed to a timestamped subfolder within a `project_output` directory at the project root. This prevents contamination of source data and provides a predictable location for all artifacts.

- **Dependency-Light CLI:** To avoid issues with unstable or complex external packages, command-line argument parsing is handled with a simple, manual implementation directly within `Program.cs`. This provides sufficient functionality for the tool's needs while minimizing external dependencies.

## Testing & Validation
- **Real Data Testing:** All new tests written for `parpToolbox` must use real game data to ensure the system is validated against real-world conditions.

- **Focused Test Suites:** New, focused test suites will be created for `parpToolbox` to validate its functionality. Legacy tests will serve as a reference for identifying important test cases.
