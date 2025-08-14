# Technology Context for parpToolbox

## Core Framework
- **Primary Framework:** .NET 9.0
- **Language:** C#

## Key Libraries & Dependencies
- **`wow.tools.local`**: The primary dependency for all low-level World of Warcraft file format parsing and handling.
- **Image Processing:** `SixLabors.ImageSharp` will likely be used for image data manipulation, particularly for texture extraction.
- **Database:** `Microsoft.Data.Sqlite` may be used for local data storage if needed.

*Note: Other dependencies like `BLPSharp`, `CascLib`, and `DBCD` are managed within `wow.tools.local` and will be used transitively.*

## Project Structure & Build
- **Solution File:** The primary solution is `parpToolbox.sln`, located in the project root.
- **Main Project:** `parpToolbox` is a .NET 9.0 console application located in `src/parpToolbox/`. This is the entry point and will contain all high-level orchestration and business logic.
- **Core Dependency:** The `wow.tools.local` project, located in `src/lib/wow.tools.local/`, is included in the solution and referenced by `parpToolbox`.
- **Legacy Reference:** The old `WoWToolbox` project structure remains in the `PM4Tool` directory for historical and reference purposes but is not part of the active build.

## Output Management
- The `ProjectOutput` utility has been implemented. It directs all generated files (e.g., OBJ exports) to a unified, timestamped `project_output` directory, preventing contamination of source data folders.
