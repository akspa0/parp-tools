# WoWFormatLib Audit – 2025-07-13

This document inventories capabilities of **wow.tools.local / WoWFormatLib** and maps them against PM4Tool requirements. It will guide migration planning.

## 1. Core Architecture
* **FileProviders** – abstraction over storage back-ends (CASC, TACT, Wago CDN). Good entry for live game data; we can wrap a PM4/PD4 provider here later.
* **FileReaders** – strongly-typed readers per file/asset type (WMOReader, M2Reader, etc.). Return struct-based models from `Structs/*`.
* **Structs** – auto-generated `[StructLayout]` layouts mirroring Blizzard binary formats for fast `BinaryReader` deserialisation.
* **Utils** – helpers (colors, versioning, listfile, DataStore).

## 2. Relevant Readers Present
| Format | Reader class | Status |
|--------|--------------|--------|
| WMO (v17) | `FileReaders.WMOReader` | Present, uses `Structs.WMO.Struct` layouts (includes MOPY flag enum). |
| M2 / M3 | `M2Reader`, `M3Reader` | Present. |
| ADT | `ADTReader` | Present. |
| Textures | `BLPReader`, `TEXReader` | Present. |
| World defs | `WDTReader`, `WDLReader` | Present. |
| Skeleton/Skin | `SKELReader`, `SKINReader` | Present. |

No native reader for **PM4 / PD4** – gap to fill.

## 3. Gaps / Action Items
1. **PM4/PD4 Support** – design new structs & reader following WoWFormatLib patterns.• Add `Structs/PM4.Struct.cs` & `FileReaders/PM4Reader.cs`.
2. **OBJ / Tooling Adapters** – our exporters rely on Core.v2 domain models; create thin adapters mapping WoWFormatLib structs → existing DTOs until we refactor exporters.
3. **Dependency Management** – add WoWFormatLib project reference to the solution; avoid source-level forks.
4. **Namespace Clash** – ensure our future `Foundation.WMO.*` doesn’t conflict with `WoWFormatLib.Structs.WMO`.
5. **Performance Testing** – benchmark WoWFormatLib WMOReader against current prototype to ensure no regressions.

## 4. Proposed Migration Steps
1. Reference `WoWFormatLib` in `WoWToolbox.Core.v2` solution. (Add as ProjectReference or NuGet if published.)
2. Spike: load sample WMO via `WMOReader` and pipe into existing `WmoObjExporter` (through adapter).
3. Parallel path: continue PM4Reader implementation, reusing DataStore & BinaryReaderExtensions.
4. Gradually deprecate duplicated v14/v17 structs once adapters validated.
5. Upstream contributions: open PRs to WoWFormatLib for PM4 support, bug fixes.

---
_End of initial audit_
