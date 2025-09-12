# Tech Context (Next)

- Target: .NET 9.0; Nullable enabled
- Projects:
  - `GillijimProject.Next.Core` (class library)
  - `GillijimProject.Next.Cli` (console)
  - `GillijimProject.Next.Tests` (tests)
- External Refs (ProjectReference):
  - `next/libs/Warcraft.NET/Warcraft.NET/Warcraft.NET.csproj`
  - `next/libs/wow.tools.local/DBCD/DBCD/DBCD.csproj`
- IO & Patterns:
  - Span/Stream-based IO; FourCC forward in memory, reversed on disk
  - ADT ordering: write `MCLQ` last within `MCNK`; omit `MH2O` when empty
- Liquids:
  - LVF Case 0/2 supported initially; 1/3 deferred
  - Dimensions: MCLQ (heights/depth 9x9), tiles 8x8; MH2O rectangles W/H in [1..8] with (W+1)*(H+1) vertices
- Testing:
  - xUnit; unit + round-trip + CLI integration (skip-if-missing fixtures)
