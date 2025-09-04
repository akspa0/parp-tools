# Tech Context

- Target: .NET 9 across projects
- Namespaces: `GillijimProject` root; `GillijimProject.Core` (library), `GillijimProject.Cli` (console)
- Build:
  - Solution-level `dotnet build` for all projects
  - `dotnet test` for test project(s)
  - `dotnet pack` for `GillijimProject.Core` (NuGet)
- IO: `FileStream`, `ReadOnlySpan<byte>`, `BitConverter`, `Encoding.ASCII`
- Testing: xUnit/NUnit for smoke/integration tests with known WDT/ADT fixtures
- Scope: LK-only; Cataclysm excluded for now
- FourCC: forward in memory; reversed on disk by serializer
- Writers: Prefer Warcraft.NET writer APIs via adapters/facades where applicable
- CLI: thin wrapper over the library; `-o/--out` respected
- Nullable: enabled; initialize non-nullable fields in constructors
- Method naming: PascalCase
- Current status: Parity achieved; refactor and integration phase underway
