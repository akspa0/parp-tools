# Tech Context

- Target: .NET 9 (SDK-style console app)
- Namespace root: `GillijimProject`
- Build: `dotnet build` under `src/gillijimproject-csharp`
- IO: `FileStream`, `ReadOnlySpan<byte>`, `BitConverter`, `Encoding.ASCII`
- Testing: planned smoke tests on known WDT/ADT assets
- Scope: LK-only; Cataclysm excluded for this port phase
- FourCC: forward in memory; reversed on disk via `WowFiles/Chunk.cs` during serialization; enforced across WowFiles per rules
- Writers respect `-o/--out` and write all outputs into the specified directory
- Nullable reference types enabled; requiring proper initialization of all non-nullable fields in constructors
- Method naming conventions follow C# standards (PascalCase)
- Current build status: 2 errors remain (CS1729 in `Alpha/McnkAlpha.cs:98` and `:115`) due to missing `Mcnk(string,int,byte[])` constructor in `WowFiles/Mcnk.cs`.
