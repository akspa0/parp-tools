# Project Brief

- Name: GillijimProject C# Port
- Status: 1:1 C++ → C# parity achieved for Alpha WDT → Wrath ADT conversion.
- Current Goal: Refactor into a reusable .NET 9 class library (`GillijimProject.Core`) with a thin CLI wrapper (`GillijimProject.Cli`).
- Next: Integrate modern Warcraft.NET writer APIs to improve safety and performance while preserving output compatibility.
- Scope: LK-only ADT/WDT pipeline; Cataclysm remains out of scope for now.
- Philosophy: Parity-first, then idiomatic C#. Single source of truth in code; `[PORT]` notes; XML docs for public APIs.
- Deliverables: NuGet-packaged library with documented API + CLI that uses the library.
