# Project Brief

- Name: GillijimProject C# Port
- Goal: 1:1 C++ → C# port of legacy `lib/gillijimproject` to a .NET 9 console app under `src/gillijimproject-csharp`.
- Scope (current): LK-only ADT/WDT pipeline. Parse Alpha WDT → convert to LK WDT + ADTs. Cataclysm excluded.
- Philosophy: Parity-first, then refine. Follow `.windsurf/rules/csharp-port.md`. One C++ file → one C# file; `[PORT]` notes; XML docs for public APIs.
- Deliverable: CLI that reads Alpha WDT, reports tiles and model refs, converts to LK structures; smoke-tested on known assets.
