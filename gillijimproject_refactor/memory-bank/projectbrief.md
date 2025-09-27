# Project Brief

- **Name**: GillijimProject C# Port
- **Status**: C++ → C# parity for Alpha WDT → Wrath ADT tooling is *functionally close* but mapping stability is still in flux.
- **Current Goal**: Stabilize Alpha→LK area mapping (DBCTool + AlphaWDTAnalysisTool) with strict per-map locking and zero-fallbacks for prototype maps (e.g., `mapId=17` Kalidar).
- **Next**: Revisit the reusable `.NET 9` library/CLI split once mapping parity is locked down and regression coverage is in place.
- **Scope**: LK-only ADT/WDT pipeline, including DBCTool crosswalk CSV generation. Cataclysm remains out of scope.
- **Philosophy**: Mapping data is authoritative; avoid heuristics that introduce cross-map leakage. Keep code as single source of truth with `[PORT]` notes and XML docs.
- **Deliverables**: Validated CSV crosswalks + patched ADTs with deterministic LK IDs (or explicit zeroes), followed by packaging into `GillijimProject.Core` / `GillijimProject.Cli` once stable.
