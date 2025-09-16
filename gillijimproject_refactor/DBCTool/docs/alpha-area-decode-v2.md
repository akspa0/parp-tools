# Alpha AreaTable Decode v2 — Multi‑Pass Plan

Status: draft
Owner: DBCTool
Scope: Alpha (0.5.3/0.5.5) AreaTable → LK (3.3.5) compare pipeline
Primary code: `DBCTool/Program.cs` (`CompareAreas`)

## Background

Alpha builds pack Zone/SubZone into 16‑bit halves within both `AreaNumber` and `ParentAreaNum`:
- `AreaNumber = (zone_hi16 << 16) | sub_lo16`
- `ParentAreaNum = (zone_hi16 << 16) | 0` for the true parent of a subzone

Therefore a single record can encode a zone (lo16==0) or a subzone (lo16>0). Traversing `ParentAreaNum` naïvely or trusting the leaf row’s `ContinentID` causes cross‑continent and cross‑zone collisions.

Reference: ADT v18 docs (packing confirmed). See `DBCTool/docs/001-areaid-mapping.md` for citations.

## Goals

- Deterministically decode zones and subzones from 16‑bit halves, anchored by continent.
- Validate parent relationships via `ParentAreaNum` halves (parent_ok flag).
- Produce audit outputs for decoded mapping and anomalies.
- Integrate decoded chains into `CompareAreas` V2 to eliminate cross‑continent/parent collisions.

## Terminology / Derived Fields

From each Alpha row (AreaTable 0.5.x):
- `area_hi16 = AreaNumber >> 16`
- `area_lo16 = AreaNumber & 0xFFFF`
- `parent_hi16 = ParentAreaNum >> 16`
- `parent_lo16 = ParentAreaNum & 0xFFFF`
- `zoneBase = (area_hi16 << 16)`
- `parentZoneBase = (parent_hi16 << 16)`
- `cont = ContinentID`
- `name = AreaName_lang | AreaName | Name`

Parent validation rules:
- Zone candidate (lo16==0) valid if `parent_hi16 == area_hi16 && parent_lo16 == 0`.
- Sub candidate (lo16>0) valid if `parent_hi16 == area_hi16 && parent_lo16 == 0`.

## Multi‑Pass Decode

### Pass 0: Index and Normalize

Scan all Alpha rows and build groups per `(cont, zoneBase)`:
- Zone candidates: `area_lo16 == 0` (potential zone names for this `zoneBase`).
- Sub candidates: `area_lo16 > 0` (potential sub names for `(zoneBase, area_lo16)`).
- Compute `parent_ok` per row using the validation rules above.
- Retain examples for conflict resolution and output notes.

### Pass 1: Resolve Zones (per continent)

For each `(cont, zoneBase)`:
- Choose canonical `zoneName` by majority among validated zone rows (`parent_ok==true`).
- If none validated, pick the most frequent name; tag as `weak`.
- Record: `ZoneIndex[(cont, zoneBase)] = { zoneName, parent_ok_count, total }`.

### Pass 2: Resolve Subzones (per continent)

For each `(cont, zoneBase, sub_lo16)`:
- Accept only rows with `parent_ok==true`.
- Choose canonical `subName` by majority; mark `ambiguous` on ties.
- Record: `SubIndex[(cont, zoneBase, sub_lo16)] = { subName, parent_ok, examples }`.

### Pass 3: Conflict Resolution

- If the same `zoneBase` appears across multiple continents, assign ownership to the continent with highest count of validated rows; record others as anomalies.
- If the same `(zoneBase, sub_lo16)` appears under multiple continents or zones, keep the variant under the continent/zone that owns the `zoneBase`; others become anomalies.
- Assign confidence tags: `strong` (validated), `weak` (no validation), `conflict`, `anomalous`.

### Pass 4: Outputs

Write two CSVs for audit:
- `out/compare/alpha_areaid_decode_v2.csv`
  - Columns:
    - `alpha_raw, alpha_raw_hex, cont, area_hi16, area_lo16, parent_hi16, parent_lo16, zone_base_hex, zone_name, sub_lo16, sub_name, parent_ok, confidence, notes`
- `out/compare/alpha_areaid_anomalies.csv`
  - Summaries of cross‑continent and parent inconsistencies, with examples.

## Integration into `CompareAreas` V2

Location: `DBCTool/Program.cs`, function `CompareAreas` (V2 section).

- Build decode once: `BuildAlphaAreaDecodeV2(storSrc_Area)` → `ZoneIndex`, `SubIndex`, `Anomalies`.
- For each source row in V2:
  - Derive `(cont_decoded, zoneBase, area_lo16)` from the halves.
  - Resolve `zoneName` from `ZoneIndex[(cont_decoded, zoneBase)]`.
  - If `area_lo16 > 0` and a sub exists in `SubIndex[(cont_decoded, zoneBase, area_lo16)]`, use that `subName`.
  - Build the source chain deterministically:
    - `lo16 == 0` → `[zoneName]`
    - `lo16 > 0` with sub hit → `[zoneName, subName]`
    - `lo16 > 0` without sub hit → fallback `[zoneName]` (still map‑correct)
  - Set `contResolved = cont_decoded` (stop trying `{contRaw, 0, 1}` when decode is strong).
  - Map crosswalk uses `contResolved` only; compute `mapIdX`/`mapNameX` accordingly.
  - Matching stays the same:
    - `TryMatchChainExact(mapIdX, chain)`.
    - Sub selection constrained to children of chosen zone and same map (already implemented with `ChooseTargetByName` and `ChooseSubWithinZone`).

- Outputs (already fixed):
  - `src_mapId` uses `contResolved`.
  - `src_path` includes `mapNameX` prefix only when resolved.
  - Patch CSVs are map‑locked (cross‑map guard remains in place).

## Data Structures (sketch)

```csharp
// Decode indices (in-memory)
Dictionary<(int cont, int zoneBase), ZoneRec> ZoneIndex;
Dictionary<(int cont, int zoneBase, int subLo), SubRec> SubIndex;

record ZoneRec(string zoneName, int validatedCount, int totalCount, string confidence);
record SubRec(string subName, bool parentOk, string confidence);
```

## Implementation Steps (small patches)

1) Add `BuildAlphaAreaDecodeV2(storSrc_Area)` in `Program.cs` next to `CompareAreas`:
- Performs Pass 0–4.
- Writes `alpha_areaid_decode_v2.csv` and `alpha_areaid_anomalies.csv`.

2) Integrate into V2:
- Replace `BuildSrcChainNamesPref(...)` usage with decoded chain from indices.
- Set `contResolved` from decode; only fallback when decode is missing for a row.

3) Re-run and audit:
- Verify Un’Goro (Kalimdor) cannot map to Blackrock Mountain (EK).
- Verify EK subzones never appear under Kalimdor per‑map outputs and vice versa.

## Validation Cases

- __Un’Goro Crater (Kalimdor)__ must never map to __Blackrock Mountain (EK)__.
- __Westfall, Darkshire, Moonbrook (EK)__ must remain in EK.
- __Teldrassil cluster (Kalimdor)__ stable at map 1 with correct subchains.

## Risks / Edge Cases

- Sparse/weak data (no validated parents) – choose majority name; fallback to zone‑only chain.
- Duplicate names across continents – decode’s continent lock prevents cross‑map matches.
- Alpha rows with inconsistent `ContinentID` per duplicate `AreaNumber` – resolved by majority of validated rows.

## Open Questions

- Do we need continent‑biased tie‑breakers for rare ambiguous cases after decode? (If yes, apply within the same continent only.)
- Should we propagate confidence tags into V2 outputs for downstream review?

## Next Actions

- Implement `BuildAlphaAreaDecodeV2` (Pass 0–4) and write both CSVs.
- Integrate into V2 chain building.
- Re‑run `--compare-area-v2` and validate the prior problem rows.

---

Appendix: Related functions
- `CompareAreas` (V2 section): wiring site
- `TryMatchChainExact(mapIdX, chain)`
- `ChooseTargetByName(srcName, mapIdX, requireMap, topLevelOnly)`
- `ChooseSubWithinZone(subName, zoneId)`

## Architecture Refactor Plan (Modular, Testable, Non‑Monolithic)

We will refactor DBCTool into a small, composable program with clear modules and boundaries. For a console tool, a Clean Architecture–style layering works better than classic MVC. The goal is to keep domain/pipeline code independent from I/O and make each piece unit‑testable in isolation.

- **Core principles**
  - Small modules with single responsibility.
  - Dependency inversion: core depends on interfaces; concrete implementations are injected at the edge.
  - Avoid static “god functions”; split `CompareAreas` into orchestrations + services.
  - Maximize deterministic pure functions; isolate I/O.

- **Target module boundaries and responsibilities**
  - Domain (Models + Contracts)
    - Models: `AreaRow`, `MapRow`, `ZoneRec`, `SubRec`, match results, export line DTOs.
    - Contracts (interfaces):
      - `IAlphaDecoder`: produces `ZoneIndex`, `SubIndex`, anomalies.
      - `IChainBuilder`: builds source chains from decoded indices.
      - `IMapCrosswalk`: Alpha→LK map crosswalk.
      - `IAreaMatcher`: chain matching on LK (top‑level zone constraint, same‑map child constraint).
      - `IExporter`: mapping/unmatched/patch/diagnostic CSV writers.
      - `IDbcdProvider`: abstraction over DBCD access.
  - AlphaDecode (Pipeline)
    - Implements `IAlphaDecoder` (multi‑pass decode: Pass 0–4).
    - Emits `alpha_areaid_decode_v2.csv` and anomalies CSV.
  - Mapping (Compare V2)
    - Implements `IChainBuilder`, `IAreaMatcher`.
    - Applies zone‑locked and map‑locked rules:
      - Top‑level zones: LK `ParentAreaID == ID`.
      - Subzones: must be child of the chosen zone and same map.
  - Crosswalk (Map ID Resolver)
    - Implements `IMapCrosswalk` with deterministic report (existing crosswalk CSV).
  - IO (Storage)
    - `CsvWriter`, `PathResolver`, `Clock`.
    - DBCD wrappers implementing `IDbcdProvider`.
  - CLI (Composition)
    - DI container (`Microsoft.Extensions.DependencyInjection`).
    - Command parsing (`System.CommandLine` or `Spectre.Console.Cli`).
    - Wires flags (e.g. `--compare-area-v2`) to services.

- **Solution structure (incremental)**
  - Phase 1 (in‑place namespaces, same project)
    - Create folders/namespaces:
      - `DBCTool.Domain/` (models, interfaces)
      - `DBCTool.AlphaDecode/`
      - `DBCTool.Mapping/`
      - `DBCTool.Crosswalk/`
      - `DBCTool.IO/`
      - `DBCTool.Cli/` (Program.cs)
    - Move large local functions into small files/classes under these namespaces.
  - Phase 2 (split projects if needed)
    - `DBCTool.Cli` (net9.0) – executable
    - `DBCTool.Core` (Domain + Crosswalk contracts)
    - `DBCTool.AlphaDecode` (impl)
    - `DBCTool.Mapping` (impl)
    - `DBCTool.IO` (impl)
    - `DBCTool.Tests` (xUnit/NUnit)
  - DI wiring
    - Register interfaces → implementations:
      - `IAlphaDecoder`, `IChainBuilder`, `IMapCrosswalk`, `IAreaMatcher`, `IExporter`, `IDbcdProvider`.
    - Keep Program.cs thin (parse options → call services).

- **Refactor plan (small PRs)**
  - Step A: Extract Domain models + interfaces into `DBCTool.Domain` namespace.
  - Step B: Move Map crosswalk into `DBCTool.Crosswalk` (pure function + report).
  - Step C: Implement `IAlphaDecoder` (multi‑pass decode), emit new CSVs.
  - Step D: Replace V2’s parent traversal with `IChainBuilder` using decoded indices.
  - Step E: Extract `IAreaMatcher` (chain match + constraints) and `IExporter`.
  - Step F: Introduce DI in `Program.cs`; replace hidden statics.
  - Step G: Optional: split into projects; add `DBCTool.Tests` and unit tests per module.

- **Testing strategy**
  - Unit tests:
    - AlphaDecode: parent validation, conflict resolution, Un’Goro vs Blackrock protection.
    - Mapping: `TryMatchChainExact`, `ChooseTargetByName`, `ChooseSubWithinZone` constraints.
    - Crosswalk: dir token & name matching fallbacks.
  - Golden files:
    - Compare subsets of CSV outputs across revisions to guard regressions.

- **Acceptance criteria**
  - No giant monolithic methods; `CompareAreas` becomes orchestration only.
  - Module boundaries covered by tests.
  - Cross‑continent mismatches eliminated (e.g., Un’Goro → Blackrock).
  - Per‑map patch CSVs stable and map‑locked.

- **Migration notes**
  - Phase 1 keeps single project to minimize disruption; namespaces + files first.
  - Phase 2 splits projects when stable to reduce rebuild churn and to improve reuse.

- **Tools and libraries**
  - `Microsoft.Extensions.DependencyInjection`
  - `Microsoft.Extensions.Logging`
  - `System.CommandLine` (or `Spectre.Console.Cli`)
  - Testing: `xUnit` or `NUnit`; `FluentAssertions` optional.

- **Updated TODOs**
  - Implement the multi‑pass decoder and outputs (this document’s Pass 0–4).
  - Begin Phase 1 refactor (namespaces + files) after alpha decode v2 is implemented.
  - Add tests for AlphaDecode, Mapping, Crosswalk.
