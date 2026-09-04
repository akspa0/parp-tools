# Spec 221: Converter Regression Harness — Corpus Gates & Real-Client Validation

**Feature Branch**: `221-converter-validation-harness` (authored on `v0.5.3`; specs 193–220 follow the same convention)

## Overview & User Intent

Substantial implementation has landed since the converters were last exercised (spec 219's `TileContentTransform`, phase-layer composition, the 2026-09-03 tile-offset magnitude revert, WMO/WDL writers used by the restored-map exporter). The operator wants the map and object converters **proven**, not assumed, via three instruments:

1. **A corpus gate** — run every converter's round trip over a real-client corpus with pass/fail budgets, so "we think it still works" becomes a number that must stay green.
2. **An oracle harness** — where a second implementation exists (gillijimproject refactor, the frozen Alpha readers/writers), cross-check our converter's output against it on real data.
3. **A real-client harness** — stage the minimal file set that boots the 0.5.3 executable, load a converted map, and verify observable in-client behavior (geometry present, objects placed, **collisions working**). Execution is operator-owned per workspace policy; the spec prepares the manifest, staged client, commands, and verification checklist.

## Measured baseline (authored from source, 2026-09-04)

- **Converter tool**: 25 commands in [`WowViewer.Tool.Converter`](file:///I:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs), including [`ValidateRoundTripCommand`](file:///I:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/ValidateRoundTripCommand.cs) (LK↔Alpha terrain, real MPQ data, height/alpha epsilons) and `WmoV14ToV17Command` / `WmoV17ToV14Command` / `M2ToMdxCommand` / `MdxToM2Command` — **but the object converters have no corpus validator**, only unit tests.
- **Unit coverage exists**: `LkToAlphaRoundTripTests`, `WmoV14ToV17ConverterTests`, `WmoV17ToV14ConverterTests`, `M2ToMdxConverterTests`, `MdxToM2ConverterTests`, `AdtTerrainWriterTests`, `WdlWriterTests` — these are the "last tested" surface; how many run on **real bytes vs synthetic** is the Phase 0 measurement.
- **Oracles available**: `gillijimproject_refactor/next` (Alpha WDT reader/writer with tests), frozen `AlphaWdtWriter.cs`, and the proven Alpha-era decoders in `WowViewer.Core.IO/Maps`.
- **Clients staged**: `H:\CLIENTS\Vanilla\0.x\0_5_3_3368` (and 0.5.5/0.6.0/0.12 siblings) — the real-executable harness targets 0.5.3 first.

## User Stories

### US-1: Converter corpus gate (Priority: P1)
- **As an** operator,
- **I want** one command that runs every converter round trip over a real-client corpus (bounded per converter) and prints a per-converter pass/fail/error table with budgets,
- **So that** regressions from recent implementation are found before they reach the viewer or an export.

### US-2: Object converter round-trip validator (Priority: P1)
- **As an** operator,
- **I want** WMO V14↔V17 and M2↔MDX corpus round trips with field-level diffing (doodads, sets, names, bounds, sequences, textures) and byte-equality on untouched chunks,
- **So that** object conversion quality is measured by the same standard terrain already is.

### US-3: Oracle cross-check (Priority: P2)
- **As an** operator,
- **I want** our Alpha/LK converters cross-checked against the gillijimproject oracle on shared inputs, with disagreements reported per chunk/field,
- **So that** a wrong assumption in one implementation is caught by the other instead of silently propagating.

### US-4: Real-client smoke & collision harness (Priority: P2, operator-executed)
- **As an** operator,
- **I want** a documented minimal-file 0.5.3 client staging (executable + MPQs needed to reach menu/console and load a map), a converted map installed into it, and a checklist verifying in-client: terrain renders, placements appear, and the character collides with converted geometry where the source says it should,
- **So that** "converted correctly" is judged by the only authority that matters — the real client.

## Acceptance Criteria

- **AC-001**: `converter validate-corpus --client-root H:\CLIENTS\...` runs all converter round trips and exits non-zero when any converter exceeds its budget; output is a machine-readable table (converter, files, pass, fail, worst-delta).
- **AC-002**: Object round trips diff every spec-relevant chunk field; a corpus run over ≥50 WMOs and ≥50 M2/MDX models reports zero unexplained field drift against the pre-overhaul baseline, or names the field and the implementation change responsible.
- **AC-003**: Oracle disagreements are triaged: each is either (a) a bug fixed with a failing test, or (b) documented as oracle-wrong with evidence. No disagreement is left unclassified.
- **AC-004**: The 0.5.3 minimal-file manifest and collision checklist are committed; the operator's staged-client run either confirms the checklist or files the first defect with a converted artifact attached.
- **AC-005**: Every fix lands with a regression test pinned to the real bytes that exposed it (synthetic-only tests may not close a corpus failure).

## Technical Constraints & Invariants

1. **Fix nothing during Phase 0** — measurement first; the baseline run defines the regression bar.
2. **Epsilons are explicit and committed** — height 0.5 yd / alpha 0.05 as in `ValidateRoundTripCommand`; object converters default to exact equality with named exceptions.
3. **Client roots are runtime configuration** — never hardcode `H:\CLIENTS`; every harness takes `--client-root`.
4. **The real client is operator-owned** — the harness prepares staging + commands and stops; no automated launch of the game binary from tests.
5. **Converters stay thin over Core** — the gate calls the same Core converters the viewer uses; no test-only conversion paths.
