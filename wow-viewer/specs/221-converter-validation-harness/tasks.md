# Spec 221: Tasks — Converter Regression Harness

## Phase 0: Baseline measurement (fix nothing)

- [ ] 221-T001: Inventory audit — for each of the 25 converter commands, record: last-touched commit, owning Core converter, existing unit tests (synthetic vs real bytes), and whether a round-trip validator exists. Emit `evidence/converter-inventory.md`.
- [ ] 221-T002: Run the existing `validate-roundtrip` (LK + Alpha modes) over `H:\CLIENTS\Vanilla\0.x\0_5_3_3368` Azeroth (bounded, e.g. 64 tiles each) and record the table as the terrain baseline.
- [ ] 221-T003: Run the existing WMO/M2 unit suites and record green/red; identify which tests use synthetic data only.
- [ ] 221-T004: Gate: baseline committed; the regression bar is now "no worse than baseline".

## Phase 1: Object corpus round-trip validator

- [ ] 221-T101: Add `converter validate-object-roundtrip` (WMO V14↔V17, M2↔MDX modes): enumerate doodad-bearing WMOs + M2/MDX models from the client catalog, bounded by `--limit`; round-trip each; diff all spec-relevant chunks (MODD/MODN/MODS/MOHD/bounds, MDXX/SKIN headers, sequences, textures) with exact equality + named exceptions.
- [ ] 221-T102: Byte-equality assertion on untouched chunks (groups, geometry) — regenerate-only-what-changed or fail.
- [ ] 221-T103: Corpus run (≥50 WMOs, ≥50 models); commit the table; every drift either fixed with a pinned real-bytes regression test or attributed to a documented implementation change.
- [ ] 221-T104: Gate: object validator green within budget; integrated into the Phase 2 aggregate.

## Phase 2: Corpus gate aggregation

- [ ] 221-T201: Add `converter validate-corpus`: orchestrates terrain (LK + Alpha) + object validators against one `--client-root`, prints the unified pass/fail/worst-delta table, exits non-zero over budget.
- [ ] 221-T202: Wire per-converter budgets (default exact for objects, committed epsilons for terrain); document how to tighten one without loosening others.
- [ ] 221-T203: Gate: `validate-corpus` green on 0.5.3 Azeroth + a second map with WMO-heavy tiles (e.g. Kalimdor cities row).

## Phase 3: Oracle cross-check

- [ ] 221-T301: Harness `converter oracle-crosscheck --oracle <gillijimproject-output>`: run shared Alpha WDT inputs through both implementations; report per-chunk/per-field disagreements.
- [ ] 221-T302: Triage ledger: every disagreement classified as fixed-here (with test) or oracle-wrong (with evidence) in `evidence/oracle-triage.md`; zero unclassified.
- [ ] 221-T303: Gate: triage ledger complete for the shared corpus slice.

## Phase 4: Real-client smoke & collision harness (operator-executed)

- [ ] 221-T401: Author `docs/real-client-harness-053.md`: the minimal file set (exe + required MPQs/WDB) to reach menu/console and load a map, staged from `H:\CLIENTS\Vanilla\0.x\0_5_3_3368` into a writable sandbox; install steps for a converted Alpha WDT + its ADTs/WMOs/M2s.
- [ ] 221-T402: Prepare exact PowerShell 7 commands (staging copy, install, launch) and stop — operator runs them per workspace policy.
- [ ] 221-T403: Collision checklist: enter the converted map, walk/swim the fixture coordinates, verify terrain collision, placement solidity, and liquid surface behavior against the source-map expectations; record pass/fail per checkpoint with screenshots.
- [ ] 221-T404: Gate: operator-run checklist either confirms AC-004 or files the first defect with the converted artifact attached; defect becomes a pinned regression test (AC-005).
- [ ] 221-T405: Register Spec 221 in [`specs/STATUS.md`](file:///I:/parp/parp-tools/wow-viewer/specs/STATUS.md) status and update [`memory-bank/activeContext.md`](file:///I:/parp/parp-tools/wow-viewer/memory-bank/activeContext.md) in the same pass as each phase gate.
