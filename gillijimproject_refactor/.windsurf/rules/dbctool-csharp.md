---
trigger: always_off
---

# 🧭 Windsurf Rules — DBCTool Concise C# (net9.0)

## 1) Purpose
Keep the codebase modular, testable, and constraint-driven. Favor small edits, explicit outputs, and deterministic logic (no cross-map/continent leaks).

## 2) Core Working Rules
- Break work into small, reviewable steps. Produce or update a plan doc first if scope > ~50 LOC.
- Never introduce “god methods”. If a function grows > 150 LOC or > 3 responsibilities, split it.
- Prefer pure functions; isolate I/O to thin edges. Inject dependencies via interfaces.

## 3) Architecture Boundaries (enforced)
- Domain contracts only depend on abstractions:
  - IAlphaDecoder, IChainBuilder, IMapCrosswalk, IAreaMatcher, IExporter, IDbcdProvider.
- Mapping rules:
  - Top-level zone matches must be LK ParentAreaID == ID (zone-locked).
  - Subzone matches must be children of the chosen zone and same map (map-locked).
- Alpha decode rules:
  - When available, always build chains from decoded halves (hi16/lo16) and use decode’s continent.
  - Do not traverse ParentAreaNum naïvely when decode is available.

## 4) Implementation Rules
- Compute contResolved and use it for:
  - src_mapId in all V2 outputs
  - crosswalk mapIdX
  - path composition (prefix with mapNameX when known)
- Chain building:
  - If lo16 == 0: chain = [zoneName]
  - If lo16 > 0 and SubIndex hit: chain = [zoneName, subName]; else fall back to [zoneName]
- Matching:
  - TryMatchChainExact(mapIdX, chain) only; do not accept cross-map results.
  - If a suggested target’s map != mapIdX, mark as violation and discard.
- File outputs:
  - Patch/mapping/unmatched CSVs must share the same header and use contResolved.

## 5) Code Style (C#)
- Target: net9.0; nullable enabled; file-scoped namespaces.
- Use `var` when RHS is obvious; explicit type for public APIs.
- `readonly` for static tables and collections; prefer `static` local functions if no instance state.
- Guard clauses; early returns; no deep nesting.
- Logging: structured (`ILogger`), actionable messages; no debug spam in hot loops.

## 6) Tests & CI Expectations
- Unit tests per module (AlphaDecode, Mapping, Crosswalk).
- Golden-file checks for CSV outputs on representative subsets.
- No “WIP” merges: each PR must pass tests and lint.

## 7) Edits & Commits
- One logical change per PR.
- If an edit exceeds ~200 LOC, split it.
- Commit messages:
  - `feat(alpha-decode): ...`
  - `fix(mapping): ...`
  - `refactor(crosswalk): ...`
  - `docs(plan): ...`

## 8) Non‑Negotiables (hard constraints)
- Do not emit cross-continent mappings. If unsure: unmatched with rationale.
- Never overwrite explicit mappings with heuristics.
- Keep CSV schema stable unless the plan doc was updated first.

## 9) Documentation
- Update [DBCTool/docs/alpha-area-decode-v2.md](cci:7://file:///i:/parp-tools/pm4next-branch/parp-tools/gillijimproject_refactor/DBCTool/docs/alpha-area-decode-v2.md:0:0-0:0) when decoding or mapping rules change.
- Any new module: add a short “How it works” paragraph in the code (XML doc or top-of-file comment).
