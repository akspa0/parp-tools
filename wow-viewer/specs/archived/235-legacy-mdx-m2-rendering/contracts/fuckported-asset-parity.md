# Contract: Fuckported-Asset Parity Check (NEW — US4)

The evidence artifact for FR-008/FR-009. No prior spec defined this; it is genuinely new to Spec
235.

## Producer

A comparison over one asset: this project's own reader result versus an external reference
parser's result (Warcraft.NET, or Benilla for 1.x-specific cases) for the same file. Driven from
`tools/inspect` (new `m2 fuckport-check` command, thin wrapper per Library-First).

## Guarantees

1. **Parity is the target, not perfection.** This project's reader is only required to render an
   asset that *some* external reference can already parse — it is not required to invent tolerance
   for a file no reference implementation can make sense of either.
2. **A failure is always specific.** When this reader fails an asset the external reference
   succeeds on, the record names the divergence (which chunk, which field, what's non-standard
   about it) — "doesn't parse" alone is not a complete record.
3. **A double failure is reported, not hidden.** When neither this reader nor the external
   reference can parse an asset, that is recorded as its own outcome (FR-009) — it is not
   collapsed into "fuckported" or silently dropped from the corpus.
4. **The check names which external reference was used.** Warcraft.NET and Benilla are not
   interchangeable — a record states which one produced the comparison result.

## Shape

```json
{
  "assetPath": "World\\Some\\NonStandardAsset.m2",
  "declaredFormat": { "magic": "MD20", "version": "0x108" },
  "thisReaderResult": { "state": "Failed", "detail": "submesh triangle range out of bounds at offset 0x1A40" },
  "externalReferenceResult": { "reference": "WarcraftNET", "state": "Succeeded" },
  "divergenceDetail": "chunk MD21 sub-block length field does not match the standard 0x108 layout; a third-party tool appears to have re-packed the skin section without updating the length prefix"
}
```

## Consumers

- Phase 4 (plan.md) — diagnoses `divergenceDetail` and decides whether a bounds/validation
  relaxation or a distinct repair step is the right fix.
- The regression test suite — each confirmed fuckported-asset case gets a pinned test using this
  record's shape as its assertion target.

## Non-goals

- Not a general asset-corpus scanner. This feature validates against the specific fuckported
  asset(s) identified in Phase 0, not every asset that might exist (spec.md Scale/Scope).
- Not a new converter or repair tool in itself (spec.md Out of Scope: "Building new
  fuckported-style converters"). If Phase 4 concludes a repair step is genuinely needed, that
  repair lives beside the existing converters (FR-015's reconciliation target), not as new
  standalone tooling.
