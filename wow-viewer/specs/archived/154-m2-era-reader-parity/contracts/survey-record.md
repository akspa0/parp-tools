# Contract: Survey Record

The evidence artifact for US1. Every support claim in this feature traces to one of these rows.

## Producer

A read-attempt over one model in one configured archive root, driven from `tools/inspect`. The tool is
a thin wrapper; the reporting capability lives in `WowViewer.Core.IO`.

## Guarantees

1. **A record is always produced.** A model that cannot be read yields a record describing the failure.
   Reading never terminates the process (FR-007). This is the contract's central promise — the current
   behaviour violates it on at least two staged builds.
2. **Provenance is mandatory.** Every record carries the build identity and the configured root label
   it was read from (FR-001).
3. **Layout selection is justified.** Every record states which layout was applied and what evidence in
   *that file* selected it (FR-002).
4. **Per-section outcomes.** One section failing does not suppress the others. Each section reports
   independently.
5. **Failures are positioned.** A failure inside an indexed array reports the element index (FR-006).
6. **Absent is not failed.** A section legitimately not present is distinct from one that failed to
   read (FR-005).

## Shape

```json
{
  "build": { "version": "3.0.1", "buildNumber": "8303", "rootLabel": "<configured root>" },
  "modelPath": "CHARACTER\\BloodElf\\Male\\BloodElfMale.M2",
  "layout": {
    "declaredMagic": "MD20",
    "declaredVersion": "0x107",
    "selectedLayout": "<layout applied>",
    "selectionEvidence": "<what in this file selected it>"
  },
  "sections": [
    { "section": "identity",  "state": "Succeeded" },
    { "section": "skeleton",  "state": "Failed", "elementIndex": 10, "detail": "non-finite pivot component" },
    { "section": "sequences", "state": "Succeeded" },
    { "section": "geometry",  "state": "Failed", "detail": "skeleton unavailable" },
    { "section": "cameras",   "state": "NotPresent" }
  ],
  "readAt": "2026-08-15T00:00:00Z"
}
```

## Consumers

- The Phase 0 build enumeration, which is complete only when every staged build has a full set of rows.
- Phase 3 scoping — which builds exist in the currently-refusing range.
- Regression comparison: the Phase 1 survey is the baseline that later phases diff against.

## Non-goals

- Not a performance record. No timings.
- Not an archive inventory. A fixed model set across builds, not a full sweep.
- Carries no client file content — paths, outcomes, and provenance only, so records are safe to commit
  as evidence.
