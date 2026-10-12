# Contract: Confirmed Match Library

Persisted record shape. Identifiers, paths, and provenance only — never client asset bytes (FR-016),
consistent with every other generated record in this project (e.g. Spec 155's sweep reports).

```json
{
  "pm4Identity": { "tileX": 32, "tileY": 48, "ck24": 1310724, "objectPart": 0 },
  "placementIdentity": {
    "build": "0.5.3.3368",
    "map": "Azeroth",
    "tileX": 32,
    "tileY": 48,
    "placementKind": "Modf",
    "uniqueId": 184023,
    "assetPath": "World\\wmo\\Azeroth\\Buildings\\Human_Farm\\Farm.wmo"
  },
  "confirmedAtUtc": "2026-08-16T18:22:04Z",
  "reason": "Footprint and roofline match exactly; same tile, same orientation, verified by clicking both.",
  "status": "Confirmed"
}
```

A retraction appends an event rather than deleting the original (FR-013):

```json
{
  "pm4Identity": { "tileX": 32, "tileY": 48, "ck24": 1310724, "objectPart": 0 },
  "event": "Retracted",
  "atUtc": "2026-08-17T09:01:55Z",
  "reason": "Was actually the adjacent barn, not the farmhouse."
}
```

**Contract rules**:

- `reason` is required on both confirmation and retraction. A match with no stated basis is not a
  confirmation this library will store — FR-011 makes the evidence part of the record, not optional
  metadata.
- `status` is **derived** from the event history, never written as a mutable field that could drift out of
  sync with it.
- No entry may be written by any automatic process. A score, proximity, or shared-fingerprint signal can
  only ever produce a *candidate* for review (FR-012, FR-015, SC-009) — candidates live outside this
  contract precisely so a candidate can never be mistaken for a confirmation.
- Rejecting a surfaced candidate is itself recorded durably, so a false candidate is not re-surfaced as
  new every session (spec.md edge case).
