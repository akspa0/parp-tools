# Contract: WTF Command Execution Outcome

Shape of one command's execution result, as produced by `WtfCommandDispatcher` (Phase 2) and reported by
`wtf run` (the CLI dry-run surface) or the in-viewer command runner. Built from a Spec 159 `WtfLine`, never
a re-parse of raw text.

```json
{
  "sourceLine": "worldport 0 1234.5 -678.9 12.3",
  "kind": "Worldport",
  "mapId": 0,
  "position": { "x": 1234.5, "y": -678.9, "z": 12.3 },
  "coordinatesPlausible": true,
  "status": "Applied",
  "detail": null
}
```

A teleport (no map ID) example:

```json
{
  "sourceLine": "teleport 45.0 -12.0 100.0",
  "kind": "Teleport",
  "mapId": null,
  "position": { "x": 45.0, "y": -12.0, "z": 100.0 },
  "coordinatesPlausible": true,
  "status": "NoCurrentMap",
  "detail": "no map is currently loaded; teleport has nothing to reposition on"
}
```

**Contract rules**:

- `status` is always one of `Applied`, `MapLoadFailed`, `NoCurrentMap`, `Unrecognized` — never a bare
  boolean. A consumer must be able to tell *why* a command didn't apply, not just that it didn't.
- `coordinatesPlausible` is carried through even on success — a command can apply and still be flagged as
  numerically implausible, since this feature does not get to decide the source data is wrong (spec.md).
- `sourceLine` is always the original text, unmodified — traceability back to the real file is required
  by FR-007, not optional diagnostic sugar.
