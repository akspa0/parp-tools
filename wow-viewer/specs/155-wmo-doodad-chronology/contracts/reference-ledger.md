# Contract: Reference Ledger

Every claim the game data makes that an asset exists, with what actually happened when that claim was
checked.

## Producer

Extraction over one referencing asset, or over a whole build's corpus. Extraction reuses the existing
world-object doodad and texture readers and the model texture tables; it adds no parsing.

## Guarantees

1. **Three resolution states, never two.** `Present`, `Absent`, and `Unreadable` are distinct.
   Collapsing `Unreadable` into `Absent` manufactures missing assets, which is the one output of this
   feature that must never be wrong in that direction.
2. **Resolution is independent of the catalogue.** Whether a listfile names an asset has no bearing on
   whether the probe reports it present. The two facts are compared later, deliberately, as the
   feature's whole point.
3. **Every reference carries its source.** The referencing asset and the reference kind travel with the
   claim, so a finding can be traced to the object that made it.
4. **Provenance is mandatory.** Build identity and configured root on every record.
5. **No content.** Paths, outcomes and provenance only. No client file bytes.

## Shape

```json
{
  "build": { "version": "0.5.3", "buildNumber": "3368", "rootLabel": "<configured root>" },
  "source": {
    "path": "World\\wmo\\Azeroth\\<object>.wmo",
    "kind": "WorldObject",
    "readState": "Read",
    "routeBlocked": false
  },
  "references": [
    { "kind": "PlacedDoodad",       "targetPath": "World\\...\\<model>.mdx", "resolution": "Present" },
    { "kind": "WorldObjectTexture", "targetPath": "World\\...\\<tex>.blp",   "resolution": "Present" },
    { "kind": "ModelTexture",       "targetPath": "World\\...\\<spray>.blp", "resolution": "Absent" }
  ]
}
```

## Consumers

- The three-set comparison, which joins these against the catalogued set.
- Candidate matching, which works from `Absent` references.
- Cross-build chronology, which works from the per-build reference sets.

## Non-goals

- Not a rendering input. Nothing draws from a ledger.
- Not a judgement of intent. An `Absent` reference may be deliberate; the ledger records the fact only.
- Not a full asset index. It records what was *referenced*, not everything a build contains.
