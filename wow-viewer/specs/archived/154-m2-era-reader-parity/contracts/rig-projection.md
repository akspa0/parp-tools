# Contract: Rig Projection

The cross-route comparable shape for US4. One projection per model, identical in meaning whichever
reading route produced it.

## Producer

A projection over an already-read model document. It performs no reading of its own — if a route
cannot read a model, there is no projection, and that is the survey's business to report.

## Guarantees

1. **Route-independent meaning.** A field means the same thing whether the model came from the `MDLX`
   route or an `MD20` route (FR-010). A consumer must not need to know which route produced it.
2. **Provenance attached.** Build identity and source model travel with the projection, so any
   comparison names both sides (SC-006).
3. **Validated skeleton only.** A projection is produced only from a skeleton that passed the structural
   rules — complete count, finite pivots, in-range acyclic parents. A rejected skeleton yields no
   projection rather than a suspect one.
4. **No motion data.** Bone hierarchy, pivots, and the sequence table only. No keyframes, no tracks, no
   exported motion — that is out of scope for this feature.

## Shape

```json
{
  "build": { "version": "0.5.3", "buildNumber": "3368", "rootLabel": "<configured root>" },
  "modelPath": "Creature\\HighElf\\HighElfMale_Warrior.mdx",
  "route": "<route that produced it>",
  "skeleton": {
    "boneCount": 54,
    "bones": [ { "index": 0, "identity": "<key or name>", "parent": -1, "pivot": [0.0, 0.0, 0.0] } ]
  },
  "sequences": {
    "count": 106,
    "entries": [ { "identity": "Walk", "durationMs": 1000, "isAlias": false } ]
  }
}
```

## Comparison rules

**Structural, never by count.** The two working routes yield 54 bones (0.5.3 High Elf) and 151 bones
(3.3.0 Blood Elf) for rigs that may well be related. A count comparison would report difference where
correspondence exists.

The question a comparison answers is: **does the smaller bone set appear within the larger, with
corresponding parent structure and pivots?** Not: are they the same size.

Consequently:

- A comparison reports correspondence, difference, and what did not correspond — never a single
  similarity number with no structure behind it.
- Controls are mandatory. Comparing two rigs in isolation cannot distinguish "these are related" from
  "all humanoid rigs share a base". Unrelated rigs must be run alongside, and a comparison that reports
  correspondence for the controls too has demonstrated nothing.

## Non-goals

- Not an interchange format. It exists to be compared, not imported into other tools.
- Not a motion export. BVH, FBX, and pose clips are explicitly out of scope.
- Not a renderer input. Nothing draws from a projection.
