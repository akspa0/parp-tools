# Contract: Sweep Report

The header for one build's sweep. **Its primary job is to make incompleteness impossible to miss.**

## Why this contract exists separately

The dangerous output of this feature is not a wrong answer, it is a confident empty one. A build whose
models cannot be read finds no texture references and therefore reports no missing textures — which
reads exactly like a healthy build. The same shape has already bitten this project twice: a documented
tool that did not exist, and a corpus reported as 1 when it was 532.

So coverage is not a footnote on the results. It is a required part of them, and a consumer cannot
read the findings without also receiving the limits.

## Guarantees

1. **Examined counts are always reported**, including zero. An under-counted sweep is visible rather
   than silent.
2. **Blocked routes are a distinct state.** "This build's model route does not read yet" is recorded
   separately from "no references were found" and from "an individual asset failed".
3. **Unreadable assets do not abort the sweep.** They are recorded and the sweep continues.
4. **Swept reference sources are enumerated.** Any consumer drawing conclusions about what is
   *unreferenced* receives the list of what was actually examined.
5. **A report with blocked routes is not a complete picture**, and any consumer presenting its findings
   must surface that.

## Shape

```json
{
  "build": { "version": "0.5.3", "buildNumber": "3368", "rootLabel": "<configured root>" },
  "worldObjectsExamined": 532,
  "modelsExamined": 5545,
  "assetsUnreadable": { "count": 0, "paths": [] },
  "routesBlocked": [],
  "referenceSourcesSwept": ["PlacedDoodad", "WorldObjectTexture", "ModelTexture"],
  "complete": true
}
```

A blocked build, for contrast — note that `modelsExamined: 0` is meaningless without the block record
beside it:

```json
{
  "build": { "version": "3.0.1", "buildNumber": "8303", "rootLabel": "<configured root>" },
  "worldObjectsExamined": 9711,
  "modelsExamined": 0,
  "assetsUnreadable": { "count": 0, "paths": [] },
  "routesBlocked": [
    { "route": "<model route for this build>", "assetCount": 17296, "reason": "route does not read yet (Spec 154)" }
  ],
  "referenceSourcesSwept": ["PlacedDoodad", "WorldObjectTexture"],
  "complete": false
}
```

## Validation

- A sweep must cover the whole corpus. Reporting one world object for the earliest staged build
  indicates the internal-listfile index was used instead of the archive access layer.
- Known missing-asset instances, such as the Mt. Hyjal effect objects, are a useful sanity read on a
  completed report — if a sweep of their build does not contain them, something is wrong with coverage.
  This is a check performed **on output that already exists**; it is not a target, and no work is
  scoped around locating any individual object.

## Non-goals

- Not a performance record. No timings.
- Not a completeness claim about the build. It reports what this feature examined, nothing more.
