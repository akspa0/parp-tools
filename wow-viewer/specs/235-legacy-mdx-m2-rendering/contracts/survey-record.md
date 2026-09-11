# Contract: Survey Record (extended from Spec 154)

Reused from `specs/154-m2-era-reader-parity/contracts/survey-record.md` with one extension. See
that file for the full producer/guarantees/shape/consumers/non-goals definition — this document
states only the delta.

## Extension: MDX coverage + `lightEffect` section

Spec 154's contract scoped the survey to M2 assets only. This feature's US1 extends it to also
cover MDX assets across the 1.0.0–3.0.1 range (per spec.md FR-001), and adds a `lightEffect`
section outcome (data-model.md) alongside the existing `identity`/`skeleton`/`sequences`/
`geometry`/`cameras` sections.

Extended shape (delta from Spec 154's JSON example):

```json
{
  "build": { "version": "0.10.3892", "buildNumber": "3892", "rootLabel": "<configured root>" },
  "modelPath": "Creature\\SomeTorch\\SomeTorch.mdx",
  "layout": {
    "declaredMagic": "MDLX",
    "declaredVersion": "<mdx version>",
    "selectedLayout": "<layout applied>",
    "selectionEvidence": "<what in this file selected it>"
  },
  "sections": [
    { "section": "identity",    "state": "Succeeded" },
    { "section": "geometry",    "state": "Succeeded" },
    { "section": "lightEffect", "state": "Failed", "detail": "light type Omni2 not yet modeled by effect pipeline" }
  ],
  "readAt": "2026-09-10T00:00:00Z"
}
```

## Guarantees (unchanged from Spec 154, restated for completeness)

1. A record is always produced, even for a model that cannot be read at all — reading never
   terminates the process.
2. Provenance (`BuildIdentity` + configured root label) is mandatory on every record.
3. Layout selection is justified with evidence from the file itself, never inferred from a sibling
   build's version word.
4. Per-section outcomes are independent — one section failing does not suppress the others.
5. Failures inside an indexed array report the element index.
6. A legitimately-absent section (`NotPresent`) is distinct from one that failed to read
   (`Failed`).

## Non-goals

Same as Spec 154's contract: not a performance record, not a full archive inventory, carries no
client file content (paths/outcomes/provenance only — safe to commit as evidence). Does not cover
the `FuckportedAssetCheck`/`LightEmitterEffect` records themselves — those are separate contracts
(see `fuckported-asset-parity.md`) that a survey row may reference by outcome but does not embed.
