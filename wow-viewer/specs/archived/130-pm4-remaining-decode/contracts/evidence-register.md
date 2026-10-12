# Contract: Decode Evidence Register

**Phase**: 1 | **Satisfies**: FR-007, FR-008, FR-012, SC-004, SC-007

**Path**: `output/pm4-decode/evidence-register.json`

The durable record of what is known, what is ruled out, and how confident each claim is. Nine
questions have been open long enough that some will end as eliminations rather than answers; without
this file those eliminations are re-searched by the next person.

## Schema

```jsonc
{
  "schemaVersion": 1,
  "generatedUtc": "2026-08-03T00:00:00Z",
  "corpusSignature": "test_data/development/World/Maps/development@616",
  "findings": [
    {
      "key": "MSLK.MspiIndexCount",
      "status": "Partial",
      "confidence": "Low",
      "statement": "Window length into MSPI. Whether a window is a polyline, a triangle run, or something else is undecided.",
      "evidence": [
        {
          "claim": "active windows resolve into MSPI without exception",
          "fits": 598882,
          "misses": 0,
          "fileCount": 309,
          "source": "pm4 unknowns"
        }
      ],
      "eliminations": [
        {
          "candidate": "trianglesOnly counter as a discriminator",
          "reason": "trianglesMode implies indicesMode for all first>=0, count>=0, so the trianglesOnly bucket cannot be non-zero for any input. Its reported 0 is a property of the inequality, not of the format.",
          "evidence": {
            "claim": "3*first + 3*count >= first + count holds for every record in the corpus",
            "fits": 0,
            "misses": 0,
            "fileCount": 309,
            "source": "research.md R3"
          }
        }
      ],
      "nextStep": "Discriminate with window-size histogram by TypeFlags family, closure test, and degenerate-triple test.",
      "corpusSignature": "test_data/development/World/Maps/development@616"
    }
  ]
}
```

## Field rules

| field | rule |
|---|---|
| `key` | Unique within the register. Once published, never renamed — rename means a new key plus an elimination on the old one. |
| `status` | `Open` · `Partial` · `Resolved` · `Eliminated` · `NoSemanticMeaning` |
| `confidence` | `None` · `Low` · `Medium` · `High` · `Verified`. Required. |
| `statement` | The interpretation, or the question when `Open`. Never empty. |
| `evidence[]` | May be empty only when `status == Open`. |
| `eliminations[]` | Required non-empty when `status == Eliminated`. |
| `nextStep` | Must be `null` when `status` is `Resolved`, `Eliminated`, or `NoSemanticMeaning`. |
| `corpusSignature` | Per finding as well as per document, so a stale finding is detectable after a corpus change. |

## Write-time validation

These are enforced by the register, not left to the caller — a rule that depends on discipline is a
rule that gets skipped at 2am.

1. A status change out of `Open` with zero evidence items is **rejected**.
2. `status == Eliminated` with zero eliminations is **rejected**.
3. A terminal status with a non-null `nextStep` is **rejected**.
4. Merging never deletes an elimination. An incoming finding's eliminations are unioned with the
   stored ones, keyed on `candidate`.
5. Writing a finding whose `corpusSignature` differs from the document's emits a warning and keeps
   both — it does not silently overwrite.

## Confidence vocabulary

Aligned with what the decoder already publishes in `Pm4TerminologyCatalog`, so the two never drift:

- `MSLK.TypeFlags` — "medium, partial, not corpus-closed"
- `MSUR.GroupKey` — "low, local research alias only"
- `MSLK.GroupObjectId` — "low, not a confirmed object identity field"

Phase 1's gate is that these round-trip verbatim. If the register cannot express what the catalog
already says, the register is wrong.

## Seeding

Phase 1 seeds the register from the nine findings `Pm4ResearchUnknownsAnalyzer` already emits as
`Pm4UnknownFinding` records, preserving each one's status, evidence string, and next step exactly.
The register starts as a faithful restatement of what is already known and grows from there.
