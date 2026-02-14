# Ghidra Baseline-Diff Workflow (ADT/WMO/MDX)

## Goal
Analyze a new pre-release binary by comparing it to the **current working implementation**.

Do **not** re-document unchanged systems.
Extract only differences that can affect parser correctness.

---

## Baseline sources (authoritative)
- Code baseline:
  - `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
  - `src/MdxViewer/Terrain/FormatProfileRegistry.cs`
- Profile architecture:
  - `specifications/versioning/Parser_Profile_Architecture.md`
- Current known profile evidence:
  - `specifications/0.9.1/Parser_Profile_0.9.1.3810_Binary.md`
  - `specifications/0.9.1/Parser_Profile_0.9.1.3810_Field_Map.md`
- Stability policy:
  - `specifications/versioning/Stability_Matrix_0.6.0_to_0.9.x.md`

---

## Non-goals
- No full reverse engineering pass per executable.
- No reprocessing binaries already covered unless a contradiction is found.
- No new architecture docs unless baseline profile fields cannot express a discovered delta.

---

## Workflow

## Step 1: Build target profile header
For target binary `<build>` define:
- `BuildId`
- `BaselineProfileComparedAgainst` (`060_070`, `080_*`, `091_3810`, etc.)
- `Scope` (`ADT`, `WMO`, `MDX`)

If build is already fully covered by existing docs, stop.

## Step 2: Function anchor map (addresses only)
Find native functions equivalent to baseline parser chain:
- Large WDT or ADT root/create, MCIN, MCNK, MCLQ/MH2O, placements
- WMO root/group/liquids
- MDX geometry/material/animation/texture load

Record address + purpose only.
Do not deep-document fields yet.

## Step 3: Delta extraction only
For each domain field/chunk contract, classify as:
- `same` (ignore)
- `changed` (record exact delta)
- `new/unknown` (record evidence + confidence)

Capture only changed/unknown items.

## Step 4: Impact mapping to code
For each delta, map to:
- break class: `hard parse fail | wrong geometry | wrong placement | visual artifact`
- exact file/method touchpoint in repo
- profile field change required

## Step 5: Patch intent summary
Produce minimal implementation intent:
- profile additions/edits
- guardrail changes
- fallback behavior
- diagnostics counter additions (if needed)

---

## Required output (single file per build)
Create:
- `specifications/outputs/<build>/baseline-diff-<build>.md`

Use this exact template:

```md
# Build Delta Report — <build>

## Baseline compared against
- AdtProfile: <id>
- WmoProfile: <id>
- MdxProfile: <id>

## ADT deltas (only changed/unknown)
| Item | Baseline | <build> | Evidence (addr/snippet) | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|

## WMO deltas (only changed/unknown)
| Item | Baseline | <build> | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|

## MDX deltas (only changed/unknown)
| Item | Baseline | <build> | Evidence | Code touchpoint | Severity | Confidence |
|---|---|---|---|---|---|---|

## Required profile edits
- <exact field additions/changes>

## Implementation targets
- <file + method list>

## Open unknowns
- <unknown + required proof function>
```

---

## Delta checklist by domain

## ADT
- Root token/offset policy changed?
- MCIN entry size/field use changed?
- MCNK required subchunks changed?
- MCLQ stride/offset/flow block changed?
- MH2O policy changed?
- Placement record sizes or indirection semantics changed?

## WMO
- Root required chunk order changed?
- Group (`MOGP`) required/optional chunk gates changed?
- `MLIQ` layout/flags/height/mask semantics changed?
- Placement/def chain interpretation changed?

## MDX
- Required chunk order/seek behavior changed?
- Geometry stream layout changed?
- Material/layer policy changed?
- Animation sequence/keyframe/compression policy changed?
- Texture/UV/wrap/replaceable handling changed?

---

## Evidence rules
Every changed/unknown item must include:
1. function address
2. proving snippet/pseudocode
3. constant/offset/record-size values
4. confidence (`High|Medium|Low`)

If no evidence: mark `unknown` and do not force implementation.

---

## Diagnostics requirements
For any profile change, ensure reporting can surface:
- `InvalidChunkSignatureCount`
- `InvalidChunkSizeCount`
- `MissingRequiredChunkCount`
- `UnknownFieldUsageCount`
- `UnsupportedProfileFallbackCount`

Include in log context:
- build id
- profile id
- file path
- chunk family (`ADT|WMO|MDX`)

---

## Definition of done
A new binary analysis is done when:
- unchanged baseline assumptions are not duplicated,
- all discovered divergences map to explicit profile/code changes,
- unknowns are listed with concrete proof tasks,
- implementation can proceed via minimal delta patching.
