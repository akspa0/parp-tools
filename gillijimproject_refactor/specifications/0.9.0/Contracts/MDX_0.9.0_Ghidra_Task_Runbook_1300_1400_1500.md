# MDX 0.9.0 Ghidra Task Runbook (1300 vs 1400 vs 1500)

## Objective
Resolve version-specific MDX regressions with evidence-first tasks:

- **1300**: baseline (works)
- **1400**: sequence names corrupted
- **1500**: loads but does not render

Use this as a strict task checklist. Capture only findings that map to a concrete parser/converter fix.

---

## Inputs
- One known-good **v1300** MDX sample
- One failing-name **v1400** MDX sample
- One invisible **v1500** MDX sample

For each sample record:
- path
- file size
- first 64 bytes (hex)

---

## Task 1 - Locate version routing and parse entry
### Goal
Find where MDX version controls parser behavior.

### Ghidra prompt
```text
Task: Locate the model parsing entrypoint and identify where MDX version values (1300, 1400, 1500) influence control flow.

Steps:
1) Find all xrefs to "VERS", "SEQS", "GEOS", "MDLX".
2) Identify the top-level chunk dispatch function.
3) Identify where version is read/stored and where it is compared.
4) Produce a table:
   - compare site address
   - compared constant(s)
   - taken branch target per version
   - function name/address handling each version

Output format:
- "Version switch map" with exact addresses and decompiled pseudocode snippets.
```

### Done when
- A single table exists showing exact branch split for 1300/1400/1500.

---

## Task 2 - Explain 1400 sequence-name corruption
### Goal
Prove the first SEQS decode divergence between 1300 and 1400.

### Ghidra prompt
```text
Task: Determine how SEQS names are decoded for version 1400 and why names become garbage.

Steps:
1) In the SEQS handler, trace record framing logic:
   - count source
   - stride source
   - per-record field offsets
2) Identify the exact source of sequence display name (byte range/offset).
3) Compare the path for version 1300 vs 1400.
4) Mark first divergence in:
   - stride value
   - name offset
   - alignment/padding assumptions

Output:
- "SEQS field provenance map" with:
  (version, stride, name_offset, interval_offset, flags_offset, bounds_offset)
- one minimal hypothesis for why 1400 names break.
```

### Done when
- You have one concrete address-level mismatch explaining bad 1400 names.

---

## Task 3 - Explain 1500 invisible render
### Goal
Find first render-reject branch for v1500.

### Ghidra prompt
```text
Task: Determine why v1500 reaches load/animation but fails render.

Steps:
1) Trace GEOS parse for version 1500:
   - VRTX interpreted count
   - PVTX interpreted count
   - max index
   - vertex count
2) Trace handoff to render prep and find reject conditions.
3) Identify the first branch that suppresses draw submission.

Output:
- "First-fail branch report":
  - function/address
  - predicate
  - concrete operands (left/right)
  - expected vs actual domain
```

### Done when
- One specific branch and predicate is identified as first-fail for v1500.

---

## Task 4 - Prove GEOS subchunk length semantics per version
### Goal
Determine bytes-vs-elements behavior by tag and version.

### Ghidra prompt
```text
Task: Resolve whether each GEOS subchunk length field is bytes or elements for v1300/v1400/v1500.

Subchunks:
VRTX, NRMS, PVTX, GNDX, MTGC, MATS, UVBS (and wrapper UVAS if applicable)

Output table:
| Version | Subchunk | DeclaredLengthInterpretation | ElementStride | DerivedCount | EvidenceAddress |
```

### Done when
- Table is complete for all three versions and core tags.

---

## Task 5 - Build operand source map for reject gate(s)
### Goal
Map static provenance of both compare operands at each critical reject gate.

### Ghidra prompt
```text
Task: For the render rejection compare that prevents drawing, map static provenance of both operands.

For each compare:
1) show CMP site
2) operand A source chain (register <- stack/field)
3) operand B source chain
4) branch polarity (reject if A>=B, A>B, etc.)
5) associated literal/log/assert push site

Output tuple format:
(compare_site, operandA_source, operandB_source, branch_polarity, literal_binding)
```

### Done when
- Every first-fail compare has a complete operand tuple.

---

## Task 6 - Produce strict 1300/1400/1500 differential
### Goal
Turn findings into patch-ready differential.

### Ghidra prompt
```text
Task: Produce a strict differential report: what is different between 1300 (good), 1400 (bad names), 1500 (invisible).

Required sections:
A) Parser path differences by version
B) SEQS record layout differences
C) GEOS/PVTX interpretation differences
D) First render gate hit differences
E) Minimal code change candidates ranked by confidence
```

### Done when
- You can point to one minimal fix candidate per symptom class.

---

## Task 7 - Patch handoff (minimal fixes only)
### Goal
Generate implementation notes that map directly to code edits.

### Ghidra prompt
```text
Task: Based on proven branch evidence only, propose minimal parser/converter fixes.

Constraints:
- no renderer workaround unless parser/converter parity cannot be achieved
- no speculative rewrites
- preserve hard-fail checks
- separate fix for 1400 SEQS names and 1500 render issue if root causes differ

Output:
1) exact function + line-range equivalent
2) before/after logic in pseudocode
3) why 1300 remains unaffected
4) regression tests to run
```

### Done when
- Fix list is concrete enough for direct patching in C#.

---

## Evidence capture template (fill per sample)
```text
Asset:
Version:
Path:
Size:
Header64:

[VersionRouting]
CompareSite:
BranchTaken:
ParserFunction:

[SEQS]
Stride:
NameOffset:
IntervalOffsets:
FlagsOffset:
BoundsOffsets:
DivergenceFrom1300:

[GEOS]
VRTX semantics/count:
PVTX semantics/count:
maxIndex:
vertCount:
DivergenceFrom1300:

[RenderGate]
FirstFailFunction:
FirstFailCompare:
OperandA source/value:
OperandB source/value:
BranchPolarity:
Outcome:

[FixCandidate]
TargetFunction:
MinimalChange:
Risk:
```

---

## Stop condition (do not over-investigate)
Stop and patch when both are true:
1. A single proven **1400 SEQS divergence** explains bad names.
2. A single proven **1500 first-fail render branch** explains invisibility.

Anything beyond that is optional cleanup, not blocker work.
