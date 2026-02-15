# MDX 0.9.x Ground-Truth Investigation Playbook (Ghidra)

## 1) Goal
Establish **exact client-native behavior** for model loading and decode decisions so we can fix MDX failures from first principles:

- MDX that load but do not render (geosets/vertices missing or rejected)
- bad/garbled sequence names on newer M2-style MDX assets
- routing ambiguity between MDLX-style and MD20/M2-style containers

This playbook is designed to produce decision-grade evidence before code changes.

---

## 2) Scope and non-goals
### In scope
- Binary-level tracing of model load entrypoints
- Chunk dispatch and per-chunk decode behavior for `SEQS`, `GEOS`, and core geometry subchunks
- Runtime validation branches that suppress rendering (index/vertex consistency, geoset rejection)

### Out of scope
- speculative parser rewrites before evidence
- broad refactors or architecture changes

---

## 3) Inputs (asset matrix)
Use a fixed matrix every run:

1. `kelthuzad.mdx` (known problematic)
2. one newer M2-style `.mdx` that currently shows bad sequence labels
3. one newer M2-style `.mdx` that currently loads but renders nothing
4. one known-good control MDX (renders correctly and sequence labels sane)

For each sample, keep:
- full path
- file size
- first 64 bytes (hex)
- expected behavior in current toolchain

---

## 4) Ghidra project setup
1. Create a dedicated Ghidra project: `mdx-09x-ground-truth`.
2. Import target client executable(s) and model-related DLLs.
3. Run full analysis with:
   - string references
   - decompiler parameter ID
   - switch analysis
4. Create bookmarks folder structure:
   - `ENTRYPOINTS`
   - `CHUNK_DISPATCH`
   - `SEQS`
   - `GEOS`
   - `RENDER_GATES`
   - `ROUTING`

---

## 5) Fast function discovery strategy
Use this exact sequence:

1. Search strings for model extensions and chunk tags:
   - `.mdx`, `.mdl`, `.m2`, `.skin`
   - `MDLX`, `MD20`, `SEQS`, `GEOS`, `VRTX`, `PVTX`, `NRMS`, `UVBS`
2. For each hit, XREF back to:
   - file/container detection code
   - chunk-loop dispatch code (`read FourCC + size + payload` patterns)
3. Mark candidate functions and rename with neutral names first:
   - `ModelLoad_Entry_A`
   - `ModelParse_ChunkLoop_A`
   - `ModelParse_SEQS_A`
   - `ModelParse_GEOS_A`

Do not over-rename until behavior is confirmed.

---

## 6) What to prove in binary (critical questions)

### Q1: Container routing
- Does extension control routing, or does magic/header control routing?
- If both are used, what is precedence when they disagree?

### Q2: `SEQS` structure expectations
- Is there a leading count?
- Is record size fixed or variant?
- Which record sizes are accepted/rejected?
- Which fields are used for UI/display sequence naming?

### Q3: `GEOS`/geometry payload semantics
- For each geometry subchunk (`VRTX`, `PVTX`, etc.), is payload interpreted as:
  - byte length
  - element count
  - element count * element-stride
- What branches trigger geoset skip/reject?

### Q4: Render suppression gates
- Which validation check causes “loaded but invisible”?
- Are there hard constraints around index bounds, degenerate counts, or material/flags?

---

## 7) Runtime capture checklist (per asset)
When stepping through parser code, capture these values exactly.

### 7.1 Routing snapshot
- file extension
- magic / root signature bytes
- selected parse path/function
- fallback path attempted? (Y/N)

### 7.2 `SEQS` snapshot
- chunk start offset and chunk size
- parser assumption branch (e.g., counted fixed-size vs legacy/raw)
- computed record count
- computed record stride
- first 2 sequence records parsed fields:
  - name bytes / interpreted name
  - start/end interval
  - move speed
  - flags
  - frequency
  - bounds/radius fields used

### 7.3 `GEOS` snapshot
- geoset count (if explicit)
- for each key subchunk (`VRTX`, `PVTX`, `NRMS`, `UVBS`):
  - declared length
  - parser interpretation mode (bytes/elements)
  - resulting element count
- final vertex count and index count

### 7.4 Render gate snapshot
- any branch that marks geoset invalid/hidden/skipped
- exact condition values at decision:
  - max index
  - vertex count
  - material id
  - flags relevant to visibility

---

## 8) Evidence table template
Copy per asset:

```text
Asset:
Path:
Size:
Header (hex 64B):

[Routing]
Extension:
Magic:
Selected Path:
Fallback Attempted:

[SEQS]
ChunkOffset:
ChunkSize:
Branch:
RecordCount:
RecordStride:
NameDecodeRule:
Record0Summary:
Record1Summary:

[GEOS]
ChunkOffset:
ChunkSize:
Branch:
VRTX Declared/Interpreted:
PVTX Declared/Interpreted:
NRMS Declared/Interpreted:
UVBS Declared/Interpreted:
FinalVertexCount:
FinalIndexCount:

[RenderGate]
GateFunction:
Condition:
ObservedValues:
Outcome:

[Conclusion]
RootCauseClass: (routing | seqs-layout | geos-payload | render-gate | mixed)
Confidence: (low/med/high)
```

---

## 9) Decision matrix for code fixes
Only patch after this matrix is populated.

1. If binary `SEQS` accepts only stride `X` in path `Y`:
   - normalize converter output and parser expectations to `X` for that path.
2. If binary differentiates `SEQS` layouts by signature/count framing:
   - implement the same discriminator order.
3. If binary interprets geometry subchunk lengths per-tag (bytes vs elements):
   - mirror per-tag interpretation exactly, not globally.
4. If invisibility is caused by one render gate:
   - patch upstream data shaping to satisfy gate first.

Priority rule: **upstream parse/conversion parity > downstream rendering workaround**.

---

## 10) Patch protocol after investigation
1. Make smallest possible parser/converter change that matches observed binary behavior.
2. Re-test full asset matrix.
3. Verify:
   - geosets render
   - sequence names are sane
   - no regressions in known-good MDX
4. Record final mapping in versioned contract notes.

---

## 11) Practical notes for this repo
- Keep fixes minimal and local to parser/converter hot paths.
- Preserve strict bounds checks while aligning semantics.
- Prefer deterministic branch selection over heuristic fallback when binary evidence is clear.

---

## 12) Deliverables from investigation run
Minimum artifacts before coding:

1. Filled evidence table for each matrix asset
2. One-page summary of confirmed binary rules:
   - routing rule
   - `SEQS` accepted layout(s)
   - `GEOS` payload interpretation rule(s)
   - render suppression gate(s)
3. Patch plan with exact code touchpoints and expected behavior change
