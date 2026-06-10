# Task 7 - Patch handoff (minimal fixes only)

## Fix 1 - 1400 SEQS names

### Target function (equivalent in C# codebase)
- SEQS chunk parser corresponding to binary `FUN_007a9c70`.

### Before (current behavior)
```text
numSeq = ReadU32()
require(sectionBytes - 4 == numSeq * 0x8C)
for each seq:
  name = ReadBytes(0x50)
  read fixed fields at hardcoded offsets
```

### After (minimal)
```text
numSeq = ReadU32()
version = Model.Version
stride, nameSize, fieldLayout = ResolveSeqLayout(version)
require(sectionBytes - 4 == numSeq * stride)
for each seq:
  name = ReadBytes(nameSize)
  read remaining fields using layout table for version
```

### Why 1300 stays unaffected
- `ResolveSeqLayout(1300)` returns existing `stride=0x8C`, `nameSize=0x50`, current offsets.

### Regression tests
- Parse known-good v1300 sample; assert sequence names unchanged.
- Parse failing v1400 sample; assert non-garbage names and stable record count.

---

## Fix 2 - 1500 invisible render

### Target function (equivalent in C# codebase)
- Data flow that populates per-geoset visibility state consumed by scene gate equivalent to `FUN_004349b0`.

### Before (current behavior)
```text
if (visibilityState[geosetId].enabledByte == 0)
    skip draw submission
```

### After (minimal)
```text
Ensure geosetId mapping and visibility-state record stride/offset are version-correct for 1500
so enabledByte is sourced from the intended field.
Do not bypass gate; fix operand provenance.
```

### Why 1300 stays unaffected
- Keep existing mapping for 1300 path unchanged; add conditional only for 1500 layout delta.

### Regression tests
- v1500 sample: load+animate+nonzero submitted draw geosets.
- v1300 sample: submitted geoset count unchanged.

---

## Additional validation to run
- Confirm version check still rejects `>1500`.
- Confirm SEQS section integrity checks remain hard-fail.
- Confirm no renderer workaround added (parser/converter parity only).
