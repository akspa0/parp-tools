# Task 3 - 1500 invisible render: first fail branch

## First static draw-suppression gate
- Function: `FUN_004349b0` (scene submission path)
- Branch site: start predicate in function body
  - `if (*(char *)(param_4 + 3 + param_3[6] * 0x14) != '\0') { ... submit draw ... }`
- Effect:
  - **true**: continues through material checks and emits draw entries.
  - **false**: returns without draw submission (silent suppress).

## Why this is the first suppress gate
- It executes before downstream geoset-batch build and before render-queue insertion.
- The branch guards the entire add-to-scene block; no alternative path submits that geoset when false.

## Predicate and operand domains
- Left operand: visibility/enable byte at `param_4 + 3 + geosetId*0x14`.
- Right operand: literal zero.
- Polarity: reject/suppress when left operand `== 0`.

## Relation to v1500 symptom
- No explicit `version==1500` branch was found nearby; parser/layout drift can still zero/offset this operand provenance, triggering suppress.
- This is the first concrete static predicate that maps directly to “loads but does not render”.
