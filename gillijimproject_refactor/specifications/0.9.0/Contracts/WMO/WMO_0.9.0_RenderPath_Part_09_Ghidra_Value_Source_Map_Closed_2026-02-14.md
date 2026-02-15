# WMO 0.9.0 Render Path — Part 09 (Ghidra Value Source Map, Closed)

Date: 2026-02-14  
Method: static-only (Ghidra disassembly + decompile)

## Target branches

### Gate A — Group color/light linkage mismatch gate
- **Function:** `FUN_006cee60`
- **Function address:** `0x006cee60`
- **Disassembly window:** `0x006cef02..0x006cef28`
- **Literal binding candidate:** `mapObjDefGroup->lightLinkList.Head() == 0` (`0x008942e8`)

### Gate B — Liquid layout/index-domain mismatch gate
- **Function:** `FUN_006dedc0`
- **Function address:** `0x006dedc0`
- **Disassembly window:** `0x006def00..0x006def30`
- **Literal binding candidate:** `(idxBase[i] - vtxSub) < (uint) (group->liquidVerts.x * group->liquidVerts.y)` (`0x008958f8`)

---

## Operand provenance (static)

### Gate A — Group color/light linkage mismatch (`FUN_006cee60`)

#### Operand A1 — list-head tagged pointer (light-link domain)
- **Source load:** `0x006cef02: MOV EAX, dword ptr [EDI + 0xb8]`
- **Base provenance:** `EDI` is `param_1` (`mapObjDefGroup*`) set at function entry (`0x006cee62`).
- **Meaning:** Reads `mapObjDefGroup->lightLinkList.Head()` tagged-pointer value.

#### Operand B1 — required-empty sentinel / tag semantics
- **Source checks:**
  - `0x006cef08: TEST AL, 0x1` (tag bit)
  - `0x006cef0c: TEST EAX, EAX` (null check)
- **Branch polarity:**
  - If tagged (`AL & 1`) **or** zero (`EAX==0`) ⇒ allowed path (`JNZ/JZ` to `0x006cef2d`)
  - Else ⇒ mismatch/assert path.

#### Compare/log usage
- Assert/log call sequence:
  - `0x006cef10: PUSH 0x1`
  - `0x006cef12: PUSH 0x0`
  - `0x006cef14: PUSH 0x8942e8` (literal address)
  - `0x006cef19: PUSH 0x1ea`
  - `0x006cef1e: PUSH 0x89408c`
  - `0x006cef23: PUSH 0x85100000`
  - `0x006cef28: CALL 0x006685d0`

#### Gate tuple (required output format)
- `(0x006cef02..0x006cef28, [EDI+0xb8], (tag==1 || ptr==0), mismatch path calls 0x006685d0, literal 0x008942e8)`

---

### Gate B — Liquid layout/index-domain mismatch (`FUN_006dedc0`)

#### Operand A2 — computed liquid index (`idxBase[i] - vtxSub`)
- **Source load/compute:**
  - `0x006def00: MOVZX EDX, word ptr [ESI + EDI*2]` (current `idxBase[i]`)
  - `0x006def04: SUB EDX, dword ptr [EBP + 0x10]` (`vtxSub` / base index offset)
- **Base provenance:**
  - `ESI` initialized from `param_2` (index buffer out) at `0x006dede6` / restored at `0x006deeec`
  - Loop index `EDI` is running index-element counter.

#### Operand B2 — expected liquid vertex-domain size (`group->liquidVerts.x * group->liquidVerts.y`)
- **Source load/compute:**
  - `0x006def07: MOV ECX, dword ptr [EBX + 0xf0]` (`liquidVerts.x`)
  - `0x006def0d: IMUL ECX, dword ptr [EBX + 0xf4]` (`liquidVerts.y`)
- **Base provenance:**
  - `EBX` is group pointer (`param_1`) established at `0x006dedcc`.

#### Compare/log usage
- Compare and branch:
  - `0x006def14: CMP EDX, ECX`
  - `0x006def16: JC 0x006def38` (valid if `EDX < ECX`)
- Mismatch/assert path:
  - `0x006def18: PUSH 0x1`
  - `0x006def1a: PUSH 0x0`
  - `0x006def1c: PUSH 0x8958f8` (literal address)
  - `0x006def21: PUSH 0x59c`
  - `0x006def26: PUSH 0x895778`
  - `0x006def2b: PUSH 0x85100000`
  - `0x006def30: CALL 0x006685d0`

#### Gate tuple (required output format)
- `(0x006def14..0x006def30, idxBase[i]-vtxSub from [ESI+EDI*2], liquidVerts.x*liquidVerts.y from [EBX+0xf0]*[EBX+0xf4], valid-branch=JC, literal 0x008958f8)`

---

## Closure status
1. Every target gate compare site has static source chain: **Yes**.
2. Mismatch-branch literal binding via `PUSH literal` captured: **Yes**.
3. Runtime correlation (Part 08) still required for full parser↔renderer contract closure: **Pending runtime pass**.
