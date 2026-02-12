# MPQ Patching Behavior — WoW 0.7.0.3694

## Why this matters
Your custom MPQ reader currently opens single archives but does not implement WoW’s **multi-archive overlay** behavior. In 0.7, normal game patching is archive precedence (override-on-first-hit), not an in-client binary-delta transform step.

## Evidence (0.7.0.3694 client)

- Archive bootstrap: `FUN_00403550` (`Client.cpp` assertions in same function)
- Dynamic patch path formatting: `FUN_00403810` (`"Data\\%s"`)
- Archive open wrapper: `FUN_0064e690` -> `FUN_0065eca0` -> `FUN_0065ecc0`
- Mounted-archive insertion (priority-sorted): `FUN_0065f250`
- File lookup across mounted archives: `FUN_0065df70`
- Archive shutdown: `FUN_00403990`
- Separate updater check for `wow-patch.mpq`: `FUN_004b29d0` (not the normal game data mount path)
- `wow-patch.mpq` command script reader: `FUN_004b28c0` (`prepatch.lst`, `extract `, `execute `)
- MPQ file materialization for updater commands: `FUN_004b2820` -> `FUN_0065eb40` -> `FUN_0065eb70`
- Process spawn for updater command execution: `FUN_004b28b0` -> `FUN_0064b020`

---

## 1) How 0.7 mounts archives

### A. Dynamic patch archive list (pre-base)

`FUN_00403550` first collects a dynamic list through callback `FUN_00403810`, which formats candidates as:

`Data\\%s`

The same function references `patch.MPQ` (`0x00847af8`), indicating this pre-base list is where patch archive mounting enters normal startup.

### B. Fixed data archive list (always attempted)

Then it mounts a fixed 10-entry table at `PTR_s_Data_model_MPQ_00846eb0`:

1. `Data\\model.MPQ`
2. `Data\\texture.MPQ`
3. `Data\\terrain.MPQ`
4. `Data\\wmo.MPQ`
5. `Data\\sound.MPQ`
6. `Data\\misc.MPQ`
7. `Data\\interface.MPQ`
8. `Data\\fonts.MPQ`
9. `Data\\speech.MPQ`
10. `Data\\dbc.MPQ`

Missing archives are logged and represented as null entries; startup continues.

---

## 2) Priority model (critical)

Each mounted archive receives a numeric priority in `FUN_00403550` (passed as `param_2` into `FUN_0064e690/FUN_0065eca0`).

`FUN_0065f250` inserts archives into a global list sorted by this value (`archive+0x148`):

- Higher priority appears earlier in the lookup chain.
- For equal priority, newer insertion is placed before existing equal entries.

In practice, archives mounted earlier in startup (dynamic patch list) get higher effective precedence than later base archives.

---

## 3) How file override works

`FUN_0065df70` walks mounted archives in list order and returns the first archive/hash entry containing the requested file.

That means patching semantics are:

**overlay + first-hit wins**

No additional merge step is required for normal data files; whichever mounted archive resolves first supplies bytes.

---

## 4) `patch.MPQ` vs `wow-patch.mpq`

- `patch.MPQ`: part of normal data mount flow (`FUN_00403550` path).
- `wow-patch.mpq`: checked in `FUN_004b29d0` via `FUN_0065eca0(..., priority=100, ...)` in patch/update service logic; this is not the standard base archive loop.

Treat these as separate roles in tooling:

- data override (`patch.MPQ`)
- patch/update package workflow (`wow-patch.mpq`)

---

## 5) Does 0.7 use BSDIFF inside patch MPQs?

### Result

For the code paths analyzed in this client build, there is **no evidence** that game data patching uses BSDIFF-style binary deltas.

### Evidence

- No string hits for `bsdiff`, `BSDIFF`, or `BSDIFF40` in the binary string table.
- No string hits for common patch-delta markers (`PTCH`, `RTPatch`) in this build.
- Normal game file open path (`FUN_0065fbf0` -> `FUN_0065df70` -> `FUN_00660040`) is archive lookup + MPQ block read/decompress.
- `wow-patch.mpq` flow is command-driven (`prepatch.lst`) with `extract` and `execute`; no delta op handler is visible in this path.

Interpretation: in 0.7, patched game files are primarily delivered as replacement entries in higher-priority archives (`patch.MPQ`), not as byte-wise transforms of base files during read.

---

## 6) `wow-patch.mpq` updater workflow (intricate behavior)

`FUN_004b29d0` does the following:

1. Opens `wow-patch.mpq` with high priority (`FUN_0065eca0(..., 100, ..., &handle)`).
2. Runs archive verification/status check (`FUN_0065c420(handle, &status)`).
3. If status is acceptable (`status == 0 || status > 4`), it executes `FUN_004b28c0`.
4. On success, calls `thunk_FUN_004130e0()` (transition/exit into patch application path).
5. Otherwise falls back to normal login/game service flow (`FUN_004b2a80(); FUN_00450e70();`).

`FUN_004b28c0` command processing details:

- Reads `prepatch.lst` from the opened archive (`FUN_0065eb40`).
- Tokenizes lines and recognizes command prefixes:
    - `extract ` -> `FUN_004b2820`
    - `execute ` -> `FUN_004b28b0`
- Stops with failure if a command handler fails.

`extract` handler details (`FUN_004b2820`):

- Reads the named file from the MPQ into memory (`FUN_0065eb40`/`FUN_0065eb70`).
- Creates/truncates destination output via file API wrapper (`FUN_00450910`).
- Writes bytes directly (`FUN_00450aa0` -> `WriteFile`).

`execute` handler details (`FUN_004b28b0`):

- Launches an external process (`FUN_0064b020` -> `CreateProcess` path).

This is a scripted updater package model, not a per-file read-time binary patch transform.

---

## 7) Implementation spec for your MPQ reader

## Required behavior

1. Build a **mount list**, not a single archive.
2. Mount dynamic patch archives before base archives.
3. Keep a priority-ordered archive chain.
4. On file open, search chain head->tail and return first match.
5. Preserve case-insensitive normalized paths (Storm-style behavior).
6. Allow missing archives without aborting entire load.

## Suggested API

```text
MountArchive(path, priority)
OpenFile(path) -> (archiveId, fileHandle)
ReadFile(handle) -> bytes
```

## Suggested algorithm

```text
mounts = []

# Dynamic patch phase (e.g. patch.MPQ discovered candidates)
for p in dynamicPatchCandidates:
    MountArchive(p, nextHighPriority())

# Fixed base phase
for p in [model, texture, terrain, wmo, sound, misc, interface, fonts, speech, dbc]:
    try MountArchive(p, nextPriority())
    except NotFound: log and continue

def OpenFile(virtualPath):
    for archive in mounts_sorted_high_to_low:
        if archive.contains(virtualPath):
            return archive.open(virtualPath)
    raise FileNotFound
```

---

## 8) What is still unknown vs what is confirmed

### Confirmed
- Priority-sorted global archive chain exists.
- Lookup stops at first match.
- `patch.MPQ` is wired into normal mount flow.
- `wow-patch.mpq` is used by patch service path.
- `wow-patch.mpq` patch flow uses `prepatch.lst` commands `extract` / `execute`.
- No BSDIFF/PTCH/RTPatch indicators were found in analyzed 0.7 paths.

### Not fully resolved yet
- Exact wildcard/pattern source used to build all dynamic patch candidates before callback `FUN_00403810` (the callback behavior is confirmed, the scanner inputs need one more pass).

This unknown does **not** block implementing compatible overlay semantics; it only affects auto-discovery breadth.

---

## 9) Confidence boundaries (important)

### What is proven from WoW.exe 0.7.0.3694

- Runtime game asset resolution is MPQ overlay by archive precedence (`FUN_0065df70` first-hit behavior).
- The client patch-service path for `wow-patch.mpq` is script-driven (`prepatch.lst`) with only `extract` and `execute` command handlers in `FUN_004b28c0`.
- No in-client BSDIFF marker/signature strings were found in this build (`bsdiff`, `BSDIFF40`, `PTCH`, `RTPatch`).

### What is not proven yet

- Whether an **external executable** launched via `execute` performs its own binary-delta patching after extraction.

This distinction matters: the client itself can still be overlay-only even if a separate patcher EXE uses deltas offline.

### How to close the remaining uncertainty

1. Extract `prepatch.lst` from a real `wow-patch.mpq`.
2. Enumerate every `execute` target in that script.
3. Reverse those target binaries (or scan for `BSDIFF40`/delta magic).
4. Confirm whether they rewrite base files using deltas or simply replace files.

Until those binaries are inspected, the strongest accurate statement is:

> WoW.exe 0.7 runtime patch resolution is overlay-first; any delta patching, if present, would be in external tools invoked by `prepatch.lst` commands.

---

## 10) Practical recommendation for your reader now

Implement patch capability in two stages:

1. **Core compatibility now**: manual mount list with explicit `patch.MPQ` + base MPQs using first-hit-wins.
2. **Discovery parity later**: replicate client’s dynamic patch candidate discovery once scanner-input function is fully mapped.

This yields correct patched-file behavior immediately for nearly all workflows.

If you also want launcher/updater parity, add optional support for:

- Opening `wow-patch.mpq`
- Reading `prepatch.lst`
- Executing `extract` commands to materialize files
- (Careful/sandboxed) handling of `execute` commands