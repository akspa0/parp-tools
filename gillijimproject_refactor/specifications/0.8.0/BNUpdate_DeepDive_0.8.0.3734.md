# BNUpdate Deep-Dive — WoW Era 0.8.0.3734

## Summary
`BNUpdate.exe` in this build is a **real patch applicator**, not just a launcher/downloader shim. It processes patch command/list files from patch archives, applies file-level transformations (including a clear `BSDIFF40` path), stages outputs, and finalizes files/patch MPQ artifacts.

- **Binary identified**: `Blizzard BNUpdate v2.89 compiled on Jul  1 2004`
- **Key source paths embedded**: `G:\Projects\Bliz\Tools\Patcher\bnupdate\*.cpp`

## High-Confidence Findings

1. **BNUpdate performs patch application itself**
   - Evidence strings:
     - `ERROR: unable to apply patch to file '%s'`
     - `ERROR: patch file '%s' in archive '%s' is corrupt`
     - `patch.lst`, `patch.cmd`, `patch.txt`, `prepatch.log`, `bnupdate.log`
   - Patch results are explicitly emitted (`Patch successful` / `Patch failed`).

2. **It includes a `BSDIFF40` algorithm path**
   - `BSDIFF40` marker string is present and used in code.
   - Function `FUN_004088f0` checks for `BSDIFF40` and executes a bsdiff-style control/diff/extra reconstruction routine.
   - Wrapper `FUN_00408880` routes patch formats into that routine.

3. **Patch format supports multiple modes**
   - `FUN_00408010` dispatches by patch subtype (header fields):
     - direct copy/replace mode,
     - bsdiff path (`FUN_00408880` / `FUN_004088f0`),
     - additional custom delta modes (`FUN_004082e0`, `FUN_004084a0`).
   - This indicates BNUpdate supports more than pure overlay replacement.

4. **Patch execution is script/list-driven from archives**
   - `FUN_00406fb0` orchestrates processing of `patch.lst`, `patch.cmd`, `delete.lst`, `revert.lst`, `hdfiles.lst`.
   - `FUN_00407200` opens archive(s), validates/authenticates, iterates list entries, and runs callback handlers per command.
   - `FUN_00406ed0` tokenizes per-line operations and invokes per-file patch handlers.

5. **Output is staged then committed**
   - Temp archive creation path in `FUN_004034c0`.
   - Final move/cleanup/rename behavior in `FUN_00402f60`.
   - Handles `Data\patch000.mpq` and `Data\patch.mpq`, and writes patch metadata/log files.

---

## Does BNUpdate Download Patches?

### Observed in this binary
- I found no clear `http://` / `ftp://` endpoint strings in BNUpdate.
- No direct "patchdownload" command/event strings in this binary (those appeared in `WoW.exe`, not BNUpdate).
- The visible BNUpdate flow is strongly local/archive-script oriented.

### Interpretation
- For this build, BNUpdate appears primarily to be the **local patch execution engine**.
- Download orchestration may be external (Battle.net agent/client flow) and then BNUpdate is invoked to apply.
- If any network path exists here, it is not obvious in string-level evidence and is not central to the observed patch pipeline.

---

## Practical Conclusion vs Prior MPQ Overlay Question
- `WoW.exe` runtime patch behavior: mostly **overlay precedence** for patch MPQs.
- `BNUpdate.exe` patch creation/apply behavior: **actual file patching logic exists**, including **`BSDIFF40`** and custom delta processing.

So both are true in different stages:
- **Runtime consumption**: overlay semantics.
- **Patch installation**: binary-diff/transform application is present in BNUpdate.

---

## Key Functions (for follow-up reversing)
- `FUN_00406fb0` — top-level patch list/cmd orchestration
- `FUN_00407200` — archive/list interpreter and per-entry dispatch
- `FUN_00406ed0` — per-line command expansion/dispatch
- `FUN_00408010` — patch payload mode dispatcher
- `FUN_00408880` / `FUN_004088f0` — bsdiff path (`BSDIFF40`)
- `FUN_004082e0`, `FUN_004084a0` — custom delta/transform routines
- `FUN_004034c0`, `FUN_00402f60` — staging + commit/finalization

---

## Confidence
- **High**: BNUpdate applies patches and includes BSDIFF40 handling.
- **Medium**: exact semantics of each non-bsdiff delta subtype without additional field annotation.
- **Medium-High**: downloader role is limited/non-primary in this binary based on available evidence.

---

## Appendix — Ghidra Function Rename Map

The following symbols were renamed during this pass to make future reversing/navigation faster.

| Address | New Symbol | Role |
|---|---|---|
| `0x00406FB0` | `BNUpdate_RunPatchPipeline` | Top-level orchestrator for patch list/cmd/revert/delete flow |
| `0x00407200` | `BNUpdate_ProcessArchiveCommandList` | Opens archive(s), validates, executes command-list entries |
| `0x00406ED0` | `BNUpdate_ProcessPatchCommandLines` | Tokenizes patch command lines and dispatches handlers |
| `0x00408010` | `BNUpdate_ApplyPatchPayload` | Main patch payload dispatcher (copy/delta/bsdiff/custom) |
| `0x00408880` | `BNUpdate_ApplyPatchPayload_FormatWrapper` | Wrapper selecting patch payload format path |
| `0x004088F0` | `BNUpdate_ApplyPatchPayload_BSDIFF40` | `BSDIFF40` apply routine |
| `0x004082E0` | `BNUpdate_ApplyPatchPayload_CustomDeltaA` | Custom delta transform routine A |
| `0x004084A0` | `BNUpdate_ApplyPatchPayload_CustomDeltaB` | Custom delta transform routine B |
| `0x004034C0` | `BNUpdate_CreateTemporaryPatchArchive` | Creates temporary archive/staging context |
| `0x00402F60` | `BNUpdate_FinalizePatchedFiles` | Commit/rename/cleanup pass for patched outputs |
| `0x00405E70` | `BNUpdate_ReadPatchVersionFromArchive` | Reads `patch.txt` from archive and extracts version |
| `0x00406BB0` | `BNUpdate_WritePatchTxtSummary` | Writes/updates local `patch.txt` summary |
| `0x004063F0` | `BNUpdate_LoadPatchConfigAndPaths` | Initializes patch paths, launcher, source/dest config |
| `0x004038C0` | `BNUpdate_ReadPatchIniSections` | Reads `Patches`/`SrcData`/`DstData`/`Launcher` entries |
| `0x00403690` | `BNUpdate_OpenPatchMpqAndProcessEmbeddedLists` | Opens patch MPQ and processes embedded list files |
| `0x00406C90` | `BNUpdate_ReportPatchResult` | Emits final success/failure result and summary output |