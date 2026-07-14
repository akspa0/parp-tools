# Quickstart: Legacy M2 rendering investigation (0.11 – 2.4.3)

How to run the investigation and validate each phase. All C# builds from repo root; the viewer is the
primary validation surface. **The USER runs the viewer / debugger steps**; agent prepares reader changes
and commands.

## Ground truth

- Staged clients: `output/tmp/wowarchive-clients/` (constitution VI — never `H:\CLIENTS`).
- Reference for M2 knowledge (read-only, never edit): `gillijimproject_refactor`.
- Format oracle for documented versions: wowdev.wiki (M2, M2/.skin) + an open reference renderer.

## 0. Confirm the version of a staged model (before any code change)

Identify the M2 format version each staged client emits — this is the discriminator (research Decision 1).
Read the `uint32` at offset `0x04` of any `.m2` from the client. Record it against the client build so the
reader's version branch targets the right value. (A tiny inspect command or a hex view of the first 8
bytes suffices; the magic `MD20` is at 0x00, version at 0x04.)

## 1. P1 — documented versions (2.4.3, 1.12.1)

```text
# Build the viewer (from repo root)
dotnet build wow-viewer/WowViewer.slnx -c Release

# Load a known 2.4.3 model in the viewer (USER runs), e.g. a creature or simple prop,
# and confirm CURRENT behavior: empty bounding box (the baseline this phase fixes).
```

Then implement the reader steps (plan.md Phase 1, steps 2–10) in
[M2ModelReader.cs](../../src/core/WowViewer.Core.IO/M2/M2ModelReader.cs):

- Verify header offsets for the version against a hex dump of the loaded model (research U1).
- Read the embedded skin profile; parse submeshes + triangle indices + texture units.
- Feed them to the render path.

**Validate**: the same 2.4.3 model now renders textured mesh, and it visually matches a reference
renderer's output of the same file. Repeat for 1.12.1 (version 256). Record confirmed offsets/structs in
[contracts/m2-format-profile.md](contracts/m2-format-profile.md). Gate: SC-001, SC-002.

## 2. P2 — mid-range (2.0.0 alpha, 2.1, 2.2, 2.3)

```text
# Load representative models from each mid-range staged client (USER runs).
```

- Confirm they render via the P1 path; where one doesn't, isolate the offset/struct delta and add a
  version-specific branch.
- Record each version's profile (or delta) in the contract. Gate: SC-003.

## 3. P3 — early alphas (1.0.0, 0.12, 0.11) via x64dbg

Only where documentation runs out. Per version with a staged client:

```text
# 1. Confirm a staged client exists for the version; if not → mark "blocked", skip.
# 2. Try the current reader; capture exactly where parsing goes wrong.
# 3. Dynamic trace (USER drives x64dbg; agent can drive via the MCP bridge):
#    - launch x64dbg on the alpha client exe
#    - mcp__x64dbg__start_session → mcp__x64dbg__connect
#    - set a breakpoint on the M2 load routine; step and read memory/registers to
#      observe how the client walks the M2 header to reach geometry
#    - recover the true field offsets / skin-profile struct
```

- Record the recovered layout + the trace as evidence in the contract (FR-007).
- Add the version branch; the model renders or the residual unknown is documented. Gate: SC-004, SC-005.

## x64dbg MCP quick reference

Configured in repo-root `.mcp.json` (points at `C:\x64dbg`); confirmed responding (`list_sessions`
returns cleanly with no session attached). To use: launch x64dbg with the target client, then
`mcp__x64dbg__start_session` / `connect`, then `set_breakpoint`, `go`, `read_memory`, `get_all_registers`,
`step_into`/`step_over`. Ghidra is **not installed** — static disassembly is a separate, user-approved
setup step if dynamic tracing proves insufficient.

## 4. Consolidate

- Every in-scope version has a contract entry (SC-005).
- Re-check a spread of WotLK+ (≥ 264) models still render — no regression (SC-006, FR-009).
- Update `memory-bank/activeContext.md` + `progress.md` with format findings and reader state.

## Definition of done (per phase)

A phase is done only when its versions **render mesh + materials against real staged clients** and the
evidence is recorded — not when the code compiles (constitution: One Phase at a Time).
