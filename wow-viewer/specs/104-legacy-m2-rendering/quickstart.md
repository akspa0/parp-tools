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

## 1. Current P1 — 1.0.0 M2 (`MD20`, version `0x100`)

```text
# Build the viewer (from repo root)
dotnet build wow-viewer/WowViewer.slnx -c Release

# USER runs: load a known staged 1.0.0 `.m2` model in the viewer.
# It must be treated as M2, not redirected to .mdx/.mdl. Record the exact model path and outcome.
```

The 1.0.0 branch lives in
[`M2Era100ModelReader.cs`](../../src/core/WowViewer.Core.IO/M2Era100/M2Era100ModelReader.cs),
with classification in
[`M2ModelReaderDispatcher.cs`](../../src/core/WowViewer.Core.IO/M2Chunked/M2ModelReaderDispatcher.cs).

- Confirm the dispatcher selects the era-100 branch, not the 1.12.1 layout.
- Confirm the info panel identifies `Renderer: M2Renderer`; seeing `MdxRenderer` is a failure of
  this phase, even if triangles appear.
- Confirm the viewer reports a version-specific M2 reader failure if parsing fails; it must not
  advise MDX/MDL or silently parse through another layout.
- Validate mesh and material output against the same staged source model.

**Validate**: the 1.0.0 model renders visible mesh/materials, and record the staged client/model
path plus confirmed layout in [contracts/m2-format-profile.md](contracts/m2-format-profile.md).
This is not signoff for 1.12.1 merely because it shares `0x100`.

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
