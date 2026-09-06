# Session A - Fresh Runtime Investigation (2026-04-01)

## Intent

This packet is a clean-room investigation log for A/B comparison.
It intentionally starts from live observations in this session and does not rely on prior-session conclusions.

## Scope

- Target binary: Win32 WoW.exe 3.3.5.12340
- Static analysis tool: Ghidra (offline decompilation)
- Runtime analysis tool: x64dbg via x64dbg-mcp
- Renderer focus: M2 load/profile/combiner path with emphasis on invisible or malformed world objects
- Location used for runtime: Stormwind Harbor (chosen for high density of later-branch M2 content)

## Outputs in this folder

- 01-runtime-log.md: chronological live actions and outcomes
- 02-win32-m2-anchor-map.md: function anchors and breakpoint addresses
- 03-console-and-render-controls.md: discovered command/cvar controls relevant to rendering debug
- 04-next-steps.md: immediate execution plan from current process state

## Evidence policy

- Each claim must have either:
  - a concrete decompilation anchor, or
  - a concrete runtime capture point (breakpoint hit/register/stack/disassembly)
- Build success alone is not treated as rendering proof.
