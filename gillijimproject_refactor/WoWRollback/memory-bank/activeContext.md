# Active Context

- **Focus**: Establish bidirectional MCAL/MCLY handling between Alpha ↔ LK, including terrain holes and flag parity.
- **Current Plan**:
  1. Unify alpha decoding using a shared MCAL reader that mirrors Noggit's handling of compressed, big, and small alpha maps (respecting `do_not_fix_alpha_map`).
  2. Implement the reverse encoder to pack Alpha 64×64 masks into LK formats, with regression tests against Noggit output.
  3. Achieve MCNK flag parity by parsing/emitting hole maps (`high_res_holes`, low-res bitmaps) and related flags in both builders.
  4. Extend diagnostics/CLI utilities to export and import alpha masks (PNG/RAW) for cross-tool validation.
  5. Build regression coverage for compressed alpha, big alpha, small alpha, hole maps, and `do_not_fix_alpha_map` scenarios.
- **Notes**:
  - Reuse `lib/noggit-red/src/noggit/Alphamap.cpp` logic as the reference implementation.
  - Ensure `McalDumpUtility` (or follow-on tooling) supports quick validation of newly encoded masks.
