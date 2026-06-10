# WDT Chunks — Build 0.5.3.3368

This folder documents WDT chunks for **0.5.3.3368** in the same style as 0.7.0, with 0.5.3 monolithic caveats.

## Chunks documented
- `MVER.md`
- `MPHD.md`
- `MAIN.md`
- `MDNM.md`
- `MONM.md`
- `MARE.md`
- `MAOF.md`
- `WDT500.md`
- `MONOLITHIC_LAYOUT.md`

## Notes
- Root-level WDT parser assertions are strongly evidenced in the binary.
- 0.5.3 root parsing transitions into ADT-like chunk handling (`MHDR`, `MCIN`, `MCNK`, etc.), unlike later fully split layouts.
- Use `MONOLITHIC_LAYOUT.md` as the control-flow anchor when implementing parser adapters.
