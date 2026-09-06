# Progress — wow-viewer

Last updated: 2026-09-06

## 2026-09-06 — Context and documentation reduction pass

- Created [docs/README.md](../docs/README.md) as the canonical documentation router; legacy
  `DOCUMENTATION-STATUS.md` and `PLANS-OVERVIEW.md` are now redirects.
- Preserved, rather than deleted, high-confidence historical material under `docs/archive/`:
  the 49-file 2026 game-viewer plan pack, the consumed M2 investigation packet, the stale
  2026-08-01 spec audit, and the intact legacy MdxViewer tarball.
- Moved only clearly superseded specs 080, 145, and 195 to
  `specs/archived/superseded/` with successor pointers. They are not asserted complete.
- Added [Spec 228](../specs/228-source-decomposition/plan.md) planning artifacts. The first source
  extraction remains blocked by the Spec 227 T004 UI-authority gate.
- The full Spec 224 receipt audit is still open; this pass recorded its archive and routing work
  without checking its cleanup tasks or silently closing Gate 1.

## 2026-09-06 — Current implementation handoff

- Spec 227 T001/T002 source documentation is receipted. The next task is the operator-owned T003
  screenshot/input matrix, then T004 gate.
- Spec 223's fog/WMO/capture acceptance retest remains separately operator-owned.

Earlier same-day narrative was preserved in
[memory-bank/archive](archive/README.md), not discarded.
