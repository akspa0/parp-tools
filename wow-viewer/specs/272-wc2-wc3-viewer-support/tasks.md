# WC3 File Browser Implementation

**Operator-directed implementation, 2026-10-14:** implement the planned file-only slice now; source/pixel/UI proof stays open rather than blocking all code behind further planning. Existing MPQ readers and world rendering remain untouched.

## Phase A — Bounded Working File Browser

- [x] T001 Shared loose/MPQ source, image/text preview, and safe selected-file extraction; focused tests. Code/test receipt: [implementation evidence](evidence/receipt-wc3-implementation.md); real beta compatibility is not closed.
- [ ] T002 Integrate the owned WC3 service into the existing File Browser with root/listfile selection, image/text preview, extraction, and lifetime cleanup. **Code landed and compiles; operator UI/lifetime proof owed.**
- [x] T003 Build the viewer and rerun focused source/preview/extraction plus existing MPQ regressions; record exact receipts. [Receipt](evidence/receipt-wc3-implementation.md): solution build 0 errors, 77 focused tests passed; not runtime proof.
- [ ] T004 Operator opens the supplied beta root, checks source image/text and extraction, then confirms existing-world isolation (SC-001–SC-005; MPQ proof requires a real archive/listfile).
- [x] T005 Operator-reported BLP0 failure: resolve shared-header/external mip data and verify real Crown1 mip decoding; [receipt](evidence/receipt-beta-blp0-and-mdx-2026-10-15.md), real 128×32 and 16×4 outputs and 86 focused tests passing.
- [ ] T006 Operator-requested MDX loading: existing reader/renderer routed into isolated file preview; WC3 framing/texture adapters implemented and real model parses (6 geosets / 90 vertices / 3 sequences). **Operator visual/model appearance proof still owed**; [receipt](evidence/receipt-beta-blp0-and-mdx-2026-10-15.md).

Code/test completion is not real-beta visual signoff. [Receipt](evidence/receipt-wc3-implementation.md).
