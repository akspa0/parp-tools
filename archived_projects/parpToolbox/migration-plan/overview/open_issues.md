# Open Technical Issues / TODOs

A running list of verifiable tasks or unknowns. Keep concise. Strike through when resolved.

- [ ] Verify **MSVI** index validity for tiles with >65k vertices (possible 32→16-bit overflow).
- [ ] Extend exporter to include **MSVT** geometry for full render model.
- [ ] Confirm whether `MSLK.LinkPadWord` ever differs from `0xFFFF` in real data sets.
- [ ] Validate `MOVI` index parsing in `_FullV14Converter` – currently unverified.
- [ ] Update `SixLabors.ImageSharp` to ≥ 3.2 (security CVE).
- [ ] Build error in `WmoObjExporter.cs` (misplaced brace at line 68) – fix & compile.

> List intentionally short; add only confirmed actionable items.
