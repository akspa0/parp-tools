# Draft: wowdev.wiki ADT/v18 Follow-Ups From Alpha Work

Status: draft addendum, not a full rewrite of `ADT/v18`.

## Why This Exists

Recent Alpha stabilization work did not mainly change `ADT/v18`, but it did clarify a few spots where the later page is the right long-term home for terminology that the Alpha page currently under-explains.

## Proposed Follow-Ups

### 1. `predominantTexture` should explicitly mention the Alpha page

The `MCNK` header entry at `0x040` on `ADT/v18` already documents:

> `predominantTexture` / `predTex` / `ReallyLowQualityTextureingMap`

Add a short cross-note:

> The Alpha page's `predTex[8]` field is the earlier member of the same data family.

That would stop Alpha readers from treating `predTex` as isolated or mysterious.

Editorial rule for this case:

> Reuse the existing `ADT/v18` terminology here. This is not a case for introducing a new tooling-driven name.

### 2. Keep the low-detail terrain wording broader than only detail doodads

The current `ADT/v18` note emphasizes detail doodads. That is useful, but for future editors it should stay broad enough to cover terrain-Lod-facing behavior too.

Suggested wording change:

> This field selects the predominant layer per low-detail subcell and is used by low-detail terrain behavior, including detail-doodad selection.

This matches current Alpha-era runtime observations better than a doodad-only description.

### 3. `MDDF` / `MODF` coordinate notes should stay compatible with Alpha-native wording

The `ADT/v18` page already carries the right coordinate-family explanation. When the Alpha page is refreshed, the two pages should not drift on these points:

- `MapOrigin = 32 * TILESIZE`
- file-space `x/z` are inverted into world planar coordinates
- both doodads and WMOs carry the 180-degree yaw offset in the native placement transform

The pages do not need identical prose, but they should not imply different native transforms.

### 4. `MCRF` culling notes should remain prominent

The `ADT/v18` page already says that objects missing from `MCRF` are not drawn. That same rule mattered heavily in Alpha work, so this is a place to preserve strong wording, not weaken it.

Suggested editorial stance:

> Keep the visibility and culling warning direct. Tool authors routinely underestimate how authoritative `MCRF` is for runtime object visibility.

## Repo-Backed Sources For This Draft

- `wow-viewer/docs/architecture/alpha-wdt-ghidra-research-2026-05-10.md`
- `wow-viewer/docs/architecture/alpha-placement-coordinate-transforms-2026-05-09.md`
