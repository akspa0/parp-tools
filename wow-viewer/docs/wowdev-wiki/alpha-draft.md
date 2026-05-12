# Draft: wowdev.wiki Alpha Page Refresh

Status: concise handoff draft for the existing wowdev Alpha page.

## Proposed Changes

1. Add a top-level WDT layout note:

> For Alpha 0.5.3, top-level WDT chunks are contiguous. Do not odd-byte pad `MDNM`, `MONM`, or other top-level chunks between headers.

1. Make `MAIN` ordering explicit:

> The 64x64 `MAIN` table should state its cell ordering explicitly. Current shared Alpha tooling uses row-major tile indexing, and the embedded 16x16 chunk tables are likewise handled row-major.

1. Replace the current `MCRF` description with:

> `MCRF` is one contiguous array. The first `nDoodadRefs` entries index the ADT-part `MDDF` table, and the next `nMapObjRefs` entries index the ADT-part `MODF` table. Name resolution still comes through `MDNM` and `MONM` via the placement tables.

1. Add the verified Alpha placement transform under `MDDF` and `MODF`:

```text
world.x = MapOrigin - filePos.z
world.y = MapOrigin - filePos.x
world.z = filePos.y

rot.x = fileRot.z
rot.y = fileRot.x
rot.z = fileRot.y + 180 degrees
```

Native order:

```text
Translate -> RotateZ -> RotateY -> RotateX
```

1. Tighten the `MCLQ` note:

> In 0.5.3, the payload is a fixed-size `0x324`-byte block per chunk when present.

1. Expand the `predTex` note conservatively:

> `predTex[8]` is the Alpha-era predominant-texture / low-detail terrain map, matching the same data family later documented on `ADT/v18` as `predominantTexture` / `predTex` / `ReallyLowQualityTextureingMap`.

Optional follow-up sentence:

> Runtime observation suggests this data participates in low-detail terrain or texture selection as full terrain detail fades in, but the exact native distance constant still needs a cleaner symbol-level note.

1. Cross-link `noEffectDoodad`:

> `noEffectDoodad[8]` is the Alpha-era bitset matching the later `noEffectDoodad` / `MCDD` family and should be documented as an explicit behavior-control field.

## Alignment Notes

- Keep `predTex` terminology aligned with `ADT/v18` `predominantTexture`.
- Call out that Alpha `MCVT` heights are absolute.
- Keep Alpha `MCLQ` specifics local to the Alpha page.

## Repo-Backed Sources

- `wow-viewer/docs/architecture/alpha-wdt-ghidra-research-2026-05-10.md`
- `wow-viewer/docs/architecture/alpha-placement-coordinate-transforms-2026-05-09.md`
- `wow-viewer/docs/architecture/alpha-mcnk-flags-and-metadata-plan.md`
