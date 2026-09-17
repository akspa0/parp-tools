# Research: Format Conformance Audit (wowdev.wiki × current readers)

**Spec**: [spec.md](spec.md) · **Date**: 2026-09-17 · **Build measured**: `wow_classic_beta` 1.60.1.69876 (local CASC + CDN fill)

Every row below is either **MEASURED** (a count from real client data, with the command that produced it)
or **CODE** (a fact about this repository, with the file). Wiki statements are quoted as claims, not facts:
the wiki is incomplete and has been wrong for this project before (MH2O "≥42" threshold, MCNR axis order).

## R1. Sources

| Source | What it is | License | Use |
|---|---|---|---|
| wowdev.wiki `WMO`, `ADT/v18`, `M2`, `BLP`, `WDT` | Community format documentation | CC-BY-SA (wiki) | Field layouts and chunk inventories to check against data |
| [Marlamin/WoWFormatLib](https://github.com/Marlamin/WoWFormatLib) | C# readers: ADT, LODADT, ANIM, BLS, GFAT, M2, M3, MDX, SKEL, SKIN, TEX, WDB, WDL, WDT, WMO, WWF | **No license file** in the repository (GitHub reports none) | Read for behaviour only. Do not copy code unless a license is added or permission is given |
| [Marlamin/wow.tools.local](https://github.com/Marlamin/wow.tools.local) (WTL) | Local wow.tools: CASC browsing, DB2, listfile, naming, map/model controllers. Submodules TACTSharp, DBCD, WoWFormatLib, WoWNamingLib | MIT | Behavioural reference for CASC product handling, listfile/naming, file linking; model rendering is WebWowViewerCpp (Emscripten), not C# |
| [wowdev/BLPSharp](https://github.com/wowdev/BLPSharp) | Successor of SereniaBLPLib (the copy vendored here), uses TinyBCSharp | MIT | Candidate replacement for BC5 and other formats (see R5) |
| TACTSharp, DBCD, WoWDBDefs | Already vendored/used here | MIT | — |

**CODE**: WoWFormatLib `WMOReader.LoadWMO(stream, lod)` slices GFID as `nGroups * lodLevel` and falls back to a lower LOD when an id is 0 — the same model adopted in `WmoV17ToV14Converter.ReadGroupFileDataIds` (commit 251026d7).
**CODE**: WoWFormatLib `M2Reader` handles chunks MD21, AFID, BFID, SFID, PFID, SKID, TXID, RPID, GPID, PCOL and skips TXAC, EXPT, EXP2, PABC, PADC, PEDC, PSBC, PGD1, WFV1–3, LDV1, PFDC, EDGF, DETL, NERF, DBOC, AFRA, DPIV, TEXL.

## R2. WMO

| Item | Wiki claim | State here | Evidence |
|---|---|---|---|
| MOBA `material_id_large` | uint16 at batch offset 0x0A, used when flag 0x2 (≥Legion) | **Fixed** 251026d7 | MEASURED: 252,161 / 393,397 batches set 0x2; 242,028 differ from the 8-bit id; max id 473 (`inspect casc wmo-survey`) |
| GFID per LOD | `nGroups × (numLod or 3)` ids when MOHD flag lod | **Fixed** 251026d7 (LOD 0 only) | MEASURED: ND_Dalaran 91 groups / 364 ids; 11DL_Dalaran 126 / 504; LOD 0 has no zero ids; survey `notLocal` 1,667 → 0 after the fix |
| Groups without MOBA / MOPL-only | (not described) | **Fixed** (parse, draw nothing) | MEASURED: 275 of 9,860 roots failed before, 0 after |
| MOMT shader 23 | "UnkDFShader… can use additional texture file IDs from color_2, flags_2 and runTimeData" | **Partial**: base texture = texture_2 | MEASURED: 22,380 shader-23 materials; 10,785 with texture_1 = 0; texture_1 when set is `PRETTYCOLORS*.BLP`; texture_2/3/color_2/flags_2 decode as colour maps (Orgrimmar2FrontGate, 10DU_HallUldamanUprez_Main01). **Blend rule unknown** |
| Shader enum 0–22 | two-layer, env, emissive, parallax, … | **Missing**: renderer draws texture_1 (base) only for every shader | CODE: `WmoRenderer` single sampler |
| MOTV ×2/×3, MOCV ×2 | TVERTS2 0x2000000, TVERTS3 0x40000000, CVERTS2 0x1000000; "only the alpha values from [MOCV] are used to blend the textures" | **Missing**: first set only | CODE: `WmoV17ToV14Converter` keeps first MOTV/MOCV |
| MOHD flags | unified render path, liquid type DBC id, do-not-fix vertex colour alpha, lod | **Partial**: lod now used; others unread | CODE |
| MGI2 | `flags2, lodIndex` per group; overrides LOD loading | **Unread** | MEASURED: the 8-byte reading yields implausible lodIndex values (e.g. 11468802) — the struct as documented does not fit this build |
| Split groups (≥9.2) | MOGP parent/child indices, child groups have no portals | **Unread** | CODE: header bytes 0x40–0x43 ignored |
| MOMX | "seems to be 0x10… just a guess" | **Unread** | MEASURED: 2,641 roots; exactly 16 bytes per MOMT material in all of them |
| MAVG/MAVD, MNLD, MFED, MBVD, MDDI, MOPE, MOLV | ambient volumes, new lights, fog extra, … | **Skipped** | CODE: known-and-skipped list |
| MOGX + MOQG, MOBS, MDAL | query face start / ground type, collision batches, detail doodads | **Skipped** | MEASURED: present in survey layouts |
| MPY2, MOVX | uint16 material per face; 32-bit indices | **Partial**: down-converted to MOPY/MOVI; material > 0xFE lost in the face fallback, MOVX only when indices fit 16 bits | CODE |

## R3. ADT (split, v18 + FileDataID era)

| Item | Wiki claim | State here | Evidence |
|---|---|---|---|
| MCNK `high_res_holes` (flag 0x10000) | 64-bit hole map replaces the 16-bit one | **Missing**: only the uint16 at 0x3C is read | CODE: `Lk/Mcnk.cs` `Holes = ToUInt16(0x3C)` — **unmeasured** how many chunks set it |
| nLayers | "max 4 (pre-Midnight), then 8"; MCMT "Midnight (12.0.0.63534) limits to 4 layers only" | **Fixed** to 8 in the tile renderer | MEASURED: Azeroth 1.60.1 has 2,176 / 295 / 14 / 1 chunks with 5 / 6 / 7 / 8 layers. The MCMT note conflicts with this data |
| MCLY flags | animation rotation/speed/enabled, overbright, use_cube_map_reflection, 0x800/0x1000 texture_scale | **Unused** in rendering | CODE |
| MTXP / MHID (`_h` height blending) | heightScale/heightOffset, `_h.blp` alpha drives blend | **Parsed, not rendered** | CODE: `MopAdtChunkParser` parses; renderer ignores |
| MTXF texture_scale | `1 << texture_scale` | **Unused** in rendering | CODE |
| MDDF/MODF scale, MODF 0x80 + MWDR/MWDS | uint16 scale/1024; doodad sets from MWDS | **Scale used**, MWDR/MWDS **unread** | CODE: `StandardTerrainAdapter` `scale / 1024f`; 0 matches for MWDR/MWDS |
| `_lod.adt` (MLHD/MLVH/MLVI/MLLL) | LOD terrain | **Unread** (no reader) | CODE: 0 matches |
| MCLV, MCBB, MCDD, MLMB | light values, blend batches, detail doodad disable, WMO bytes | MCLV/MCBB/MCDD **parsed only**, MLMB **unread** | CODE |

## R4. M2 (chunked MD21)

| Item | Wiki claim | State here | Evidence |
|---|---|---|---|
| SFID / TXID | skin + texture ids | **Supported** | 3,069 v272 + 30 v274 models build (map-survey) |
| SKID (`.skel`) / BFID (`.bone`) / AFID (`.anim`) | skeleton, bone, external animation file ids | **Missing** | CODE: 0 matches for SKID/BFID/AFID; external `.anim` loading exists only by path (`M2ToMdxConverter.EnumerateExternalAnimationPaths`) |
| LDV1 | LOD skin selection | **Missing** | CODE |
| RPID / GPID | particle model ids | **Missing** | CODE |
| TXAC, EXP2, PABC, PADC, PSBC, PEDC, PGD1, WFV1–3, EDGF, NERF, DETL, TEXL, PFDC | extended particle, texture weights, waterfall/PBR, edge fade, light cookies… | **Missing** | CODE |
| "ok but 0 sections" | — | 11 v272 + 4 v274 models on Azeroth | MEASURED (map-survey) — cause not investigated |

## R5. BLP

| Item | Wiki claim | State here | Evidence |
|---|---|---|---|
| PIXEL_BC5 = 11 | listed in the enum | **Missing**: SereniaBLPLib maps DXT by alpha size, no BC5 path | CODE: `BlpFile.cs` DXT switch — **unmeasured** prevalence |
| ARGB8888 encoding 3/4 | colour encoding | Supported | CODE |
| DXT5 1024² WMO textures | — | Decode correctly | MEASURED: `130065` decoded and inspected |

## R6. WDT

| Item | Wiki claim | State here |
|---|---|---|
| MAID 8 fields | root, obj0, obj1, tex0, lod, mapTexture, mapTextureN, minimap | **Supported** (root/obj/tex/minimap aliases) |
| MPHD FileDataID fields | lgt, occ, fogs, mpv, tex, wdl, pd4 | **Unread** |
| MPHD 0x80 height texturing | enables `_h` blending | **Unread** |
| `_lgt.wdt`, `_occ.wdt`, `_fogs.wdt`, `_mpv.wdt` | lights, occlusion, volumetric fog, particulates | **Unread** |
