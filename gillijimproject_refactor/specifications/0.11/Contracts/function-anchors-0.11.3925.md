# Function Anchor Map — 0.11.3925

## ADT anchors
- `0x006b2920` (`FUN_006b2920`): root ADT chunk contract gate (`MVER`, `MHDR`, `MCIN`, `MTEX`, `MMDX`, `MMID`, `MWMO`, `MWID`, `MDDF`, `MODF`).
- `0x006b3be0` (`FUN_006b3be0`): direct MCIN entry indexer/chunk materializer (`entry = base + 8 + (tileIndex*0x10)`; chunk ptr from entry offset).
- `0x006a23d0` (`FUN_006a23d0`): ADT chunk bring-up chain and per-layer loop.
- `0x006a2840` (`FUN_006a2840`): MCNK required-subchunk assertions (`MCVT`, `MCNR`, `MCLY`, `MCRF`, `MCSH`, `MCAL`, `MCLQ`, `MCSE`).
- `0x006a26e0` (`FUN_006a26e0`): ADT placement/object record fan-out from chunk-local tables.

## WMO anchors
- `0x006b4dd0` (`FUN_006b4dd0`): WMO group root contract (`MVER`, `MOGP`, parent/link checks).
- `0x006b5260` (`FUN_006b5260`): conditional subchunk gate parser (`MOLR`, `MODR`, `MOBN/MOBR`, `MOCV`, `MLIQ`, `MORI/MORB`).
- `0x006b5260` (`FUN_006b5260`): `MLIQ` block decode and sample/mask pointer derivation (`chunk+0x26`, bytes `f8*f4*8` + `f0*fc`).
- `0x006962f0` (`FUN_006962f0`): liquid vertex-grid builder from `MLIQ` samples (`f4/f8` dims, `104/108` XY origins, sample height at `+4`).
- `0x006aa0c0` (`FUN_006aa0c0`): liquid index-strip builder from mask stream (`fc/100` dims), validates indices against `(f4*f8)`.
- `0x00678c40` (`FUN_00678c40`): liquid sound/query pass consuming mask nibble classes and sample heights.
- `0x006aa390` (`FUN_006aa390`): draw path using `(ushort)(+0x110)` as liquid/material class index (`*0x40` stride).
- `0x00679000` (`FUN_00679000`): world liquid proximity query over loaded chunk liquids.
- `0x006905f0` (`FUN_006905f0`): point liquid-flag query on chunk-local liquid grids.
- `0x006907d0` (`FUN_006907d0`): point liquid height/material query on chunk-local liquid grids.
- `0x0067b4f0` (`FUN_0067b4f0`), `0x0067bc50` (`FUN_0067bc50`): liquid index construction/query helpers over mask grids.

## MDX/M2 anchors
- `0x00706a40` (`FUN_00706a40`): model load entry (`CM2Model` construction path).
- `0x00710890` (`FUN_00710890`): extension normalization (`.mdx/.mdl` coercion toward `.m2`) and shared-model cache dispatch.
- `0x0070db30` (`FUN_0070db30`): async/shared model IO setup.
- `0x0070dcc0` (`FUN_0070dcc0`): post-read validation/init callback invoking header validator.
- `0x0070d500` (`FUN_0070d500`): core model binary validator (`MD20`, version `0x100`, table bounds checks).
- `0x0070e4d0`, `0x0070f140`, `0x0070f4e0`, `0x0070f690`, `0x0070f970`: nested fixed-stride validators (`0x6C`, `0xD4`, `0x7C`, `0xDC`, `0x1F8`).
- `0x0070e350`, `0x0070e450`, `0x0070e680`, `0x0070f460`, `0x0070fdc0`, `0x0070ff40`, `0x0070ffc0`, `0x0070fe40`, `0x0070fec0`: typed span validators (1/2/4/8/0x0C/0x10/0x24-byte element contracts).
