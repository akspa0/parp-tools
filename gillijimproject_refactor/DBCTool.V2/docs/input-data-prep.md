# Input Data Preparation (tree/)

This guide explains the exact input layout expected by DBCTool.V2 and (for context) by AlphaWDTAnalysisTool. We standardize on a central `test_data/` directory with versioned subfolders (e.g., `0.5.3/`, `0.5.5/`, `0.6.0/`, `3.3.5/`). Under each version there is a `tree/` folder that contains the fully extracted game data. This keeps the repo root free for notes while making inputs predictable for tooling.

- Purpose: define a reproducible folder structure for Alpha 0.5.3/0.5.5/0.6.0 and LK 3.3.5 inputs.
- Philosophy: minimal assumptions, preservation-first; no DBC edits, numeric-only mapping.

> Note: This document only prepares inputs. No CSVs or outputs are created until you run the tools.
> Next step: follow the Quick start in [areaid-restoration-approach.md](areaid-restoration-approach.md).

---

## What DBCTool.V2 consumes

DBCTool.V2 operates only on DBCs to generate crosswalk and audit CSVs. It does not read WDT/ADT.

- Required:
  - LK 3.3.5 `DBFilesClient/*.dbc` (e.g., `AreaTable.dbc`, `Map.dbc`)
  - Alpha 0.5.3 and/or 0.5.5 `DBFilesClient/*.dbc` (when available)
  - Optional 0.6.0 `DBFilesClient/*.dbc` for pivoting

Outputs are written to `DBCTool.V2/dbctool_outputs/session_*/compare/` and `compare/v2/`.

## What AlphaWDTAnalysisTool consumes (context)

AlphaWDTAnalysisTool remaps AreaIDs on LK ADT exports using numeric crosswalks and the Alpha ADTs’ captured area values. It benefits from:

- Alpha `World/Maps/<map>/<map>.wdt` (see manual extraction below)
- Alpha ADT files for the source terrain (to capture `alpha_raw = zone<<16 | sub`)
- LK 3.3.5 `DBFilesClient/*.dbc` (for verify labels/names only)

---

## Standard directory layout (example)

```
<repo-root>/test_data/
  0.5.3/
    tree/
      DBFilesClient/                # if available for your build
      World/Maps/
        Kalimdor/Kalimdor.wdt       # manually extracted (see below)
        Azeroth/Azeroth.wdt         # ...
        # ADT files as available for Alpha
  0.5.5/
    tree/
      DBFilesClient/
      World/Maps/
        Kalimdor/Kalimdor.wdt
        Azeroth/Azeroth.wdt
  0.6.0/                            # optional pivot source
    tree/
      DBFilesClient/
  3.3.5/
    tree/
      DBFilesClient/
        AreaTable.dbc
        Map.dbc
        # other DBCs as shipped
```

You can choose different version folder names if needed, but the structure must be `test_data/<version>/tree/...`. The important part is that each build’s `DBFilesClient/` and (for Alpha) `World/Maps/<map>/<map>.wdt` live under that `tree/` subfolder.

---

## Manual extraction of Alpha WDT from listfile-less map MPQs

Alpha map WDTs are stored in listfile-less MPQs located at:

```
World/Maps/<mapname>/<mapname>.wdt.mpq
```

Each such MPQ typically contains exactly two unnamed files:

- A 16-byte MD5 checksum (not required by our tools)
- The WDT binary payload

Because the archive has no listfile and no embedded names, you must manually extract and rename files:

1) Open the MPQ with a tool that handles listfile-less archives. We use MPQEditor by Ladik. You can find it at: https://www.zezula.net/en/mpq/download.html
2) Extract the larger unnamed file (the WDT) and rename it to `<mapname>.wdt`.
3) Optionally extract the small 16-byte file and save it as `<mapname>.wdt.md5` for your records (not used by our tooling).
4) Place the renamed WDT at:
   - `test_data/0.5.x/tree/World/Maps/<mapname>/<mapname>.wdt`
5) Repeat for every map you need (e.g., `Kalimdor`, `Azeroth`).

Notes:
- There is no safe, general automation for these listfile-less MPQs; manual extraction is expected.
- Ensure the filename matches the folder and map (case-sensitive on some platforms).

### About WDL (World Level Data)

You may also see a `.wdl` alongside WDT in some MPQs. WDL stores heightmap data. We already have a separate tool for WDL, but it is not used in DBCTool.V2’s crosswalk generation or in the strict AreaID remap path described in this repository. It’s optional to extract/keep it here.

---

## Running the tools with this layout

- DBCTool.V2
  - Point input flags or environment to versioned roots like `test_data/0.5.3/tree`, `test_data/0.5.5/tree`, `test_data/0.6.0/tree` (optional), and `test_data/3.3.5/tree`.
  - Generate crosswalks and dumps; outputs appear under `DBCTool.V2/dbctool_outputs/session_*/compare/`.

- AlphaWDTAnalysisTool
  - Use the per-map crosswalks in a single `--dbctool-patch-dir` (include both 0.5.3 and 0.5.5 via060, plus overrides if any).
  - Provide LK 3.3.5 DBCs via `--dbctool-lk-dir` to enrich verify names.
  - Verify and aggregate with `tools/agg-area-verify.ps1`.

---

## Gotchas and tips

- File names must match map names exactly: `<map>/<map>.wdt`.
- If a region’s ADTs only carry `lo16 == 0` (zone-only), strict mapping will produce zone-level LK IDs. This is expected for many Kalimdor areas in Alpha 0.5.3.
- Silithus in Alpha presents as an On-Map-Dungeon (no numeric `src_areaNumber`), so strict mapping results in 0 for those chunks.
- Keep your own `notes/` under `tree/` or repo root; the tools won’t read it.
- If you want modern naming painted over Alpha terrain, that is a separate, documented alternative (LK-ADT painting), not part of this preservation track.

---

## Tools used

- MPQEditor (by Ladik) — used to open listfile-less Alpha map MPQs and manually extract the unnamed WDT payloads.
  - Download: https://www.zezula.net/en/mpq/download.html
  - Why needed: the `World/Maps/<map>/<map>.wdt.mpq` archives contain unnamed entries (typically a 16-byte MD5 and the WDT) and no listfile. Automation is unreliable; use MPQEditor to extract and then rename to `<map>.wdt` before placing it under `tree/alpha-05x/World/Maps/<map>/`.
