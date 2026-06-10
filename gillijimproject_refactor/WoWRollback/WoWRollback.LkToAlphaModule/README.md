# WoWRollback.LkToAlphaModule

## Overview
Library for converting LK ADT data to Alpha-compatible structures and back. Includes liquids support and terrain subchunk handling.

## Highlights
- MH2O/MCLQ support (liquids end-to-end)
- MCNK mask/shadow handling helpers
- Builders/extractors used by CLI export/patch commands

## Usage
Library-only. Add a `ProjectReference` and use the extract/build APIs as seen in `WoWRollback.Cli`.

## See Also
- `../docs/Alpha-WDT-Conversion-Spec.md`
- `../docs/Fix-MCLY-MCAL-MCSH-Format.md`
- `../docs/MCNK-SubChunk-Audit.md`
