# WoW File Format Specifications

Definitive format documentation derived from our verified viewer implementations and Ghidra reverse engineering.

## Specifications

| File | Description | Source of Truth |
|------|-------------|-----------------|
| [alpha-053-terrain.md](alpha-053-terrain.md) | Alpha 0.5.3 WDT/ADT/MCNK terrain | `AlphaTerrainAdapter.cs`, `WdtAlpha`, `AdtAlpha`, `McnkAlpha` |
| [alpha-053-models.md](alpha-053-models.md) | Alpha 0.5.3 MDX models and WMO v14 | `MdxFile.cs`, `WmoV14ToV17Converter.cs` |
| [alpha-053-coordinates.md](alpha-053-coordinates.md) | Coordinate systems and conversions | `WoWConstants.cs`, `AlphaTerrainAdapter.cs` |
| [lk-335-terrain.md](lk-335-terrain.md) | LK 3.3.5 WDT/ADT/MCNK terrain | `StandardTerrainAdapter.cs`, `Mcnk.cs` |
| [mpq-070-patching.md](mpq-070-patching.md) | WoW 0.7.0.3694 MPQ patch overlay behavior (`patch.MPQ`) | Ghidra decompilation of mount/open/lookup flow |

## Ghidra Reverse Engineering Prompts

| File | Target Binary | Notes |
|------|--------------|-------|
| [ghidra/prompt-053.md](ghidra/prompt-053.md) | WoWClient.exe 0.5.3.3368 | Has PDB! Best starting point |
| [ghidra/prompt-055.md](ghidra/prompt-055.md) | WoWClient.exe 0.5.5.3494 | No PDB, use 0.5.3 as reference |
| [ghidra/prompt-060.md](ghidra/prompt-060.md) | WoWClient.exe 0.6.0.3592 | Transitional format |
| [ghidra/prompt-335.md](ghidra/prompt-335.md) | Wow.exe 3.3.5.12340 | LK reference build |
| [ghidra/prompt-400.md](ghidra/prompt-400.md) | Wow.exe 4.0.0.11927 | Cataclysm Alpha (3.3.5-style data formats) |

## Unknowns & Research Targets

See [unknowns.md](unknowns.md) for a prioritized list of format unknowns that need Ghidra investigation.
