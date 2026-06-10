# WoW 0.8.0.3734 — Profile Index

This file is the entry point for parser-profile documentation and binary contracts for build `0.8.0.3734`.

## Primary profile docs
- [Parser_Profile_0.8.0.3734_Binary.md](Parser_Profile_0.8.0.3734_Binary.md)
- [Parser_Profile_0.8.0.3734_Field_Map.md](Parser_Profile_0.8.0.3734_Field_Map.md)

## Contracts (authoritative evidence extracts)
- [Contracts/baseline-diff-0.8.0.3734.md](Contracts/baseline-diff-0.8.0.3734.md)
- [Contracts/Ghidra_Function_Anchors_0.8.0.3734.md](Contracts/Ghidra_Function_Anchors_0.8.0.3734.md)
- [Contracts/ADT_Binary_Contract_0.8.0.3734.md](Contracts/ADT_Binary_Contract_0.8.0.3734.md)
- [Contracts/WMO_Binary_Contract_0.8.0.3734.md](Contracts/WMO_Binary_Contract_0.8.0.3734.md)
- [Contracts/MDX_Binary_Contract_0.8.0.3734.md](Contracts/MDX_Binary_Contract_0.8.0.3734.md)

## Supporting 0.8.0 research notes
- [ADT_MCLQ_0.8.0.3734.md](ADT_MCLQ_0.8.0.3734.md)
- [ADT_Unknown_Field_Resolution_0.8.0.3734.md](ADT_Unknown_Field_Resolution_0.8.0.3734.md)
- [ADT_Semantic_Diff_0.8.0.3734_to_0.9.1.3810.md](ADT_Semantic_Diff_0.8.0.3734_to_0.9.1.3810.md)
- [WMO_Format_0.8.0.3734.md](WMO_Format_0.8.0.3734.md)
- [MDX_Format_0.8.0.3734.md](MDX_Format_0.8.0.3734.md)
- [MPQ_Patching_0.8.0.3734.md](MPQ_Patching_0.8.0.3734.md)
- [BNUpdate_DeepDive_0.8.0.3734.md](BNUpdate_DeepDive_0.8.0.3734.md)

## Recommended implementation order
1. Apply profile constants from [Parser_Profile_0.8.0.3734_Field_Map.md](Parser_Profile_0.8.0.3734_Field_Map.md).
2. Validate parser behavior against contract deltas in [Contracts/baseline-diff-0.8.0.3734.md](Contracts/baseline-diff-0.8.0.3734.md).
3. Resolve remaining unknowns listed in profile/contract docs before widening build-range coverage.

## Current known open items
- ADT MCIN entry-size/count derivation proof for this build.
- Full MDX top-level chunk order under dispatcher path.
- Precise semantic naming for some WMO `MLIQ` header words.
