# Project Brief

## Mission
**WoWRollback** is a comprehensive toolchain designed to **read, write, and convert** all file formats for the **World of Warcraft Alpha 0.5.3 (3368)** client. The primary goal is to enable full bidirectional asset transfer: bringing modern WoW content into the Alpha client and extracting Alpha content for analysis or use in modern clients.

## Core Objectives
1.  **Full Read/Write Support**: Implement robust readers and writers for all key Alpha formats:
    *   **WDT/ADT**: Monolithic format with embedded tiles (MCNK) and legacy liquids (MCLQ).
    *   **WMO**: Alpha v14 format (monolithic).
    *   **M2/MDX**: Alpha MDX format.
    *   **BLP**: BLP2 format with DXT compression.
2.  **Bidirectional Conversion**:
    *   **Modern → Alpha**: Downporting assets (LK 3.3.5 and newer) to 0.5.3 specifications.
    *   **Alpha → Modern**: Up-porting historical assets to 3.3.5+ formats.
3.  **Diagnostic Tooling**: Provide a standalone toolchain (`AlphaWdtInspector`) to diagnose format issues, verify packing integrity, and ensure "byte-perfect" output where possible.
4.  **No Guesswork**: Maintain definitive, verified specifications (`memory-bank/specs/Alpha-0.5.3-Format.md`) derived from clean-room analysis and battle-tested code.

## Key Components
- **WoWRollback.Core**: Shared library for format handling.
- **AlphaLkToAlphaStandalone**: Dedicated LK → Alpha converter and roundtrip validator.
- **AlphaWdtInspector**: Diagnostic CLI for WDT/ADT analysis.
- **BlpResizer**: Batch texture converter/resizer for Alpha compatibility.

## Success Criteria
- **In-Game Validation**: Converted worlds and assets load and render correctly in the original 0.5.3 client without crashes.
- **Data Integrity**: Roundtrip conversions (Alpha → LK → Alpha) preserve critical metadata like AreaIDs, liquid types, and object placements.
- **Deterministic Output**: Toolchains produce consistent, reproducible results.

