# 073b — Tools Tab Converter Integration

## Goal
Surface the major `WowViewer.Tool.Converter` commands inside the viewer's **Tools > Converters** sub-tab so users can run conversions without dropping to a terminal.

## Constraints
- No removal of existing functionality.
- Legacy mode (`View > Legacy UI`) untouched.
- Build `WoWViewer.csproj` clean after every phase.
- Run conversions by launching the existing converter executable; do NOT reimplement conversion logic in the viewer.

## Converter commands to expose

1. **Alpha ↔ LK**: `convert-alpha-to-lk`, `convert-lk-to-alpha`
2. **Model formats**: `convert-m2-to-mdx`, `convert-mdx-to-m2`
3. **WMO versions**: `convert-wmo-v14-to-v17`, `convert-wmo-v17-to-v14`
4. **ADT utilities**: `split-adt-to-lk`, `patch-terrain-adt`
5. **Validation**: `validate-round-trip`

## Phase 1: Discover command-line contracts

- [ ] T001 Read `Program.cs` in `WowViewer.Tool.Converter` to map command verbs to `Run` methods.
- [ ] T002 Read one command file per group above to confirm required/optional args.
- [ ] T003 Document each verb's args in this tasks.md.

## Phase 2: Add Converters sub-tab

- [ ] T004 Add `ConvertersBottomTab` to `ToolsBottomTab` enum (or reuse `UtilitiesBottomTab` if more appropriate).
- [ ] T005 Add label mapping in `WorkbenchNavigator`.
- [ ] T006 Add `DrawToolsConvertersSubTab` method in `ViewerApp` (or `ViewerApp_Tools.cs` if it exists).
- [ ] T007 Register the sub-tab in the Tools tab dispatch.

## Phase 3: Per-converter UI cards

- [ ] T008 Create a reusable `DrawConverterCard(title, description, inputs[], runAction)` helper.
- [ ] T009 Add Alpha→LK card: input WDT path, output dir, optional area crosswalk, verbose toggle.
- [ ] T010 Add LK→Alpha card: input WDT path, output dir, verbose toggle.
- [ ] T011 Add M2↔MDX cards: input file, output file/dir.
- [ ] T012 Add WMO V14↔V17 cards: input file, output file/dir.
- [ ] T013 Add Split ADT to LK card: input ADT/WDT, output dir.
- [ ] T014 Add Terrain Patch ADT card: input ADT, patch options.
- [ ] T015 Add Validate Round Trip card: input path, expected format.

## Phase 4: Run integration

- [ ] T016 Locate converter executable relative to viewer executable (`../tools/converter/...` or project output path).
- [ ] T017 Run converter via `Process.Start` with args, capture stdout/stderr, show status in the card.
- [ ] T018 Disable Run button while a conversion is in progress.
- [ ] T019 Show last result (success/failure + truncated output) in the card.

## Phase 5: Validation

- [ ] T020 Build `WoWViewer.csproj` with 0 errors.
- [ ] T021 Open Tools > Converters in tab UI; verify cards render without overlap.
- [ ] T022 Toggle `View > Legacy UI`; verify no regression.
- [ ] T023 Update this tasks.md; commit + push.
