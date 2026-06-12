# Tasks: 059 M2 MD20 v0x109+ Cataclysm Support

## Phase 1: Era Tag Split + Dispatch Update

- [x] T001 Add `Md20_4X_V109 = 4` to `M2Era1121EraTag` enum and `"4.x / Cata+ (MD20 v0x109)"` display string in `M2Era1121EraTag.cs`

- [x] T002 Split `>= 0x108` in `DetectEra`: version == 0x108 → `Md20_3X_V108`, version >= 0x109 → `Md20_4X_V109` in `M2ModelReaderDispatcher.cs`

- [x] T003 Add `Md20_4X_V109` to dispatch switch — uses same `M2ModelReader` as 3.3.5

- [x] T004 [P] Add `Dispatcher_4X_Version_GoesToCataTag` test (synthetic 0x109 header)

- [x] T005 [P] Add `Dispatcher_10A_Version_AlsoGoesToCataTag` test (synthetic 0x10A header, open-ended upper bound)

- [x] T006 Add `CreateSyntheticMd20_4X` helper method to test class

- [x] T007 Full test pass: all 11 M2 era tests green

## Phase 2: Ghidra Confirmation

- [ ] T008 When Ghidra analysis completes, locate M2 loader function and read version gate in 4.0.0.11927 binary

- [ ] T009 [P] Document finding in `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`

## Phase 3: Real-Data Validation

- [ ] T010 Extract a `.m2` file from staged 4.0.0.11927 MPQ archives

- [ ] T011 Run `WowViewer.Tool.Inspect m2 inspect` on extracted file, verify era tag is `4.x / Cata+`

- [ ] T012 [P] Full build + test pass
