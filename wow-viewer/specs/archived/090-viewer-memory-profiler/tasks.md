# Tasks: Spec 090 Viewer Memory Profiler

**Input**: `spec.md`, `plan.md`

## Phase 1: Runtime Memory Counters (P1)

**Goal**: Make the memory spike visible inside the viewer.

- [x] T001 [US1] Add process working set/private bytes and GC heap counters to Runtime Stats in `ViewerApp_Sidebars.cs`.
- [x] T002 [US1] Show MPQ read-cache count and byte total in Runtime Stats.
- [x] T003 [US1] Show world asset raw-cache count and byte total in Runtime Stats.

## Phase 2: Raw Asset Cache Byte Cap (P1)

**Goal**: Prevent the local raw asset byte cache from growing by entry count alone.

- [x] T004 [US2] Track raw file-cache bytes in `WorldAssetManager.cs`.
- [x] T005 [US2] Add a byte cap to `WorldAssetManager` raw file-cache LRU eviction.
- [x] T006 [US2] Include raw cache byte totals in `WorldAssetReadStats`.

## Phase 3: Validation (P1)

- [x] T007 [US1] Build with `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`.
- [ ] T008 [US1] Manual check Runtime Stats with no world loaded.
- [ ] T009 [US1] Manual check staged `4_0_0_11927` Stormwind/Azeroth memory counters.

## Phase 4: Next Optimization Decision (P2)

- [ ] T010 [US3] If world raw cache approaches the cap, lower or make the cap configurable.
- [ ] T011 [US3] If managed heap grows outside cache counters, profile parsed WMO/M2 object graphs.
- [ ] T012 [US3] If private bytes grow but managed/cache counters do not, use native memory profiling.
- [ ] T013 [US3] If GPU/driver memory is implicated, capture a frame with NVIDIA Nsight Graphics.

## Notes

- Do not change live renderer eviction in this slice.
- Keep all validation on staged clients under `output/tmp/wowarchive-clients/`.
