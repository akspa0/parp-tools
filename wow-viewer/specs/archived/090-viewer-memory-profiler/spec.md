# Feature Specification: 090 Viewer Memory Profiler And 4.0.0 Map Leak Control

**Feature Branch**: `090-viewer-memory-profiler`

**Created**: 2026-07-05

**Status**: Draft

**Input**: User description: "The viewer has a serious memory leak when loading 4.0.0 maps. Loading Stormwind jumps to about 34 GB RAM. Add profiling or similar performance tooling; evaluate whether external GPU tools such as NVIDIA Nsight help with C#/Silk.NET."

## User Scenarios & Testing

### User Story 1 - See Memory While Loading A Map (Priority: P1)

As a viewer user, I need the Runtime Stats surface to show process memory, managed heap memory, and major asset-cache sizes while a map is loading, so I can tell what is growing when a 4.0.0 map explodes in RAM.

**Why this priority**: Without memory counters, every leak fix is guesswork.

**Independent Test**: Open Runtime Stats, load a staged `4_0_0_11927` Stormwind/Azeroth location, and watch working set, private bytes, managed heap, MPQ read cache, and world asset raw cache update during load.

**Acceptance Scenarios**:
1. **Given** a world scene is loaded, **When** Runtime Stats is open, **Then** it shows process working set, private bytes, managed heap, total allocated bytes, and Gen0/Gen1/Gen2 collection counts.
2. **Given** MPQ and world asset caches are active, **When** files are loaded, **Then** Runtime Stats shows raw cache counts and byte totals.
3. **Given** a memory spike occurs, **When** the user reports the Runtime Stats values, **Then** the maintainer can distinguish process/native growth from managed heap and raw-byte cache growth.

### User Story 2 - Bound Raw Asset Cache Growth (Priority: P1)

As a viewer user, I need raw file-byte caches to have byte budgets, not only entry budgets, so loading a dense 4.0.0 city cannot keep unlimited large WMO/group/texture byte arrays alive.

**Why this priority**: The current world asset cache is capped by entry count only, which is unsafe for Cataclysm-era split WMO/object-heavy maps.

**Independent Test**: Load a dense map and verify the world asset raw cache byte total remains under the configured cap while the viewer continues rendering.

**Acceptance Scenarios**:
1. **Given** more raw asset files are read than fit the byte budget, **When** the LRU eviction runs, **Then** old raw byte arrays are removed until both entry and byte budgets are respected.
2. **Given** renderer objects are already loaded, **When** their original raw bytes are evicted, **Then** the loaded renderer remains usable.

### User Story 3 - Decide The Next Profiler Tool (Priority: P2)

As a maintainer, I need clear guidance on when to use built-in counters, dotnet diagnostics, NVIDIA Nsight Graphics, or Visual Studio profiling, so we do not waste time with the wrong tool.

**Why this priority**: Nsight can inspect Silk.NET/OpenGL GPU work, but a 34 GB RAM jump is first a process/heap/cache problem.

**Independent Test**: Read the plan and follow the profiling route for a 4.0.0 Stormwind load.

## Edge Cases

- A map may load many small files or a few huge files; cache budgeting must handle both.
- Process private bytes can grow while managed heap stays flat; that indicates native/GPU/driver or unmanaged image/bitmap allocations, not ordinary C# object retention.
- Managed heap can grow while cache byte counters stay flat; that indicates parsed object graphs or retained renderer/model data.
- GPU VRAM can grow without process working set matching it; use GPU tooling only after CPU/process memory is characterized.

## Requirements

### Functional Requirements

- **FR-001**: Runtime Stats MUST display process working set, private bytes, managed heap, total allocated bytes, and GC collection counts.
- **FR-002**: Runtime Stats MUST display MPQ read-cache count and byte total.
- **FR-003**: Runtime Stats MUST display world asset raw file-cache count and byte total.
- **FR-004**: World asset raw file caching MUST enforce both an entry cap and a byte cap.
- **FR-005**: The first profiling plan MUST identify which external profiler class applies to CPU heap, managed allocations, native process memory, GPU memory, and OpenGL draw-call timing.
- **FR-006**: The first source slice MUST NOT change renderer eviction for live M2/WMO renderers until the Runtime Stats counters identify renderer residency as the growth source.

### Key Entities

- **Process Memory Snapshot**: Working set, private bytes, managed heap, total allocated bytes, and GC collection counts for the viewer process.
- **World Asset Raw Cache**: `WorldAssetManager` raw byte cache for model, skin, WMO group, and texture source files.
- **MPQ Read Cache**: `MpqDataSource` bounded raw-byte read cache.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Runtime Stats shows memory/caches without requiring an external profiler.
- **SC-002**: World asset raw cache remains below its byte budget during dense map loads.
- **SC-003**: The 4.0.0 Stormwind memory investigation can be reported using at least five numbers: working set, private bytes, managed heap, world asset raw cache bytes, and MPQ read cache bytes.
- **SC-004**: The feature builds with `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`.

## Assumptions

- The reported Stormwind case uses staged `4_0_0_11927` client data under `output/tmp/wowarchive-clients/`.
- The first fix should improve observability and obvious raw-cache growth before changing renderer residency or object streaming semantics.
- NVIDIA Nsight Graphics can be useful later for OpenGL/Silk.NET GPU frame analysis, but it is not the first tool for a 34 GB process RAM spike.
