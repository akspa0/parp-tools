# Implementation Plan: Spec 090 Viewer Memory Profiler

**Branch**: `v0.5.0-dev` | **Date**: 2026-07-05 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `wow-viewer/specs/090-viewer-memory-profiler/spec.md`

## Summary

Add first-class memory visibility to Runtime Stats and cap the world asset raw-byte cache by bytes as well as entries. This creates immediate evidence for the 4.0.0 Stormwind RAM spike and reduces the easiest unbounded source without changing live renderer eviction.

## Technical Context

**Language/Version**: C#/.NET, ImGui.NET, Silk.NET OpenGL

**Primary Dependencies**: `ViewerApp_Sidebars.cs`, `WorldAssetManager.cs`, `MpqDataSource.cs`, `System.Diagnostics.Process`, `GC`

**Storage**: In-memory diagnostics only

**Testing**: `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`; manual Runtime Stats check on staged 4.0.0 map

**Target Platform**: Windows desktop viewer

**Project Type**: Desktop viewer app

**Performance Goals**: Keep raw world asset byte cache under a fixed budget during dense map loads; expose enough counters to locate the next growth source.

**Constraints**:
- New code goes in `wow-viewer`.
- Do not use non-staged clients.
- Do not evict live renderers until a measured profile proves renderer residency is the main growth source.

## Profiling Tool Decision

- **Built-in Runtime Stats**: first line for every map load. It distinguishes process memory, managed heap, MPQ cache, world raw cache, asset queues, and renderer counts.
- **dotnet-counters / Visual Studio managed memory profiler**: use when managed heap grows with process memory.
- **VMMap / Visual Studio native memory view**: use when private bytes grow but managed heap and cache counters stay flat.
- **NVIDIA Nsight Graphics**: useful for Silk.NET/OpenGL frame captures, draw calls, GPU resources, and VRAM behavior. It is not the first tool for a 34 GB CPU RAM jump unless Runtime Stats points away from managed/native process memory and toward GPU/driver resources.

## Project Structure

```text
wow-viewer/
├── specs/090-viewer-memory-profiler/
│   ├── spec.md
│   ├── plan.md
│   └── tasks.md
└── src/viewer/WoWViewer/
    ├── ViewerApp_Sidebars.cs
    └── Terrain/WorldAssetManager.cs
```

## Phased Delivery

### Phase A: Built-In Memory Counters

- Add process and GC memory counters to Runtime Stats.
- Show MPQ read cache count/bytes.
- Show world asset raw cache count/bytes.
- Build validate.

### Phase B: Raw Cache Byte Budget

- Track `WorldAssetManager` raw cache bytes.
- Evict raw cache entries until both entry and byte caps are respected.
- Keep renderer residency unchanged.
- Build validate.

### Phase C: Stormwind Measurement

- Load staged `4_0_0_11927` Stormwind/Azeroth.
- Record before-load, after-terrain-load, after-asset-queue-drain, and after-camera-idle counters.
- Choose the next optimization based on the largest retained owner.

## Validation Plan

1. `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
2. Open Runtime Stats before loading a map and confirm process/GC counters render.
3. Load a staged 4.0.0 Stormwind route and capture memory counter values.
4. If working set/private bytes remain high while managed/cache counters are low, switch to native/GPU profiling instead of changing C# caches.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |
