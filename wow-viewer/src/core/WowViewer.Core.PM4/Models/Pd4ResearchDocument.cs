using System.Numerics;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Models;

public sealed record Pd4ResearchDocument(
    string? SourcePath,
    uint Version,
    uint Mcrc,
    IReadOnlyList<Pm4ChunkRecord> Chunks,
    Pd4KnownChunkSet KnownChunks,
    IReadOnlyList<string> Diagnostics);

public sealed record Pd4KnownChunkSet(
    IReadOnlyList<Pd4MslkEntry> Mslk,
    IReadOnlyList<Vector3> Mspv,
    IReadOnlyList<uint> Mspi,
    IReadOnlyList<Vector3> Msvt,
    IReadOnlyList<uint> Msvi,
    IReadOnlyList<Pd4MsurEntry> Msur,
    IReadOnlyList<Vector3> Mscn);
