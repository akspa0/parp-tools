using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Research;

public sealed record Pm4ResearchDocument(
    string? SourcePath,
    uint Version,
    IReadOnlyList<Pm4ChunkRecord> Chunks,
    Pm4KnownChunkSet KnownChunks,
    IReadOnlyList<string> Diagnostics);