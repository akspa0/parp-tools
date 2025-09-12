using System.Collections.Generic;

namespace GillijimProject.Next.Core.WowFiles.Alpha;

/// <summary>
/// Minimal domain model for Alpha-era ADT summarization.
/// Contains per-chunk (MCNK) metadata and optional MCVT stats.
/// </summary>
public sealed record AlphaAdt(
    string Path,
    int PresentChunks,
    IReadOnlyList<AlphaAdtChunk> Chunks
);

public sealed record AlphaAdtChunk(
    int Index,
    long McnkOffset,
    uint McnkSize,
    bool HasMcvt,
    int McvtFloatCount,
    bool HasMclq
);
