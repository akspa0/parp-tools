namespace ParpToolbox.Formats.PM4
{
    /// <summary>
    /// Immutable high-level representation of a PM4 file after parsing and adaption.
    /// Will be populated by <see cref="ParpToolbox.Services.PM4.Pm4Adapter"/>.
    /// </summary>
    public sealed record Pm4Scene
{
    /// <summary>All vertices in world space (already X-flipped).</summary>
    public IReadOnlyList<System.Numerics.Vector3> Vertices { get; init; } = Array.Empty<System.Numerics.Vector3>();
    /// <summary>Global triangle list referencing <see cref="Vertices"/> (absolute indices).</summary>
    public IReadOnlyList<(int A,int B,int C)> Triangles { get; init; } = Array.Empty<(int,int,int)>();

    /// <summary>Per-MSUR surface grouping (optional, empty if MSUR absent).</summary>
    public IReadOnlyList<SurfaceGroup> Groups { get; init; } = Array.Empty<SurfaceGroup>();

    /// <summary>Optional link‐table information from MSLK.</summary>
    public IReadOnlyList<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry> Links { get; init; } = Array.Empty<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry>();

    /// <summary>Flat index buffer from MSVI (already triangulated ordering), used for diagnostics.</summary>
    public IReadOnlyList<int> Indices { get; init; } = Array.Empty<int>();

    /// <summary>Raw MSUR surface entries for diagnostics / advanced grouping.</summary>
    public IReadOnlyList<ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry> Surfaces { get; init; } = Array.Empty<ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry>();

    /// <summary>Optional property records from MPRR.</summary>
    public IReadOnlyList<ParpToolbox.Formats.P4.Chunks.Common.MprrChunk.Entry> Properties { get; init; } = Array.Empty<ParpToolbox.Formats.P4.Chunks.Common.MprrChunk.Entry>();
}

/// <summary>Represents a contiguous set of indices defined by an MSUR entry.</summary>
public sealed record SurfaceGroup(
    byte GroupKey,
    string Name,
    IReadOnlyList<(int A,int B,int C)> Faces,
    ushort RawFlags);


}
