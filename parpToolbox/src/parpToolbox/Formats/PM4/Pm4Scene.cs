namespace ParpToolbox.Formats.PM4
{
    /// <summary>
    /// Immutable high-level representation of a PM4 file after parsing and adaption.
    /// Will be populated by <see cref="ParpToolbox.Services.PM4.Pm4Adapter"/>.
    /// </summary>
    public sealed class Pm4Scene
{
    /// <summary>All vertices in world space (already X-flipped).</summary>
    public List<System.Numerics.Vector3> Vertices { get; init; } = new();
    /// <summary>Global triangle list referencing <see cref="Vertices"/> (absolute indices).</summary>
    public List<(int A,int B,int C)> Triangles { get; init; } = new();

    /// <summary>Per-MSUR surface grouping (optional, empty if MSUR absent).</summary>
    public List<SurfaceGroup> Groups { get; init; } = new();

    /// <summary>Optional link‐table information from MSLK.</summary>
    public List<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry> Links { get; init; } = new();

    /// <summary>Flat index buffer from MSVI (already triangulated ordering), used for diagnostics.</summary>
    public List<int> Indices { get; init; } = new();

    /// <summary>Raw MSUR surface entries for diagnostics / advanced grouping.</summary>
    public List<ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry> Surfaces { get; init; } = new();

    /// <summary>Raw MSPI chunks for diagnostics / advanced grouping.</summary>
    public List<ParpToolbox.Formats.P4.Chunks.Common.MspiChunk> Spis { get; init; } = new();

    /// <summary>Optional property records from MPRR.</summary>
    public List<ParpToolbox.Formats.P4.Chunks.Common.MprrChunk.Entry> Properties { get; init; } = new();

    /// <summary>Optional placement records from MPRL (likely doodad/prop positions).</summary>
    public List<ParpToolbox.Formats.P4.Chunks.Common.MprlChunk.Entry> Placements { get; init; } = new();

    /// <summary>Any additional raw chunks that were parsed but not yet surfaced, e.g. MSCN.</summary>
    public List<ParpToolbox.Formats.P4.Chunks.Common.IIffChunk> ExtraChunks { get; init; } = new();
}

/// <summary>Represents a contiguous set of indices defined by an MSUR entry.</summary>
public sealed record SurfaceGroup(
    byte GroupKey,
    string Name,
    IReadOnlyList<(int A,int B,int C)> Faces,
    ushort RawFlags);


}
