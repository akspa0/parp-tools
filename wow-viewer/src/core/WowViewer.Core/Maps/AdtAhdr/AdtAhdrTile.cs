using System.Numerics;

namespace WowViewer.Core.Maps.AdtAhdr;

/// <summary>
/// Spec 237: one decoded AHDR-family terrain file (DAT v26, and the wiki's v22/v23 relatives).
/// Fields documented as measured are proven on the DAT v26 corpus; everything else is kept raw.
/// See docs/architecture/adt-v26-format.md.
/// </summary>
public sealed class AdtAhdrTile
{
    public required string SourcePath { get; init; }

    /// <summary>MVER version when the file starts with MVER (26 in the corpus); null for AHDR-first files.</summary>
    public uint? MverVersion { get; init; }

    public uint Version { get; init; }
    public int VerticesX { get; init; }
    public int VerticesY { get; init; }
    public int ChunksX { get; init; }
    public int ChunksY { get; init; }

    /// <summary>AHDR +0x14.. (11 uint32). v26: first value 8396383, rest 0; meaning open.</summary>
    public uint[] HeaderReserved { get; init; } = [];

    /// <summary>Raw ALOC (5 uint32) or null when absent.</summary>
    public uint[]? Aloc { get; init; }

    /// <summary>Measured: ALOC[1] is tile X (the outer grid's column axis).</summary>
    public int? TileX => Aloc is { Length: >= 3 } ? (int)Aloc[1] : null;

    /// <summary>Measured: ALOC[2] is tile Y (the outer grid's row axis).</summary>
    public int? TileY => Aloc is { Length: >= 3 } ? (int)Aloc[2] : null;

    /// <summary>Row-major outer heights, VerticesY rows × VerticesX columns (measured for v26).</summary>
    public float[] OuterHeights { get; init; } = [];

    /// <summary>Inner heights, (VerticesY-1) × (VerticesX-1); order assumed to match the outer grid (unverified).</summary>
    public float[] InnerHeights { get; init; } = [];

    /// <summary>Raw ANRM bytes (encoding unverified for v26).</summary>
    public byte[]? NormalsRaw { get; init; }

    /// <summary>Raw ACVT bytes (channel order unverified).</summary>
    public byte[]? VertexShadingRaw { get; init; }

    public IReadOnlyList<string> TextureNames { get; init; } = [];
    public IReadOnlyList<string> ModelNames { get; init; } = [];
    public IReadOnlyList<AdtAhdrChunk> Chunks { get; init; } = [];
    public IReadOnlyList<uint[]> Adst { get; init; } = [];
    public IReadOnlyList<string> Diagnostics { get; init; } = [];
}

public sealed class AdtAhdrChunk
{
    /// <summary>Measured for v26: i % 16 for the i-th ACNK in file order.</summary>
    public int IndexX { get; init; }

    /// <summary>Measured for v26: i / 16 for the i-th ACNK in file order.</summary>
    public int IndexY { get; init; }

    public byte[] HeaderRaw { get; init; } = [];
    public IReadOnlyList<AdtAhdrLayer> Layers { get; init; } = [];

    /// <summary>Raw ASHD payload (512 bytes in v26; bit layout unverified).</summary>
    public byte[]? ShadowRaw { get; init; }

    public IReadOnlyList<AdtAhdrObjectDefinition> Objects { get; init; } = [];
}

/// <summary>ALYR. v26: flags 0x100 on every layer; AMAP is 4096 bytes (8-bit 64×64).</summary>
public sealed record AdtAhdrLayer(int TextureIndex, uint Flags, byte[]? AlphaMap);

/// <summary>
/// ACDO, read with the wiki v22 field names. Only <see cref="ModelIndex"/> and the record size are
/// confirmed for v26; position frame, rotation units and the "scale" fields are unverified.
/// </summary>
public sealed record AdtAhdrObjectDefinition(
    int ModelIndex,
    Vector3 Position,
    Vector3 Rotation,
    Vector3 ScaleField,
    float Unknown28,
    uint UniqueId,
    byte[] RawRecord);
