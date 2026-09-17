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

    /// <summary>Raw AOCH payload (2048 bytes, all zero in the v26 corpus), kept so files can be rewritten exactly.</summary>
    public byte[]? AochRaw { get; init; }

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

    /// <summary>ADST rows: (uniqueId, model FileDataID, 1). See <see cref="AdtAhdrModelFileReference"/>.</summary>
    public IReadOnlyList<AdtAhdrModelFileReference> ModelFileReferences { get; init; } = [];
    public IReadOnlyList<string> Diagnostics { get; init; } = [];
}

public sealed class AdtAhdrChunk
{
    /// <summary>Measured for v26: i % 16 for the i-th ACNK in file order.</summary>
    public int IndexX { get; init; }

    /// <summary>Measured for v26: i / 16 for the i-th ACNK in file order.</summary>
    public int IndexY { get; init; }

    public byte[] HeaderRaw { get; init; } = [];

    /// <summary>
    /// Predominant texture layer (0..3) of 8×8 cell (<paramref name="cellX"/>, <paramref name="cellY"/>), from the
    /// 2-bit map at ACNK header +0x12 (LSB first, row-major). MEASURED for DAT v26: matches the dominant blended
    /// layer computed from AMAP in 95.0% of 239,808 cells (control 67.9%; script acnk_predominant_v26.py).
    /// </summary>
    public int PredominantLayer(int cellX, int cellY)
    {
        if (HeaderRaw.Length < 0x22 || (uint)cellX > 7 || (uint)cellY > 7)
            return 0;

        int bit = (cellY * 8 + cellX) * 2;
        return (HeaderRaw[0x12 + bit / 8] >> (bit % 8)) & 3;
    }
    public IReadOnlyList<AdtAhdrLayer> Layers { get; init; } = [];

    /// <summary>Raw ASHD payload (512 bytes in v26; bit layout unverified).</summary>
    public byte[]? ShadowRaw { get; init; }

    public IReadOnlyList<AdtAhdrObjectDefinition> Objects { get; init; } = [];
}

/// <summary>
/// ALYR. v26: flags 0x100 on every layer; AMAP is 4096 bytes (8-bit 64×64) holding this layer's blend <b>weight</b>
/// (weights of all layers sum to 255 per pixel, layer 0 included). Convert with <c>AdtAhdrAlpha</c>.
/// </summary>
public sealed record AdtAhdrLayer(int TextureIndex, uint Flags, byte[]? AlphaMap);

/// <summary>
/// ACDO: one object placement. Layout MEASURED on the DAT v26 corpus (5,309 records in 18 files; script
/// specs/237-adt-v26-terrain/evidence/scripts/acdo_chunk_frame_v26.py):
/// <list type="bullet">
/// <item>+0x00 model index into ADOO (0..284, every index valid).</item>
/// <item>+0x04/+0x08/+0x0C position in <b>inches</b>, relative to the chunk that stores the record: +0x04 along the
/// grid column axis (ALOC tile X) and +0x0C along the row axis (ALOC tile Y), both from the chunk centre (range
/// ±600 = half a 1200-inch chunk); +0x08 vertical, from the mean of the chunk's 145 AVTX heights. Placed this
/// way, objects sit on the terrain: median |error| 0.31 in, signed median 0.00, versus 103.9 in when objects
/// are moved to random chunks.</item>
/// <item>+0x10/+0x14/+0x18 rotation in degrees, same axis order as the position (+0x14 is about the vertical).</item>
/// <item>+0x1C scale (0.1..3.47). +0x20 always 1.0, +0x24 always 0, +0x28 float mostly 0 (meaning open).</item>
/// <item>+0x2C uniqueId (distinct for every record).</item>
/// <item>+0x30 count of trailing uint32 values (0 in 56-byte records, 1 in the 11 60-byte records, all WMOs);
/// +0x34 uint32 (0, 1, 2, 3 or 65536; open); +0x38.. the trailing values (1 or 2; open, possibly a doodad set).</item>
/// </list>
/// </summary>
public sealed record AdtAhdrObjectDefinition(
    int ModelIndex,
    Vector3 LocalPositionInches,
    Vector3 RotationDegrees,
    float Scale,
    float Field20,
    uint Field24,
    float Field28,
    uint UniqueId,
    uint Field34,
    uint[] TrailingValues,
    byte[] RawRecord);

/// <summary>
/// ADST: (uniqueId, model FileDataID, 1). MEASURED on the DAT v26 corpus: 321 rows in 7 files; every
/// FileDataID names a model in the community listfile (e.g. 190719 World/critter/BIRDS/Bird01.m2), the last
/// field is always 1, and no ADST uniqueId matches any ACDO uniqueId in the 699 files. The rows carry no
/// position, so they cannot be placed on their own.
/// </summary>
public sealed record AdtAhdrModelFileReference(uint UniqueId, uint FileDataId, uint Field8);
