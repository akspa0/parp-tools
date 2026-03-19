using System.Numerics;

namespace WoWMapConverter.Core.Formats.PM4;

/// <summary>
/// Canonical MSHD header decoded from a PM4 file.
/// </summary>
public record MshdChunk(
    uint Field00,
    uint Field04,
    uint Field08,
    uint Field0C,
    uint Field10,
    uint Field14,
    uint Field18,
    uint Field1C);

/// <summary>
/// Canonical MSLK entry. This keeps the rollback decoder semantics instead of the older core guesses.
/// </summary>
public record MslkChunk(
    byte TypeFlags,
    byte Subtype,
    ushort Padding,
    uint GroupObjectId,
    int MspiFirstIndex,
    byte MspiIndexCount,
    uint LinkId,
    ushort RefIndex,
    ushort SystemFlag);

/// <summary>
/// Canonical MSUR entry with CK24 accessors decoded from PackedParams.
/// </summary>
public record MsurChunk(
    byte GroupKey,
    byte IndexCount,
    byte AttributeMask,
    byte Padding,
    Vector3 Normal,
    float Height,
    uint MsviFirstIndex,
    uint MdosIndex,
    uint PackedParams)
{
    public uint Ck24 => (PackedParams >> 8) & 0xFFFFFF;

    public byte Ck24Type => (byte)((PackedParams >> 24) & 0xFF);

    public ushort Ck24ObjectId => (ushort)(Ck24 & 0xFFFF);
}

/// <summary>
/// Canonical MPRL entry. The layout is 24 bytes, not the older 16-byte approximation.
/// </summary>
public record MprlChunk(
    ushort Unk00,
    short Unk02,
    ushort Unk04,
    ushort Unk06,
    Vector3 Position,
    short Unk14,
    ushort Unk16);

/// <summary>
/// Canonical MPRR edge/reference entry.
/// </summary>
public record MprrChunk(
    ushort Value1,
    ushort Value2)
{
    public bool IsSentinel => Value1 == 0xFFFF;
}

/// <summary>
/// Canonical single-file PM4 decode result used for future cross-tile aggregation work.
/// </summary>
public record Pm4FileStructure(
    uint Version,
    MshdChunk? Header,
    List<MslkChunk> LinkEntries,
    List<uint> PathIndices,
    List<Vector3> PathVertices,
    List<uint> MeshIndices,
    List<Vector3> MeshVertices,
    List<MsurChunk> Surfaces,
    List<Vector3> SceneNodes,
    List<MprlChunk> PositionRefs,
    List<MprrChunk> GraphEntries,
    Dictionary<string, uint> ChunkSizes,
    List<string> UnparsedChunks);