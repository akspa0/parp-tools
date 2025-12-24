using System.Numerics;

namespace WoWRollback.PM4Module.Decoding;

/// <summary>MSLK entry - Object catalog linking surfaces to geometry.</summary>
public record MslkChunk(
    byte TypeFlags,      // Obstacle type (1=walkable, 2=walls, etc)
    byte Subtype,        // Hypothesis: floor level
    ushort Padding,
    uint GroupObjectId,
    int MspiFirstIndex,  // 24-bit, -1 = no geometry
    byte MspiIndexCount,
    uint LinkId,         // Cross-tile link
    ushort RefIndex,     // → MPRL if < count, else → MSVT
    ushort SystemFlag    // Always 0x8000
);

/// <summary>MSUR entry - Surface definition with CK24 grouping.</summary>
public record MsurChunk(
    byte GroupKey,       // 0 = M2/portal candidate
    byte IndexCount,     // Triangle count x 3
    byte AttributeMask,  // bit7 = liquid?
    byte Padding,
    Vector3 Normal,
    float Height,
    uint MsviFirstIndex,
    uint MdosIndex,      // **CRITICAL**: Link to MSCN node array
    uint PackedParams    // CK24 = (PackedParams >> 8) & 0xFFFFFF
) {
    public uint CK24 => (PackedParams >> 8) & 0xFFFFFF;
    public byte CK24Type => (byte)((PackedParams >> 24) & 0xFF);
    public ushort CK24ObjectId => (ushort)(CK24 & 0xFFFF);
}

/// <summary>MPRL entry - Position reference with metadata (purpose of non-position fields under investigation).</summary>
public record MprlChunk(
    ushort Unk00,         // Always 0 observed
    short Unk02,          // -1 in all entries for some tiles
    ushort Unk04,         // Varies, may be pitch/tilt or index
    ushort Unk06,         // Often 0x8000
    Vector3 Position,
    short Unk14,          // Signed, -1 to ~7 observed - may be yaw or level index
    ushort Unk16          // Varies, may be flags
);

/// <summary>MPRR entry - Reference graph edge.</summary>
public record MprrChunk(
    ushort Value1,        // Index (MPRL, MSVT) or sentinel (0xFFFF)
    ushort Value2         // Type/Flag
);

/// <summary>MSHD Header - File metadata.</summary>
public record MshdChunk(
    uint Field00, // Tile X extent (approx)
    uint Field04, // Tile Y extent (approx)
    uint Field08, // Tile Z extent (approx)
    uint Field0C,
    uint Field10,
    uint Field14,
    uint Field18,
    uint Field1C
);

public record Pm4FileStructure(
    uint Version,
    MshdChunk? Header,
    List<MslkChunk> LinkEntries,
    List<uint> PathIndices,      // MSPI
    List<Vector3> PathVertices,  // MSPV
    List<uint> MeshIndices,      // MSVI
    List<Vector3> MeshVertices,  // MSVT
    List<MsurChunk> Surfaces,    // MSUR
    List<Vector3> SceneNodes,    // MSCN
    List<MprlChunk> PositionRefs,// MPRL
    List<MprrChunk> GraphEntries // MPRR
);
