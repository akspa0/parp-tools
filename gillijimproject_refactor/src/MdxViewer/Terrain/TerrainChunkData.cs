using System.Numerics;

namespace MdxViewer.Terrain;

/// <summary>
/// GPU-ready terrain data for a single MCNK chunk, extracted from Alpha parsers.
/// </summary>
public class TerrainChunkData
{
    /// <summary>Tile X in the 64×64 grid.</summary>
    public int TileX { get; init; }

    /// <summary>Tile Y in the 64×64 grid.</summary>
    public int TileY { get; init; }

    /// <summary>Chunk X within the tile (0-15).</summary>
    public int ChunkX { get; init; }

    /// <summary>Chunk Y within the tile (0-15).</summary>
    public int ChunkY { get; init; }

    /// <summary>145 height values in interleaved order (9-8-9-8... rows).</summary>
    public float[] Heights { get; init; } = Array.Empty<float>();

    /// <summary>145 normals in interleaved order.</summary>
    public Vector3[] Normals { get; init; } = Array.Empty<Vector3>();

    /// <summary>Hole mask from MCNK header (16-bit, one bit per 2×2 cell group).</summary>
    public int HoleMask { get; init; }

    /// <summary>Texture layers for this chunk (up to 4).</summary>
    public TerrainLayer[] Layers { get; init; } = Array.Empty<TerrainLayer>();

    /// <summary>Alpha map data per layer (layer index → 64×64 byte array). Layer 0 has no alpha.</summary>
    public Dictionary<int, byte[]> AlphaMaps { get; init; } = new();

    /// <summary>MCSH shadow map: 64×64 bytes (expanded from 1-bit-per-cell). 0=lit, 255=shadowed.</summary>
    public byte[]? ShadowMap { get; init; }

    /// <summary>Parsed MCLQ liquid data for this chunk. Null if no liquid present.</summary>
    public LiquidChunkData? Liquid { get; set; }

    /// <summary>World-space position of this chunk's corner.</summary>
    public Vector3 WorldPosition { get; init; }

    /// <summary>AreaID from MCNK header (Alpha: Unknown3 at offset 0x38). Used for AreaTable DBC lookup.</summary>
    public int AreaId { get; init; }

    /// <summary>Raw MCNK header flags. Bit 2=Water, Bit 3=Ocean, Bit 4=Magma, Bit 5=Slime, etc.</summary>
    public int McnkFlags { get; init; }
}

/// <summary>
/// A single texture layer within a terrain chunk.
/// </summary>
public struct TerrainLayer
{
    /// <summary>Index into the tile's MTEX texture name table.</summary>
    public int TextureIndex;

    /// <summary>Layer flags from MCLY.</summary>
    public uint Flags;

    /// <summary>Offset into MCAL for this layer's alpha data.</summary>
    public uint AlphaOffset;

    /// <summary>Ground effect ID.</summary>
    public uint EffectId;
}
