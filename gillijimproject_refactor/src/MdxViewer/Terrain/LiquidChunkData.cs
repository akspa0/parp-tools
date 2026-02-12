using System.Numerics;

namespace MdxViewer.Terrain;

/// <summary>
/// Parsed MCLQ inline liquid data for a single terrain chunk (Alpha 0.5.3 format).
/// Ghidra-verified: inline data (no FourCC), 804 bytes per liquid instance.
/// 9×9 vertex grid (81 vertices × 8 bytes) + 4×4 tile grid (16 floats) + flow vectors.
/// Liquid type determined from MCNK header flags bits 2-5, NOT from tile data.
/// Up to 4 liquid instances per chunk (one per type: Water/Ocean/Magma/Slime).
/// </summary>
public class LiquidChunkData
{
    /// <summary>Size of one MCLQ inline data block: 804 bytes (0x324).</summary>
    public const int InstanceSize = 804;

    /// <summary>Minimum height of the liquid surface.</summary>
    public float MinHeight { get; init; }

    /// <summary>Maximum height of the liquid surface.</summary>
    public float MaxHeight { get; init; }

    /// <summary>9×9 vertex heights (81 values). Index = y*9+x.</summary>
    public float[] Heights { get; init; } = Array.Empty<float>();

    /// <summary>9×9 vertex data words (81 values). Flags/depth per vertex.</summary>
    public uint[] VertexData { get; init; } = Array.Empty<uint>();

    /// <summary>4×4 tile grid (16 floats). Coarse tile data (heights or flags).</summary>
    public float[] TileGrid { get; init; } = Array.Empty<float>();

    /// <summary>8×8 per-tile flags (64 bytes). Null = render all tiles.
    /// Per 0.8.0 Ghidra spec: (flag &amp; 0x0F) == 0x0F means no liquid at tile.</summary>
    public byte[]? TileFlags { get; init; }

    /// <summary>Liquid type for this instance (from MCNK header flags bits 2-5).</summary>
    public LiquidType Type { get; init; } = LiquidType.Water;

    /// <summary>World-space position of this chunk's corner (same as parent terrain chunk).</summary>
    public Vector3 WorldPosition { get; init; }

    /// <summary>Tile X in the 64×64 grid.</summary>
    public int TileX { get; init; }

    /// <summary>Tile Y in the 64×64 grid.</summary>
    public int TileY { get; init; }

    /// <summary>Chunk X within the tile (0-15).</summary>
    public int ChunkX { get; init; }

    /// <summary>Chunk Y within the tile (0-15).</summary>
    public int ChunkY { get; init; }
}

/// <summary>
/// Liquid type classification for rendering color selection.
/// </summary>
public enum LiquidType
{
    Water = 0,
    Ocean = 1,
    Magma = 2,
    Slime = 3
}
