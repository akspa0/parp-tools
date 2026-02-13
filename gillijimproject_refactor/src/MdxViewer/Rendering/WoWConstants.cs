namespace MdxViewer.Rendering;

/// <summary>
/// Centralized constants from the WoW Alpha 0.5.3 client, verified via Ghidra reverse engineering.
/// </summary>
public static class WoWConstants
{
    // ── Terrain ──────────────────────────────────────────────────────────
    /// <summary>World units per terrain chunk (533.3333…).</summary>
    public const float ChunkSize = 533.33333f;

    /// <summary>World units per terrain cell (66.6667…). 8 cells per chunk.</summary>
    public const float CellSize = 66.66667f;

    /// <summary>Half a chunk in world units (266.6667…).</summary>
    public const float ChunkOffset = 266.66667f;

    /// <summary>Inverse of ChunkSize (1/533.3333).</summary>
    public const float ChunkScale = 1.0f / 533.33333f;

    /// <summary>Maximum chunk index in the 64×64 grid (0-based).</summary>
    public const int MaxChunk = 63;

    /// <summary>Number of cells per chunk edge.</summary>
    public const int NumCells = 8;

    /// <summary>Number of outer vertices per chunk edge (9×9 = 81 outer).</summary>
    public const int NumOuterVertices = 9;

    /// <summary>Number of inner vertices per chunk edge (8×8 = 64 inner).</summary>
    public const int NumInnerVertices = 8;

    /// <summary>Total vertices per MCNK chunk: 81 outer + 64 inner = 145.</summary>
    public const int VerticesPerChunk = 145;

    /// <summary>Triangles per chunk: 8×8 cells × 2 tris × 4 sub-triangles / 4 = 128 (simplified: 256 total with center vertex).</summary>
    public const int TrianglesPerChunk = 256;

    /// <summary>Chunks per ADT tile edge (16×16 = 256 chunks per tile).</summary>
    public const int ChunksPerTileEdge = 16;

    /// <summary>Total chunks per ADT tile.</summary>
    public const int ChunksPerTile = 256;

    /// <summary>Tiles per map edge (64×64 = 4096 tiles per map).</summary>
    public const int TilesPerMapEdge = 64;

    /// <summary>World units per ADT tile (16 chunks × 533.3333 = 8533.3333).</summary>
    public const float TileSize = ChunksPerTileEdge * ChunkSize;

    /// <summary>Map origin offset: world coordinate of tile (0,0) corner.</summary>
    public const float MapOrigin = 17066.66666f;

    // ── Movement ─────────────────────────────────────────────────────────
    public const float WalkSpeed = 3.5f;
    public const float RunSpeed = 7.0f;
    public const float SwimSpeed = 3.5f;
    public const float JumpVelocity = 8.0f;
    public const float Gravity = 9.8f;

    // ── Liquids ──────────────────────────────────────────────────────────
    public const uint LiquidWater = 0x0;
    public const uint LiquidOcean = 0x1;
    public const uint LiquidMagma = 0x2;
    public const uint LiquidSlime = 0x3;
    public const uint LiquidNone  = 0xF;

    // ── Detail Doodads ───────────────────────────────────────────────────
    public const float DetailDoodadDistance = 100.0f;
    public const int MaxDetailDoodads = 64;

    // ── Rendering ────────────────────────────────────────────────────────
    /// <summary>Alpha test threshold for AlphaKey blend mode (Ghidra: 0.5).</summary>
    public const float AlphaKeyThreshold = 0.5f;

    /// <summary>Water particle scale (1/36).</summary>
    public const float WaterParticleScale = 0.027777778f;

    /// <summary>Magma particle scale (1/9).</summary>
    public const float MagmaParticleScale = 0.11111111f;
}
