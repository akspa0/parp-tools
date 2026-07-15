using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Runtime.World.Terrain;

public sealed class WorldTerrainChunkData
{
    public WorldTerrainChunkData(
        int chunkIndex,
        int indexX,
        int indexY,
        uint areaId,
        uint flags,
        int layerCount,
        ushort holeMask,
        bool hasLiquidFlags,
        bool hasVertexColors,
        float[]? heights,
        IReadOnlyList<AdtTextureChunkLayer>? textureLayers = null,
        Vector3[]? normals = null,
        byte[]? shadowMap = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(chunkIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(indexX);
        ArgumentOutOfRangeException.ThrowIfNegative(indexY);
        ArgumentOutOfRangeException.ThrowIfNegative(layerCount);
        if (heights is not null && heights.Length != 145)
            throw new ArgumentException("Terrain chunk height payloads must contain exactly 145 MCVT samples.", nameof(heights));
        if (normals is not null && normals.Length != 145)
            throw new ArgumentException("Terrain chunk normal payloads must contain exactly 145 MCNR samples.", nameof(normals));
        if (shadowMap is not null && shadowMap.Length != 64 * 64)
            throw new ArgumentException("Terrain chunk shadow payloads must contain exactly 4096 expanded MCSH texels.", nameof(shadowMap));

        ChunkIndex = chunkIndex;
        IndexX = indexX;
        IndexY = indexY;
        AreaId = areaId;
        Flags = flags;
        DeclaredLayerCount = layerCount;
        HoleMask = holeMask;
        HoleMaskState = new WorldTerrainHoleMask(holeMask);
        HasLiquidFlags = hasLiquidFlags;
        HasVertexColors = hasVertexColors;
        Heights = heights;
        TextureLayers = textureLayers ?? [];
        Normals = normals;
        ShadowMap = shadowMap;
        CellGrid = WorldTerrainCellGrid.CreateDefault(holeMask);
    }

    public int ChunkIndex { get; }

    public int IndexX { get; }

    public int IndexY { get; }

    public uint AreaId { get; }

    public uint Flags { get; }

    public int DeclaredLayerCount { get; }

    public int LayerCount => TextureLayers.Count > 0 ? TextureLayers.Count : DeclaredLayerCount;

    public ushort HoleMask { get; }

    public bool HasHoles => HoleMask != 0;

    public WorldTerrainHoleMask HoleMaskState { get; }

    public bool HasLiquidFlags { get; }

    public bool HasVertexColors { get; }

    public float[]? Heights { get; }

    public bool HasHeights => Heights is { Length: 145 };

    public Vector3[]? Normals { get; }

    public bool HasNormals => Normals is { Length: 145 };

    public IReadOnlyList<AdtTextureChunkLayer> TextureLayers { get; }

    public bool HasTextureLayers => TextureLayers.Count > 0;

    /// <summary>Expanded MCSH mask: 0 is lit and 255 is shadowed.</summary>
    public byte[]? ShadowMap { get; }

    public bool HasShadowMap => ShadowMap is { Length: 64 * 64 };

    public WorldTerrainCellGrid CellGrid { get; }
}
