using System;
using System.Collections.Generic;
using System.Linq;

namespace WoWRollback.LkToAlphaModule.Liquids;

/// <summary>
/// MCLQ tile liquid types (lower nibble values).
/// </summary>
public enum MclqLiquidType : byte
{
    None = 0,
    Ocean = 1,
    River = 4,
    Slime = 3,
    Magma = 6,
}

/// <summary>
/// MCLQ per-tile flags (upper nibble values).
/// Matches Alpha-era conventions: fatigue, no render, etc.
/// </summary>
[Flags]
public enum MclqTileFlags : byte
{
    None = 0x00,
    Fatigue = 0x10,
    NoRender = 0x20,
    ForcedSwim = 0x40,
    Unk80 = 0x80,
}

/// <summary>
/// MCLQ payload (9x9 vertex grids + 8x8 tile metadata) for a single MCNK.
/// </summary>
public sealed class MclqData
{
    public const int VertexGrid = 9;
    public const int TileGrid = 8;

    public float[] Heights { get; }
    public byte[] Depth { get; }
    public MclqLiquidType[,] Types { get; }
    public MclqTileFlags[,] Flags { get; }

    public MclqData(float[] heights, byte[] depth, MclqLiquidType[,] types, MclqTileFlags[,] flags)
    {
        Heights = heights ?? throw new ArgumentNullException(nameof(heights));
        Depth = depth ?? throw new ArgumentNullException(nameof(depth));
        Types = types ?? throw new ArgumentNullException(nameof(types));
        Flags = flags ?? throw new ArgumentNullException(nameof(flags));
        ValidateDimensions();
    }

    private void ValidateDimensions()
    {
        if (Heights.Length != VertexGrid * VertexGrid)
            throw new ArgumentException("Heights must be 9x9 = 81 elements", nameof(Heights));
        if (Depth.Length != VertexGrid * VertexGrid)
            throw new ArgumentException("Depth must be 9x9 = 81 elements", nameof(Depth));
        if (Types.GetLength(0) != TileGrid || Types.GetLength(1) != TileGrid)
            throw new ArgumentException("Types must be 8x8", nameof(Types));
        if (Flags.GetLength(0) != TileGrid || Flags.GetLength(1) != TileGrid)
            throw new ArgumentException("Flags must be 8x8", nameof(Flags));
    }
}

/// <summary>
/// Liquid vertex format constants (mirrors ADT v18 cases).
/// </summary>
public enum LiquidVertexFormat : ushort
{
    HeightDepth = 0,
    HeightUv = 1,
    DepthOnly = 2,
    HeightUvDepth = 3,
}

/// <summary>
/// Mapping between MCLQ tile types and LK LiquidType IDs.
/// </summary>
public sealed class LiquidTypeMapping
{
    private readonly Dictionary<MclqLiquidType, ushort> _mclqToLiquidType = new();
    private readonly Dictionary<ushort, MclqLiquidType> _liquidTypeToMclq = new();

    public LiquidTypeMapping(Dictionary<MclqLiquidType, ushort> mclqToLiquidType)
    {
        foreach (var kv in mclqToLiquidType)
        {
            _mclqToLiquidType[kv.Key] = kv.Value;
            _liquidTypeToMclq[kv.Value] = kv.Key;
        }
    }

    public ushort ToLiquidTypeId(MclqLiquidType type)
        => _mclqToLiquidType.TryGetValue(type, out var id) ? id : (ushort)0;

    public MclqLiquidType ToMclqType(ushort liquidTypeId)
        => _liquidTypeToMclq.TryGetValue(liquidTypeId, out var t) ? t : MclqLiquidType.None;

    public static LiquidTypeMapping CreateDefault() => new(new Dictionary<MclqLiquidType, ushort>
    {
        { MclqLiquidType.None, 0 },
        { MclqLiquidType.Ocean, 1 },
        { MclqLiquidType.River, 4 },
        { MclqLiquidType.Slime, 3 },
        { MclqLiquidType.Magma, 6 },
    });
}

/// <summary>
/// Options controlling liquid conversion behavior.
/// </summary>
public sealed class LiquidsOptions
{
    public bool EnableLiquids { get; init; } = true;

    public IReadOnlyList<MclqLiquidType> Precedence { get; init; }
        = new[] { MclqLiquidType.Magma, MclqLiquidType.Slime, MclqLiquidType.River, MclqLiquidType.Ocean };

    public bool GreenLava { get; init; } = false;

    public LiquidTypeMapping Mapping { get; init; } = LiquidTypeMapping.CreateDefault();

    public LiquidsOptions WithMapping(LiquidTypeMapping mapping) => new()
    {
        EnableLiquids = EnableLiquids,
        Precedence = Precedence.ToArray(),
        GreenLava = GreenLava,
        Mapping = mapping
    };
}

/// <summary>
/// MH2O per-chunk attributes (8x8 bitmasks).
/// </summary>
public sealed record Mh2oAttributes(ulong FishableMask, ulong DeepMask);

/// <summary>
/// MH2O liquid instance rectangle and vertex data.
/// </summary>
public sealed class Mh2oInstance
{
    public ushort LiquidTypeId { get; init; }
    public LiquidVertexFormat Lvf { get; init; }
    public float MinHeightLevel { get; init; }
    public float MaxHeightLevel { get; init; }
    public byte XOffset { get; init; }
    public byte YOffset { get; init; }
    public byte Width { get; init; }
    public byte Height { get; init; }
    public byte[]? ExistsBitmap { get; init; }
    public float[]? HeightMap { get; init; }
    public byte[]? DepthMap { get; init; }
    public int VertexCount => (Width + 1) * (Height + 1);

    public void Validate()
    {
        if (Width is < 1 or > 8) throw new ArgumentOutOfRangeException(nameof(Width));
        if (Height is < 1 or > 8) throw new ArgumentOutOfRangeException(nameof(Height));
        if (XOffset > 7) throw new ArgumentOutOfRangeException(nameof(XOffset));
        if (YOffset > 7) throw new ArgumentOutOfRangeException(nameof(YOffset));
        if (XOffset + Width > 8) throw new ArgumentOutOfRangeException(null, "XOffset+Width must be <= 8");
        if (YOffset + Height > 8) throw new ArgumentOutOfRangeException(null, "YOffset+Height must be <= 8");

        int v = VertexCount;
        switch (Lvf)
        {
            case LiquidVertexFormat.HeightDepth:
                if (HeightMap is null || DepthMap is null)
                    throw new InvalidOperationException("HeightDepth requires both HeightMap and DepthMap");
                if (HeightMap.Length != v) throw new ArgumentException("HeightMap length mismatch", nameof(HeightMap));
                if (DepthMap.Length != v) throw new ArgumentException("DepthMap length mismatch", nameof(DepthMap));
                break;
            case LiquidVertexFormat.DepthOnly:
                if (DepthMap is null) throw new InvalidOperationException("DepthOnly requires DepthMap");
                if (DepthMap.Length != v) throw new ArgumentException("DepthMap length mismatch", nameof(DepthMap));
                if (HeightMap is not null && HeightMap.Length != v)
                    throw new ArgumentException("HeightMap length mismatch", nameof(HeightMap));
                break;
            case LiquidVertexFormat.HeightUv:
            case LiquidVertexFormat.HeightUvDepth:
                // UV handling deferred for initial implementation.
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(Lvf));
        }

        if (ExistsBitmap is not null)
        {
            int expectedBits = Width * Height;
            int expectedBytes = (expectedBits + 7) / 8;
            if (ExistsBitmap.Length != expectedBytes)
                throw new ArgumentException("ExistsBitmap length mismatch", nameof(ExistsBitmap));
        }
    }
}

/// <summary>
/// MH2O chunk container (per MCNK) comprising optional attributes and instances.
/// </summary>
public sealed class Mh2oChunk
{
    public List<Mh2oInstance> Instances { get; } = new();
    public Mh2oAttributes? Attributes { get; init; }
    public bool IsEmpty => Instances.Count == 0;

    public void Add(Mh2oInstance instance)
    {
        instance.Validate();
        Instances.Add(instance);
    }
}
