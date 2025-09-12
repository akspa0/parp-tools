using System.Collections.Generic;

namespace GillijimProject.Next.Core.Domain.Liquids;

/// <summary>
/// Mapping between MCLQ tile types and LK LiquidType IDs. Default values are placeholders
/// and should be validated against LiquidType.dbc when available.
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

    /// <summary>
    /// Default mapping. Values are the MCLQ codes as placeholders; override via CLI mapping when available.
    /// </summary>
    public static LiquidTypeMapping CreateDefault() => new(new Dictionary<MclqLiquidType, ushort>
    {
        { MclqLiquidType.None, 0 },
        { MclqLiquidType.Ocean, 1 },
        { MclqLiquidType.River, 4 },
        { MclqLiquidType.Slime, 3 },
        { MclqLiquidType.Magma, 6 },
    });
}
