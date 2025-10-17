using System;
using System.Collections.Generic;

namespace WoWRollback.LkToAlphaModule.Models;

/// <summary>
/// Strongly typed container describing all data required to build a Lich King MCNK chunk.
/// </summary>
public sealed class LkMcnkSource
{
    public int IndexX { get; init; }
    public int IndexY { get; init; }
    public uint Flags { get; init; }
    public uint AreaId { get; init; }
    public ushort HolesLowRes { get; init; }
    public float Radius { get; init; }
    public uint DoodadRefCount { get; init; }
    public uint MapObjectRefs { get; init; }
    public ulong NoEffectDoodad { get; init; }
    public uint OffsLiquid { get; init; }
    public uint OffsSndEmitters { get; init; }
    public uint SndEmitterCount { get; init; }

    public byte[] McvtRaw { get; set; } = Array.Empty<byte>();
    public byte[] McnrRaw { get; set; } = Array.Empty<byte>();
    public byte[] MclyRaw { get; set; } = Array.Empty<byte>();
    public byte[] McrfRaw { get; set; } = Array.Empty<byte>();
    public byte[] McshRaw { get; set; } = Array.Empty<byte>();
    public byte[] McseRaw { get; set; } = Array.Empty<byte>();

    public ushort[] PredictedTextures { get; init; } = new ushort[8];

    /// <summary>
    /// Alpha layers associated with MCLY entries. `LayerIndex` matches the 16-byte entry order.
    /// </summary>
    public List<LkMcnkAlphaLayer> AlphaLayers { get; } = new();

    public int LayerCount => MclyRaw.Length / 16;
}

public sealed class LkMcnkAlphaLayer
{
    public int LayerIndex { get; init; }
    public uint? OverrideFlags { get; init; }
    public byte[] ColumnMajorAlpha { get; init; } = Array.Empty<byte>();
}
