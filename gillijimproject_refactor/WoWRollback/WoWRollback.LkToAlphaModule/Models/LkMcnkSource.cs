using System;
using System.Collections.Generic;

namespace WoWRollback.LkToAlphaModule.Models;

/// <summary>
/// Strongly typed container describing all data required to build a Lich King MCNK chunk while retaining Alpha parity information.
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
    public ulong NoEffectDoodadMask { get; init; }
    public byte NoEffectDoodad { get; init; }
    public uint OffsSndEmitters { get; init; }
    public uint SndEmitterCount { get; init; }
    public uint OffsLiquid { get; init; }

    public byte[] McvtRaw { get; set; } = Array.Empty<byte>();
    public byte[] McnrRaw { get; set; } = Array.Empty<byte>();
    public byte[] MclyRaw { get; set; } = Array.Empty<byte>();
    public byte[] McrfRaw { get; set; } = Array.Empty<byte>();
    public byte[] McshRaw { get; set; } = Array.Empty<byte>();
    public byte[] McseRaw { get; set; } = Array.Empty<byte>();
    public byte[] MclqRaw { get; set; } = Array.Empty<byte>();

    public List<int> DoodadReferenceIndices { get; } = new();
    public List<int> WmoReferenceIndices { get; } = new();

    public ushort[] PredictedTextures { get; init; } = new ushort[8];
    public byte[] NoEffectDoodadFlags { get; init; } = new byte[8];

    public List<byte[]> LiquidHeightMaps { get; } = new();
    public List<byte[]> LiquidFlags { get; } = new();
    public List<byte[]> SoundEmitters { get; } = new();

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
