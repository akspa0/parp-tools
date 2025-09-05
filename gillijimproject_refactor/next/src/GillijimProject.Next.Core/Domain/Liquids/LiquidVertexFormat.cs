using System;

namespace GillijimProject.Next.Core.Domain.Liquids;

/// <summary>
/// Liquid vertex format (LVF) used by MH2O instances. Mirrors ADT v18 wiki cases.
/// </summary>
public enum LiquidVertexFormat : ushort
{
    /// <summary>Case 0: Height + Depth data (float heightmap, byte depthmap)</summary>
    HeightDepth = 0,

    /// <summary>Case 1: Height + UV data (float heightmap, uv_map_entry[]). TODO(PORT)</summary>
    HeightUv = 1,

    /// <summary>Case 2: Depth-only data (byte depthmap); height is 0.0</summary>
    DepthOnly = 2,

    /// <summary>Case 3: Height + UV + Depth data. TODO(PORT)</summary>
    HeightUvDepth = 3,
}
