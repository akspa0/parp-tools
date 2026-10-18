using System.Text;

namespace WowViewer.Core.Runtime.World;

/// <summary>
/// Ground-truth flag decoders for WDT, WMO, M2, and ADT formats, verified against
/// native client decompilation (WoW 4.3.4 build 15595).
/// </summary>
public static class FormatFlagsDescriber
{
    /// <summary>
    /// Decodes WDT MPHD header flags (SMMapHeader.m_flags).
    /// </summary>
    public static string DescribeWdtMphdFlags(uint flags)
    {
        if (flags == 0) return "None";
        var parts = new List<string>();

        if ((flags & 0x01) != 0) parts.Add("GlobalMapObj(0x1)");
        if ((flags & 0x02) != 0) parts.Add("VertexFormat2(0x2)");
        if ((flags & 0x04) != 0) parts.Add("EnvMapShader/MultiPass(0x4)");
        if ((flags & 0x08) != 0) parts.Add("SortedDoodads(0x8)");
        if ((flags & 0x10) != 0) parts.Add("VertexFormat16/Normals(0x10)");
        if ((flags & 0x20) != 0) parts.Add("FlipGround(0x20)");

        uint unk = flags & ~0x3Fu;
        if (unk != 0) parts.Add($"Unk=0x{unk:X}");

        return string.Join(", ", parts);
    }

    /// <summary>
    /// Decodes WDT MAIN tile chunk flags (SMAreaInfo.m_flags).
    /// </summary>
    public static string DescribeWdtMainFlags(uint flags)
    {
        if (flags == 0) return "Empty(0x0)";
        var parts = new List<string>();

        if ((flags & 0x01) != 0) parts.Add("HasAdt(0x1)");
        if ((flags & 0x02) != 0) parts.Add("InfiniteOcean(0x2)");

        uint unk = flags & ~0x03u;
        if (unk != 0) parts.Add($"Unk=0x{unk:X}");

        return string.Join(", ", parts);
    }

    /// <summary>
    /// Decodes root WMO header flags (SMOHeader.m_flags).
    /// </summary>
    public static string DescribeWmoRootFlags(uint flags)
    {
        if (flags == 0) return "None";
        var parts = new List<string>();

        if ((flags & 0x01) != 0) parts.Add("NoVertexAtten(0x1)");
        if ((flags & 0x02) != 0) parts.Add("UnifiedLighting(0x2)");
        if ((flags & 0x04) != 0) parts.Add("DirectLiquidType(0x4)");
        if ((flags & 0x08) != 0) parts.Add("PreserveMocvAlpha(0x8)");

        uint unk = flags & ~0x0Fu;
        if (unk != 0) parts.Add($"Unk=0x{unk:X}");

        return string.Join(", ", parts);
    }

    /// <summary>
    /// Decodes WMO group header flags (SMOGroupHeader.m_flags).
    /// </summary>
    public static string DescribeWmoGroupFlags(uint flags, string? name = null)
    {
        var parts = new List<string>();

        if ((flags & 0x0000001) != 0) parts.Add("HasBsp(0x1)");
        if ((flags & 0x0000004) != 0) parts.Add("HasMocv(0x4)");
        if ((flags & 0x0000008) != 0) parts.Add("Exterior(0x8)");
        if ((flags & 0x0000040) != 0) parts.Add("ExteriorLit(0x40)");
        if ((flags & 0x0000080) != 0) parts.Add("NoCollision/Occluder(0x80)");
        if ((flags & 0x0001000) != 0) parts.Add("HasLiquid(0x1000)");
        if ((flags & 0x0010000) != 0) parts.Add("AlwaysVisible(0x10000)");
        if ((flags & 0x0040000) != 0) parts.Add("Skybox(0x40000)");
        if ((flags & 0x0080000) != 0) parts.Add("Shader14(0x80000)");
        if ((flags & 0x4000000) != 0) parts.Add("Antiportal/Occluder(0x4M)");

        bool isAntiportalByName = !string.IsNullOrEmpty(name) && (
            name.Contains("antiportal", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("anti_portal", StringComparison.OrdinalIgnoreCase));
        if (isAntiportalByName && (flags & 0x4000000) == 0)
        {
            parts.Add("AntiportalByName");
        }

        uint knownMask = 0x0000001u | 0x0000004u | 0x0000008u | 0x0000040u | 0x0000080u
                       | 0x0001000u | 0x0010000u | 0x0040000u | 0x0080000u | 0x4000000u;
        uint unk = flags & ~knownMask;
        if (unk != 0) parts.Add($"Unk=0x{unk:X}");

        return parts.Count > 0 ? string.Join(", ", parts) : "None";
    }

    /// <summary>
    /// Decodes M2 model header flags (M2Data.m_flags).
    /// </summary>
    public static string DescribeM2ModelFlags(uint flags)
    {
        if (flags == 0) return "None";
        var parts = new List<string>();

        if ((flags & 0x03) == 0x03) parts.Add("SurfaceBasis(0x3)");
        else if ((flags & 0x01) != 0) parts.Add("TiltNormal/AlignUp(0x1)");

        if ((flags & 0x80) != 0) parts.Add("CastShadows(0x80)");

        uint unk = flags & ~0x83u;
        if (unk != 0) parts.Add($"Unk=0x{unk:X}");

        return string.Join(", ", parts);
    }

    /// <summary>
    /// Decodes M2 material flags (M2Material.m_flags).
    /// </summary>
    public static string DescribeM2MaterialFlags(ushort flags)
    {
        if (flags == 0) return "None";
        var parts = new List<string>();

        if ((flags & 0x001) != 0) parts.Add("Unlit(0x1)");
        if ((flags & 0x002) != 0) parts.Add("Unfogged(0x2)");
        if ((flags & 0x004) != 0) parts.Add("TwoSided(0x4)");
        if ((flags & 0x010) != 0) parts.Add("DepthWriteDisabled(0x10)");
        if ((flags & 0x020) != 0) parts.Add("HitTestTransparent(0x20)");
        if ((flags & 0x040) != 0) parts.Add("DisableShadowCasting(0x40)");
        if ((flags & 0x080) != 0) parts.Add("ShadowBatchCustom(0x80)");
        if ((flags & 0x100) != 0) parts.Add("DisableShadowReceiving(0x100)");
        if ((flags & 0x400) != 0) parts.Add("ShadowBatchUntextured(0x400)");

        uint unk = (uint)flags & ~0x5F7u;
        if (unk != 0) parts.Add($"Unk=0x{unk:X}");

        return string.Join(", ", parts);
    }

    /// <summary>
    /// Decodes ADT MCNK terrain chunk flags.
    /// </summary>
    public static string DescribeMcnkFlags(int rawFlags)
    {
        uint flags = unchecked((uint)rawFlags);
        if (flags == 0) return "None";
        var parts = new List<string>();

        if ((flags & 0x00000001) != 0) parts.Add("HasShadows(0x1)");
        if ((flags & 0x00000002) != 0) parts.Add("Impassable(0x2)");
        if ((flags & 0x00000004) != 0) parts.Add("River(0x4)");
        if ((flags & 0x00000008) != 0) parts.Add("Ocean(0x8)");
        if ((flags & 0x00000010) != 0) parts.Add("Magma(0x10)");
        if ((flags & 0x00000020) != 0) parts.Add("Slime(0x20)");
        if ((flags & 0x00000040) != 0) parts.Add("HasMccv(0x40)");
        if ((flags & 0x00008000) != 0) parts.Add("FixedAlpha(0x8000)");
        if ((flags & 0x00020000) != 0) parts.Add("HasBakedShadows(0x20000)");

        uint knownMask = 0x00000001u | 0x00000002u | 0x00000004u | 0x00000008u
                       | 0x00000010u | 0x00000020u | 0x00000040u | 0x00008000u
                       | 0x00020000u;
        uint unk = flags & ~knownMask;
        if (unk != 0) parts.Add($"Unk=0x{unk:X}");

        return string.Join(", ", parts);
    }
}
