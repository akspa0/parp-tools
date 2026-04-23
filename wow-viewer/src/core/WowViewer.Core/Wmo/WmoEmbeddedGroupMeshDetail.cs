namespace WowViewer.Core.Wmo;

public sealed class WmoEmbeddedGroupMeshDetail
{
    public WmoEmbeddedGroupMeshDetail(
        int groupIndex,
        long groupHeaderOffset,
        WmoGroupSummary groupSummary,
        WmoGroupMeshDetail mesh,
        WmoGroupLiquidSummary? liquidSummary,
        IReadOnlyList<ushort> doodadRefs,
        IReadOnlyList<ushort> lightRefs)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(groupIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(groupHeaderOffset);
        ArgumentNullException.ThrowIfNull(groupSummary);
        ArgumentNullException.ThrowIfNull(mesh);
        ArgumentNullException.ThrowIfNull(doodadRefs);
        ArgumentNullException.ThrowIfNull(lightRefs);

        GroupIndex = groupIndex;
        GroupHeaderOffset = groupHeaderOffset;
        GroupSummary = groupSummary;
        Mesh = mesh;
        LiquidSummary = liquidSummary;
        DoodadRefs = doodadRefs;
        LightRefs = lightRefs;
    }

    public int GroupIndex { get; }

    public long GroupHeaderOffset { get; }

    public WmoGroupSummary GroupSummary { get; }

    public WmoGroupMeshDetail Mesh { get; }

    public WmoGroupLiquidSummary? LiquidSummary { get; }

    public IReadOnlyList<ushort> DoodadRefs { get; }

    public IReadOnlyList<ushort> LightRefs { get; }
}