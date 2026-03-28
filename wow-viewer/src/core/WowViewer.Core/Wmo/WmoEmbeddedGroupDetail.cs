namespace WowViewer.Core.Wmo;

public sealed class WmoEmbeddedGroupDetail
{
    public WmoEmbeddedGroupDetail(
        int groupIndex,
        long groupHeaderOffset,
        WmoGroupSummary groupSummary,
        WmoGroupLiquidSummary? liquidSummary,
        WmoGroupBatchSummary? batchSummary,
        WmoGroupFaceMaterialSummary? faceMaterialSummary,
        WmoGroupUvSummary? uvSummary,
        WmoGroupVertexColorSummary? vertexColorSummary,
        WmoGroupDoodadRefSummary? doodadRefSummary,
        WmoGroupLightRefSummary? lightRefSummary,
        WmoGroupIndexSummary? indexSummary,
        WmoGroupVertexSummary? vertexSummary,
        WmoGroupNormalSummary? normalSummary,
        WmoGroupBspNodeSummary? bspNodeSummary,
        WmoGroupBspFaceSummary? bspFaceSummary,
        WmoGroupBspFaceRangeSummary? bspFaceRangeSummary)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(groupIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(groupHeaderOffset);
        ArgumentNullException.ThrowIfNull(groupSummary);

        GroupIndex = groupIndex;
        GroupHeaderOffset = groupHeaderOffset;
        GroupSummary = groupSummary;
        LiquidSummary = liquidSummary;
        BatchSummary = batchSummary;
        FaceMaterialSummary = faceMaterialSummary;
        UvSummary = uvSummary;
        VertexColorSummary = vertexColorSummary;
        DoodadRefSummary = doodadRefSummary;
        LightRefSummary = lightRefSummary;
        IndexSummary = indexSummary;
        VertexSummary = vertexSummary;
        NormalSummary = normalSummary;
        BspNodeSummary = bspNodeSummary;
        BspFaceSummary = bspFaceSummary;
        BspFaceRangeSummary = bspFaceRangeSummary;
    }

    public int GroupIndex { get; }

    public long GroupHeaderOffset { get; }

    public WmoGroupSummary GroupSummary { get; }

    public WmoGroupLiquidSummary? LiquidSummary { get; }

    public WmoGroupBatchSummary? BatchSummary { get; }

    public WmoGroupFaceMaterialSummary? FaceMaterialSummary { get; }

    public WmoGroupUvSummary? UvSummary { get; }

    public WmoGroupVertexColorSummary? VertexColorSummary { get; }

    public WmoGroupDoodadRefSummary? DoodadRefSummary { get; }

    public WmoGroupLightRefSummary? LightRefSummary { get; }

    public WmoGroupIndexSummary? IndexSummary { get; }

    public WmoGroupVertexSummary? VertexSummary { get; }

    public WmoGroupNormalSummary? NormalSummary { get; }

    public WmoGroupBspNodeSummary? BspNodeSummary { get; }

    public WmoGroupBspFaceSummary? BspFaceSummary { get; }

    public WmoGroupBspFaceRangeSummary? BspFaceRangeSummary { get; }
}