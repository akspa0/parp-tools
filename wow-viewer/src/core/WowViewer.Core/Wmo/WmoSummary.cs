using System.Numerics;

namespace WowViewer.Core.Wmo;

public sealed class WmoSummary
{
    public WmoSummary(
        string sourcePath,
        uint? version,
        int reportedMaterialCount,
        int materialEntryCount,
        int reportedGroupCount,
        int groupInfoCount,
        int reportedPortalCount,
        int reportedLightCount,
        int textureNameCount,
        int reportedDoodadNameCount,
        int doodadNameTableCount,
        int reportedDoodadPlacementCount,
        int doodadPlacementEntryCount,
        int reportedDoodadSetCount,
        int doodadSetEntryCount,
        uint flags,
        Vector3 boundsMin,
        Vector3 boundsMax)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(reportedMaterialCount);
        ArgumentOutOfRangeException.ThrowIfNegative(materialEntryCount);
        ArgumentOutOfRangeException.ThrowIfNegative(reportedGroupCount);
        ArgumentOutOfRangeException.ThrowIfNegative(groupInfoCount);
        ArgumentOutOfRangeException.ThrowIfNegative(reportedPortalCount);
        ArgumentOutOfRangeException.ThrowIfNegative(reportedLightCount);
        ArgumentOutOfRangeException.ThrowIfNegative(textureNameCount);
        ArgumentOutOfRangeException.ThrowIfNegative(reportedDoodadNameCount);
        ArgumentOutOfRangeException.ThrowIfNegative(doodadNameTableCount);
        ArgumentOutOfRangeException.ThrowIfNegative(reportedDoodadPlacementCount);
        ArgumentOutOfRangeException.ThrowIfNegative(doodadPlacementEntryCount);
        ArgumentOutOfRangeException.ThrowIfNegative(reportedDoodadSetCount);
        ArgumentOutOfRangeException.ThrowIfNegative(doodadSetEntryCount);

        SourcePath = sourcePath;
        Version = version;
        ReportedMaterialCount = reportedMaterialCount;
        MaterialEntryCount = materialEntryCount;
        ReportedGroupCount = reportedGroupCount;
        GroupInfoCount = groupInfoCount;
        ReportedPortalCount = reportedPortalCount;
        ReportedLightCount = reportedLightCount;
        TextureNameCount = textureNameCount;
        ReportedDoodadNameCount = reportedDoodadNameCount;
        DoodadNameTableCount = doodadNameTableCount;
        ReportedDoodadPlacementCount = reportedDoodadPlacementCount;
        DoodadPlacementEntryCount = doodadPlacementEntryCount;
        ReportedDoodadSetCount = reportedDoodadSetCount;
        DoodadSetEntryCount = doodadSetEntryCount;
        Flags = flags;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int ReportedMaterialCount { get; }

    public int MaterialEntryCount { get; }

    public int ReportedGroupCount { get; }

    public int GroupInfoCount { get; }

    public int ReportedPortalCount { get; }

    public int ReportedLightCount { get; }

    public int TextureNameCount { get; }

    public int ReportedDoodadNameCount { get; }

    public int DoodadNameTableCount { get; }

    public int ReportedDoodadPlacementCount { get; }

    public int DoodadPlacementEntryCount { get; }

    public int ReportedDoodadSetCount { get; }

    public int DoodadSetEntryCount { get; }

    public uint Flags { get; }

    public Vector3 BoundsMin { get; }

    public Vector3 BoundsMax { get; }
}
