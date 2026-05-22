using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoRenderDocumentReader
{
    public static WmoRenderDocument Read(string path, Func<string, byte[]?>? assetReader = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path), assetReader);
    }

    public static WmoRenderDocument Read(Stream stream, string sourcePath = "<memory>", Func<string, byte[]?>? assetReader = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        WmoSummary summary = WmoSummaryReader.Read(stream, sourcePath);
        IReadOnlyList<WmoMaterialDetail> materials = WmoMaterialDetailReader.Read(stream, sourcePath);
        IReadOnlyList<WmoEmbeddedGroupMeshDetail> groups = LoadGroups(stream, sourcePath, summary, assetReader);
        IReadOnlyList<WmoPortalVertexDetail> portalVertices = summary.ReportedPortalCount > 0
            ? WmoPortalDetailReader.ReadVertices(stream, sourcePath)
            : [];
        IReadOnlyList<WmoPortalDetail> portals = summary.ReportedPortalCount > 0
            ? WmoPortalDetailReader.ReadPortals(stream, sourcePath)
            : [];
        IReadOnlyList<WmoPortalReferenceDetail> portalReferences = summary.ReportedPortalCount > 0
            ? WmoPortalDetailReader.ReadReferences(stream, sourcePath)
            : [];
        IReadOnlyList<WmoDoodadSetDetail> doodadSets = summary.ReportedDoodadSetCount > 0
            ? WmoDoodadDetailReader.ReadSets(stream, sourcePath)
            : [];
        IReadOnlyList<WmoDoodadPlacementDetail> doodadPlacements = summary.ReportedDoodadPlacementCount > 0
            ? WmoDoodadDetailReader.ReadPlacements(stream, sourcePath)
            : [];
        return new WmoRenderDocument(sourcePath, summary.Version, summary, materials, groups, portalVertices, portals, portalReferences, doodadSets, doodadPlacements);
    }

    private static IReadOnlyList<WmoEmbeddedGroupMeshDetail> LoadGroups(
        Stream stream,
        string sourcePath,
        WmoSummary summary,
        Func<string, byte[]?>? assetReader)
    {
        IReadOnlyList<WmoEmbeddedGroupMeshDetail> embeddedGroups = WmoEmbeddedGroupMeshDetailReader.Read(stream, sourcePath);
        if (embeddedGroups.Count > 0 || summary.ReportedGroupCount <= 0)
            return embeddedGroups;

        List<WmoEmbeddedGroupMeshDetail> externalGroups = new(summary.ReportedGroupCount);
        for (int groupIndex = 0; groupIndex < summary.ReportedGroupCount; groupIndex++)
        {
            string groupPath = BuildGroupPath(sourcePath, groupIndex);
            byte[]? groupBytes = TryReadGroupBytes(groupPath, assetReader);
            if (groupBytes is not { Length: > 0 })
                continue;

            try
            {
                using MemoryStream groupStream = new(groupBytes, writable: false);
                externalGroups.Add(WmoEmbeddedGroupMeshDetailReader.ReadGroup(groupStream, groupPath, groupIndex));
            }
            catch
            {
            }
        }

        return externalGroups.Count > 0 ? externalGroups : embeddedGroups;
    }

    private static string BuildGroupPath(string sourcePath, int groupIndex)
    {
        string baseName = Path.GetFileNameWithoutExtension(sourcePath);
        string groupName = $"{baseName}_{groupIndex:D3}.wmo";
        string? directory = Path.GetDirectoryName(sourcePath);
        return string.IsNullOrEmpty(directory) ? groupName : Path.Combine(directory, groupName);
    }

    private static byte[]? TryReadGroupBytes(string groupPath, Func<string, byte[]?>? assetReader)
    {
        if (assetReader is not null)
        {
            byte[]? assetBytes = assetReader(groupPath);
            if (assetBytes is { Length: > 0 })
                return assetBytes;
        }

        return File.Exists(groupPath) ? File.ReadAllBytes(groupPath) : null;
    }
}