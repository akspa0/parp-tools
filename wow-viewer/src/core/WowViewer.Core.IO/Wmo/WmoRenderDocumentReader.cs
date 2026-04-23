using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoRenderDocumentReader
{
    public static WmoRenderDocument Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoRenderDocument Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        WmoSummary summary = WmoSummaryReader.Read(stream, sourcePath);
        IReadOnlyList<WmoMaterialDetail> materials = WmoMaterialDetailReader.Read(stream, sourcePath);
        IReadOnlyList<WmoEmbeddedGroupMeshDetail> groups = WmoEmbeddedGroupMeshDetailReader.Read(stream, sourcePath);
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
}