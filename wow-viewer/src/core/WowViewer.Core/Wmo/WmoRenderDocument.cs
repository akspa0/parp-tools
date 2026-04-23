namespace WowViewer.Core.Wmo;

public sealed class WmoRenderDocument
{
    public WmoRenderDocument(
        string sourcePath,
        uint? version,
        WmoSummary summary,
        IReadOnlyList<WmoMaterialDetail> materials,
        IReadOnlyList<WmoEmbeddedGroupMeshDetail> groups,
        IReadOnlyList<WmoPortalVertexDetail> portalVertices,
        IReadOnlyList<WmoPortalDetail> portals,
        IReadOnlyList<WmoPortalReferenceDetail> portalReferences,
        IReadOnlyList<WmoDoodadSetDetail> doodadSets,
        IReadOnlyList<WmoDoodadPlacementDetail> doodadPlacements)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentNullException.ThrowIfNull(summary);
        ArgumentNullException.ThrowIfNull(materials);
        ArgumentNullException.ThrowIfNull(groups);
        ArgumentNullException.ThrowIfNull(portalVertices);
        ArgumentNullException.ThrowIfNull(portals);
        ArgumentNullException.ThrowIfNull(portalReferences);
        ArgumentNullException.ThrowIfNull(doodadSets);
        ArgumentNullException.ThrowIfNull(doodadPlacements);

        SourcePath = sourcePath;
        Version = version;
        Summary = summary;
        Materials = materials;
        Groups = groups;
        PortalVertices = portalVertices;
        Portals = portals;
        PortalReferences = portalReferences;
        DoodadSets = doodadSets;
        DoodadPlacements = doodadPlacements;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public WmoSummary Summary { get; }

    public IReadOnlyList<WmoMaterialDetail> Materials { get; }

    public IReadOnlyList<WmoEmbeddedGroupMeshDetail> Groups { get; }

    public IReadOnlyList<WmoPortalVertexDetail> PortalVertices { get; }

    public IReadOnlyList<WmoPortalDetail> Portals { get; }

    public IReadOnlyList<WmoPortalReferenceDetail> PortalReferences { get; }

    public IReadOnlyList<WmoDoodadSetDetail> DoodadSets { get; }

    public IReadOnlyList<WmoDoodadPlacementDetail> DoodadPlacements { get; }
}