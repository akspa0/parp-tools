using System.Numerics;

namespace WowViewer.Core.Wmo;

public sealed class WmoGroupSummary
{
    public WmoGroupSummary(
        string sourcePath,
        uint? version,
        int headerSizeBytes,
        uint nameOffset,
        uint descriptiveNameOffset,
        uint flags,
        Vector3 boundsMin,
        Vector3 boundsMax,
        int portalStart,
        int portalCount,
        int transparentBatchCount,
        int interiorBatchCount,
        int exteriorBatchCount,
        uint groupLiquid,
        int faceMaterialCount,
        int vertexCount,
        int indexCount,
        int normalCount,
        int primaryUvCount,
        int additionalUvSetCount,
        int batchCount,
        int vertexColorCount,
        int doodadRefCount,
        bool hasLiquid)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(headerSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(portalStart);
        ArgumentOutOfRangeException.ThrowIfNegative(portalCount);
        ArgumentOutOfRangeException.ThrowIfNegative(transparentBatchCount);
        ArgumentOutOfRangeException.ThrowIfNegative(interiorBatchCount);
        ArgumentOutOfRangeException.ThrowIfNegative(exteriorBatchCount);
        ArgumentOutOfRangeException.ThrowIfNegative(faceMaterialCount);
        ArgumentOutOfRangeException.ThrowIfNegative(vertexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(indexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(normalCount);
        ArgumentOutOfRangeException.ThrowIfNegative(primaryUvCount);
        ArgumentOutOfRangeException.ThrowIfNegative(additionalUvSetCount);
        ArgumentOutOfRangeException.ThrowIfNegative(batchCount);
        ArgumentOutOfRangeException.ThrowIfNegative(vertexColorCount);
        ArgumentOutOfRangeException.ThrowIfNegative(doodadRefCount);

        SourcePath = sourcePath;
        Version = version;
        HeaderSizeBytes = headerSizeBytes;
        NameOffset = nameOffset;
        DescriptiveNameOffset = descriptiveNameOffset;
        Flags = flags;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
        PortalStart = portalStart;
        PortalCount = portalCount;
        TransparentBatchCount = transparentBatchCount;
        InteriorBatchCount = interiorBatchCount;
        ExteriorBatchCount = exteriorBatchCount;
        GroupLiquid = groupLiquid;
        FaceMaterialCount = faceMaterialCount;
        VertexCount = vertexCount;
        IndexCount = indexCount;
        NormalCount = normalCount;
        PrimaryUvCount = primaryUvCount;
        AdditionalUvSetCount = additionalUvSetCount;
        BatchCount = batchCount;
        VertexColorCount = vertexColorCount;
        DoodadRefCount = doodadRefCount;
        HasLiquid = hasLiquid;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int HeaderSizeBytes { get; }

    public uint NameOffset { get; }

    public uint DescriptiveNameOffset { get; }

    public uint Flags { get; }

    public Vector3 BoundsMin { get; }

    public Vector3 BoundsMax { get; }

    public int PortalStart { get; }

    public int PortalCount { get; }

    public int TransparentBatchCount { get; }

    public int InteriorBatchCount { get; }

    public int ExteriorBatchCount { get; }

    public int DeclaredBatchCount => TransparentBatchCount + InteriorBatchCount + ExteriorBatchCount;

    public uint GroupLiquid { get; }

    public int FaceMaterialCount { get; }

    public int VertexCount { get; }

    public int IndexCount { get; }

    public int NormalCount { get; }

    public int PrimaryUvCount { get; }

    public int AdditionalUvSetCount { get; }

    public int BatchCount { get; }

    public int VertexColorCount { get; }

    public int DoodadRefCount { get; }

    public bool HasLiquid { get; }
}
