using System.Numerics;
using WowViewer.Core.IO.Files;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;
using WowViewer.Core.Wmo;
using WowViewer.Core.IO.Wmo;

internal sealed record WmoFingerprintBuildResult(
    string WmoPath,
    int GroupCount,
    Pm4FingerprintRecord? RootFingerprint,
    IReadOnlyList<Pm4FingerprintRecord> GroupFingerprints,
    IReadOnlyList<string> Warnings);

internal static class Pm4FingerprintBuildSupport
{
    public static WmoFingerprintBuildResult BuildWmoFingerprints(
        string wmoPath,
        byte[] rootBytes,
        Func<string, byte[]?> assetReader)
    {
        List<string> warnings = [];

        WmoRenderDocument renderDoc;
        try
        {
            using MemoryStream rootStream = new(rootBytes, writable: false);
            renderDoc = WmoRenderDocumentReader.Read(rootStream, wmoPath, assetReader);
        }
        catch (Exception ex) when (ex is InvalidDataException or IOException)
        {
            warnings.Add($"WMO read failed: {ex.Message}");
            return new WmoFingerprintBuildResult(wmoPath, 0, null, [], warnings);
        }

        int groupCount = renderDoc.Groups.Count;
        if (groupCount == 0)
        {
            warnings.Add("WMO has no groups.");
            return new WmoFingerprintBuildResult(wmoPath, 0, null, [], warnings);
        }

        List<Pm4FingerprintRecord> groupFingerprints = [];
        List<Vector3> mergedVertices = [];
        List<int> mergedIndices = [];
        int vertexOffset = 0;

        foreach (WmoEmbeddedGroupMeshDetail group in renderDoc.Groups)
        {
            IReadOnlyList<Vector3> groupVerts = group.Mesh.Vertices;
            IReadOnlyList<ushort> groupIndices = group.Mesh.Indices;

            if (groupVerts.Count < 3 || groupIndices.Count < 3)
            {
                warnings.Add($"Group {group.GroupIndex} has insufficient geometry ({groupVerts.Count} verts, {groupIndices.Count} indices).");
                continue;
            }

            int[] groupIndicesInt = new int[groupIndices.Count];
            for (int i = 0; i < groupIndices.Count; i++)
                groupIndicesInt[i] = groupIndices[i];

            Pm4FingerprintRecord? groupFp = Pm4FingerprintExtractor.ExtractFromGeometry(
                groupVerts,
                groupIndicesInt,
                surfaceCount: group.Mesh.FaceMaterials.Count,
                ck24Type: 0x42,
                typeFlagsProfile: new Dictionary<byte, int>(),
                assetId: $"{wmoPath}#group{group.GroupIndex}",
                assetPath: wmoPath,
                assetKind: "wmo",
                groupCount: 1,
                sourceLabel: $"wmo-group-{group.GroupIndex}");

            if (groupFp is not null)
                groupFingerprints.Add(groupFp);
            else
                warnings.Add($"Group {group.GroupIndex} fingerprint extraction failed (degenerate geometry).");

            foreach (Vector3 v in groupVerts)
                mergedVertices.Add(v);
            foreach (ushort idx in groupIndices)
                mergedIndices.Add(idx + vertexOffset);
            vertexOffset += groupVerts.Count;
        }

        Pm4FingerprintRecord? rootFp = null;
        if (mergedVertices.Count >= 3 && mergedIndices.Count >= 3)
        {
            int totalFaces = renderDoc.Groups.Sum(static g => g.Mesh.FaceMaterials.Count);
            rootFp = Pm4FingerprintExtractor.ExtractFromGeometry(
                mergedVertices,
                mergedIndices,
                surfaceCount: totalFaces,
                ck24Type: 0x42,
                typeFlagsProfile: new Dictionary<byte, int>(),
                assetId: wmoPath,
                assetPath: wmoPath,
                assetKind: "wmo",
                groupCount: groupCount,
                sourceLabel: "wmo-root-merged");
        }
        else
        {
            warnings.Add("Merged root geometry has insufficient vertices/indices for fingerprint.");
        }

        return new WmoFingerprintBuildResult(wmoPath, groupCount, rootFp, groupFingerprints, warnings);
    }

    public static Pm4FingerprintDatabase BuildDatabase(
        string archiveRoot,
        ArchiveCatalogBootstrapOptions bootstrapOptions,
        IReadOnlyList<string> wmoPaths,
        Action<string>? progress = null)
    {
        List<Pm4FingerprintRecord> records = [];
        int processed = 0;
        int succeeded = 0;
        int failed = 0;

        Func<string, byte[]?> assetReader = virtualPath =>
        {
            try
            {
                return ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, [archiveRoot], bootstrapOptions);
            }
            catch
            {
                return null;
            }
        };

        foreach (string wmoPath in wmoPaths)
        {
            processed++;
            try
            {
                string normalizedPath = wmoPath.Replace('\\', '/').TrimStart('/').ToLowerInvariant();
                byte[] rootBytes = ArchiveVirtualFileReader.ReadVirtualFile(
                    normalizedPath, [archiveRoot], bootstrapOptions);

                WmoFingerprintBuildResult result = BuildWmoFingerprints(wmoPath, rootBytes, assetReader);

                if (result.RootFingerprint is not null)
                {
                    records.Add(result.RootFingerprint);
                    succeeded++;
                }
                foreach (Pm4FingerprintRecord groupFp in result.GroupFingerprints)
                    records.Add(groupFp);

                if (result.Warnings.Count > 0 && succeeded % 100 == 0)
                    progress?.Invoke($"  Warnings on {wmoPath}: {string.Join("; ", result.Warnings.Take(2))}");
            }
            catch (Exception ex) when (ex is FileNotFoundException or InvalidDataException or IOException)
            {
                failed++;
            }

            if (processed % 200 == 0)
                progress?.Invoke($"  Processed {processed}/{wmoPaths.Count} WMOs ({succeeded} succeeded, {failed} failed)...");
        }

        progress?.Invoke($"Processed {processed} WMOs: {succeeded} succeeded, {failed} failed, {records.Count} fingerprints.");

        return new Pm4FingerprintDatabase(
            archiveRoot,
            DateTime.UtcNow.ToString("o"),
            succeeded,
            records);
    }
}
