using System.Numerics;
using WowViewer.Core.IO.Files;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;
using WowViewer.Core.Wmo;
using WowViewer.Core.IO.Wmo;

internal sealed record WmoSurfaceBuildResult(
    string WmoPath,
    int GroupCount,
    SurfaceCorrelationFingerprint? RootFingerprint,
    IReadOnlyList<SurfaceCorrelationFingerprint> GroupFingerprints,
    IReadOnlyList<string> Warnings);

internal static class Pm4SurfaceBuildSupport
{
    public static WmoSurfaceBuildResult BuildWmoSurfaceFingerprints(
        string wmoPath,
        byte[] rootBytes,
        Func<string, byte[]?> assetReader,
        float edgeBinSize = 1.0f,
        float areaBinSize = 1.0f)
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
            return new WmoSurfaceBuildResult(wmoPath, 0, null, [], warnings);
        }

        int groupCount = renderDoc.Groups.Count;
        if (groupCount == 0)
        {
            warnings.Add("WMO has no groups.");
            return new WmoSurfaceBuildResult(wmoPath, 0, null, [], warnings);
        }

        List<SurfaceCorrelationFingerprint> groupFingerprints = [];

        foreach (WmoEmbeddedGroupMeshDetail group in renderDoc.Groups)
        {
            SurfaceCorrelationFingerprint? groupFp = Pm4SurfaceCorrelationExtractor.ExtractFromWmoGroup(
                group.Mesh.Vertices,
                group.Mesh.Indices,
                group.Mesh.FaceMaterials,
                group.GroupIndex,
                wmoPath,
                edgeBinSize,
                areaBinSize);

            if (groupFp is not null)
                groupFingerprints.Add(groupFp);
            else
                warnings.Add($"Group {group.GroupIndex} surface extraction failed (degenerate geometry).");
        }

        SurfaceCorrelationFingerprint? rootFp = Pm4SurfaceCorrelationExtractor.MergeWmoGroups(groupFingerprints, wmoPath, edgeBinSize);

        return new WmoSurfaceBuildResult(wmoPath, groupCount, rootFp, groupFingerprints, warnings);
    }

    public static SurfaceCorrelationDatabase BuildSurfaceDatabase(
        string archiveRoot,
        ArchiveCatalogBootstrapOptions bootstrapOptions,
        IReadOnlyList<string> wmoPaths,
        float edgeBinSize = 1.0f,
        float areaBinSize = 1.0f,
        Action<string>? progress = null)
    {
        List<SurfaceCorrelationFingerprint> records = [];
        int processed = 0;
        int succeeded = 0;
        int failed = 0;

        Func<string, byte[]?> assetReader = virtualPath =>
        {
            try
            {
                return ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, [archiveRoot], bootstrapOptions);
            }
            catch { return null; }
        };

        foreach (string wmoPath in wmoPaths)
        {
            processed++;
            try
            {
                string normalizedPath = wmoPath.Replace('\\', '/').TrimStart('/').ToLowerInvariant();
                byte[] rootBytes = ArchiveVirtualFileReader.ReadVirtualFile(
                    normalizedPath, [archiveRoot], bootstrapOptions);

                WmoSurfaceBuildResult result = BuildWmoSurfaceFingerprints(wmoPath, rootBytes, assetReader, edgeBinSize, areaBinSize);

                if (result.RootFingerprint is not null)
                {
                    records.Add(result.RootFingerprint);
                    succeeded++;
                }
                foreach (SurfaceCorrelationFingerprint g in result.GroupFingerprints)
                    records.Add(g);
            }
            catch (Exception ex) when (ex is FileNotFoundException or InvalidDataException or IOException)
            {
                failed++;
            }

            if (processed % 200 == 0)
                progress?.Invoke($"  Processed {processed}/{wmoPaths.Count} WMOs ({succeeded} succeeded, {failed} failed)...");
        }

        progress?.Invoke($"Processed {processed} WMOs: {succeeded} succeeded, {failed} failed, {records.Count} surface fingerprints.");

        return new SurfaceCorrelationDatabase(
            archiveRoot,
            DateTime.UtcNow.ToString("o"),
            succeeded,
            edgeBinSize,
            areaBinSize,
            records);
    }

    public static List<SurfaceCorrelationFingerprint> ExtractPm4SurfaceFingerprints(
        string pm4Dir,
        float edgeBinSize = 1.0f,
        float areaBinSize = 1.0f,
        Action<string>? progress = null)
    {
        string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(pm4Dir);
        string[] pm4Files = Directory.GetFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly);

        List<SurfaceCorrelationFingerprint> fingerprints = [];
        int processed = 0;
        int groupsFound = 0;

        foreach (string pm4File in pm4Files.OrderBy(Path.GetFileName))
        {
            if (!Pm4CoordinateService.TryParseTileCoordinates(pm4File, out int tileX, out int tileY))
                continue;

            try
            {
                Pm4ResearchDocument doc = Pm4ResearchReader.ReadFile(pm4File);
                IReadOnlyList<Pm4MsurEntry> msur = doc.KnownChunks.Msur;
                IReadOnlyList<uint> msvi = doc.KnownChunks.Msvi;
                IReadOnlyList<Vector3> msvt = doc.KnownChunks.Msvt;

                var groups = msur
                    .Where(static s => s.Ck24 != 0 && s.IndexCount >= 3)
                    .GroupBy(static s => s.Ck24);

                foreach (IGrouping<uint, Pm4MsurEntry> group in groups)
                {
                    List<Pm4MsurEntry> surfaces = group.ToList();
                    byte ck24Type = surfaces[0].Ck24Type;
                    uint ck24 = group.Key;
                    string assetId = $"tile{tileX}_{tileY}_ck24_0x{ck24:X6}";
                    string assetPath = Path.GetFileName(pm4File);
                    string assetKind = ck24Type switch
                    {
                        0x42 or 0x43 or 0xC0 or 0xC1 or 0xC2 or 0xC3 => "wmo",
                        0x40 or 0x41 => "m2",
                        _ => "unknown",
                    };

                    SurfaceCorrelationFingerprint? fp = Pm4SurfaceCorrelationExtractor.ExtractFromPm4Group(
                        msvt, msvi, surfaces, ck24, ck24Type, assetId, assetPath, assetKind, edgeBinSize, areaBinSize);

                    if (fp is not null)
                    {
                        fingerprints.Add(fp);
                        groupsFound++;
                    }
                }

                processed++;
                if (processed % 100 == 0)
                    progress?.Invoke($"  Processed {processed}/{pm4Files.Length}...");
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error reading {Path.GetFileName(pm4File)}: {ex.Message}");
            }
        }

        progress?.Invoke($"Processed {processed} PM4 files, {groupsFound} surface fingerprints extracted.");
        return fingerprints;
    }
}
