using System.Numerics;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;
using WowViewer.Core.Wmo;

internal sealed record Pm4GeneratorValidationResult(
    string Pm4Path,
    string AdtPath,
    int RealCk24GroupCount,
    int AdtWmoPlacementCount,
    int GeneratedGroupCount,
    int MatchedGroupCount,
    double MeanSymmetricScore,
    double MeanPm4Coverage,
    double MeanWmoCoverage,
    IReadOnlyList<Pm4GeneratorGroupValidation> GroupValidations,
    IReadOnlyList<string> Warnings);

internal sealed record Pm4GeneratorGroupValidation(
    string WmoPath,
    Vector3 Position,
    Vector3 Rotation,
    float Scale,
    int GeneratedTriangleCount,
    int GeneratedSurfaceCount,
    uint? MatchedRealCk24,
    double? SymmetricScore,
    double? Pm4Coverage,
    double? WmoCoverage,
    string Status);

internal static class Pm4GeneratorValidationSupport
{
    public static Pm4GeneratorValidationResult ValidateTile(
        string pm4Path,
        string adtPath,
        string archiveRoot,
        ArchiveCatalogBootstrapOptions bootstrapOptions,
        float edgeBinSize = 1.0f,
        float areaBinSize = 1.0f,
        float normalBinSize = 0.1f,
        float heightBinSize = 1.0f,
        Action<string>? progress = null)
    {
        List<string> warnings = [];

        Pm4ResearchDocument realDoc = Pm4ResearchReader.ReadFile(pm4Path);
        IReadOnlyList<SurfaceFingerprint> realFingerprints = ExtractPm4Fingerprints(
            realDoc, edgeBinSize, areaBinSize, normalBinSize, heightBinSize);

        progress?.Invoke($"Real PM4: {realFingerprints.Count} CK24 groups, {realFingerprints.Sum(static f => f.TriangleCount)} triangles.");

        AdtPlacementCatalog placements = AdtPlacementReader.Read(adtPath);
        progress?.Invoke($"ADT placements: {placements.WorldModelPlacements.Count} WMOs, {placements.ModelPlacements.Count} M2s.");

        List<SurfaceFingerprint> generatedFingerprints = [];
        List<Pm4GeneratorGroupValidation> groupValidations = [];

        int processed = 0;
        foreach (AdtWorldModelPlacement placement in placements.WorldModelPlacements)
        {
            processed++;
            try
            {
                SurfaceFingerprint? generated = GenerateWmoPlacementFingerprint(
                    placement, archiveRoot, bootstrapOptions, edgeBinSize, areaBinSize, normalBinSize, heightBinSize, warnings);

                if (generated is null)
                {
                    groupValidations.Add(new Pm4GeneratorGroupValidation(
                        placement.ModelPath, placement.Position, placement.Rotation, 1f,
                        0, 0, null, null, null, null, "no_geometry"));
                    continue;
                }

                generatedFingerprints.Add(generated);

                (SurfaceFingerprint? bestReal, double score, double pm4Coverage, double wmoCoverage) = FindBestMatch(generated, realFingerprints);

                uint? matchedCk24 = bestReal?.Ck24;
                string status = score >= 0.50 ? "matched" : bestReal is not null ? "subthreshold" : "no_overlap";

                groupValidations.Add(new Pm4GeneratorGroupValidation(
                    placement.ModelPath, placement.Position, placement.Rotation, 1f,
                    generated.TriangleCount, generated.SurfaceCount, matchedCk24, score, pm4Coverage, wmoCoverage, status));

                if (processed % 50 == 0)
                    progress?.Invoke($"  Processed {processed}/{placements.WorldModelPlacements.Count} WMO placements...");
            }
            catch (Exception ex) when (ex is FileNotFoundException or InvalidDataException or IOException)
            {
                warnings.Add($"Placement '{placement.ModelPath}': {ex.Message}");
                groupValidations.Add(new Pm4GeneratorGroupValidation(
                    placement.ModelPath, placement.Position, placement.Rotation, 1f,
                    0, 0, null, null, null, null, "error"));
            }
        }

        int matchedCount = groupValidations.Count(static g => g.Status == "matched");
        double meanScore = groupValidations
            .Where(static g => g.SymmetricScore.HasValue)
            .Select(static g => g.SymmetricScore!.Value)
            .DefaultIfEmpty(0)
            .Average();
        double meanPm4Coverage = groupValidations
            .Where(static g => g.Pm4Coverage.HasValue)
            .Select(static g => g.Pm4Coverage!.Value)
            .DefaultIfEmpty(0)
            .Average();
        double meanWmoCoverage = groupValidations
            .Where(static g => g.WmoCoverage.HasValue)
            .Select(static g => g.WmoCoverage!.Value)
            .DefaultIfEmpty(0)
            .Average();

        return new Pm4GeneratorValidationResult(
            pm4Path,
            adtPath,
            realFingerprints.Count,
            placements.WorldModelPlacements.Count,
            generatedFingerprints.Count,
            matchedCount,
            meanScore,
            meanPm4Coverage,
            meanWmoCoverage,
            groupValidations,
            warnings);
    }

    private static SurfaceFingerprint? GenerateWmoPlacementFingerprint(
        AdtWorldModelPlacement placement,
        string archiveRoot,
        ArchiveCatalogBootstrapOptions bootstrapOptions,
        float edgeBinSize,
        float areaBinSize,
        float normalBinSize,
        float heightBinSize,
        List<string> warnings)
    {
        string normalizedPath = placement.ModelPath.Replace('\\', '/').Trim().TrimStart('/').ToLowerInvariant();

        byte[] wmoBytes = ArchiveVirtualFileReader.ReadVirtualFile(normalizedPath, [archiveRoot], bootstrapOptions);

        Func<string, byte[]?> assetReader = virtualPath =>
        {
            try
            {
                return ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, [archiveRoot], bootstrapOptions);
            }
            catch { return null; }
        };

        WmoRenderDocument renderDoc = WmoRenderDocumentReader.Read(
            new MemoryStream(wmoBytes, writable: false), normalizedPath, assetReader);

        List<Vector3> allVerts = [];
        List<ushort> allIndices = [];
        int vertOffset = 0;

        foreach (WmoEmbeddedGroupMeshDetail group in renderDoc.Groups)
        {
            foreach (Vector3 v in group.Mesh.Vertices)
                allVerts.Add(v);

            foreach (ushort idx in group.Mesh.Indices)
                allIndices.Add((ushort)(idx + vertOffset));

            vertOffset += group.Mesh.Vertices.Count;
        }

        if (allVerts.Count == 0 || allIndices.Count < 3)
            return null;

        Pm4GenerationData genData = Pm4Generator.GenerateFromCollisionMesh(
            allVerts, allIndices, placement.Position, placement.Rotation, scale: 1f,
            ck24Type: 0x43, ck24ObjectId: 1, regionId: 0);

        if (genData.Msur.Count == 0)
            return null;

        List<TriangleFeature> triangles = [];
        foreach (Pm4GenerationMsur surface in genData.Msur)
        {
            int first = (int)surface.MsviFirstIndex / 4;
            int count = surface.IndexCount;
            if (first + count > genData.Msvi.Count)
                continue;

            for (int i = 1; i < count - 1; i++)
            {
                Vector3 a = genData.Msvt[(int)genData.Msvi[first]];
                Vector3 b = genData.Msvt[(int)genData.Msvi[first + i]];
                Vector3 c = genData.Msvt[(int)genData.Msvi[first + i + 1]];
                triangles.Add(TriangleFeature.FromTriangle(a, b, c));
            }
        }

        if (triangles.Count == 0)
            return null;

        Dictionary<string, int> histogram = BuildAbsoluteHistogram(triangles, edgeBinSize, areaBinSize, normalBinSize, heightBinSize);

        return new SurfaceFingerprint(
            0x43,
            1,
            placement.ModelPath,
            genData.Msur.Count,
            triangles.Count,
            histogram);
    }

    private static IReadOnlyList<SurfaceFingerprint> ExtractPm4Fingerprints(
        Pm4ResearchDocument doc,
        float edgeBinSize,
        float areaBinSize,
        float normalBinSize,
        float heightBinSize)
    {
        IReadOnlyList<Pm4MsurEntry> msur = doc.KnownChunks.Msur;
        IReadOnlyList<uint> msvi = doc.KnownChunks.Msvi;
        IReadOnlyList<Vector3> msvt = doc.KnownChunks.Msvt;

        List<SurfaceFingerprint> fingerprints = [];

        var groups = msur
            .Where(static s => s.Ck24 != 0 && s.IndexCount >= 3)
            .GroupBy(static s => s.Ck24);

        foreach (IGrouping<uint, Pm4MsurEntry> group in groups)
        {
            List<Pm4MsurEntry> surfaces = group.ToList();
            byte ck24Type = surfaces[0].Ck24Type;
            uint ck24 = group.Key;

            List<TriangleFeature> triangles = [];
            foreach (Pm4MsurEntry surface in surfaces)
            {
                int first = checked((int)surface.MsviFirstIndex);
                int count = surface.IndexCount;
                if (first + count > msvi.Count)
                    continue;

                List<int> localVerts = [];
                for (int i = 0; i < count; i++)
                {
                    int vi = checked((int)msvi[first + i]);
                    if ((uint)vi < (uint)msvt.Count)
                        localVerts.Add(vi);
                }

                if (localVerts.Count < 3)
                    continue;

                for (int i = 1; i < localVerts.Count - 1; i++)
                {
                    Vector3 a = msvt[localVerts[0]];
                    Vector3 b = msvt[localVerts[i]];
                    Vector3 c = msvt[localVerts[i + 1]];
                    triangles.Add(TriangleFeature.FromTriangle(a, b, c));
                }
            }

            if (triangles.Count == 0)
                continue;

            Dictionary<string, int> histogram = BuildAbsoluteHistogram(triangles, edgeBinSize, areaBinSize, normalBinSize, heightBinSize);

            fingerprints.Add(new SurfaceFingerprint(
                ck24Type,
                ck24,
                doc.SourcePath ?? "unknown",
                surfaces.Count,
                triangles.Count,
                histogram));
        }

        return fingerprints;
    }

    private static Dictionary<string, int> BuildAbsoluteHistogram(
        IReadOnlyList<TriangleFeature> triangles,
        float edgeBinSize,
        float areaBinSize,
        float normalBinSize,
        float heightBinSize)
    {
        Dictionary<string, int> histogram = new();
        foreach (TriangleFeature t in triangles)
        {
            string key = t.AbsoluteKey(edgeBinSize, areaBinSize, normalBinSize, heightBinSize);
            histogram[key] = histogram.TryGetValue(key, out int v) ? v + 1 : 1;
        }

        return histogram;
    }

    private static (SurfaceFingerprint? Best, double Score, double Pm4Coverage, double WmoCoverage) FindBestMatch(
        SurfaceFingerprint generated,
        IReadOnlyList<SurfaceFingerprint> realFingerprints)
    {
        SurfaceFingerprint? best = null;
        double bestScore = 0;
        double bestPm4Coverage = 0;
        double bestWmoCoverage = 0;

        foreach (SurfaceFingerprint real in realFingerprints)
        {
            if (generated.Histogram.Count == 0 || real.Histogram.Count == 0)
                continue;

            int pm4Matched = 0;
            int pm4Total = generated.TriangleCount;
            int wmoMatched = 0;
            int wmoTotal = real.TriangleCount;

            foreach (var kv in generated.Histogram)
            {
                if (real.Histogram.TryGetValue(kv.Key, out int realCount))
                {
                    int matched = Math.Min(kv.Value, realCount);
                    pm4Matched += matched;
                    wmoMatched += matched;
                }
            }

            double pm4Coverage = (double)pm4Matched / pm4Total;
            double wmoCoverage = (double)wmoMatched / wmoTotal;
            double score = pm4Coverage * wmoCoverage > 0
                ? 2.0 * pm4Coverage * wmoCoverage / (pm4Coverage + wmoCoverage)
                : 0;

            if (score > bestScore)
            {
                bestScore = score;
                best = real;
                bestPm4Coverage = pm4Coverage;
                bestWmoCoverage = wmoCoverage;
            }
        }

        return (best, bestScore, bestPm4Coverage, bestWmoCoverage);
    }

    private sealed record SurfaceFingerprint(
        byte Ck24Type,
        uint Ck24,
        string AssetPath,
        int SurfaceCount,
        int TriangleCount,
        IReadOnlyDictionary<string, int> Histogram);
}
