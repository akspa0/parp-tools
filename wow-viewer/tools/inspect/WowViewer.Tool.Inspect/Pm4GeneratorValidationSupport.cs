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
        float normalAlignmentBinSize = 0.0f,
        float planarOffsetBinSize = 0.0f,
        Action<string>? progress = null)
    {
        List<string> warnings = [];

        Pm4ResearchDocument realDoc = Pm4ResearchReader.ReadFile(pm4Path);
        IReadOnlyList<SurfaceCorrelationFingerprint> realFingerprints = ExtractPm4Fingerprints(
            realDoc,
            ck24Type: 0x43,
            edgeBinSize,
            areaBinSize,
            normalAlignmentBinSize,
            planarOffsetBinSize);

        progress?.Invoke($"Real PM4: {realFingerprints.Count} CK24 groups, {realFingerprints.Sum(static f => f.TriangleCount)} triangles.");

        AdtPlacementCatalog placements = AdtPlacementReader.Read(adtPath);
        progress?.Invoke($"ADT placements: {placements.WorldModelPlacements.Count} WMOs, {placements.ModelPlacements.Count} M2s.");

        List<Pm4GeneratorGroupValidation> groupValidations = [];
        int processed = 0;

        foreach (AdtWorldModelPlacement placement in placements.WorldModelPlacements)
        {
            processed++;
            try
            {
                SurfaceCorrelationFingerprint? generated = GenerateWmoPlacementFingerprint(
                    placement, archiveRoot, bootstrapOptions,
                    edgeBinSize, areaBinSize, normalAlignmentBinSize, planarOffsetBinSize,
                    warnings);

                if (generated is null)
                {
                    groupValidations.Add(new Pm4GeneratorGroupValidation(
                        placement.ModelPath, placement.Position, placement.Rotation, 1f,
                        0, 0, null, null, null, null, "no_geometry"));
                    continue;
                }

                (SurfaceCorrelationFingerprint? bestReal, double score, double pm4Coverage, double wmoCoverage) =
                    FindBestMatch(generated, realFingerprints);

                uint? matchedCk24 = null;
                if (bestReal is not null)
                {
                    ReadOnlySpan<char> ck24Span = bestReal.AssetId.AsSpan(bestReal.AssetId.LastIndexOf('-') + 1);
                    if (ck24Span.StartsWith("0x"))
                        ck24Span = ck24Span.Slice(2);
                    matchedCk24 = uint.Parse(ck24Span, System.Globalization.NumberStyles.HexNumber);
                }
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
            groupValidations.Count(static g => g.GeneratedTriangleCount > 0),
            matchedCount,
            meanScore,
            meanPm4Coverage,
            meanWmoCoverage,
            groupValidations,
            warnings);
    }

    private static SurfaceCorrelationFingerprint? GenerateWmoPlacementFingerprint(
        AdtWorldModelPlacement placement,
        string archiveRoot,
        ArchiveCatalogBootstrapOptions bootstrapOptions,
        float edgeBinSize,
        float areaBinSize,
        float normalAlignmentBinSize,
        float planarOffsetBinSize,
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

        Pm4GenerationData genData = Pm4Generator.GenerateFromWmo(
            renderDoc, placement.Position, placement.Rotation, scale: 1f,
            ck24Type: 0x43, ck24ObjectId: 1, regionId: 0);

        if (genData.Msur.Count == 0)
            return null;

        List<Pm4MsurEntry> msurEntries = genData.Msur
            .Select(static s => new Pm4MsurEntry(
                GroupKey: 0,
                IndexCount: s.IndexCount,
                AttributeMask: 0,
                Padding: 0,
                s.Normal,
                s.Height,
                s.MsviFirstIndex,
                _0x18: 0,
                PackedParams: s.PackedParams))
            .ToList();

        return Pm4SurfaceCorrelationExtractor.ExtractFromPm4Group(
            genData.Msvt,
            genData.Msvi,
            msurEntries,
            ck24: 1,
            ck24Type: 0x43,
            assetId: $"gen-{placement.UniqueId}",
            assetPath: placement.ModelPath,
            assetKind: "generated",
            edgeBinSize,
            areaBinSize,
            normalAlignmentBinSize,
            planarOffsetBinSize);
    }

    private static IReadOnlyList<SurfaceCorrelationFingerprint> ExtractPm4Fingerprints(
        Pm4ResearchDocument doc,
        byte ck24Type,
        float edgeBinSize,
        float areaBinSize,
        float normalAlignmentBinSize,
        float planarOffsetBinSize)
    {
        List<SurfaceCorrelationFingerprint> fingerprints = [];

        var groups = doc.KnownChunks.Msur
            .Where(s => s.Ck24 != 0 && s.IndexCount >= 3 && s.Ck24Type == ck24Type)
            .GroupBy(static s => s.Ck24);

        foreach (IGrouping<uint, Pm4MsurEntry> group in groups)
        {
            uint ck24 = group.Key;
            SurfaceCorrelationFingerprint? fingerprint = Pm4SurfaceCorrelationExtractor.ExtractFromPm4Group(
                doc.KnownChunks.Msvt,
                doc.KnownChunks.Msvi,
                group.ToList(),
                ck24,
                ck24Type,
                assetId: $"pm4-ck24-0x{ck24:X6}",
                assetPath: doc.SourcePath ?? "unknown",
                assetKind: "pm4",
                edgeBinSize,
                areaBinSize,
                normalAlignmentBinSize,
                planarOffsetBinSize);

            if (fingerprint is not null)
                fingerprints.Add(fingerprint);
        }

        return fingerprints;
    }

    private static (SurfaceCorrelationFingerprint? Best, double Score, double Pm4Coverage, double WmoCoverage) FindBestMatch(
        SurfaceCorrelationFingerprint generated,
        IReadOnlyList<SurfaceCorrelationFingerprint> realFingerprints)
    {
        SurfaceCorrelationFingerprint? best = null;
        double bestScore = 0;
        double bestPm4Coverage = 0;
        double bestWmoCoverage = 0;

        foreach (SurfaceCorrelationFingerprint real in realFingerprints)
        {
            if (generated.TriangleHistogram.Count == 0 || real.TriangleHistogram.Count == 0)
                continue;

            int pm4Matched = 0;
            int pm4Total = generated.TriangleCount;
            int wmoMatched = 0;
            int wmoTotal = real.TriangleCount;

            foreach (var kv in generated.TriangleHistogram)
            {
                if (real.TriangleHistogram.TryGetValue(kv.Key, out int realCount))
                {
                    int matched = Math.Min(kv.Value, realCount);
                    pm4Matched += matched;
                    wmoMatched += matched;
                }
            }

            double pm4Coverage = pm4Total > 0 ? (double)pm4Matched / pm4Total : 0;
            double wmoCoverage = wmoTotal > 0 ? (double)wmoMatched / wmoTotal : 0;
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
}
