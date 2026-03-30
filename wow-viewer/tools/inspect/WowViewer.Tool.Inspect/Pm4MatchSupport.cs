using System.Numerics;
using System.Text.Json;
using System.Text.Json.Serialization;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Maps;
using WowViewer.Core.Mdx;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;
using WowViewer.Core.Wmo;

internal sealed record Pm4MatchResult(
    string Pm4Path,
    string PlacementPath,
    string ArchiveRoot,
    int TileX,
    int TileY,
    int Pm4ObjectCount,
    int WmoPlacementCount,
    int M2PlacementCount,
    float SearchRange,
    IReadOnlyList<Pm4ObjectMatch> Pm4ObjectMatches,
    IReadOnlyList<Pm4PlacementMatchPlacement> WmoPlacements,
    IReadOnlyList<Pm4PlacementMatchPlacement> M2Placements,
    IReadOnlyList<string> Notes);

internal sealed record Pm4ObjectMatch(
    uint Ck24,
    byte Ck24Type,
    ushort Ck24ObjectId,
    int ObjectPartId,
    uint LinkGroupObjectId,
    int SurfaceCount,
    int LinkedPositionRefCount,
    byte DominantGroupKey,
    byte DominantAttributeMask,
    uint DominantMdosIndex,
    float AverageSurfaceHeight,
    float FootprintArea,
    int AnchorPointCount,
    int AnchorNormalCount,
    float? AnchorHeadingMeanDegrees,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    Vector3 Center,
    int NearbyCandidateCount,
    int ExportedCandidateCount,
    IReadOnlyList<Pm4ObjectSurfaceData> Surfaces,
    IReadOnlyList<Pm4ObjectPlacementCandidate> PossibleMatches);

internal sealed record Pm4ObjectSurfaceData(
    int SurfaceIndex,
    byte GroupKey,
    byte AttributeMask,
    byte IndexCount,
    float Height,
    uint MsviFirstIndex,
    uint MdosIndex,
    uint PackedParams0x1C,
    uint Ck24,
    byte Ck24Type,
    ushort Ck24ObjectId,
    Vector3 Normal);

internal sealed record Pm4ObjectPlacementCandidate(
    string Kind,
    int UniqueId,
    string ModelPath,
    bool AssetResolved,
    string? AssetSource,
    Vector3 PlacementPosition,
    Vector3 PlacementRotation,
    float PlacementScale,
    float AnchorPlanarGap,
    float PlanarGap,
    float VerticalGap,
    float CenterDistance,
    float PlanarOverlapRatio,
    float VolumeOverlapRatio,
    float FootprintOverlapRatio,
    float FootprintAreaRatio,
    float FootprintDistance,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    Vector3 Center);

internal sealed record Pm4PlacementMatchPlacement(
    string Kind,
    int UniqueId,
    string ModelPath,
    Vector3 PlacementPosition,
    Vector3 PlacementRotation,
    float PlacementScale,
    Vector3 WorldBoundsMin,
    Vector3 WorldBoundsMax,
    bool AssetResolved,
    string? AssetSource,
    int CandidateCount,
    IReadOnlyList<Pm4PlacementMatchCandidate> Matches);

internal sealed record Pm4PlacementMatchCandidate(
    uint Ck24,
    byte Ck24Type,
    ushort Ck24ObjectId,
    int ObjectPartId,
    uint LinkGroupObjectId,
    int SurfaceCount,
    int LinkedPositionRefCount,
    byte DominantGroupKey,
    byte DominantAttributeMask,
    uint DominantMdosIndex,
    float AverageSurfaceHeight,
    float AnchorPlanarGap,
    float PlanarGap,
    float VerticalGap,
    float CenterDistance,
    float PlanarOverlapRatio,
    float VolumeOverlapRatio,
    float FootprintOverlapRatio,
    float FootprintAreaRatio,
    float FootprintDistance,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    Vector3 Center);

internal sealed record Pm4InspectObjectState(
    Pm4CorrelationObjectState State,
    IReadOnlyList<Vector2> AnchorPlanarPoints,
    Pm4LinkedPositionRefSummary AnchorSummary,
    IReadOnlyList<Pm4ObjectSurfaceData> Surfaces);

internal static class Pm4MatchSupport
{
    public static Pm4MatchResult Run(
        string pm4Path,
        string placementPath,
        string archiveRoot,
        string? listfilePath,
        int maxMatchesPerPlacement,
        float searchRange)
    {
        Pm4ResearchDocument document = Pm4ResearchReader.ReadFile(pm4Path);
        if (!Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int tileX, out int tileY))
            throw new InvalidOperationException($"Could not parse tile coordinates from '{pm4Path}'.");

        AdtPlacementCatalog placements = AdtPlacementReader.Read(placementPath);
        IReadOnlyList<Pm4InspectObjectState> pm4Objects = BuildPm4ObjectStates(document, tileX, tileY);
        int resolvedMaxMatches = Math.Max(1, maxMatchesPerPlacement);

        List<string> notes =
        [
            $"Primary output is PM4-object-centered nearby-candidate search within a {searchRange:F1} unit range, not exact PM4-to-placement identity.",
            "MSCN is actively used here through MSUR.MdosIndex-backed connector keys when grouping PM4 candidates.",
            "MPRR is not used for placement matching in this command yet; it remains research-only in the current shared PM4 stack.",
            "Candidate ranking uses existing footprint-overlap scoring first, then linked MPRL anchor proximity when an object carries placement refs.",
            "MPRL heading is exported only as supporting evidence; the per-object artifact keeps the selected MSUR surface slice and raw PackedParams (0x1C) for later reconstruction work.",
            "Zero-CK24 candidates are split by the current viewer-style seed path (GroupKey+AttributeMask, then connectivity) so they do not collapse into one tile-wide bucket."
        ];

        List<Pm4PlacementMatchPlacement> wmoPlacements = new(placements.WorldModelPlacements.Count);
        for (int index = 0; index < placements.WorldModelPlacements.Count; index++)
            wmoPlacements.Add(BuildWmoPlacementMatch(placements.WorldModelPlacements[index], archiveRoot, listfilePath, pm4Objects, resolvedMaxMatches));

        List<Pm4PlacementMatchPlacement> m2Placements = new(placements.ModelPlacements.Count);
        for (int index = 0; index < placements.ModelPlacements.Count; index++)
            m2Placements.Add(BuildM2PlacementMatch(placements.ModelPlacements[index], archiveRoot, listfilePath, pm4Objects, resolvedMaxMatches));

        List<Pm4PlacementMatchPlacement> allPlacements = new(wmoPlacements.Count + m2Placements.Count);
        allPlacements.AddRange(wmoPlacements);
        allPlacements.AddRange(m2Placements);
        List<Pm4ObjectMatch> pm4ObjectMatches = BuildPm4ObjectMatches(pm4Objects, allPlacements, searchRange, resolvedMaxMatches);

        return new Pm4MatchResult(
            Path.GetFullPath(pm4Path),
            Path.GetFullPath(placementPath),
            Path.GetFullPath(archiveRoot),
            tileX,
            tileY,
            pm4Objects.Count,
            wmoPlacements.Count,
            m2Placements.Count,
            searchRange,
            pm4ObjectMatches,
            wmoPlacements,
            m2Placements,
            notes);
    }

    public static void Print(Pm4MatchResult result)
    {
        Console.WriteLine("WowViewer.Tool.Inspect PM4 match report");
        Console.WriteLine($"PM4: {result.Pm4Path}");
        Console.WriteLine($"Placements: {result.PlacementPath}");
        Console.WriteLine($"Archive root: {result.ArchiveRoot}");
        Console.WriteLine($"Tile: ({result.TileX}, {result.TileY}) PM4 objects={result.Pm4ObjectCount} WMO placements={result.WmoPlacementCount} M2 placements={result.M2PlacementCount} searchRange={result.SearchRange:F1}");

        PrintPm4ObjectMatches(result.Pm4ObjectCount, result.Pm4ObjectMatches, result.SearchRange);

        PrintPlacementBucket("WMO", result.WmoPlacements);
        PrintPlacementBucket("M2", result.M2Placements);

        foreach (string note in result.Notes)
            Console.WriteLine($"Note: {note}");
    }

    public static string ToJson(Pm4MatchResult result)
    {
        return JsonSerializer.Serialize(result, CreateJsonOptions());
    }

    public static IReadOnlyList<string> WriteObjectArtifacts(Pm4MatchResult result, string outputDirectory)
    {
        string resolvedOutputDirectory = Path.GetFullPath(outputDirectory);
        Directory.CreateDirectory(resolvedOutputDirectory);

        List<string> writtenPaths = new(result.Pm4ObjectMatches.Count + 1);
        List<object> manifestObjects = new(result.Pm4ObjectMatches.Count);

        foreach (Pm4ObjectMatch objectMatch in result.Pm4ObjectMatches)
        {
            string fileName = BuildObjectArtifactFileName(objectMatch);
            string objectPath = Path.Combine(resolvedOutputDirectory, fileName);
            var payload = new
            {
                schemaVersion = "pm4-match-object-v1",
                tile = new
                {
                    result.TileX,
                    result.TileY,
                    result.Pm4Path,
                    result.PlacementPath,
                    result.ArchiveRoot,
                    result.SearchRange,
                },
                notes = result.Notes,
                objectMatch,
            };

            File.WriteAllText(objectPath, JsonSerializer.Serialize(payload, CreateJsonOptions()));
            writtenPaths.Add(objectPath);

            manifestObjects.Add(new
            {
                objectMatch.ObjectPartId,
                objectMatch.Ck24,
                objectMatch.Ck24Type,
                objectMatch.Ck24ObjectId,
                objectMatch.SurfaceCount,
                objectMatch.NearbyCandidateCount,
                objectMatch.ExportedCandidateCount,
                fileName,
            });
        }

        string manifestPath = Path.Combine(resolvedOutputDirectory, $"tile_{result.TileX}_{result.TileY}_manifest.json");
        var manifest = new
        {
            schemaVersion = "pm4-match-manifest-v1",
            tile = new
            {
                result.TileX,
                result.TileY,
                result.Pm4Path,
                result.PlacementPath,
                result.ArchiveRoot,
                result.SearchRange,
                result.Pm4ObjectCount,
                result.WmoPlacementCount,
                result.M2PlacementCount,
            },
            notes = result.Notes,
            objects = manifestObjects,
        };

        File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest, CreateJsonOptions()));
        writtenPaths.Add(manifestPath);

        return writtenPaths;
    }

    private static void PrintPlacementBucket(string label, IReadOnlyList<Pm4PlacementMatchPlacement> placements)
    {
        int resolved = placements.Count(static placement => placement.AssetResolved);
        Console.WriteLine($"{label}: placements={placements.Count} assetResolved={resolved}");
    }

    private static void PrintPm4ObjectMatches(int pm4ObjectCount, IReadOnlyList<Pm4ObjectMatch> objectMatches, float searchRange)
    {
        int matchedCount = objectMatches.Count(static match => match.NearbyCandidateCount > 0);
        Console.WriteLine($"PM4 object candidate search: matched={matchedCount}/{pm4ObjectCount} withinRange={searchRange:F1}");

        IEnumerable<Pm4ObjectMatch> ranked = matchedCount > 0
            ? objectMatches.Where(static match => match.NearbyCandidateCount > 0)
            : objectMatches;

        foreach (Pm4ObjectMatch objectMatch in ranked
            .OrderByDescending(static match => match.NearbyCandidateCount)
            .ThenByDescending(static match => match.PossibleMatches.Count > 0 ? match.PossibleMatches[0].FootprintOverlapRatio : 0f)
            .ThenBy(static match => match.PossibleMatches.Count > 0 ? match.PossibleMatches[0].AnchorPlanarGap : float.MaxValue)
            .ThenBy(static match => match.PossibleMatches.Count > 0 ? match.PossibleMatches[0].CenterDistance : float.MaxValue)
            .Take(Math.Min(10, objectMatches.Count)))
        {
            Console.WriteLine($"  ck24=0x{objectMatch.Ck24:X6} part={objectMatch.ObjectPartId} nearby={objectMatch.NearbyCandidateCount} groupKey={objectMatch.DominantGroupKey} attr=0x{objectMatch.DominantAttributeMask:X2} mdos={objectMatch.DominantMdosIndex}");
            foreach (Pm4ObjectPlacementCandidate placement in objectMatch.PossibleMatches.Take(Math.Min(3, objectMatch.PossibleMatches.Count)))
            {
                Console.WriteLine($"    {placement.Kind} {placement.ModelPath} uid={placement.UniqueId} resolved={placement.AssetResolved} center={placement.CenterDistance:F1} anchorGap={placement.AnchorPlanarGap:F1} planarGap={placement.PlanarGap:F1} footprint={placement.FootprintOverlapRatio:F3}");
            }
        }
    }

    private static Pm4PlacementMatchPlacement BuildWmoPlacementMatch(AdtWorldModelPlacement placement, string archiveRoot, string? listfilePath, IReadOnlyList<Pm4InspectObjectState> pm4Objects, int maxMatches)
    {
        bool assetResolved = false;
        string? assetSource = null;
        Vector3 worldBoundsMin = placement.BoundsMin;
        Vector3 worldBoundsMax = placement.BoundsMax;
        Vector2[] footprintHull = BuildAabbFootprintHull(worldBoundsMin, worldBoundsMax);
        float footprintArea = Pm4CorrelationMath.ComputeFootprintArea(footprintHull);

        try
        {
            byte[] bytes = ArchiveVirtualFileReader.ReadVirtualFile(NormalizeVirtualPath(placement.ModelPath), [archiveRoot], listfilePath);
            using MemoryStream stream = new(bytes, writable: false);
            WmoSummary summary = WmoSummaryReader.Read(stream, placement.ModelPath);
            Matrix4x4 transform = BuildWmoTransform(placement.Position, placement.Rotation);
            TransformBounds(summary.BoundsMin, summary.BoundsMax, transform, out worldBoundsMin, out worldBoundsMax);
            footprintHull = BuildTransformedAabbFootprintHull(summary.BoundsMin, summary.BoundsMax, transform);
            footprintArea = Pm4CorrelationMath.ComputeFootprintArea(footprintHull);
            assetResolved = true;
            assetSource = placement.ModelPath;
        }
        catch
        {
        }

        List<Pm4PlacementMatchCandidate> matches = BuildMatches(worldBoundsMin, worldBoundsMax, placement.Position, footprintHull, footprintArea, pm4Objects, maxMatches);

        return new Pm4PlacementMatchPlacement("wmo", placement.UniqueId, placement.ModelPath, placement.Position, placement.Rotation, 1f, worldBoundsMin, worldBoundsMax, assetResolved, assetSource, matches.Count, matches);
    }

    private static Pm4PlacementMatchPlacement BuildM2PlacementMatch(AdtModelPlacement placement, string archiveRoot, string? listfilePath, IReadOnlyList<Pm4InspectObjectState> pm4Objects, int maxMatches)
    {
        bool assetResolved = false;
        string? assetSource = null;
        Vector3 worldBoundsMin = placement.Position - new Vector3(2f);
        Vector3 worldBoundsMax = placement.Position + new Vector3(2f);
        Vector2[] footprintHull = BuildAabbFootprintHull(worldBoundsMin, worldBoundsMax);
        float footprintArea = Pm4CorrelationMath.ComputeFootprintArea(footprintHull);

        try
        {
            byte[] bytes = ArchiveVirtualFileReader.ReadVirtualFile(NormalizeVirtualPath(placement.ModelPath), [archiveRoot], listfilePath);
            using MemoryStream stream = new(bytes, writable: false);
            MdxSummary summary = MdxSummaryReader.Read(stream, placement.ModelPath);
            Vector3? localBoundsMin = summary.Collision?.BoundsMin ?? summary.BoundsMin;
            Vector3? localBoundsMax = summary.Collision?.BoundsMax ?? summary.BoundsMax;
            if (localBoundsMin.HasValue && localBoundsMax.HasValue)
            {
                Matrix4x4 transform = BuildM2Transform(placement.Position, placement.Rotation, placement.Scale);
                TransformBounds(localBoundsMin.Value, localBoundsMax.Value, transform, out worldBoundsMin, out worldBoundsMax);
                footprintHull = BuildTransformedAabbFootprintHull(localBoundsMin.Value, localBoundsMax.Value, transform);
                footprintArea = Pm4CorrelationMath.ComputeFootprintArea(footprintHull);
                assetResolved = true;
                assetSource = placement.ModelPath;
            }
        }
        catch
        {
        }

        List<Pm4PlacementMatchCandidate> matches = BuildMatches(worldBoundsMin, worldBoundsMax, placement.Position, footprintHull, footprintArea, pm4Objects, maxMatches);

        return new Pm4PlacementMatchPlacement("m2", placement.UniqueId, placement.ModelPath, placement.Position, placement.Rotation, placement.Scale, worldBoundsMin, worldBoundsMax, assetResolved, assetSource, matches.Count, matches);
    }

    private static List<Pm4ObjectMatch> BuildPm4ObjectMatches(IReadOnlyList<Pm4InspectObjectState> pm4Objects, IReadOnlyList<Pm4PlacementMatchPlacement> placements, float searchRange, int maxMatches)
    {
        List<Pm4ObjectMatch> matches = new(pm4Objects.Count);

        foreach (Pm4InspectObjectState pm4Object in pm4Objects)
        {
            List<object> rankedCandidates = placements
                .Select(placement =>
                {
                    Vector3 placementCenter = (placement.WorldBoundsMin + placement.WorldBoundsMax) * 0.5f;
                    Vector2[] placementFootprintHull = BuildAabbFootprintHull(placement.WorldBoundsMin, placement.WorldBoundsMax);
                    float placementFootprintArea = Pm4CorrelationMath.ComputeFootprintArea(placementFootprintHull);
                    Pm4CorrelationMetrics metrics = Pm4CorrelationMath.EvaluateMetrics(
                        pm4Object.State.BoundsMin,
                        pm4Object.State.BoundsMax,
                        pm4Object.State.Center,
                        pm4Object.State.FootprintHull,
                        pm4Object.State.FootprintArea,
                        placement.WorldBoundsMin,
                        placement.WorldBoundsMax,
                        placementCenter,
                        placementFootprintHull,
                        placementFootprintArea);
                    float anchorPlanarGap = ComputeAnchorPlanarGap(pm4Object.AnchorPlanarPoints, placement.PlacementPosition);

                    return new
                    {
                        Placement = placement,
                        Metrics = metrics,
                        AnchorPlanarGap = anchorPlanarGap,
                        Score = new Pm4CorrelationCandidateScore(true, metrics, placement.WorldBoundsMin, placement.WorldBoundsMax, placementCenter),
                    };
                })
                .OrderBy(entry => entry.Score, Comparer<Pm4CorrelationCandidateScore>.Create(Pm4CorrelationMath.CompareCandidateScores))
                .ThenBy(entry => entry.AnchorPlanarGap)
                .Cast<object>()
                .ToList();

            int nearbyCandidateCount = rankedCandidates.Count(entry =>
            {
                dynamic current = entry;
                return IsWithinSearchRange(current.Metrics, current.AnchorPlanarGap, searchRange);
            });

            List<Pm4ObjectPlacementCandidate> candidates = rankedCandidates
                .Take(maxMatches)
                .Select(entry =>
                {
                    dynamic current = entry;
                    return new Pm4ObjectPlacementCandidate(
                        current.Placement.Kind,
                        current.Placement.UniqueId,
                        current.Placement.ModelPath,
                        current.Placement.AssetResolved,
                        current.Placement.AssetSource,
                        current.Placement.PlacementPosition,
                        current.Placement.PlacementRotation,
                        current.Placement.PlacementScale,
                        current.AnchorPlanarGap,
                        current.Metrics.PlanarGap,
                        current.Metrics.VerticalGap,
                        current.Metrics.CenterDistance,
                        current.Metrics.PlanarOverlapRatio,
                        current.Metrics.VolumeOverlapRatio,
                        current.Metrics.FootprintOverlapRatio,
                        current.Metrics.FootprintAreaRatio,
                        current.Metrics.FootprintDistance,
                        current.Placement.WorldBoundsMin,
                        current.Placement.WorldBoundsMax,
                        (current.Placement.WorldBoundsMin + current.Placement.WorldBoundsMax) * 0.5f);
                })
                .ToList();

            matches.Add(new Pm4ObjectMatch(
                pm4Object.State.Object.Ck24,
                pm4Object.State.Object.Ck24Type,
                pm4Object.State.Object.Ck24ObjectId,
                pm4Object.State.Object.ObjectPartId,
                pm4Object.State.Object.LinkGroupObjectId,
                pm4Object.State.Object.SurfaceCount,
                pm4Object.State.Object.LinkedPositionRefCount,
                pm4Object.State.Object.DominantGroupKey,
                pm4Object.State.Object.DominantAttributeMask,
                pm4Object.State.Object.DominantMdosIndex,
                pm4Object.State.Object.AverageSurfaceHeight,
                pm4Object.State.FootprintArea,
                pm4Object.AnchorPlanarPoints.Count,
                pm4Object.AnchorSummary.NormalCount,
                pm4Object.AnchorSummary.HasNormalHeadings ? pm4Object.AnchorSummary.HeadingMeanDegrees : null,
                pm4Object.State.BoundsMin,
                pm4Object.State.BoundsMax,
                pm4Object.State.Center,
                nearbyCandidateCount,
                candidates.Count,
                pm4Object.Surfaces,
                candidates));
        }

        return matches;
    }

    private static bool IsWithinSearchRange(Pm4CorrelationMetrics metrics, float anchorPlanarGap, float searchRange)
    {
        return metrics.PlanarOverlapRatio > 0f
            || metrics.FootprintOverlapRatio > 0f
            || metrics.CenterDistance <= searchRange
            || metrics.FootprintDistance <= searchRange
            || metrics.PlanarGap <= searchRange
            || anchorPlanarGap <= searchRange;
    }

    private static List<Pm4PlacementMatchCandidate> BuildMatches(Vector3 referenceBoundsMin, Vector3 referenceBoundsMax, Vector3 referenceCenter, IReadOnlyList<Vector2> referenceFootprintHull, float referenceFootprintArea, IReadOnlyList<Pm4InspectObjectState> pm4Objects, int maxMatches)
    {
        return pm4Objects
            .Select(candidate =>
            {
                Pm4CorrelationMetrics metrics = Pm4CorrelationMath.EvaluateMetrics(referenceBoundsMin, referenceBoundsMax, referenceCenter, referenceFootprintHull, referenceFootprintArea, candidate.State.BoundsMin, candidate.State.BoundsMax, candidate.State.Center, candidate.State.FootprintHull, candidate.State.FootprintArea);
                float anchorPlanarGap = ComputeAnchorPlanarGap(candidate.AnchorPlanarPoints, referenceCenter);
                return new
                {
                    Candidate = candidate,
                    AnchorPlanarGap = anchorPlanarGap,
                    Score = new Pm4CorrelationCandidateScore(true, metrics, candidate.State.BoundsMin, candidate.State.BoundsMax, candidate.State.Center),
                };
            })
            .OrderBy(static entry => entry.Score, Comparer<Pm4CorrelationCandidateScore>.Create(Pm4CorrelationMath.CompareCandidateScores))
            .ThenBy(entry => entry.AnchorPlanarGap)
            .Take(maxMatches)
            .Select(entry => new Pm4PlacementMatchCandidate(
                entry.Candidate.State.Object.Ck24,
                entry.Candidate.State.Object.Ck24Type,
                entry.Candidate.State.Object.Ck24ObjectId,
                entry.Candidate.State.Object.ObjectPartId,
                entry.Candidate.State.Object.LinkGroupObjectId,
                entry.Candidate.State.Object.SurfaceCount,
                entry.Candidate.State.Object.LinkedPositionRefCount,
                entry.Candidate.State.Object.DominantGroupKey,
                entry.Candidate.State.Object.DominantAttributeMask,
                entry.Candidate.State.Object.DominantMdosIndex,
                entry.Candidate.State.Object.AverageSurfaceHeight,
                entry.AnchorPlanarGap,
                entry.Score.Metrics.PlanarGap,
                entry.Score.Metrics.VerticalGap,
                entry.Score.Metrics.CenterDistance,
                entry.Score.Metrics.PlanarOverlapRatio,
                entry.Score.Metrics.VolumeOverlapRatio,
                entry.Score.Metrics.FootprintOverlapRatio,
                entry.Score.Metrics.FootprintAreaRatio,
                entry.Score.Metrics.FootprintDistance,
                entry.Candidate.State.BoundsMin,
                entry.Candidate.State.BoundsMax,
                entry.Candidate.State.Center))
            .ToList();
    }

    private static IReadOnlyList<Pm4InspectObjectState> BuildPm4ObjectStates(Pm4ResearchDocument document, int tileX, int tileY)
    {
        IReadOnlyList<Vector3> meshVertices = document.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = document.KnownChunks.Msvi;
        List<IndexedSurface> indexedSurfaces = document.KnownChunks.Msur
            .Select(static (surface, surfaceIndex) => new IndexedSurface(surfaceIndex, surface))
            .Where(static indexed => indexed.Surface.IndexCount >= 3)
            .ToList();
        if (indexedSurfaces.Count == 0)
            return Array.Empty<Pm4InspectObjectState>();

        List<Pm4CorrelationObjectInput> inputs = [];
        List<IReadOnlyList<Vector2>> anchorPlanarPoints = [];
        List<Pm4LinkedPositionRefSummary> anchorSummaries = [];
        List<IReadOnlyList<Pm4ObjectSurfaceData>> objectSurfaces = [];
        List<SeedGroup> seedGroups = BuildSeedGroups(indexedSurfaces);
        bool fallbackTileLocal = Pm4PlacementMath.IsLikelyTileLocal(meshVertices);
        int nextObjectPartId = 0;

        foreach (SeedGroup seedGroup in seedGroups)
        {
            List<Pm4MsurEntry> seedSurfaces = seedGroup.Surfaces.Select(static surface => surface.Surface).ToList();
            Pm4AxisConvention axisConvention = Pm4PlacementMath.DetectAxisConventionBySurfaceNormals(meshVertices, meshIndices, seedSurfaces);
            List<Pm4MprlEntry> seedRefs = CollectLinkedPositionRefs(document, seedGroup.Surfaces);
            Pm4CoordinateModeResolution coordinateModeResolution = Pm4PlacementMath.ResolveCoordinateMode(meshVertices, meshIndices, seedSurfaces, seedRefs, seedRefs, tileX, tileY, axisConvention, fallbackTileLocal ? Pm4CoordinateMode.TileLocal : Pm4CoordinateMode.WorldSpace);
            Pm4PlacementSolution seedPlacement = Pm4PlacementMath.ResolvePlacementSolution(meshVertices, meshIndices, seedSurfaces, seedRefs, seedRefs, tileX, tileY, coordinateModeResolution.CoordinateMode, axisConvention);
            List<List<IndexedSurface>> linkedGroups = seedGroup.RequiresConnectivitySeedSplit ? SplitByConnectivity(document, seedGroup.Surfaces) : SplitByMslkGroupObjectId(document, seedGroup.Surfaces);

            foreach (List<IndexedSurface> linkedGroup in linkedGroups)
            {
                if (linkedGroup.Count == 0)
                    continue;

                uint dominantLinkGroupObjectId = SelectDominantGroupObjectId(document, linkedGroup);
                List<Pm4MsurEntry> linkedSurfaces = linkedGroup.Select(static surface => surface.Surface).ToList();
                List<Pm4MprlEntry> linkedRefs = CollectLinkedPositionRefs(document, linkedGroup);
                Pm4CoordinateModeResolution linkedModeResolution = Pm4PlacementMath.ResolveCoordinateMode(meshVertices, meshIndices, linkedSurfaces, linkedRefs, linkedRefs, tileX, tileY, axisConvention, coordinateModeResolution.CoordinateMode);
                Pm4PlacementSolution linkedPlacement = Pm4PlacementMath.ResolvePlacementSolution(meshVertices, meshIndices, linkedSurfaces, linkedRefs, linkedRefs, tileX, tileY, linkedModeResolution.CoordinateMode, axisConvention);

                List<List<Pm4MsurEntry>> mdosGroups = !seedGroup.RequiresConnectivitySeedSplit
                    ? linkedSurfaces.GroupBy(static surface => surface.MdosIndex).Select(static group => group.ToList()).ToList()
                    : [linkedSurfaces];

                foreach (List<Pm4MsurEntry> mdosGroup in mdosGroups)
                {
                    List<IndexedSurface> matchingIndexed = linkedGroup.Where(surface => mdosGroup.Contains(surface.Surface)).ToList();
                    List<List<Pm4MsurEntry>> components = !seedGroup.RequiresConnectivitySeedSplit
                        ? SplitByConnectivity(document, matchingIndexed).Select(static component => component.Select(static indexed => indexed.Surface).ToList()).ToList()
                        : [mdosGroup];

                    foreach (List<Pm4MsurEntry> component in components)
                    {
                        List<Vector3> vertices = Pm4PlacementMath.CollectSurfaceVertices(meshVertices, meshIndices, component);
                        if (vertices.Count == 0)
                            continue;

                        List<Vector3> worldPoints = new(vertices.Count);
                        for (int index = 0; index < vertices.Count; index++)
                            worldPoints.Add(Pm4PlacementMath.ConvertPm4VertexToWorld(vertices[index], linkedPlacement));

                        byte dominantGroupKey = SelectDominantSurfaceValue(component, static surface => surface.GroupKey);
                        byte dominantAttributeMask = SelectDominantSurfaceValue(component, static surface => surface.AttributeMask);
                        uint dominantMdosIndex = SelectDominantSurfaceValue(component, static surface => surface.MdosIndex);
                        float averageSurfaceHeight = component.Count > 0 ? component.Average(static surface => surface.Height) : 0f;
                        List<IndexedSurface> componentIndexed = matchingIndexed.Where(surface => component.Contains(surface.Surface)).ToList();
                        List<Pm4MprlEntry> componentRefs = CollectLinkedPositionRefs(document, componentIndexed);
                        int objectPartId = nextObjectPartId++;
                        uint internalGroupCk24 = seedGroup.DisplayCk24 == 0u ? 0x80000000u | (uint)objectPartId : seedGroup.DisplayCk24;

                        inputs.Add(new Pm4CorrelationObjectInput(
                            tileX,
                            tileY,
                            new Pm4ObjectGroupKey(tileX, tileY, internalGroupCk24),
                            new Pm4CorrelationObjectDescriptor(seedGroup.DisplayCk24, seedGroup.DisplayCk24Type, objectPartId, dominantLinkGroupObjectId, component.Count, componentRefs.Count, dominantGroupKey, dominantAttributeMask, dominantMdosIndex, averageSurfaceHeight),
                            worldPoints,
                            seedPlacement.WorldPivot));
                        anchorPlanarPoints.Add(BuildAnchorPlanarPoints(componentRefs));
                        anchorSummaries.Add(Pm4PlacementMath.SummarizeLinkedPositionRefs(componentRefs));
                        objectSurfaces.Add(componentIndexed.Select(static surface => new Pm4ObjectSurfaceData(
                            surface.SurfaceIndex,
                            surface.Surface.GroupKey,
                            surface.Surface.AttributeMask,
                            surface.Surface.IndexCount,
                            surface.Surface.Height,
                            surface.Surface.MsviFirstIndex,
                            surface.Surface.MdosIndex,
                            surface.Surface._0x1C,
                            surface.Surface.Ck24,
                            surface.Surface.Ck24Type,
                            surface.Surface.Ck24ObjectId,
                            surface.Surface.Normal)).ToList());
                    }
                }
            }
        }

        IReadOnlyList<Pm4CorrelationObjectState> states = Pm4CorrelationMath.BuildObjectStates(inputs);
        List<Pm4InspectObjectState> result = new(states.Count);
        for (int index = 0; index < states.Count; index++)
            result.Add(new Pm4InspectObjectState(states[index], anchorPlanarPoints[index], anchorSummaries[index], objectSurfaces[index]));

        return result;
    }

    private static string BuildObjectArtifactFileName(Pm4ObjectMatch objectMatch)
    {
        string ck24Text = objectMatch.Ck24 != 0u
            ? $"ck24_{objectMatch.Ck24:X6}"
            : $"zero_g{objectMatch.DominantGroupKey:X2}_a{objectMatch.DominantAttributeMask:X2}";

        return $"object_{objectMatch.ObjectPartId:D5}_{ck24Text}.json";
    }

    private static JsonSerializerOptions CreateJsonOptions()
    {
        JsonSerializerOptions options = new()
        {
            WriteIndented = true,
            IncludeFields = true,
        };
        options.Converters.Add(new Vector3JsonConverter());
        return options;
    }

    private sealed class Vector3JsonConverter : JsonConverter<Vector3>
    {
        public override Vector3 Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException("Vector3 JSON value must be an object.");

            float x = 0f;
            float y = 0f;
            float z = 0f;

            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject)
                    return new Vector3(x, y, z);

                if (reader.TokenType != JsonTokenType.PropertyName)
                    throw new JsonException("Vector3 JSON object contains an invalid token.");

                string propertyName = reader.GetString() ?? string.Empty;
                reader.Read();
                float value = reader.GetSingle();
                switch (propertyName.ToLowerInvariant())
                {
                    case "x":
                        x = value;
                        break;
                    case "y":
                        y = value;
                        break;
                    case "z":
                        z = value;
                        break;
                }
            }

            throw new JsonException("Vector3 JSON object was not closed.");
        }

        public override void Write(Utf8JsonWriter writer, Vector3 value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();
            writer.WriteNumber("x", value.X);
            writer.WriteNumber("y", value.Y);
            writer.WriteNumber("z", value.Z);
            writer.WriteEndObject();
        }
    }

    private static IReadOnlyList<Vector2> BuildAnchorPlanarPoints(IReadOnlyList<Pm4MprlEntry> refs)
    {
        if (refs.Count == 0)
            return Array.Empty<Vector2>();

        List<Vector2> points = new(refs.Count);
        for (int index = 0; index < refs.Count; index++)
        {
            Vector3 position = refs[index].Position;
            points.Add(new Vector2(position.X, position.Z));
        }

        return points;
    }

    private static float ComputeAnchorPlanarGap(IReadOnlyList<Vector2> anchorPlanarPoints, Vector3 referencePosition)
    {
        if (anchorPlanarPoints.Count == 0)
            return float.MaxValue;

        Vector2 target = new(referencePosition.X, referencePosition.Y);
        float bestDistanceSquared = float.PositiveInfinity;
        for (int index = 0; index < anchorPlanarPoints.Count; index++)
        {
            float distanceSquared = Vector2.DistanceSquared(anchorPlanarPoints[index], target);
            if (distanceSquared < bestDistanceSquared)
                bestDistanceSquared = distanceSquared;
        }

        return float.IsFinite(bestDistanceSquared) ? MathF.Sqrt(bestDistanceSquared) : float.MaxValue;
    }

    private static List<SeedGroup> BuildSeedGroups(IReadOnlyList<IndexedSurface> indexedSurfaces)
    {
        List<SeedGroup> groups = [];

        foreach (IGrouping<uint, IndexedSurface> group in indexedSurfaces.Where(static surface => surface.Surface.Ck24 != 0).GroupBy(static surface => surface.Surface.Ck24).OrderBy(static group => group.Key))
            groups.Add(new SeedGroup(group.Key, (byte)(group.Key >> 16), false, group.OrderBy(static item => item.SurfaceIndex).ToList()));

        foreach (IGrouping<(byte groupKey, byte attributeMask), IndexedSurface> group in indexedSurfaces.Where(static surface => surface.Surface.Ck24 == 0).GroupBy(static surface => (surface.Surface.GroupKey, surface.Surface.AttributeMask)).OrderBy(static group => group.Key.GroupKey).ThenBy(static group => group.Key.AttributeMask))
            groups.Add(new SeedGroup(0u, 0, true, group.OrderBy(static item => item.SurfaceIndex).ToList()));

        return groups;
    }

    private static List<Pm4MprlEntry> CollectLinkedPositionRefs(Pm4ResearchDocument document, IReadOnlyList<IndexedSurface> surfaces)
    {
        if (document.KnownChunks.Mprl.Count == 0 || document.KnownChunks.Mslk.Count == 0 || surfaces.Count == 0)
            return [];

        HashSet<int> surfaceIndices = surfaces.Select(static surface => surface.SurfaceIndex).ToHashSet();
        HashSet<int> seenRefs = [];
        List<Pm4MprlEntry> refs = [];

        for (int index = 0; index < document.KnownChunks.Mslk.Count; index++)
        {
            Pm4MslkEntry link = document.KnownChunks.Mslk[index];
            if (!surfaceIndices.Contains(link.RefIndex) || (uint)link.RefIndex >= (uint)document.KnownChunks.Mprl.Count || !seenRefs.Add(link.RefIndex))
                continue;

            refs.Add(document.KnownChunks.Mprl[link.RefIndex]);
        }

        return refs;
    }

    private static uint SelectDominantGroupObjectId(Pm4ResearchDocument document, IReadOnlyList<IndexedSurface> surfaces)
    {
        HashSet<int> surfaceIndices = surfaces.Select(static surface => surface.SurfaceIndex).ToHashSet();
        Dictionary<uint, int> counts = [];
        uint bestValue = 0;
        int bestCount = 0;

        for (int index = 0; index < document.KnownChunks.Mslk.Count; index++)
        {
            Pm4MslkEntry link = document.KnownChunks.Mslk[index];
            if (link.GroupObjectId == 0 || !surfaceIndices.Contains(link.RefIndex))
                continue;

            int nextCount = counts.TryGetValue(link.GroupObjectId, out int existing) ? existing + 1 : 1;
            counts[link.GroupObjectId] = nextCount;
            if (nextCount > bestCount)
            {
                bestCount = nextCount;
                bestValue = link.GroupObjectId;
            }
        }

        return bestValue;
    }

    private static List<List<IndexedSurface>> SplitByMslkGroupObjectId(Pm4ResearchDocument document, IReadOnlyList<IndexedSurface> surfaces)
    {
        if (surfaces.Count <= 1 || document.KnownChunks.Mslk.Count == 0)
            return [surfaces.ToList()];

        Dictionary<int, int> localIndices = new(surfaces.Count);
        for (int index = 0; index < surfaces.Count; index++)
            localIndices[surfaces[index].SurfaceIndex] = index;

        Dictionary<uint, HashSet<int>> membersByGroupId = [];
        for (int index = 0; index < document.KnownChunks.Mslk.Count; index++)
        {
            Pm4MslkEntry link = document.KnownChunks.Mslk[index];
            if (link.GroupObjectId == 0 || !localIndices.TryGetValue(link.RefIndex, out int localIndex))
                continue;

            if (!membersByGroupId.TryGetValue(link.GroupObjectId, out HashSet<int>? members))
            {
                members = [];
                membersByGroupId[link.GroupObjectId] = members;
            }

            members.Add(localIndex);
        }

        if (membersByGroupId.Count == 0)
            return [surfaces.ToList()];

        int[] parent = new int[surfaces.Count];
        for (int index = 0; index < parent.Length; index++)
            parent[index] = index;

        HashSet<int> linked = [];
        foreach (HashSet<int> members in membersByGroupId.Values)
        {
            if (members.Count < 2)
                continue;

            int first = members.First();
            linked.Add(first);
            foreach (int member in members)
            {
                linked.Add(member);
                Union(parent, first, member);
            }
        }

        if (linked.Count < 2)
            return [surfaces.ToList()];

        Dictionary<int, List<IndexedSurface>> components = [];
        for (int index = 0; index < surfaces.Count; index++)
        {
            if (!linked.Contains(index))
                continue;

            int root = Find(parent, index);
            if (!components.TryGetValue(root, out List<IndexedSurface>? component))
            {
                component = [];
                components[root] = component;
            }

            component.Add(surfaces[index]);
        }

        List<List<IndexedSurface>> result = components.Values.OrderBy(static component => component.Min(static item => item.SurfaceIndex)).ToList();
        List<IndexedSurface> unlinked = surfaces.Where((_, localIndex) => !linked.Contains(localIndex)).ToList();
        if (unlinked.Count > 0)
            result.Add(unlinked);

        return result.Count > 0 ? result : [surfaces.ToList()];
    }

    private static List<List<IndexedSurface>> SplitByConnectivity(Pm4ResearchDocument document, IReadOnlyList<IndexedSurface> surfaces)
    {
        if (surfaces.Count <= 1)
            return [surfaces.ToList()];

        IReadOnlyList<uint> meshIndices = document.KnownChunks.Msvi;
        IReadOnlyList<Vector3> meshVertices = document.KnownChunks.Msvt;
        List<List<int>> surfaceVertices = new(surfaces.Count);
        Dictionary<int, List<int>> vertexToSurfaceIndices = [];

        for (int surfaceIndex = 0; surfaceIndex < surfaces.Count; surfaceIndex++)
        {
            Pm4MsurEntry surface = surfaces[surfaceIndex].Surface;
            int firstIndex = checked((int)surface.MsviFirstIndex);
            int endExclusive = Math.Min(firstIndex + surface.IndexCount, meshIndices.Count);
            List<int> vertices = [];
            HashSet<int> unique = [];

            if (surface.IndexCount > 0 && firstIndex >= 0 && endExclusive > firstIndex)
            {
                for (int index = firstIndex; index < endExclusive; index++)
                {
                    int vertexIndex = checked((int)meshIndices[index]);
                    if ((uint)vertexIndex >= (uint)meshVertices.Count || !unique.Add(vertexIndex))
                        continue;

                    vertices.Add(vertexIndex);
                    if (!vertexToSurfaceIndices.TryGetValue(vertexIndex, out List<int>? owners))
                    {
                        owners = [];
                        vertexToSurfaceIndices[vertexIndex] = owners;
                    }

                    owners.Add(surfaceIndex);
                }
            }

            surfaceVertices.Add(vertices);
        }

        bool[] visited = new bool[surfaces.Count];
        Queue<int> queue = new();
        List<List<IndexedSurface>> components = [];

        for (int start = 0; start < surfaces.Count; start++)
        {
            if (visited[start])
                continue;

            visited[start] = true;
            queue.Enqueue(start);
            List<IndexedSurface> component = [];

            while (queue.Count > 0)
            {
                int current = queue.Dequeue();
                component.Add(surfaces[current]);

                foreach (int vertexIndex in surfaceVertices[current])
                {
                    if (!vertexToSurfaceIndices.TryGetValue(vertexIndex, out List<int>? neighbors))
                        continue;

                    foreach (int neighbor in neighbors)
                    {
                        if (visited[neighbor])
                            continue;

                        visited[neighbor] = true;
                        queue.Enqueue(neighbor);
                    }
                }
            }

            components.Add(component.OrderBy(static item => item.SurfaceIndex).ToList());
        }

        return components;
    }

    private static T SelectDominantSurfaceValue<T>(IReadOnlyList<Pm4MsurEntry> surfaces, Func<Pm4MsurEntry, T> selector) where T : notnull
    {
        return surfaces.GroupBy(selector).OrderByDescending(static group => group.Count()).ThenBy(static group => group.Key).First().Key;
    }

    private static int Find(int[] parent, int index)
    {
        while (parent[index] != index)
        {
            parent[index] = parent[parent[index]];
            index = parent[index];
        }

        return index;
    }

    private static void Union(int[] parent, int left, int right)
    {
        int rootLeft = Find(parent, left);
        int rootRight = Find(parent, right);
        if (rootLeft != rootRight)
            parent[rootRight] = rootLeft;
    }

    private static string NormalizeVirtualPath(string modelPath)
    {
        return modelPath.Replace('\\', '/').Trim().TrimStart('/').ToLowerInvariant();
    }

    private static Matrix4x4 BuildM2Transform(Vector3 position, Vector3 rotationDegrees, float scale)
    {
        float rx = -rotationDegrees.Y * MathF.PI / 180f;
        float ry = -rotationDegrees.X * MathF.PI / 180f;
        float rz = rotationDegrees.Z * MathF.PI / 180f;

        return Matrix4x4.CreateRotationZ(MathF.PI)
            * Matrix4x4.CreateScale(scale)
            * Matrix4x4.CreateRotationX(rx)
            * Matrix4x4.CreateRotationY(ry)
            * Matrix4x4.CreateRotationZ(rz)
            * Matrix4x4.CreateTranslation(position);
    }

    private static Matrix4x4 BuildWmoTransform(Vector3 position, Vector3 rotationDegrees)
    {
        float rx = rotationDegrees.X * MathF.PI / 180f;
        float ry = rotationDegrees.Y * MathF.PI / 180f;
        float rz = rotationDegrees.Z * MathF.PI / 180f;

        return Matrix4x4.CreateRotationZ(MathF.PI)
            * Matrix4x4.CreateRotationX(rx)
            * Matrix4x4.CreateRotationY(ry)
            * Matrix4x4.CreateRotationZ(rz)
            * Matrix4x4.CreateTranslation(position);
    }

    private static void TransformBounds(Vector3 localMin, Vector3 localMax, Matrix4x4 transform, out Vector3 worldMin, out Vector3 worldMax)
    {
        Span<Vector3> corners = stackalloc Vector3[8]
        {
            new(localMin.X, localMin.Y, localMin.Z),
            new(localMin.X, localMin.Y, localMax.Z),
            new(localMin.X, localMax.Y, localMin.Z),
            new(localMin.X, localMax.Y, localMax.Z),
            new(localMax.X, localMin.Y, localMin.Z),
            new(localMax.X, localMin.Y, localMax.Z),
            new(localMax.X, localMax.Y, localMin.Z),
            new(localMax.X, localMax.Y, localMax.Z),
        };

        worldMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
        worldMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);
        for (int index = 0; index < corners.Length; index++)
        {
            Vector3 world = Vector3.Transform(corners[index], transform);
            worldMin = Vector3.Min(worldMin, world);
            worldMax = Vector3.Max(worldMax, world);
        }
    }

    private static Vector2[] BuildTransformedAabbFootprintHull(Vector3 localMin, Vector3 localMax, Matrix4x4 transform)
    {
        return Pm4CorrelationMath.BuildTransformedFootprintHull(BuildAabbCorners(localMin, localMax), transform);
    }

    private static Vector2[] BuildAabbFootprintHull(Vector3 boundsMin, Vector3 boundsMax)
    {
        return
        [
            new Vector2(boundsMin.X, boundsMin.Y),
            new Vector2(boundsMax.X, boundsMin.Y),
            new Vector2(boundsMax.X, boundsMax.Y),
            new Vector2(boundsMin.X, boundsMax.Y),
        ];
    }

    private static List<Vector3> BuildAabbCorners(Vector3 boundsMin, Vector3 boundsMax)
    {
        return
        [
            new(boundsMin.X, boundsMin.Y, boundsMin.Z),
            new(boundsMin.X, boundsMin.Y, boundsMax.Z),
            new(boundsMin.X, boundsMax.Y, boundsMin.Z),
            new(boundsMin.X, boundsMax.Y, boundsMax.Z),
            new(boundsMax.X, boundsMin.Y, boundsMin.Z),
            new(boundsMax.X, boundsMin.Y, boundsMax.Z),
            new(boundsMax.X, boundsMax.Y, boundsMin.Z),
            new(boundsMax.X, boundsMax.Y, boundsMax.Z),
        ];
    }

    private sealed record IndexedSurface(int SurfaceIndex, Pm4MsurEntry Surface);

    private sealed record SeedGroup(uint DisplayCk24, byte DisplayCk24Type, bool RequiresConnectivitySeedSplit, List<IndexedSurface> Surfaces);
}