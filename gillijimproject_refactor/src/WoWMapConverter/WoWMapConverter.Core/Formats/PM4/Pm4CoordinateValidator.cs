using System.Numerics;
using WoWMapConverter.Core.Formats.Cataclysm;
using WoWMapConverter.Core.Formats.Classic;

namespace WoWMapConverter.Core.Formats.PM4;

public record Pm4CoordinateValidationOptions(
    string MapDirectory,
    int? TileLimit,
    float MatchThreshold,
    float TileBoundsTolerance,
    int SampleCount)
{
    public static Pm4CoordinateValidationOptions CreateDefault() => new(
        MapDirectory: Pm4CoordinateService.DefaultDevelopmentMapDirectory,
        TileLimit: null,
        MatchThreshold: 32f,
        TileBoundsTolerance: 2f,
        SampleCount: 3);
}

public record Pm4CoordinateMatchSample(
    Vector3 RefLocalPosition,
    Vector3 RefPlacementPosition,
    Vector3 AdtPlacementPosition,
    float HorizontalDistance,
    float HeightDelta,
    string PlacementLabel,
    string PlacementKind);

public record Pm4TileCoordinateValidationResult(
    string TileName,
    int TileX,
    int TileY,
    int PlacementCount,
    int PositionRefCount,
    int InTileBoundsCount,
    int MatchedWithinThresholdCount,
    float? AverageNearestDistance,
    float? AverageMatchedDistance,
    List<Pm4CoordinateMatchSample> BestMatches);

public record Pm4CoordinateValidationReport(
    string MapDirectory,
    int TilesScanned,
    int TilesValidated,
    int TilesSkippedMissingObj0,
    int TilesSkippedMissingPlacements,
    int TotalPlacements,
    int TotalPositionRefs,
    int TotalInTileBounds,
    int TotalMatchedWithinThreshold,
    float? AverageNearestDistance,
    List<Pm4TileCoordinateValidationResult> Tiles);

public static class Pm4CoordinateValidator
{
    public static Pm4CoordinateValidationReport ValidateDirectory(Pm4CoordinateValidationOptions? options = null)
    {
        options ??= Pm4CoordinateValidationOptions.CreateDefault();

        string mapDirectory = Pm4CoordinateService.ResolveMapDirectory(options.MapDirectory);
        if (!Directory.Exists(mapDirectory))
            throw new DirectoryNotFoundException($"PM4 validation directory not found: {mapDirectory}");

        var tileResults = new List<Pm4TileCoordinateValidationResult>();
        int tilesScanned = 0;
        int tilesSkippedMissingObj0 = 0;
        int tilesSkippedMissingPlacements = 0;
        int totalPlacements = 0;
        int totalPositionRefs = 0;
        int totalInTileBounds = 0;
        int totalMatchedWithinThreshold = 0;
        float totalNearestDistance = 0f;
        int totalNearestSamples = 0;

        foreach (string pm4Path in Directory.EnumerateFiles(mapDirectory, "*.pm4").OrderBy(Path.GetFileName))
        {
            if (!Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int tileX, out int tileY))
                continue;

            tilesScanned++;

            string obj0Path = Pm4CoordinateService.GetObj0PathForPm4(pm4Path);
            if (!File.Exists(obj0Path))
            {
                tilesSkippedMissingObj0++;
                continue;
            }

            List<AdtPlacement> placements = LoadPlacements(obj0Path);
            if (placements.Count == 0)
            {
                tilesSkippedMissingPlacements++;
                continue;
            }

            Pm4File pm4 = Pm4File.FromFile(pm4Path);
            Pm4TileCoordinateValidationResult tileResult = ValidateTile(
                pm4,
                Path.GetFileNameWithoutExtension(pm4Path),
                tileX,
                tileY,
                placements,
                options);

            tileResults.Add(tileResult);
            totalPlacements += tileResult.PlacementCount;
            totalPositionRefs += tileResult.PositionRefCount;
            totalInTileBounds += tileResult.InTileBoundsCount;
            totalMatchedWithinThreshold += tileResult.MatchedWithinThresholdCount;

            if (tileResult.AverageNearestDistance.HasValue)
            {
                totalNearestDistance += tileResult.AverageNearestDistance.Value * tileResult.PositionRefCount;
                totalNearestSamples += tileResult.PositionRefCount;
            }

            if (options.TileLimit.HasValue && tileResults.Count >= options.TileLimit.Value)
                break;
        }

        float? averageNearestDistance = totalNearestSamples > 0
            ? totalNearestDistance / totalNearestSamples
            : null;

        return new Pm4CoordinateValidationReport(
            MapDirectory: mapDirectory,
            TilesScanned: tilesScanned,
            TilesValidated: tileResults.Count,
            TilesSkippedMissingObj0: tilesSkippedMissingObj0,
            TilesSkippedMissingPlacements: tilesSkippedMissingPlacements,
            TotalPlacements: totalPlacements,
            TotalPositionRefs: totalPositionRefs,
            TotalInTileBounds: totalInTileBounds,
            TotalMatchedWithinThreshold: totalMatchedWithinThreshold,
            AverageNearestDistance: averageNearestDistance,
            Tiles: tileResults);
    }

    private static Pm4TileCoordinateValidationResult ValidateTile(
        Pm4File pm4,
        string tileName,
        int tileX,
        int tileY,
        List<AdtPlacement> placements,
        Pm4CoordinateValidationOptions options)
    {
        int inTileBoundsCount = 0;
        int matchedWithinThresholdCount = 0;
        float nearestDistanceSum = 0f;
        int nearestDistanceCount = 0;
        float matchedDistanceSum = 0f;
        int matchedDistanceCount = 0;
        var bestMatches = new List<Pm4CoordinateMatchSample>();

        foreach (MprlEntry positionRef in pm4.PositionRefs)
        {
            Vector3 refPlacement = Pm4CoordinateService.MprlToAdtPlacement(positionRef.Position);

            if (Pm4CoordinateService.IsWithinPlacementTileBounds(refPlacement, tileX, tileY, options.TileBoundsTolerance))
                inTileBoundsCount++;

            AdtPlacement? nearestPlacement = null;
            float nearestDistance = float.MaxValue;
            foreach (AdtPlacement placement in placements)
            {
                float distance = HorizontalDistance(refPlacement, placement.Position);
                if (distance < nearestDistance)
                {
                    nearestDistance = distance;
                    nearestPlacement = placement;
                }
            }

            if (nearestPlacement == null)
                continue;

            nearestDistanceSum += nearestDistance;
            nearestDistanceCount++;

            if (nearestDistance <= options.MatchThreshold)
            {
                matchedWithinThresholdCount++;
                matchedDistanceSum += nearestDistance;
                matchedDistanceCount++;
            }

            bestMatches.Add(new Pm4CoordinateMatchSample(
                RefLocalPosition: positionRef.Position,
                RefPlacementPosition: refPlacement,
                AdtPlacementPosition: nearestPlacement.Position,
                HorizontalDistance: nearestDistance,
                HeightDelta: MathF.Abs(refPlacement.Y - nearestPlacement.Position.Y),
                PlacementLabel: nearestPlacement.Label,
                PlacementKind: nearestPlacement.Kind));
        }

        bestMatches = bestMatches
            .OrderBy(sample => sample.HorizontalDistance)
            .ThenBy(sample => sample.HeightDelta)
            .Take(options.SampleCount)
            .ToList();

        float? averageNearestDistance = nearestDistanceCount > 0
            ? nearestDistanceSum / nearestDistanceCount
            : null;

        float? averageMatchedDistance = matchedDistanceCount > 0
            ? matchedDistanceSum / matchedDistanceCount
            : null;

        return new Pm4TileCoordinateValidationResult(
            TileName: tileName,
            TileX: tileX,
            TileY: tileY,
            PlacementCount: placements.Count,
            PositionRefCount: pm4.PositionRefs.Count,
            InTileBoundsCount: inTileBoundsCount,
            MatchedWithinThresholdCount: matchedWithinThresholdCount,
            AverageNearestDistance: averageNearestDistance,
            AverageMatchedDistance: averageMatchedDistance,
            BestMatches: bestMatches);
    }

    private static List<AdtPlacement> LoadPlacements(string obj0Path)
    {
        AdtObj obj0 = AdtObj.Load(obj0Path);
        var placements = new List<AdtPlacement>(obj0.Doodads.Count + obj0.WorldModels.Count);

        foreach (MddfEntry doodad in obj0.Doodads)
        {
            placements.Add(new AdtPlacement(
                Kind: "m2",
                Label: ResolveName(obj0.M2Names, obj0.M2Offsets, doodad.NameId, "m2"),
                Position: doodad.Position));
        }

        foreach (ModfEntry worldModel in obj0.WorldModels)
        {
            placements.Add(new AdtPlacement(
                Kind: "wmo",
                Label: ResolveName(obj0.WmoNames, obj0.WmoOffsets, worldModel.NameId, "wmo"),
                Position: worldModel.Position));
        }

        return placements;
    }

    private static string ResolveName(List<string> names, uint[]? offsets, uint nameId, string kind)
    {
        if (nameId < names.Count)
            return names[(int)nameId];

        if (offsets != null)
        {
            int offsetIndex = Array.IndexOf(offsets, nameId);
            if (offsetIndex >= 0 && offsetIndex < names.Count)
                return names[offsetIndex];
        }

        return $"{kind}:{nameId}";
    }

    private static float HorizontalDistance(Vector3 left, Vector3 right)
    {
        float deltaX = left.X - right.X;
        float deltaZ = left.Z - right.Z;
        return MathF.Sqrt(deltaX * deltaX + deltaZ * deltaZ);
    }

    private sealed record AdtPlacement(string Kind, string Label, Vector3 Position);
}