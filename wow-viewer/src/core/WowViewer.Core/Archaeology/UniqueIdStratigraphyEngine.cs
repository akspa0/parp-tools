namespace WowViewer.Core.Archaeology;

public static class UniqueIdStratigraphyEngine
{
    public const int DefaultLayerGapThreshold = 100;

    public static MapStratigraphyProfile ComputeProfile(
        int mapId,
        string mapName,
        IReadOnlyList<UniqueIdRecord> placements,
        string? buildVersion = null,
        int layerGapThreshold = DefaultLayerGapThreshold)
    {
        ArgumentNullException.ThrowIfNull(placements);

        var validPlacements = placements
            .Where(static p => p.UniqueId > 0)
            .OrderBy(static p => p.UniqueId)
            .ToList();

        if (validPlacements.Count == 0)
        {
            return new MapStratigraphyProfile
            {
                MapId = mapId,
                MapName = mapName,
                BuildVersion = buildVersion,
                TotalPlacements = placements.Count,
                DistinctUniqueIds = 0,
                MinUniqueId = 0,
                MaxUniqueId = 0,
                P10 = 0,
                P25 = 0,
                Median = 0,
                P75 = 0,
                P90 = 0,
                Layers = [],
                Tiles = []
            };
        }

        var sortedIds = validPlacements.Select(static p => p.UniqueId).ToList();
        int minId = sortedIds[0];
        int maxId = sortedIds[^1];
        int distinctCount = sortedIds.Distinct().Count();

        double p10 = ComputeQuantile(sortedIds, 0.10);
        double p25 = ComputeQuantile(sortedIds, 0.25);
        double median = ComputeQuantile(sortedIds, 0.50);
        double p75 = ComputeQuantile(sortedIds, 0.75);
        double p90 = ComputeQuantile(sortedIds, 0.90);

        var layers = DetectLayers(validPlacements, layerGapThreshold);
        var tileSummaries = ComputeTileSummaries(validPlacements);

        return new MapStratigraphyProfile
        {
            MapId = mapId,
            MapName = mapName,
            BuildVersion = buildVersion,
            TotalPlacements = placements.Count,
            DistinctUniqueIds = distinctCount,
            MinUniqueId = minId,
            MaxUniqueId = maxId,
            P10 = p10,
            P25 = p25,
            Median = median,
            P75 = p75,
            P90 = p90,
            Layers = layers,
            Tiles = tileSummaries
        };
    }

    public static double ComputeQuantile(IReadOnlyList<int> sortedValues, double quantile)
    {
        if (sortedValues == null || sortedValues.Count == 0)
            return 0.0;

        if (sortedValues.Count == 1 || quantile <= 0.0)
            return sortedValues[0];
        if (quantile >= 1.0)
            return sortedValues[^1];

        double position = (sortedValues.Count - 1) * quantile;
        int lowerIndex = (int)Math.Floor(position);
        int upperIndex = (int)Math.Ceiling(position);

        if (lowerIndex == upperIndex)
            return sortedValues[lowerIndex];

        double fraction = position - lowerIndex;
        return sortedValues[lowerIndex] + fraction * (sortedValues[upperIndex] - sortedValues[lowerIndex]);
    }

    public static IReadOnlyList<UniqueIdLayerSummary> DetectLayers(
        IReadOnlyList<UniqueIdRecord> sortedPlacements,
        int gapThreshold = DefaultLayerGapThreshold)
    {
        if (sortedPlacements.Count == 0)
            return [];

        var layers = new List<UniqueIdLayerSummary>();
        int layerNum = 1;

        int currentStart = sortedPlacements[0].UniqueId;
        int currentEnd = currentStart;
        int previousId = currentStart;

        int totalCount = 0;
        int m2Count = 0;
        int wmoCount = 0;
        int creatureCount = 0;
        int goCount = 0;

        foreach (var p in sortedPlacements)
        {
            if (p.UniqueId - previousId > gapThreshold && totalCount > 0)
            {
                layers.Add(new UniqueIdLayerSummary(
                    layerNum++,
                    currentStart,
                    currentEnd,
                    totalCount,
                    m2Count,
                    wmoCount,
                    creatureCount,
                    goCount));

                currentStart = p.UniqueId;
                totalCount = 0;
                m2Count = 0;
                wmoCount = 0;
                creatureCount = 0;
                goCount = 0;
            }

            currentEnd = p.UniqueId;
            previousId = p.UniqueId;
            totalCount++;

            switch (p.Kind)
            {
                case PlacementOriginKind.M2: m2Count++; break;
                case PlacementOriginKind.Wmo: wmoCount++; break;
                case PlacementOriginKind.Creature: creatureCount++; break;
                case PlacementOriginKind.GameObject: goCount++; break;
            }
        }

        if (totalCount > 0)
        {
            layers.Add(new UniqueIdLayerSummary(
                layerNum,
                currentStart,
                currentEnd,
                totalCount,
                m2Count,
                wmoCount,
                creatureCount,
                goCount));
        }

        return layers;
    }

    private static IReadOnlyList<TileStratigraphySummary> ComputeTileSummaries(IReadOnlyList<UniqueIdRecord> placements)
    {
        var tileGroups = placements
            .Where(static p => p.HasTileCoordinates)
            .GroupBy(static p => (p.TileX, p.TileY))
            .OrderBy(static g => g.Key.TileY)
            .ThenBy(static g => g.Key.TileX);

        var summaries = new List<TileStratigraphySummary>();
        foreach (var group in tileGroups)
        {
            var sortedTileIds = group.Select(static p => p.UniqueId).OrderBy(static id => id).ToList();
            int min = sortedTileIds[0];
            int max = sortedTileIds[^1];
            int median = (int)Math.Round(ComputeQuantile(sortedTileIds, 0.50));

            int m2 = group.Count(static p => p.Kind == PlacementOriginKind.M2);
            int wmo = group.Count(static p => p.Kind == PlacementOriginKind.Wmo);
            int spawn = group.Count(static p => p.Kind == PlacementOriginKind.Creature || p.Kind == PlacementOriginKind.GameObject);

            summaries.Add(new TileStratigraphySummary(
                group.Key.TileX,
                group.Key.TileY,
                min,
                max,
                median,
                group.Count(),
                m2,
                wmo,
                spawn));
        }

        return summaries;
    }
}
