using System.Numerics;

namespace WowViewer.Core.Archaeology;

public static class CrossMapDivergenceEngine
{
    public const float DefaultSpatialTolerance = 0.50f;

    public static MapDivergenceReport AnalyzeDivergence(
        int targetMapId,
        string targetMapName,
        IReadOnlyList<UniqueIdRecord> targetPlacements,
        int referenceMapId,
        string referenceMapName,
        IReadOnlyList<UniqueIdRecord> referencePlacements,
        float spatialTolerance = DefaultSpatialTolerance)
    {
        ArgumentNullException.ThrowIfNull(targetPlacements);
        ArgumentNullException.ThrowIfNull(referencePlacements);

        if (targetPlacements.Count == 0 || referencePlacements.Count == 0)
        {
            return new MapDivergenceReport
            {
                TargetMapId = targetMapId,
                TargetMapName = targetMapName,
                ReferenceMapId = referenceMapId,
                ReferenceMapName = referenceMapName,
                TargetTotalPlacements = targetPlacements.Count,
                MatchedPlacementsCount = 0,
                SharedPercentage = 0.0,
                DivergenceCutoffUniqueId = 0,
                Matches = [],
                Notes = ["One or both placement collections were empty."]
            };
        }

        // 1. Build lookup tables for reference placements:
        //    a) By UniqueId
        //    b) Spatial grid index (cell size 64 units)
        const float cellSize = 64f;
        var refById = new Dictionary<int, UniqueIdRecord>();
        var refGrid = new Dictionary<(int cx, int cy), List<UniqueIdRecord>>();

        foreach (var r in referencePlacements)
        {
            if (r.UniqueId > 0 && !refById.ContainsKey(r.UniqueId))
                refById[r.UniqueId] = r;

            int cx = (int)MathF.Floor(r.Position.X / cellSize);
            int cy = (int)MathF.Floor(r.Position.Y / cellSize);
            if (!refGrid.TryGetValue((cx, cy), out var cellList))
            {
                cellList = [];
                refGrid[(cx, cy)] = cellList;
            }
            cellList.Add(r);
        }

        float toleranceSq = spatialTolerance * spatialTolerance;
        var matches = new List<CrossMapPlacementMatch>(targetPlacements.Count);
        int maxSharedUid = 0;
        int matchedCount = 0;

        foreach (var target in targetPlacements)
        {
            string targetModel = Path.GetFileName(target.ModelPath);
            UniqueIdRecord? bestMatch = null;
            float bestDistSq = float.MaxValue;
            bool idMatched = false;
            bool spatialMatched = false;

            // Check direct UniqueId match first
            if (target.UniqueId > 0 && refById.TryGetValue(target.UniqueId, out var directMatch))
            {
                string directModel = Path.GetFileName(directMatch.ModelPath);
                if (string.Equals(targetModel, directModel, StringComparison.OrdinalIgnoreCase))
                {
                    bestMatch = directMatch;
                    idMatched = true;
                    float dist = Vector3.Distance(target.Position, directMatch.Position);
                    bestDistSq = dist * dist;
                    if (bestDistSq <= toleranceSq)
                        spatialMatched = true;
                }
            }

            // If not found by ID or spatial mismatch, check spatial neighborhood
            if (bestMatch == null)
            {
                int tcx = (int)MathF.Floor(target.Position.X / cellSize);
                int tcy = (int)MathF.Floor(target.Position.Y / cellSize);

                for (int dx = -1; dx <= 1; dx++)
                {
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        if (refGrid.TryGetValue((tcx + dx, tcy + dy), out var neighbors))
                        {
                            foreach (var candidate in neighbors)
                            {
                                string candidateModel = Path.GetFileName(candidate.ModelPath);
                                if (!string.Equals(targetModel, candidateModel, StringComparison.OrdinalIgnoreCase))
                                    continue;

                                float distSq = Vector3.DistanceSquared(target.Position, candidate.Position);
                                if (distSq <= toleranceSq && distSq < bestDistSq)
                                {
                                    bestDistSq = distSq;
                                    bestMatch = candidate;
                                    spatialMatched = true;
                                    idMatched = candidate.UniqueId == target.UniqueId;
                                }
                            }
                        }
                    }
                }
            }

            DivergenceStatus status;
            if (bestMatch != null)
            {
                status = DivergenceStatus.InheritedFromOverworld;
                matchedCount++;
                maxSharedUid = Math.Max(maxSharedUid, Math.Max(target.UniqueId, bestMatch.UniqueId));
            }
            else
            {
                status = DivergenceStatus.PostForkAuthored;
            }

            matches.Add(new CrossMapPlacementMatch
            {
                TargetPlacement = target,
                ReferencePlacement = bestMatch,
                Status = status,
                Distance = bestMatch != null ? MathF.Sqrt(bestDistSq) : 0f,
                UniqueIdMatched = idMatched,
                SpatialMatched = spatialMatched
            });
        }

        double sharedPct = targetPlacements.Count > 0
            ? (double)matchedCount / targetPlacements.Count * 100.0
            : 0.0;

        var notes = new List<string>
        {
            $"Matched {matchedCount}/{targetPlacements.Count} placements ({sharedPct:F1}%) against reference map {referenceMapId} ({referenceMapName}).",
            $"Divergence boundary UID: {maxSharedUid}. Placements with UniqueId > {maxSharedUid} represent subsequent iterations authored after the map diverged."
        };

        return new MapDivergenceReport
        {
            TargetMapId = targetMapId,
            TargetMapName = targetMapName,
            ReferenceMapId = referenceMapId,
            ReferenceMapName = referenceMapName,
            TargetTotalPlacements = targetPlacements.Count,
            MatchedPlacementsCount = matchedCount,
            SharedPercentage = sharedPct,
            DivergenceCutoffUniqueId = maxSharedUid,
            Matches = matches,
            Notes = notes
        };
    }
}
