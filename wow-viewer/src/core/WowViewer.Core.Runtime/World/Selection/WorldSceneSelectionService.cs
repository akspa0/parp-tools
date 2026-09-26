namespace WowViewer.Core.Runtime.World.Selection;

/// <summary>
/// Pure hover/click selection policy for the world scene (Spec 228 T004–T006; Epic 251 U-01 E4).
/// Receives explicit snapshots and returns identifiers; it never touches the viewer, GL or ImGui.
/// Every rule here preserves the ordering the viewer used before the extraction, including the
/// tie behaviour of <see cref="List{T}.Sort(Comparison{T})"/>, so callers must keep passing
/// candidates in their collection order.
/// </summary>
public static class WorldSceneSelectionService
{
    private const float HoverRayDistanceEpsilon = 0.01f;
    private const float BrushDistanceEpsilon = 0.01f;
    private const float Pm4BrushDepthEpsilon = 0.0025f;

    /// <summary>Click or hover-ray evaluation over one snapshot.</summary>
    public static WorldSceneSelectionResult Select(in WorldSceneSelectionRequest request, WorldSceneSelectionSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        if (!request.IsValid || !snapshot.IsValid)
            return WorldSceneSelectionResult.Invalid;

        WorldSceneSelectionPolicy policy = request.Policy;
        var eligible = new List<WorldSceneSelectionCandidate>(snapshot.Candidates.Count);
        foreach (WorldSceneSelectionCandidate candidate in snapshot.Candidates)
        {
            if (policy.IsDistanceAllowed(candidate.RayDistance))
                eligible.Add(candidate);
        }

        return request.Mode == WorldSceneSelectionMode.Click
            ? RankClick(eligible, request.HasClickedChunk)
            : SelectHoverRay(eligible, request);
    }

    /// <summary>
    /// Click ranking: when a terrain chunk was clicked and some hits sit in it, only those remain;
    /// then clicked-chunk hits first, nearer chunk-grid distance, nearer selection point to the
    /// clicked world point, nearer ray distance.
    /// </summary>
    private static WorldSceneSelectionResult RankClick(List<WorldSceneSelectionCandidate> hits, bool hasClickedChunk)
    {
        if (hasClickedChunk && hits.Any(static hit => hit.SharesClickedChunk))
            hits.RemoveAll(static hit => !hit.SharesClickedChunk);

        hits.Sort(CompareClick);
        return ToResult(hits);
    }

    /// <summary>The click ordering, exposed so equivalence tests can state it exactly once.</summary>
    public static int CompareClick(WorldSceneSelectionCandidate left, WorldSceneSelectionCandidate right)
    {
        int clickedChunkCompare = right.SharesClickedChunk.CompareTo(left.SharesClickedChunk);
        if (clickedChunkCompare != 0)
            return clickedChunkCompare;

        int chunkDistanceCompare = left.ChunkGridDistance.CompareTo(right.ChunkGridDistance);
        if (chunkDistanceCompare != 0)
            return chunkDistanceCompare;

        int centroidCompare = left.SelectionPointDistanceSq.CompareTo(right.SelectionPointDistanceSq);
        if (centroidCompare != 0)
            return centroidCompare;

        return left.RayDistance.CompareTo(right.RayDistance);
    }

    /// <summary>
    /// Hover along the ray: visible kinds only, enclosing WMOs fall through to what they contain
    /// (Spec 211), then nearest ray distance first (stable for equal distances).
    /// </summary>
    private static WorldSceneSelectionResult SelectHoverRay(List<WorldSceneSelectionCandidate> hits, in WorldSceneSelectionRequest request)
    {
        bool wmosVisible = request.WmosVisible;
        bool doodadsVisible = request.DoodadsVisible;
        List<WorldSceneSelectionCandidate> visible = hits
            .Where(hit => hit.Kind switch
            {
                WorldSceneSelectionKind.Wmo or WorldSceneSelectionKind.WmoDoodad => wmosVisible,
                WorldSceneSelectionKind.Mdx => doodadsVisible,
                _ => false,
            })
            .ToList();

        IReadOnlyList<WorldSceneSelectionCandidate> fallThrough = ApplyWmoContainerFallThrough(visible);
        WorldSceneSelectionPolicy policy = request.Policy;
        List<WorldSceneSelectionCandidate> ordered = fallThrough
            .Where(hit => policy.IsDistanceAllowed(hit.RayDistance))
            .OrderBy(static hit => hit.RayDistance)
            .ToList();
        return ToResult(ordered);
    }

    private static IReadOnlyList<WorldSceneSelectionCandidate> ApplyWmoContainerFallThrough(IReadOnlyList<WorldSceneSelectionCandidate> hits)
    {
        if (hits.Count <= 1)
            return hits;

        var candidates = new WmoContainerFallThroughFilter.CandidateObject[hits.Count];
        for (int index = 0; index < hits.Count; index++)
        {
            WorldSceneSelectionCandidate hit = hits[index];
            candidates[index] = new WmoContainerFallThroughFilter.CandidateObject(
                index,
                hit.Kind == WorldSceneSelectionKind.Wmo,
                hit.BoundsMin,
                hit.BoundsMax,
                hit.SelectionPoint);
        }

        IReadOnlyList<WmoContainerFallThroughFilter.CandidateObject> filtered = WmoContainerFallThroughFilter.ApplyFallThrough(candidates);
        if (filtered.Count == hits.Count)
            return hits;

        var result = new List<WorldSceneSelectionCandidate>(filtered.Count);
        foreach (WmoContainerFallThroughFilter.CandidateObject candidate in filtered)
            result.Add(hits[candidate.Id]);
        return result;
    }

    /// <summary>
    /// Hover-ray result including liquid bodies: the best scene hit (if any) stands until a liquid
    /// body is strictly nearer; liquid bodies are considered in the given order.
    /// </summary>
    public static WorldSceneHoverRayResult ResolveHoverRay(
        in WorldSceneSelectionRequest request,
        WorldSceneSelectionSnapshot sceneSnapshot,
        IReadOnlyList<WorldSceneRayTarget> liquidBodies)
    {
        ArgumentNullException.ThrowIfNull(liquidBodies);
        var invalid = new WorldSceneHoverRayResult(WorldSceneSelectionStatus.InvalidRequest, WorldSceneHoverRayTarget.None, -1, float.MaxValue);
        if (request.Mode != WorldSceneSelectionMode.HoverRay)
            return invalid;

        WorldSceneSelectionResult scene = Select(request, sceneSnapshot);
        if (scene.Status == WorldSceneSelectionStatus.InvalidRequest)
            return invalid;

        WorldSceneHoverRayTarget target = WorldSceneHoverRayTarget.None;
        int id = -1;
        float distance = float.MaxValue;
        if (scene.Status == WorldSceneSelectionStatus.Hit)
        {
            id = scene.BestId;
            target = WorldSceneHoverRayTarget.SceneObject;
            distance = sceneSnapshot.Candidates.First(candidate => candidate.Id == id).RayDistance;
        }

        foreach (WorldSceneRayTarget body in liquidBodies)
        {
            float t = body.RayDistance;
            if (t < 0f || !request.Policy.IsDistanceAllowed(t) || t >= distance)
                continue;

            target = WorldSceneHoverRayTarget.LiquidBody;
            id = body.Id;
            distance = t;
        }

        return distance < float.MaxValue
            ? new WorldSceneHoverRayResult(WorldSceneSelectionStatus.Hit, target, id, distance)
            : new WorldSceneHoverRayResult(WorldSceneSelectionStatus.NoHit, WorldSceneHoverRayTarget.None, -1, float.MaxValue);
    }

    /// <summary>
    /// Screen-space hover: eligible targets (by camera distance) are counted; the best is the
    /// nearest to the cursor, with near-equal cursor distances (within 0.01) resolved by depth.
    /// Targets are considered in the given order.
    /// </summary>
    public static WorldSceneBrushResult SelectBrush(in WorldSceneSelectionPolicy policy, IReadOnlyList<WorldSceneBrushCandidate> candidates)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        if (!policy.IsValidForPositions)
            return new WorldSceneBrushResult(WorldSceneSelectionStatus.InvalidRequest, -1, 0, float.MaxValue, float.MaxValue);

        int bestId = -1;
        bool hasBest = false;
        float bestDistanceSq = float.MaxValue;
        float bestDepth = float.MaxValue;
        int count = 0;
        foreach (WorldSceneBrushCandidate candidate in candidates)
        {
            if (!policy.IsPositionAllowed(candidate.WorldPosition))
                continue;

            count++;
            if (!hasBest
                || candidate.ScreenDistanceSq < bestDistanceSq - BrushDistanceEpsilon
                || (MathF.Abs(candidate.ScreenDistanceSq - bestDistanceSq) <= BrushDistanceEpsilon && candidate.Depth < bestDepth))
            {
                hasBest = true;
                bestId = candidate.Id;
                bestDistanceSq = candidate.ScreenDistanceSq;
                bestDepth = candidate.Depth;
            }
        }

        return hasBest
            ? new WorldSceneBrushResult(WorldSceneSelectionStatus.Hit, bestId, count, bestDistanceSq, bestDepth)
            : new WorldSceneBrushResult(WorldSceneSelectionStatus.NoHit, -1, count, float.MaxValue, float.MaxValue);
    }

    /// <summary>
    /// Ray arbitration between the scene and the PM4 overlay: PM4 wins when it is the only hit,
    /// when depth is ignored, or when it is nearer by more than 0.01; otherwise the scene hit.
    /// </summary>
    public static WorldSceneHoverSource ChooseHoverRaySource(bool hasSceneHit, float sceneDistance, bool hasPm4Hit, float pm4Distance, bool pm4IgnoreDepth)
    {
        if (hasPm4Hit && (!hasSceneHit || pm4IgnoreDepth || pm4Distance < sceneDistance - HoverRayDistanceEpsilon))
            return WorldSceneHoverSource.Pm4;

        return hasSceneHit ? WorldSceneHoverSource.Scene : WorldSceneHoverSource.None;
    }

    /// <summary>
    /// Brush preference for a PM4 hit over a scene hit: always when depth is ignored; otherwise the
    /// clearly nearer depth (0.0025) wins, and near-equal depths go to the smaller cursor distance.
    /// </summary>
    public static bool PreferPm4Brush(float pm4DistanceSq, float pm4Depth, float sceneDistanceSq, float sceneDepth, bool pm4IgnoreDepth)
    {
        if (pm4IgnoreDepth)
            return true;

        if (sceneDepth + Pm4BrushDepthEpsilon < pm4Depth)
            return false;

        if (pm4Depth + Pm4BrushDepthEpsilon < sceneDepth)
            return true;

        return pm4DistanceSq <= sceneDistanceSq;
    }

    /// <summary>Brush arbitration: PM4 wins when it is the only hit or preferred over the scene hit.</summary>
    public static WorldSceneHoverSource ChooseHoverBrushSource(
        bool hasSceneHit, float sceneDistanceSq, float sceneDepth,
        bool hasPm4Hit, float pm4DistanceSq, float pm4Depth, bool pm4IgnoreDepth)
    {
        if (hasPm4Hit && (!hasSceneHit || PreferPm4Brush(pm4DistanceSq, pm4Depth, sceneDistanceSq, sceneDepth, pm4IgnoreDepth)))
            return WorldSceneHoverSource.Pm4;

        return hasSceneHit ? WorldSceneHoverSource.Scene : WorldSceneHoverSource.None;
    }

    private static WorldSceneSelectionResult ToResult(List<WorldSceneSelectionCandidate> ordered)
    {
        var ids = new int[ordered.Count];
        for (int i = 0; i < ordered.Count; i++)
            ids[i] = ordered[i].Id;
        return new WorldSceneSelectionResult(ids.Length > 0 ? WorldSceneSelectionStatus.Hit : WorldSceneSelectionStatus.NoHit, ids);
    }
}
