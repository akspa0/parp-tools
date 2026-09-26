using WowViewer.Core.Runtime.World.Selection;

namespace WoWViewer.Terrain;

/// <summary>
/// Viewer edge of the Core selection service (Spec 228 T008; Epic 251 U-01 E4). Maps the scene's
/// resident pick hits to snapshot candidates (identifier = position in the hit list) and applies
/// the returned order back to the same hits. It holds no state and exposes no UI route.
/// </summary>
internal static class WorldSceneSelectionAdapter
{
    /// <summary>
    /// Filters and orders click hits in place: range limit, clicked-chunk filter, click ranking.
    /// </summary>
    public static void RankClickHits(List<SceneObjectPickHit> hits, bool hasClickedChunk, in WorldSceneSelectionPolicy policy)
    {
        WorldSceneSelectionResult result = WorldSceneSelectionService.Select(
            new WorldSceneSelectionRequest(WorldSceneSelectionMode.Click, policy, hasClickedChunk),
            ToSnapshot(hits));
        Reorder(hits, result.RankedIds);
    }

    /// <summary>
    /// Hover along the ray over click-ranked hits, then liquid bodies (in their given order) when
    /// strictly nearer. Hit ids are indices into <paramref name="rankedHits"/>; liquid ids are the
    /// caller's body indices.
    /// </summary>
    public static WorldSceneHoverRayResult ResolveHoverRay(
        IReadOnlyList<SceneObjectPickHit> rankedHits,
        in WorldSceneSelectionPolicy policy,
        bool wmosVisible,
        bool doodadsVisible,
        IReadOnlyList<WorldSceneRayTarget> liquidBodies)
    {
        return WorldSceneSelectionService.ResolveHoverRay(
            new WorldSceneSelectionRequest(WorldSceneSelectionMode.HoverRay, policy, WmosVisible: wmosVisible, DoodadsVisible: doodadsVisible),
            ToSnapshot(rankedHits),
            liquidBodies);
    }

    private static WorldSceneSelectionSnapshot ToSnapshot(IReadOnlyList<SceneObjectPickHit> hits)
    {
        var candidates = new WorldSceneSelectionCandidate[hits.Count];
        for (int i = 0; i < hits.Count; i++)
        {
            SceneObjectPickHit hit = hits[i];
            candidates[i] = new WorldSceneSelectionCandidate(
                i,
                ToKind(hit.ObjectType),
                hit.Distance,
                hit.BoundsMin,
                hit.BoundsMax,
                hit.SelectionPoint,
                hit.SelectionPointDistanceSq,
                hit.SharesClickedChunk,
                hit.ChunkGridDistance);
        }

        return new WorldSceneSelectionSnapshot(candidates);
    }

    private static WorldSceneSelectionKind ToKind(ObjectType objectType) => objectType switch
    {
        ObjectType.Wmo => WorldSceneSelectionKind.Wmo,
        ObjectType.WmoDoodad => WorldSceneSelectionKind.WmoDoodad,
        ObjectType.Mdx => WorldSceneSelectionKind.Mdx,
        _ => throw new ArgumentOutOfRangeException(nameof(objectType), objectType, "Pick hits are WMO, MDX or WMO-doodad hits."),
    };

    private static void Reorder(List<SceneObjectPickHit> hits, IReadOnlyList<int> orderedIds)
    {
        SceneObjectPickHit[] original = hits.ToArray();
        hits.Clear();
        foreach (int id in orderedIds)
            hits.Add(original[id]);
    }
}
