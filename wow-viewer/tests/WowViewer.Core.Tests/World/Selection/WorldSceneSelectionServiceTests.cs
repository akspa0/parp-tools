using System.Numerics;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Selection;
using Xunit;

namespace WowViewer.Core.Tests.World.Selection;

public sealed class WorldSceneSelectionServiceTests
{
    private static readonly WorldSceneSelectionPolicy Unlimited = WorldSceneSelectionPolicy.Unlimited;

    private static WorldSceneSelectionCandidate Hit(
        int id,
        float distance,
        WorldSceneSelectionKind kind = WorldSceneSelectionKind.Mdx,
        bool sharesChunk = false,
        int chunkGrid = int.MaxValue,
        float pointDistSq = float.MaxValue,
        Vector3? point = null,
        Vector3? min = null,
        Vector3? max = null)
    {
        Vector3 p = point ?? new Vector3(id * 100f, 0f, 0f);
        return new WorldSceneSelectionCandidate(id, kind, distance, min ?? p - Vector3.One, max ?? p + Vector3.One, p, pointDistSq, sharesChunk, chunkGrid);
    }

    private static int[] Ids(WorldSceneSelectionResult result) => result.RankedIds.ToArray();

    [Fact]
    public void Click_RanksClickedChunkThenGridThenSelectionPointThenRayDistance()
    {
        var snapshot = new WorldSceneSelectionSnapshot(
        [
            Hit(0, distance: 1f, chunkGrid: 3),
            Hit(1, distance: 9f, chunkGrid: 1, pointDistSq: 50f),
            Hit(2, distance: 8f, chunkGrid: 1, pointDistSq: 10f),
            Hit(3, distance: 2f, chunkGrid: 1, pointDistSq: 10f),
        ]);

        var result = WorldSceneSelectionService.Select(new WorldSceneSelectionRequest(WorldSceneSelectionMode.Click, Unlimited), snapshot);

        Assert.Equal(WorldSceneSelectionStatus.Hit, result.Status);
        Assert.Equal([3, 2, 1, 0], Ids(result));
        Assert.Equal(3, result.BestId);
    }

    [Fact]
    public void Click_KeepsOnlyClickedChunkHits_WhenAnyShareTheClickedChunk()
    {
        var snapshot = new WorldSceneSelectionSnapshot(
        [
            Hit(0, distance: 1f, chunkGrid: 1),
            Hit(1, distance: 5f, sharesChunk: true, chunkGrid: 0),
            Hit(2, distance: 3f, sharesChunk: true, chunkGrid: 0),
        ]);

        var withChunk = WorldSceneSelectionService.Select(new WorldSceneSelectionRequest(WorldSceneSelectionMode.Click, Unlimited, HasClickedChunk: true), snapshot);
        var withoutChunk = WorldSceneSelectionService.Select(new WorldSceneSelectionRequest(WorldSceneSelectionMode.Click, Unlimited, HasClickedChunk: false), snapshot);

        Assert.Equal([2, 1], Ids(withChunk));
        Assert.Equal([2, 1, 0], Ids(withoutChunk));
    }

    [Fact]
    public void Select_DropsHitsBeyondTheHoverRange_OnlyWhenTheRangeIsLimited()
    {
        var snapshot = new WorldSceneSelectionSnapshot([Hit(0, distance: 10f), Hit(1, distance: 100f), Hit(2, distance: float.NaN)]);
        var limited = new WorldSceneSelectionPolicy(true, 50f, Vector3.Zero);

        var clipped = WorldSceneSelectionService.Select(new WorldSceneSelectionRequest(WorldSceneSelectionMode.Click, limited), snapshot);
        var open = WorldSceneSelectionService.Select(new WorldSceneSelectionRequest(WorldSceneSelectionMode.Click, Unlimited), snapshot);

        Assert.Equal([0], Ids(clipped));
        Assert.Equal(3, open.RankedIds.Count);
    }

    [Fact]
    public void HoverRay_HonoursVisibility_AndFallsThroughEnclosingWmo()
    {
        var shell = Hit(0, distance: 1f, kind: WorldSceneSelectionKind.Wmo, point: Vector3.Zero, min: new Vector3(-50f), max: new Vector3(50f));
        var inside = Hit(1, distance: 20f, kind: WorldSceneSelectionKind.Mdx, point: new Vector3(5f, 5f, 5f));
        var snapshot = new WorldSceneSelectionSnapshot([shell, inside]);

        var both = WorldSceneSelectionService.Select(new WorldSceneSelectionRequest(WorldSceneSelectionMode.HoverRay, Unlimited), snapshot);
        var wmosOnly = WorldSceneSelectionService.Select(new WorldSceneSelectionRequest(WorldSceneSelectionMode.HoverRay, Unlimited, DoodadsVisible: false), snapshot);
        var nothingVisible = WorldSceneSelectionService.Select(new WorldSceneSelectionRequest(WorldSceneSelectionMode.HoverRay, Unlimited, WmosVisible: false, DoodadsVisible: false), snapshot);

        Assert.Equal(1, both.BestId);                 // the enclosing WMO falls through to its interior object
        Assert.Equal(0, wmosOnly.BestId);             // alone, the WMO itself is hovered
        Assert.Equal(WorldSceneSelectionStatus.NoHit, nothingVisible.Status);
    }

    [Fact]
    public void HoverRay_NearestFirst_IsStableForEqualDistances()
    {
        var snapshot = new WorldSceneSelectionSnapshot([Hit(4, 7f), Hit(2, 3f), Hit(9, 3f), Hit(1, 5f)]);

        var result = WorldSceneSelectionService.Select(new WorldSceneSelectionRequest(WorldSceneSelectionMode.HoverRay, Unlimited), snapshot);

        Assert.Equal([2, 9, 1, 4], Ids(result));
    }

    [Fact]
    public void ResolveHoverRay_LiquidWinsOnlyWhenStrictlyNearerAndInRange()
    {
        var snapshot = new WorldSceneSelectionSnapshot([Hit(0, 10f)]);
        var request = new WorldSceneSelectionRequest(WorldSceneSelectionMode.HoverRay, new WorldSceneSelectionPolicy(true, 40f, Vector3.Zero));

        var equal = WorldSceneSelectionService.ResolveHoverRay(request, snapshot, [new WorldSceneRayTarget(7, 10f)]);
        var nearer = WorldSceneSelectionService.ResolveHoverRay(request, snapshot, [new WorldSceneRayTarget(7, 4f), new WorldSceneRayTarget(8, 2f), new WorldSceneRayTarget(9, -1f)]);
        var liquidOnlyOutOfRange = WorldSceneSelectionService.ResolveHoverRay(request, new WorldSceneSelectionSnapshot([]), [new WorldSceneRayTarget(7, 90f)]);

        Assert.Equal((WorldSceneHoverRayTarget.SceneObject, 0, 10f), (equal.Target, equal.Id, equal.Distance));
        Assert.Equal((WorldSceneHoverRayTarget.LiquidBody, 8, 2f), (nearer.Target, nearer.Id, nearer.Distance));
        Assert.Equal(WorldSceneSelectionStatus.NoHit, liquidOnlyOutOfRange.Status);
    }

    [Fact]
    public void Brush_NearestToCursor_ThenDepthWithinEpsilon_AndCountsEligibleTargets()
    {
        var camera = Vector3.Zero;
        var policy = new WorldSceneSelectionPolicy(true, 100f, camera);
        WorldSceneBrushCandidate[] candidates =
        [
            new(0, 5.000f, 0.9f, new Vector3(10f, 0f, 0f)),
            new(1, 5.005f, 0.2f, new Vector3(20f, 0f, 0f)),   // within 0.01 of #0 and nearer in depth
            new(2, 1.000f, 0.5f, new Vector3(500f, 0f, 0f)),  // nearest to cursor but out of range
        ];

        var result = WorldSceneSelectionService.SelectBrush(policy, candidates);

        Assert.Equal(WorldSceneSelectionStatus.Hit, result.Status);
        Assert.Equal(1, result.BestId);
        Assert.Equal(2, result.EligibleCount);
    }

    [Theory]
    [InlineData(true, 10f, true, 9.995f, false, WorldSceneHoverSource.Scene)]  // not nearer by more than 0.01
    [InlineData(true, 10f, true, 9.98f, false, WorldSceneHoverSource.Pm4)]
    [InlineData(true, 10f, true, 50f, true, WorldSceneHoverSource.Pm4)]      // depth ignored
    [InlineData(false, 0f, true, 50f, false, WorldSceneHoverSource.Pm4)]
    [InlineData(true, 10f, false, 0f, false, WorldSceneHoverSource.Scene)]
    [InlineData(false, 0f, false, 0f, false, WorldSceneHoverSource.None)]
    public void ChooseHoverRaySource_FollowsTheDepthPolicy(bool hasScene, float sceneDistance, bool hasPm4, float pm4Distance, bool ignoreDepth, WorldSceneHoverSource expected)
    {
        Assert.Equal(expected, WorldSceneSelectionService.ChooseHoverRaySource(hasScene, sceneDistance, hasPm4, pm4Distance, ignoreDepth));
    }

    [Theory]
    [InlineData(1f, 0.50f, 9f, 0.40f, false, false)]  // scene clearly nearer in depth
    [InlineData(9f, 0.40f, 1f, 0.50f, false, true)]   // PM4 clearly nearer in depth
    [InlineData(2f, 0.500f, 3f, 0.501f, false, true)] // same depth band: smaller cursor distance
    [InlineData(4f, 0.500f, 3f, 0.501f, false, false)]
    [InlineData(99f, 0.9f, 1f, 0.1f, true, true)]     // depth ignored
    public void PreferPm4Brush_FollowsTheDepthBandThenCursorDistance(float pm4DistSq, float pm4Depth, float sceneDistSq, float sceneDepth, bool ignoreDepth, bool expected)
    {
        Assert.Equal(expected, WorldSceneSelectionService.PreferPm4Brush(pm4DistSq, pm4Depth, sceneDistSq, sceneDepth, ignoreDepth));
    }

    [Fact]
    public void InvalidInputs_NeverProduceASelection()
    {
        var snapshot = new WorldSceneSelectionSnapshot([Hit(0, 1f)]);
        WorldSceneSelectionPolicy[] badPolicies =
        [
            new(true, float.NaN, Vector3.Zero),
            new(true, -1f, Vector3.Zero),
            new(true, 10f, new Vector3(float.NaN, 0f, 0f)),
        ];

        foreach (WorldSceneSelectionPolicy policy in badPolicies)
            Assert.Equal(WorldSceneSelectionStatus.InvalidRequest, WorldSceneSelectionService.SelectBrush(policy, [new WorldSceneBrushCandidate(0, 1f, 1f, Vector3.Zero)]).Status);

        // Ray picks never used the camera position, so only the range makes them invalid.
        foreach (WorldSceneSelectionPolicy policy in badPolicies[..2])
            Assert.Equal(WorldSceneSelectionStatus.InvalidRequest, WorldSceneSelectionService.Select(new WorldSceneSelectionRequest(WorldSceneSelectionMode.Click, policy), snapshot).Status);
        Assert.Equal(WorldSceneSelectionStatus.Hit, WorldSceneSelectionService.Select(new WorldSceneSelectionRequest(WorldSceneSelectionMode.Click, badPolicies[2]), snapshot).Status);

        var badMode = WorldSceneSelectionService.Select(new WorldSceneSelectionRequest((WorldSceneSelectionMode)42, Unlimited), snapshot);
        var duplicateIds = WorldSceneSelectionService.Select(new WorldSceneSelectionRequest(WorldSceneSelectionMode.Click, Unlimited), new WorldSceneSelectionSnapshot([Hit(3, 1f), Hit(3, 2f)]));
        var negativeId = WorldSceneSelectionService.Select(new WorldSceneSelectionRequest(WorldSceneSelectionMode.Click, Unlimited), new WorldSceneSelectionSnapshot([Hit(-2, 1f)]));
        var clickModeForRay = WorldSceneSelectionService.ResolveHoverRay(new WorldSceneSelectionRequest(WorldSceneSelectionMode.Click, Unlimited), snapshot, []);

        Assert.All([badMode, duplicateIds, negativeId], r => Assert.Equal((WorldSceneSelectionStatus.InvalidRequest, -1), (r.Status, r.BestId)));
        Assert.Equal(WorldSceneSelectionStatus.InvalidRequest, clickModeForRay.Status);
    }

    /// <summary>
    /// Equivalence with the pre-extraction viewer algorithm (WorldScene.CollectSceneObjectPickHits
    /// sort/filter and TryBuildHoveredSceneInfoByRay), transcribed below as the reference. Many
    /// ties are generated on purpose: the unstable List.Sort must give the same permutation.
    /// </summary>
    [Fact]
    public void RandomSnapshots_MatchThePreExtractionAlgorithm()
    {
        var random = new Random(228_004);
        for (int trial = 0; trial < 600; trial++)
        {
            int count = random.Next(0, 40);
            bool hasClickedChunk = random.Next(2) == 0;
            bool limit = random.Next(3) == 0;
            float maxDistance = random.Next(5, 60);
            bool wmosVisible = random.Next(4) != 0;
            bool doodadsVisible = random.Next(4) != 0;
            var candidates = new List<WorldSceneSelectionCandidate>(count);
            for (int i = 0; i < count; i++)
            {
                var kind = (WorldSceneSelectionKind)random.Next(3);
                Vector3 point = new(random.Next(-20, 20), random.Next(-20, 20), random.Next(-5, 5));
                Vector3 half = new(random.Next(1, 25), random.Next(1, 25), random.Next(1, 10));
                candidates.Add(new WorldSceneSelectionCandidate(
                    i, kind, random.Next(0, 80), point - half, point + half, point,
                    random.Next(3) == 0 ? float.MaxValue : random.Next(0, 6),
                    random.Next(3) == 0,
                    random.Next(3) == 0 ? int.MaxValue : random.Next(0, 4)));
            }

            var policy = new WorldSceneSelectionPolicy(limit, maxDistance, Vector3.Zero);
            var snapshot = new WorldSceneSelectionSnapshot(candidates);

            // Click: the viewer filtered by range while collecting, then chunk-filtered and sorted.
            List<WorldSceneSelectionCandidate> referenceClick = candidates.Where(c => !limit || c.RayDistance <= maxDistance).ToList();
            if (hasClickedChunk && referenceClick.Any(static hit => hit.SharesClickedChunk))
                referenceClick.RemoveAll(static hit => !hit.SharesClickedChunk);
            referenceClick.Sort(static (left, right) =>
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
            });
            var click = WorldSceneSelectionService.Select(new WorldSceneSelectionRequest(WorldSceneSelectionMode.Click, policy, hasClickedChunk), snapshot);
            Assert.Equal(referenceClick.Select(c => c.Id), click.RankedIds);

            // Hover: the viewer hovered over the click-ranked list (no clicked chunk), then filtered.
            var hoverInput = WorldSceneSelectionService.Select(new WorldSceneSelectionRequest(WorldSceneSelectionMode.Click, policy), snapshot).RankedIds
                .Select(id => candidates[id]).ToList();
            var visible = hoverInput.Where(hit => hit.Kind switch
            {
                WorldSceneSelectionKind.Wmo or WorldSceneSelectionKind.WmoDoodad => wmosVisible,
                WorldSceneSelectionKind.Mdx => doodadsVisible,
                _ => false,
            }).ToList();
            var fall = WmoContainerFallThroughFilter.ApplyFallThrough(visible
                .Select((hit, index) => new WmoContainerFallThroughFilter.CandidateObject(index, hit.Kind == WorldSceneSelectionKind.Wmo, hit.BoundsMin, hit.BoundsMax, hit.SelectionPoint))
                .ToList());
            IEnumerable<WorldSceneSelectionCandidate> filtered = fall.Count == visible.Count ? visible : fall.Select(f => visible[f.Id]);
            int? referenceBest = filtered.Where(hit => !limit || hit.RayDistance <= maxDistance)
                .OrderBy(static hit => hit.RayDistance)
                .Select(static hit => (int?)hit.Id)
                .FirstOrDefault();

            var hover = WorldSceneSelectionService.Select(
                new WorldSceneSelectionRequest(WorldSceneSelectionMode.HoverRay, policy, WmosVisible: wmosVisible, DoodadsVisible: doodadsVisible),
                new WorldSceneSelectionSnapshot(hoverInput));
            Assert.Equal(referenceBest ?? -1, hover.BestId);
        }
    }
}
