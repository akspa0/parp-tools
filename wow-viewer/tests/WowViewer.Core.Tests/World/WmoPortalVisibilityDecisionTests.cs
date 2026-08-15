using System.Numerics;
using WowViewer.Core.Runtime.World;

namespace WowViewer.Core.Tests.World;

public sealed class WmoPortalVisibilityDecisionTests
{
    [Fact]
    public void ExteriorCameraReachesInteriorOnlyThroughPortalVolume()
    {
        WmoPortalVisibilityDecision decision = WmoPortalVisibilityEvaluator.Evaluate(
            Groups,
            [DoorwayPortal],
            new Vector3(-80f, 0f, 0f),
            _ => true);

        Assert.Equal(WmoPortalVisibilityMode.Exterior, decision.Diagnostics.Mode);
        Assert.Equal([0, 1], decision.VisibleGroupIndices);
        Assert.True(decision.Diagnostics.UsedPortalClip);
        Assert.Equal(1, decision.Diagnostics.TestedPortalCount);
    }

    [Fact]
    public void OffDoorwayInteriorGroupIsRejectedFromExteriorView()
    {
        WmoPortalVisibilityDecision decision = WmoPortalVisibilityEvaluator.Evaluate(
            [
                Groups[0],
                Groups[1] with
                {
                    BoundsMin = new Vector3(6f, 4f, -1f),
                    BoundsMax = new Vector3(10f, 6f, 1f),
                },
            ],
            [DoorwayPortal],
            new Vector3(-80f, 0f, 0f),
            groupIndex => groupIndex == 0);

        Assert.Equal([0], decision.VisibleGroupIndices);
        Assert.Null(decision.Diagnostics.FallbackReason);
    }

    [Fact]
    public void ExteriorPortalRequiresCameraOnSourceSide()
    {
        WmoPortalVisibilityDecision decision = WmoPortalVisibilityEvaluator.Evaluate(
            Groups,
            [DoorwayPortal],
            new Vector3(80f, 0f, 0f),
            _ => true);

        Assert.Equal(WmoPortalVisibilityMode.Exterior, decision.Diagnostics.Mode);
        Assert.Equal([0], decision.VisibleGroupIndices);
        Assert.Equal(1, decision.Diagnostics.RejectedPortalCount);
    }

    [Fact]
    public void InteriorCameraTraversesBackThroughPortal()
    {
        WmoPortalVisibilityDecision decision = WmoPortalVisibilityEvaluator.Evaluate(
            Groups,
            [DoorwayPortal],
            new Vector3(7f, 0f, 0f),
            _ => true);

        Assert.Equal(WmoPortalVisibilityMode.Interior, decision.Diagnostics.Mode);
        Assert.Equal(1, decision.Diagnostics.SourceGroupIndex);
        Assert.Equal([0, 1], decision.VisibleGroupIndices);
    }

    [Fact]
    public void MalformedPortalFailsOpenToAllGroups()
    {
        WmoPortalVisibilityDecision decision = WmoPortalVisibilityEvaluator.Evaluate(
            Groups,
            [DoorwayPortal with { Vertices = [new Vector3(5f, 0f, 0f)] }],
            new Vector3(-80f, 0f, 0f));

        Assert.Equal(WmoPortalVisibilityMode.ConservativeFallback, decision.Diagnostics.Mode);
        Assert.Equal("portal_geometry_invalid", decision.Diagnostics.FallbackReason);
        Assert.Equal([0, 1], decision.VisibleGroupIndices);
    }

    [Fact]
    public void MissingPortalDataFailsOpen()
    {
        WmoPortalVisibilityDecision decision = WmoPortalVisibilityEvaluator.Evaluate(
            Groups,
            [],
            new Vector3(-20f, 0f, 0f));

        Assert.Equal(WmoPortalVisibilityMode.ConservativeFallback, decision.Diagnostics.Mode);
        Assert.Equal("portal_data_absent", decision.Diagnostics.FallbackReason);
        Assert.Equal([0, 1], decision.VisibleGroupIndices);
    }

    [Fact]
    public void CameraOnPortalBoundaryFallsBackConservatively()
    {
        WmoPortalVisibilityDecision decision = WmoPortalVisibilityEvaluator.Evaluate(
            Groups,
            [DoorwayPortal],
            new Vector3(5f, 0f, 0f));

        Assert.Equal(WmoPortalVisibilityMode.ConservativeFallback, decision.Diagnostics.Mode);
        Assert.Equal("camera_on_portal_plane", decision.Diagnostics.FallbackReason);
        Assert.Equal([0, 1], decision.VisibleGroupIndices);
    }

    [Fact]
    public void PortalCycleRemainsBounded()
    {
        WmoPortalVisibilityDecision decision = WmoPortalVisibilityEvaluator.Evaluate(
            Groups,
            [DoorwayPortal],
            new Vector3(7f, 0f, 0f),
            interiorMaximumDepth: 8,
            maxVisitsPerGroup: 2);

        Assert.Equal([0, 1], decision.VisibleGroupIndices);
        Assert.InRange(decision.Diagnostics.VisitedGroupCount, 1, 4);
        Assert.InRange(decision.Diagnostics.MaxDepthReached, 0, 8);
    }

    [Fact]
    public void MaximumDepthStopsPortalTraversal()
    {
        WmoPortalVisibilityDecision decision = WmoPortalVisibilityEvaluator.Evaluate(
            Groups,
            [DoorwayPortal],
            new Vector3(7f, 0f, 0f),
            interiorMaximumDepth: 0);

        Assert.Equal([1], decision.VisibleGroupIndices);
        Assert.Equal(0, decision.Diagnostics.TestedPortalCount);
    }

    [Fact]
    public void VisitCapacityFallsBackInsteadOfDroppingGeometry()
    {
        WmoPortalVisibilityDecision decision = WmoPortalVisibilityEvaluator.Evaluate(
            Groups,
            [DoorwayPortal, DoorwayPortal with { PortalIndex = 4 }],
            new Vector3(7f, 0f, 0f),
            maxVisitsPerGroup: 1);

        Assert.Equal(WmoPortalVisibilityMode.ConservativeFallback, decision.Diagnostics.Mode);
        Assert.Equal("portal_visit_capacity_reached", decision.Diagnostics.FallbackReason);
        Assert.Equal([0, 1], decision.VisibleGroupIndices);
    }

    private static readonly WmoPortalVisibilityGroup[] Groups =
    [
        new(0, 0x8, new Vector3(-30f, -3f, -3f), new Vector3(0f, 3f, 3f)),
        new(1, 0, new Vector3(6f, -1f, -1f), new Vector3(10f, 1f, 1f)),
    ];

    private static readonly WmoPortalVisibilityPortal DoorwayPortal = new(
        3,
        [
            new Vector3(5f, -2f, -2f),
            new Vector3(5f, 2f, -2f),
            new Vector3(5f, 2f, 2f),
            new Vector3(5f, -2f, 2f),
        ],
        Vector3.UnitX,
        -5f,
        [
            new WmoPortalVisibilityReference(0, -1),
            new WmoPortalVisibilityReference(1, 1),
        ]);
}
