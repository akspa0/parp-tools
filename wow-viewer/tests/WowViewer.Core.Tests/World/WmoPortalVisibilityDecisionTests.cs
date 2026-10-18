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

    [Fact]
    public void ExteriorCameraInsideExteriorBoundsDoesNotTriggerInteriorMode()
    {
        // Camera is inside Group 0 bounds [-30, 0], which is flagged Exterior (0x8)
        WmoPortalVisibilityDecision decision = WmoPortalVisibilityEvaluator.Evaluate(
            Groups,
            [DoorwayPortal],
            new Vector3(-10f, 0f, 0f),
            _ => true);

        Assert.Equal(WmoPortalVisibilityMode.Exterior, decision.Diagnostics.Mode);
        Assert.Equal([0, 1], decision.VisibleGroupIndices);
    }

    [Fact]
    public void MultipleInteriorGroupsContainCameraSeedsBothGroupsAtDepthZero()
    {
        // Two interior groups meeting at a doorway threshold [9, 11]
        WmoPortalVisibilityGroup roomA = new(1, 0, new Vector3(0f, -5f, -5f), new Vector3(10f, 5f, 5f));
        WmoPortalVisibilityGroup roomB = new(2, 0, new Vector3(9f, -5f, -5f), new Vector3(20f, 5f, 5f));

        // Camera is standing on the threshold at X = 9.5
        WmoPortalVisibilityDecision decision = WmoPortalVisibilityEvaluator.Evaluate(
            [roomA, roomB],
            [DoorwayPortal with
            {
                PortalIndex = 10,
                PlaneDistance = -9.5f,
                Vertices =
                [
                    new Vector3(9.5f, -2f, -2f),
                    new Vector3(9.5f, 2f, -2f),
                    new Vector3(9.5f, 2f, 2f),
                    new Vector3(9.5f, -2f, 2f),
                ],
                References =
                [
                    new WmoPortalVisibilityReference(1, -1),
                    new WmoPortalVisibilityReference(2, 1),
                ]
            }],
            new Vector3(9.5f, 0f, 0f),
            interiorMaximumDepth: 0);

        Assert.Equal(WmoPortalVisibilityMode.Interior, decision.Diagnostics.Mode);
        // Both rooms are seeded at depth 0 so neither room unloads when standing on the threshold
        Assert.Contains(1, decision.VisibleGroupIndices);
        Assert.Contains(2, decision.VisibleGroupIndices);
    }

    [Fact]
    public void DeepPortalChainTraversesToDepthEight()
    {
        // Linear chain of 4 rooms: 0 -> 1 -> 2 -> 3
        WmoPortalVisibilityGroup room0 = new(0, 0, new Vector3(0f, -2f, -2f), new Vector3(10f, 2f, 2f));
        WmoPortalVisibilityGroup room1 = new(1, 0, new Vector3(10f, -2f, -2f), new Vector3(20f, 2f, 2f));
        WmoPortalVisibilityGroup room2 = new(2, 0, new Vector3(20f, -2f, -2f), new Vector3(30f, 2f, 2f));
        WmoPortalVisibilityGroup room3 = new(3, 0, new Vector3(30f, -2f, -2f), new Vector3(40f, 2f, 2f));

        WmoPortalVisibilityPortal p01 = new(1,
            [new Vector3(10f, -1f, -1f), new Vector3(10f, 1f, -1f), new Vector3(10f, 1f, 1f), new Vector3(10f, -1f, 1f)],
            Vector3.UnitX, -10f,
            [new WmoPortalVisibilityReference(0, -1), new WmoPortalVisibilityReference(1, 1)]);

        WmoPortalVisibilityPortal p12 = new(2,
            [new Vector3(20f, -1f, -1f), new Vector3(20f, 1f, -1f), new Vector3(20f, 1f, 1f), new Vector3(20f, -1f, 1f)],
            Vector3.UnitX, -20f,
            [new WmoPortalVisibilityReference(1, -1), new WmoPortalVisibilityReference(2, 1)]);

        WmoPortalVisibilityPortal p23 = new(3,
            [new Vector3(30f, -1f, -1f), new Vector3(30f, 1f, -1f), new Vector3(30f, 1f, 1f), new Vector3(30f, -1f, 1f)],
            Vector3.UnitX, -30f,
            [new WmoPortalVisibilityReference(2, -1), new WmoPortalVisibilityReference(3, 1)]);

        // Depth 8 traversal sees all 4 rooms through the doors
        WmoPortalVisibilityDecision decisionDepth8 = WmoPortalVisibilityEvaluator.Evaluate(
            [room0, room1, room2, room3],
            [p01, p12, p23],
            new Vector3(5f, 0f, 0f),
            interiorMaximumDepth: 8);

        Assert.Equal([0, 1, 2, 3], decisionDepth8.VisibleGroupIndices);

        // Depth 1 traversal stops after 1 doorway (rooms 2 and 3 omitted)
        WmoPortalVisibilityDecision decisionDepth1 = WmoPortalVisibilityEvaluator.Evaluate(
            [room0, room1, room2, room3],
            [p01, p12, p23],
            new Vector3(5f, 0f, 0f),
            interiorMaximumDepth: 1);

        Assert.Equal([0, 1], decisionDepth1.VisibleGroupIndices);
    }

    [Fact]
    public void CameraOutsideExteriorPortalDoesNotTriggerInteriorMode()
    {
        // Street exterior group and shop interior group with a doorway portal at X = 5
        WmoPortalVisibilityGroup street = new(0, 0x8, new Vector3(-50f, -10f, -5f), new Vector3(5f, 10f, 10f));
        WmoPortalVisibilityGroup shop = new(1, 0, new Vector3(5f, -10f, -5f), new Vector3(30f, 10f, 10f));
        WmoPortalVisibilityPortal doorway = new(1,
            [new Vector3(5f, -2f, 0f), new Vector3(5f, 2f, 0f), new Vector3(5f, 2f, 4f), new Vector3(5f, -2f, 4f)],
            Vector3.UnitX, -5f,
            [new WmoPortalVisibilityReference(0, -1), new WmoPortalVisibilityReference(1, 1)]);

        // Camera is standing on the street at X = 4.0, outside the shop door
        WmoPortalVisibilityDecision decision = WmoPortalVisibilityEvaluator.Evaluate(
            [street, shop],
            [doorway],
            new Vector3(4f, 0f, 1.8f),
            _ => true);

        // Ground truth: camera is outdoors on the street, so Mode must be Exterior and street must remain visible
        Assert.Equal(WmoPortalVisibilityMode.Exterior, decision.Diagnostics.Mode);
        Assert.Contains(0, decision.VisibleGroupIndices);
    }

    [Fact]
    public void CameraInsideAabbOfInteriorRoomFailsToExteriorWhenOnExteriorSideOfPortal()
    {
        // Simulates an angled or overhanging interior room whose axis-aligned bounding box covers [0, 20],
        // but whose doorway portal is at X = 10 separating exterior (X <= 10) from interior (X >= 10).
        WmoPortalVisibilityGroup street = new(0, 0x8, new Vector3(-50f, -20f, -5f), new Vector3(10f, 20f, 20f));
        WmoPortalVisibilityGroup shop = new(1, 0, new Vector3(0f, -10f, -5f), new Vector3(20f, 10f, 10f));
        WmoPortalVisibilityPortal doorway = new(1,
            [new Vector3(10f, -2f, 0f), new Vector3(10f, 2f, 0f), new Vector3(10f, 2f, 4f), new Vector3(10f, -2f, 4f)],
            Vector3.UnitX, -10f,
            [new WmoPortalVisibilityReference(0, -1), new WmoPortalVisibilityReference(1, 1)]);

        // Camera is at X = 5. Even though X = 5 is inside shop's AABB [0, 20], it is on the exterior side of the portal (X < 10)
        WmoPortalVisibilityDecision decision = WmoPortalVisibilityEvaluator.Evaluate(
            [street, shop],
            [doorway],
            new Vector3(5f, 0f, 1.8f),
            _ => true);

        // Camera is outside on the street, so it must not falsely trigger interior mode and drop the street
        Assert.Equal(WmoPortalVisibilityMode.Exterior, decision.Diagnostics.Mode);
        Assert.Contains(0, decision.VisibleGroupIndices);
    }

    [Fact]
    public void CameraInsideInteriorRoomStillCorrectlyTriggersInteriorMode()
    {
        WmoPortalVisibilityGroup street = new(0, 0x8, new Vector3(-50f, -20f, -5f), new Vector3(10f, 20f, 20f));
        WmoPortalVisibilityGroup shop = new(1, 0, new Vector3(0f, -10f, -5f), new Vector3(20f, 10f, 10f));
        WmoPortalVisibilityPortal doorway = new(1,
            [new Vector3(10f, -2f, 0f), new Vector3(10f, 2f, 0f), new Vector3(10f, 2f, 4f), new Vector3(10f, -2f, 4f)],
            Vector3.UnitX, -10f,
            [new WmoPortalVisibilityReference(0, -1), new WmoPortalVisibilityReference(1, 1)]);

        // Camera is at X = 15, strictly inside the shop on the interior side of the doorway portal
        WmoPortalVisibilityDecision decision = WmoPortalVisibilityEvaluator.Evaluate(
            [street, shop],
            [doorway],
            new Vector3(15f, 0f, 1.8f),
            _ => true);

        Assert.Equal(WmoPortalVisibilityMode.Interior, decision.Diagnostics.Mode);
        Assert.Contains(1, decision.VisibleGroupIndices);
        // Traversal through doorway reaches the street as well
        Assert.Contains(0, decision.VisibleGroupIndices);
    }

    [Fact]
    public void OutdoorGroupWithoutPortalsIsClassifiedAsExterior()
    {
        WmoPortalVisibilityGroup parkWalkway = new(0, 0, new Vector3(-20f, -20f, -5f), new Vector3(20f, 20f, 5f), HasPortals: false);
        Assert.True(parkWalkway.IsExterior);
        Assert.False(parkWalkway.IsInterior);
    }

    [Fact]
    public void OutdoorGroupWithExteriorLightingIsClassifiedAsExterior()
    {
        WmoPortalVisibilityGroup outdoorPlaza = new(0, WmoPortalVisibilityGroup.UsesExteriorLightingFlag, new Vector3(-20f, -20f, -5f), new Vector3(20f, 20f, 5f), HasPortals: true);
        Assert.True(outdoorPlaza.IsExterior);
        Assert.False(outdoorPlaza.IsInterior);
    }

    [Fact]
    public void IndoorGroupWithoutExteriorFlagsIsClassifiedAsInterior()
    {
        WmoPortalVisibilityGroup indoorRoom = new(1, WmoPortalVisibilityGroup.IndoorFlag, new Vector3(0f, -10f, -5f), new Vector3(20f, 10f, 10f), HasPortals: true);
        Assert.False(indoorRoom.IsExterior);
        Assert.True(indoorRoom.IsInterior);
    }

    [Fact]
    public void CameraInOutdoorParkGroupRemainsInExteriorModeAndRendersAllFrustumGroups()
    {
        // Composite city layout:
        // Group 0: Entrance archway (HasPortals = false, Flags = 0)
        // Group 1: Park pathway (HasPortals = false, Flags = 0x40)
        // Group 2: Distant stone tower (HasPortals = false, Flags = 0x10000)
        // Group 3: Park shop interior (HasPortals = true, Flags = 0x2000)
        WmoPortalVisibilityGroup archway = new(0, 0, new Vector3(-30f, -10f, -5f), new Vector3(-10f, 10f, 10f), HasPortals: false);
        WmoPortalVisibilityGroup parkPath = new(1, WmoPortalVisibilityGroup.UsesExteriorLightingFlag, new Vector3(-10f, -10f, -5f), new Vector3(30f, 10f, 10f), HasPortals: false);
        WmoPortalVisibilityGroup distantTower = new(2, WmoPortalVisibilityGroup.AlwaysVisibleFlag, new Vector3(40f, -10f, -5f), new Vector3(60f, 10f, 30f), HasPortals: false);
        WmoPortalVisibilityGroup shop = new(3, WmoPortalVisibilityGroup.IndoorFlag, new Vector3(0f, 20f, -5f), new Vector3(20f, 40f, 10f), HasPortals: true);

        // Portal into the shop from the park path facing +X (towards the path)
        WmoPortalVisibilityPortal shopDoor = new(
            0,
            [
                new Vector3(10f, 20f, -2f),
                new Vector3(10f, 20f, 2f),
                new Vector3(10f, 22f, 2f),
                new Vector3(10f, 22f, -2f),
            ],
            Vector3.UnitX,
            -10f,
            [
                new WmoPortalVisibilityReference(1, 1),
                new WmoPortalVisibilityReference(3, -1),
            ]);

        // Camera is standing in the entrance archway at (-20, 0, 0), looking into the park
        WmoPortalVisibilityDecision decision = WmoPortalVisibilityEvaluator.Evaluate(
            [archway, parkPath, distantTower, shop],
            [shopDoor],
            new Vector3(-20f, 0f, 0f),
            groupIndex => groupIndex != 3); // shop is not in frustum, but outdoor groups are

        Assert.Equal(WmoPortalVisibilityMode.Exterior, decision.Diagnostics.Mode);
        // Frustum-visible outdoor groups must all be admitted
        Assert.Contains(0, decision.VisibleGroupIndices);
        Assert.Contains(1, decision.VisibleGroupIndices);
        Assert.Contains(2, decision.VisibleGroupIndices);
        // Shop interior is NOT in frustum or line of sight, so it is culled
        Assert.DoesNotContain(3, decision.VisibleGroupIndices);
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
