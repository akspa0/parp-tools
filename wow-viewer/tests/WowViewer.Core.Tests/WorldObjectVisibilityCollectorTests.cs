using System.Numerics;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Visibility;

namespace WowViewer.Core.Tests;

public sealed class WorldObjectVisibilityCollectorTests
{
    [Fact]
    public void CollectVisibleWmos_RejectsRearOffFrustumCandidate()
    {
        WorldVisibilityFrame frame = new();
        WorldObjectVisibilityContext context = CreateContext(cameraForward: -Vector3.UnitY);
        WorldObjectInstance instance = CreateInstance("wmo://rear", new Vector3(0f, 2000f, 0f), halfExtent: 10f);

        int culled = WorldObjectVisibilityCollector.CollectVisibleWmos(
            frame,
            [instance],
            context,
            static _ => false,
            static (_, _) => false,
            static _ => true,
            static (_, _) => { });

        Assert.Equal(1, culled);
        Assert.Empty(frame.VisibleWmos);
    }

    [Fact]
    public void CollectVisibleMdx_PreservesNearHoldCandidate()
    {
        WorldVisibilityFrame frame = new();
        WorldObjectVisibilityContext context = CreateContext(cameraForward: Vector3.UnitY);
        WorldObjectInstance instance = CreateInstance("mdx://near", new Vector3(0f, -64f, 0f), halfExtent: 2f);

        int culled = WorldObjectVisibilityCollector.CollectVisibleMdx(
            frame,
            [instance],
            context,
            static _ => false,
            static (_, _) => false,
            static _ => true,
            static (_, _) => { });

        Assert.Equal(0, culled);
        Assert.Single(frame.VisibleMdx);
        Assert.Equal(1.0f, frame.VisibleMdx[0].OpaqueFade);
        Assert.Equal(1.0f, frame.VisibleMdx[0].TransparentFade);
    }

    [Fact]
    public void CollectVisibleMdx_CullsSmallDoodadBeyondDistanceLimit()
    {
        WorldVisibilityFrame frame = new();
        WorldObjectVisibilityContext context = CreateContext(cullSmallDoodadsOnly: true);
        WorldObjectInstance instance = CreateInstance("mdx://small", new Vector3(0f, 6000f, 0f), halfExtent: 1f);

        int culled = WorldObjectVisibilityCollector.CollectVisibleMdx(
            frame,
            [instance],
            context,
            static _ => false,
            static (_, _) => true,
            static _ => true,
            static (_, _) => { });

        Assert.Equal(1, culled);
        Assert.Empty(frame.VisibleMdx);
    }

    [Fact]
    public void CollectVisibleMdx_ComputesDeterministicFadeForFarVisibleCandidate()
    {
        WorldVisibilityFrame frame = new();
        WorldObjectVisibilityContext context = CreateContext(fogEnd: 1000f);
        WorldObjectInstance instance = CreateInstance("mdx://fade", new Vector3(0f, 1000f, 0f), halfExtent: 1f);

        int culled = WorldObjectVisibilityCollector.CollectVisibleMdx(
            frame,
            [instance],
            context,
            static _ => false,
            static (_, _) => true,
            static _ => true,
            static (_, _) => { });

        Assert.Equal(0, culled);
        WorldVisibleMdxEntry visible = Assert.Single(frame.VisibleMdx);
        Assert.Equal(0.4497093f, visible.OpaqueFade, 5);
        Assert.Equal(0.4451545f, visible.TransparentFade, 5);
    }

    [Fact]
    public void CollectVisibleMdx_CountsTaxiActorInVisibilityFrame()
    {
        WorldVisibilityFrame frame = new();
        WorldObjectVisibilityContext context = CreateContext(countAsTaxiActor: true);
        WorldObjectInstance instance = CreateInstance("mdx://taxi", new Vector3(0f, 256f, 0f), halfExtent: 8f);

        int culled = WorldObjectVisibilityCollector.CollectVisibleMdx(
            frame,
            [instance],
            context,
            static _ => false,
            static (_, _) => true,
            static _ => true,
            static (_, _) => { });

        Assert.Equal(0, culled);
        Assert.Single(frame.VisibleMdx);
        Assert.True(frame.VisibleMdx[0].IsTaxiActor);
        Assert.Equal(1, frame.VisibleTaxiMdxCount);
    }

    [Fact]
    public void CollectVisibleMdx_PerformanceProfileCullsTinyFarCandidate()
    {
        WorldVisibilityFrame frame = new();
        WorldObjectVisibilityContext context = CreateContext(
            fogEnd: 5000f,
            visibilityProfile: WorldObjectVisibilityProfile.Performance,
            verticalFieldOfViewRadians: MathF.PI / 3f);
        WorldObjectInstance instance = CreateInstance("mdx://tiny-far", new Vector3(0f, 1800f, 0f), halfExtent: 0.75f);

        int culled = WorldObjectVisibilityCollector.CollectVisibleMdx(
            frame,
            [instance],
            context,
            static _ => false,
            static (_, _) => true,
            static _ => true,
            static (_, _) => { });

        Assert.Equal(1, culled);
        Assert.Empty(frame.VisibleMdx);
    }

    [Fact]
    public void CollectVisibleMdx_PerformanceProfileSkipsTinyMissingLoad()
    {
        WorldVisibilityFrame frame = new();
        WorldObjectVisibilityContext context = CreateContext(
            fogEnd: 5000f,
            visibilityProfile: WorldObjectVisibilityProfile.Performance,
            verticalFieldOfViewRadians: MathF.PI / 3f);
        WorldObjectInstance instance = CreateInstance("mdx://tiny-missing", new Vector3(0f, 1800f, 0f), halfExtent: 0.75f);
        bool queued = false;

        int culled = WorldObjectVisibilityCollector.CollectVisibleMdx(
            frame,
            [instance],
            context,
            static _ => false,
            static (_, _) => true,
            static _ => false,
            (_, _) => queued = true);

        Assert.Equal(1, culled);
        Assert.False(queued);
        Assert.Empty(frame.VisibleMdx);
    }

    [Fact]
    public void CollectVisibleMdx_PerformanceProfileQueuesFrontMeaningfulMissingLoad()
    {
        WorldVisibilityFrame frame = new();
        WorldObjectVisibilityContext context = CreateContext(
            fogEnd: 5000f,
            visibilityProfile: WorldObjectVisibilityProfile.Performance,
            verticalFieldOfViewRadians: MathF.PI / 3f);
        WorldObjectInstance instance = CreateInstance("mdx://meaningful-missing", new Vector3(0f, 900f, 0f), halfExtent: 6f);
        bool queued = false;

        int culled = WorldObjectVisibilityCollector.CollectVisibleMdx(
            frame,
            [instance],
            context,
            static _ => false,
            static (_, _) => true,
            static _ => false,
            (_, _) => queued = true);

        Assert.Equal(0, culled);
        Assert.True(queued);
        Assert.Empty(frame.VisibleMdx);
    }

    private static WorldObjectVisibilityContext CreateContext(
        Vector3? cameraForward = null,
        float fogEnd = 1200f,
        bool cullSmallDoodadsOnly = false,
        bool countAsTaxiActor = false,
        float verticalFieldOfViewRadians = MathF.PI / 3f,
        WorldObjectVisibilityProfile visibilityProfile = WorldObjectVisibilityProfile.Quality)
    {
        return new WorldObjectVisibilityContext(
            CameraPosition: Vector3.Zero,
            CameraForward: Vector3.Normalize(cameraForward ?? Vector3.UnitY),
            FogEnd: fogEnd,
            ObjectStreamingRangeMultiplier: 1.0f,
            CullSmallDoodadsOnly: cullSmallDoodadsOnly,
            CountAsTaxiActor: countAsTaxiActor,
            VerticalFieldOfViewRadians: verticalFieldOfViewRadians,
            VisibilityProfile: visibilityProfile);
    }

    private static WorldObjectInstance CreateInstance(string modelKey, Vector3 center, float halfExtent)
    {
        Vector3 extent = new(halfExtent, halfExtent, halfExtent);
        return new WorldObjectInstance
        {
            ModelKey = modelKey,
            ModelName = modelKey,
            ModelPath = modelKey,
            Transform = Matrix4x4.CreateTranslation(center),
            PlacementPosition = center,
            PlacementScale = 1.0f,
            BoundsMin = center - extent,
            BoundsMax = center + extent,
            LocalBoundsMin = -extent,
            LocalBoundsMax = extent,
            BoundsResolved = true,
        };
    }
}