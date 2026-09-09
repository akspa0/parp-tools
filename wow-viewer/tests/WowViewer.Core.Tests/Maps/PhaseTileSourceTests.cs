using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests.Maps;

/// <summary>
/// Spec 219 Phase 1: ResolveTileSource resolution order (per-tile last-wins first, then offset
/// with rotation/mirror) and the untransformed-routing invariant (SC-006's structural half).
/// </summary>
public sealed class PhaseTileSourceTests
{
    private static bool AllTilesExist(int _, int __) => true;

    [Fact]
    public void UntransformedLayer_RoutesThroughOffset()
    {
        var layer = new PhaseLayerSettings { MapName = "Donor", TileOffsetX = 2, TileOffsetY = 3 };

        PhaseTileSource source = PhaseCompositionPolicy.ResolveTileSource(layer, 10, 12, AllTilesExist);

        Assert.True(source.HasSource);
        Assert.Equal(PhaseTileSourceKind.Offset, source.Via);
        Assert.Equal(8, source.SourceTileX);
        Assert.Equal(9, source.SourceTileY);
        Assert.Empty(source.Transforms);
        Assert.Equal(PhaseRotationApproximation.None, source.Approximation);
    }

    [Fact]
    public void PerTileMapping_BeatsOffset()
    {
        var layer = new PhaseLayerSettings { MapName = "Donor", TileOffsetX = 2, TileOffsetY = 2 };
        layer.TilePlacements.Add(new PhaseTilePlacement(5, 6, 10, 12));

        PhaseTileSource source = PhaseCompositionPolicy.ResolveTileSource(layer, 10, 12, AllTilesExist);

        Assert.True(source.HasSource);
        Assert.Equal(PhaseTileSourceKind.PerTile, source.Via);
        Assert.Equal(5, source.SourceTileX);
        Assert.Equal(6, source.SourceTileY);
    }

    [Fact]
    public void PerTileMapping_LastWins_OnDuplicateTarget()
    {
        var layer = new PhaseLayerSettings { MapName = "Donor" };
        layer.TilePlacements.Add(new PhaseTilePlacement(1, 1, 10, 10));
        layer.TilePlacements.Add(new PhaseTilePlacement(2, 2, 10, 10));

        PhaseTileSource source = PhaseCompositionPolicy.ResolveTileSource(layer, 10, 10, AllTilesExist);

        Assert.Equal(2, source.SourceTileX);
        Assert.Equal(2, source.SourceTileY);
        Assert.True(source.HasConflict);
        Assert.Equal(2, source.TargetClaimCount);
    }

    [Fact]
    public void PerTileMapping_MissingDonorTile_IsEmpty()
    {
        var layer = new PhaseLayerSettings { MapName = "Donor" };
        layer.TilePlacements.Add(new PhaseTilePlacement(5, 5, 10, 10));

        PhaseTileSource source = PhaseCompositionPolicy.ResolveTileSource(
            layer, 10, 10, (_, _) => false);

        Assert.False(source.HasSource);
    }

    [Fact]
    public void UnmappedTarget_WithPerTileMappings_FallsBackToOffset()
    {
        var layer = new PhaseLayerSettings { MapName = "Donor", TileOffsetX = 1, TileOffsetY = 1 };
        layer.TilePlacements.Add(new PhaseTilePlacement(5, 5, 10, 10));

        PhaseTileSource source = PhaseCompositionPolicy.ResolveTileSource(layer, 20, 20, AllTilesExist);

        Assert.True(source.HasSource);
        Assert.Equal(PhaseTileSourceKind.Offset, source.Via);
        Assert.Equal(19, source.SourceTileX);
        Assert.Equal(19, source.SourceTileY);
    }

    [Fact]
    public void PlacedTileOnlyMode_UnmappedTarget_IsEmptyInsteadOfUsingOffset()
    {
        var layer = new PhaseLayerSettings
        {
            MapName = "Donor",
            TileOffsetX = 1,
            TileOffsetY = 1,
            UsePlacedTilesOnly = true,
        };
        layer.TilePlacements.Add(new PhaseTilePlacement(5, 5, 10, 10));

        PhaseTileSource source = PhaseCompositionPolicy.ResolveTileSource(layer, 20, 20, AllTilesExist);

        Assert.False(source.HasSource);
    }

    [Fact]
    public void PlacedTileOnlyMode_WithoutPlacements_IsEmpty()
    {
        var layer = new PhaseLayerSettings { MapName = "Donor", UsePlacedTilesOnly = true };

        PhaseTileSource source = PhaseCompositionPolicy.ResolveTileSource(layer, 20, 20, AllTilesExist);

        Assert.False(source.HasSource);
    }

    [Fact]
    public void LockedPlacement_ClaimsOnlyItsExplicitTarget()
    {
        var layer = new PhaseLayerSettings { MapName = "Donor" };
        layer.TilePlacements.Add(new PhaseTilePlacement(5, 5, 10, 10, Locked: true));

        Assert.True(PhaseCompositionPolicy.IsTargetLockedByLayer(layer, 10, 10));
        Assert.False(PhaseCompositionPolicy.IsTargetLockedByLayer(layer, 10, 11));
    }

    [Fact]
    public void EarlierLockedPlacement_BlocksOnlyLaterContributingLayers()
    {
        var lockOwner = new PhaseLayerSettings { MapName = "Owner" };
        lockOwner.TilePlacements.Add(new PhaseTilePlacement(5, 5, 10, 10, Locked: true));
        var laterLayer = new PhaseLayerSettings { MapName = "Later" };

        IReadOnlyList<PhaseLayerSettings> layers = [lockOwner, laterLayer];

        Assert.True(PhaseCompositionPolicy.IsTargetLockedByEarlierLayer(layers, 1, 10, 10));
        Assert.False(PhaseCompositionPolicy.IsTargetLockedByEarlierLayer(layers, 1, 11, 10));

        lockOwner.Enabled = false;
        Assert.False(PhaseCompositionPolicy.IsTargetLockedByEarlierLayer(layers, 1, 10, 10));
    }

    [Fact]
    public void Rotate90CW_InversesTheLookup()
    {
        // CW content rotation: target (tx, ty) is filled by rotating donor tile lookup. The exact
        // grid inverse of CW is CCW applied to (target - offset).
        var layer = new PhaseLayerSettings
        {
            MapName = "Donor",
            TileOffsetX = 0,
            TileOffsetY = 0,
            RotationDegrees = 90f,
        };

        PhaseTileSource source = PhaseCompositionPolicy.ResolveTileSource(layer, 10, 10, AllTilesExist);

        Assert.True(source.HasSource);
        Assert.Equal(PhaseTileSourceKind.Rotation, source.Via);
        Assert.Contains(TileTransformKind.Rotate90CW, source.Transforms);
        // The inverse lookup must be the CCW rotation of the target about the same origin
        // (origin defaults are adapter-supplied; the policy uses the numeric inverse).
        (int ix, int iy) = PhaseCompositionPolicy.InverseTransformTile(10, 10, layer);
        Assert.Equal(source.SourceTileX, ix);
        Assert.Equal(source.SourceTileY, iy);
    }

    [Fact]
    public void MirrorHorizontal_RoutesAndReportsTransform()
    {
        var layer = new PhaseLayerSettings { MapName = "Donor", MirrorHorizontal = true };

        PhaseTileSource source = PhaseCompositionPolicy.ResolveTileSource(layer, 10, 10, AllTilesExist);

        Assert.True(source.HasSource);
        Assert.Contains(TileTransformKind.MirrorH, source.Transforms);
        // Horizontal means left-right on the map: tile Y is the West-East column, so it flips.
        Assert.Equal(10, source.SourceTileX);
        Assert.Equal(-10, source.SourceTileY);
    }

    [Fact]
    public void InverseTransformTile_UsesConfiguredOrigin()
    {
        var layer = new PhaseLayerSettings
        {
            MapName = "Donor",
            RotationDegrees = 90f,
            RotationOriginTileX = 10f,
            RotationOriginTileY = 20f,
        };

        // Target one tile east of origin; inverse of a clockwise turn lands one tile south.
        Assert.Equal((9, 20), PhaseCompositionPolicy.InverseTransformTile(10, 21, layer));
    }

    [Fact]
    public void NegativeQuarterTurn_ComposesAsCounterClockwise()
    {
        var layer = new PhaseLayerSettings { MapName = "Donor", RotationDegrees = -90f };

        Assert.Equal(
            new[] { TileTransformKind.Rotate90CCW },
            PhaseCompositionPolicy.ComposeTileTransforms(layer));
    }

    [Fact]
    public void Clone_PreservesAllTransformStateAndCopiesMappings()
    {
        var layer = new PhaseLayerSettings
        {
            MapName = "Donor",
            RotationDegrees = 90f,
            RotationOriginTileX = 12.5f,
            RotationOriginTileY = 13.5f,
            MirrorHorizontal = true,
            UsePlacedTilesOnly = true,
        };
        layer.TilePlacements.Add(new PhaseTilePlacement(1, 2, 3, 4, Locked: true));

        PhaseLayerSettings clone = layer.Clone();

        Assert.Equal(layer.RotationDegrees, clone.RotationDegrees);
        Assert.Equal(layer.RotationOriginTileX, clone.RotationOriginTileX);
        Assert.Equal(layer.RotationOriginTileY, clone.RotationOriginTileY);
        Assert.Equal(layer.MirrorHorizontal, clone.MirrorHorizontal);
        Assert.Equal(layer.UsePlacedTilesOnly, clone.UsePlacedTilesOnly);
        Assert.Equal(layer.TilePlacements, clone.TilePlacements);
        Assert.True(Assert.Single(clone.TilePlacements).Locked);
        Assert.NotSame(layer.TilePlacements, clone.TilePlacements);
    }

    [Fact]
    public void ComposeTileTransforms_OrdersRotationBeforeMirrors()
    {
        var layer = new PhaseLayerSettings
        {
            MapName = "Donor",
            RotationDegrees = 90f,
            MirrorHorizontal = true,
            MirrorVertical = true,
        };

        IReadOnlyList<TileTransformKind> transforms = PhaseCompositionPolicy.ComposeTileTransforms(layer);

        Assert.Equal(
            new[] { TileTransformKind.Rotate90CW, TileTransformKind.MirrorH, TileTransformKind.MirrorV },
            transforms);
    }

    [Fact]
    public void ComposeTileTransforms_FreeAngle_ContributesNoChunkTransform()
    {
        var layer = new PhaseLayerSettings { MapName = "Donor", RotationDegrees = 45f };

        Assert.Empty(PhaseCompositionPolicy.ComposeTileTransforms(layer));
        Assert.Equal(
            PhaseRotationApproximation.FreeRotate,
            PhaseCompositionPolicy.ResolveRotationApproximation(layer));
    }

    [Fact]
    public void UntransformedLayer_HasTransformIsFalse()
    {
        var layer = new PhaseLayerSettings { MapName = "Donor" };

        Assert.False(layer.HasTransform);
    }

    [Fact]
    public void InvalidPlacement_IsSkipped()
    {
        var layer = new PhaseLayerSettings { MapName = "Donor" };
        layer.TilePlacements.Add(new PhaseTilePlacement(-5, 0, 10, 10)); // invalid donor

        PhaseTileSource source = PhaseCompositionPolicy.ResolveTileSource(layer, 10, 10, AllTilesExist);

        // Falls through to the offset path (zero offset here): the invalid mapping is ignored.
        Assert.True(source.HasSource);
        Assert.Equal(PhaseTileSourceKind.Offset, source.Via);
    }
}
