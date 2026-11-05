using System.Text.Json;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests.Maps;

public sealed class DonorGroupTransformServiceTests
{
    [Fact]
    public void CreatePreview_TranslatesAllPlacementsTogether_AndPreservesPlacementMetadata()
    {
        PhaseLayerSettings layer = CreateLayer(
            new PhaseTilePlacement(2, 3, 10, 12, Locked: true, Included: false, ZOffset: 4f, ZScale: 1.5f),
            new PhaseTilePlacement(5, 6, 14, 17));

        DonorGroupTransformPreview preview = DonorGroupTransformService.CreatePreview(
            layer, new DonorGroupTransform(0f, TranslationX: 2f, TranslationY: -3f));

        Assert.True(preview.CanAccept);
        Assert.Equal(new PhaseTilePlacement(2, 3, 12, 9, true, false, 4f, 1.5f), preview.PreviewLayer.TilePlacements[0]);
        Assert.Equal(new PhaseTilePlacement(5, 6, 16, 14), preview.PreviewLayer.TilePlacements[1]);
        Assert.Equal(0f, preview.AppliedTransform.RotationDegrees);
        Assert.Equal(12f, preview.AppliedTransform.PivotTileX);
        Assert.Equal(14.5f, preview.AppliedTransform.PivotTileY);
    }

    [Theory]
    [InlineData(17f, 17f)]
    [InlineData(377.5f, 17.5f)]
    [InlineData(-17f, 343f)]
    [InlineData(720f, 0f)]
    public void CreatePreview_NormalizesAngle_AndKeepsExactFractionalAngle(float requested, float expected)
    {
        PhaseLayerSettings layer = CreateLayer(
            new PhaseTilePlacement(0, 0, 10, 10),
            new PhaseTilePlacement(0, 1, 10, 14));

        DonorGroupTransformPreview preview = DonorGroupTransformService.CreatePreview(
            layer, new DonorGroupTransform(requested));

        Assert.True(preview.CanAccept);
        Assert.Equal(expected, preview.AppliedTransform.RotationDegrees);
        Assert.Equal(preview.AppliedTransform, preview.PreviewLayer.GroupTransform);
        Assert.Equal(10f, preview.AppliedTransform.PivotTileX);
        Assert.Equal(12f, preview.AppliedTransform.PivotTileY);

        if (expected == 0f)
        {
            Assert.Equal(layer.TilePlacements, preview.PreviewLayer.TilePlacements);
        }

        if (expected == 17f)
        {
            Assert.Equal((11, 10), (preview.PreviewLayer.TilePlacements[0].TargetTileX,
                preview.PreviewLayer.TilePlacements[0].TargetTileY));
            Assert.Equal((9, 14), (preview.PreviewLayer.TilePlacements[1].TargetTileX,
                preview.PreviewLayer.TilePlacements[1].TargetTileY));
        }
    }

    [Fact]
    public void CreatePreview_SingleTileUsesItsOwnCentroidAsPivot()
    {
        PhaseLayerSettings layer = CreateLayer(new PhaseTilePlacement(3, 4, 20, 21));

        DonorGroupTransformPreview preview = DonorGroupTransformService.CreatePreview(
            layer, new DonorGroupTransform(17.25f));

        Assert.True(preview.CanAccept);
        Assert.Equal(20f, preview.AppliedTransform.PivotTileX);
        Assert.Equal(21f, preview.AppliedTransform.PivotTileY);
        Assert.Equal((20, 21), (preview.PreviewLayer.TilePlacements[0].TargetTileX,
            preview.PreviewLayer.TilePlacements[0].TargetTileY));
    }

    [Fact]
    public void CreatePreview_OffGroupPivotFallsBackToCentroidWithWarning()
    {
        PhaseLayerSettings layer = CreateLayer(
            new PhaseTilePlacement(0, 0, 10, 10),
            new PhaseTilePlacement(0, 1, 10, 14));

        DonorGroupTransformPreview preview = DonorGroupTransformService.CreatePreview(
            layer, new DonorGroupTransform(17f, PivotTileX: 60f, PivotTileY: 60f));

        Assert.True(preview.CanAccept);
        Assert.Contains(preview.Issues, issue => issue.Code == DonorGroupTransformIssueCode.PivotOutsideGroup
            && issue.Severity == DonorGroupTransformIssueSeverity.Warning);
        Assert.Equal((10f, 12f), (preview.AppliedTransform.PivotTileX, preview.AppliedTransform.PivotTileY));
    }

    [Fact]
    public void CreatePreview_RejectsOutOfBoundsTransform()
    {
        PhaseLayerSettings layer = CreateLayer(new PhaseTilePlacement(0, 0, 62, 63));

        DonorGroupTransformPreview preview = DonorGroupTransformService.CreatePreview(
            layer, new DonorGroupTransform(0f, TranslationX: 2f));

        Assert.False(preview.CanAccept);
        Assert.Contains(preview.Issues, issue => issue.Code == DonorGroupTransformIssueCode.OutOfBounds
            && issue.Severity == DonorGroupTransformIssueSeverity.Error);
    }

    [Fact]
    public void CreatePreview_RejectsCollidingTargetClaims()
    {
        PhaseLayerSettings layer = CreateLayer(
            new PhaseTilePlacement(0, 0, 10, 10),
            new PhaseTilePlacement(0, 1, 10, 10));

        DonorGroupTransformPreview preview = DonorGroupTransformService.CreatePreview(
            layer, new DonorGroupTransform(0f));

        Assert.False(preview.CanAccept);
        DonorGroupTransformIssue collision = Assert.Single(preview.Issues,
            issue => issue.Code == DonorGroupTransformIssueCode.TargetCollision);
        Assert.Equal(new[] { 0, 1 }, collision.PlacementIndices);
        Assert.Equal((10, 10), (collision.TileX, collision.TileY));
    }

    [Fact]
    public void CreatePreview_ReportsGapsAsNonBlockingWarnings()
    {
        PhaseLayerSettings layer = CreateLayer(
            new PhaseTilePlacement(0, 0, 10, 10),
            new PhaseTilePlacement(0, 2, 10, 12));

        DonorGroupTransformPreview preview = DonorGroupTransformService.CreatePreview(
            layer, new DonorGroupTransform(0f));

        Assert.True(preview.CanAccept);
        Assert.Contains(preview.Issues, issue => issue.Code == DonorGroupTransformIssueCode.GroupGap
            && issue.Severity == DonorGroupTransformIssueSeverity.Warning);
    }

    [Fact]
    public void CreatePreview_UsesAnIsolatedLayerClone_AndCancelLeavesSourceByteIdentical()
    {
        PhaseLayerSettings source = CreateLayer(
            new PhaseTilePlacement(2, 3, 10, 10, Locked: true, Included: false, ZOffset: 2f, ZScale: 0.75f),
            new PhaseTilePlacement(4, 5, 10, 14));
        source.RotationDegrees = 31f;
        source.RotationOriginTileX = 7.5f;
        source.RotationOriginTileY = 8.5f;
        string before = Serialize(source);

        DonorGroupTransformPreview preview = DonorGroupTransformService.CreatePreview(
            source, new DonorGroupTransform(17f, TranslationX: 1f));

        Assert.NotSame(source, preview.PreviewLayer);
        Assert.NotSame(source.TilePlacements, preview.PreviewLayer.TilePlacements);
        Assert.NotEqual(Serialize(source), Serialize(preview.PreviewLayer));

        // Cancel is intentionally discard-only: there is no commit call and the source stays untouched.
        _ = preview;
        Assert.Equal(before, Serialize(source));
    }

    [Fact]
    public void CreatePreview_InvalidAngleReturnsBlockingIssueWithoutMutation()
    {
        PhaseLayerSettings layer = CreateLayer(new PhaseTilePlacement(0, 0, 10, 10));
        string before = Serialize(layer);

        DonorGroupTransformPreview preview = DonorGroupTransformService.CreatePreview(
            layer, new DonorGroupTransform(float.NaN));

        Assert.False(preview.CanAccept);
        Assert.Contains(preview.Issues, issue => issue.Code == DonorGroupTransformIssueCode.InvalidRotation);
        Assert.Equal(before, Serialize(layer));
    }

    [Fact]
    public void CreatePreview_EmptyGroupIsRejectedWithoutThrowing()
    {
        PhaseLayerSettings layer = CreateLayer();

        DonorGroupTransformPreview preview = DonorGroupTransformService.CreatePreview(
            layer, new DonorGroupTransform(0f));

        Assert.False(preview.CanAccept);
        Assert.Contains(preview.Issues, issue => issue.Code == DonorGroupTransformIssueCode.EmptyGroup);
    }

    private static PhaseLayerSettings CreateLayer(params PhaseTilePlacement[] placements)
    {
        var layer = new PhaseLayerSettings { MapName = "Donor" };
        foreach (PhaseTilePlacement placement in placements)
            layer.TilePlacements.Add(placement);
        return layer;
    }

    private static string Serialize(PhaseLayerSettings layer)
        => JsonSerializer.Serialize(PhaseLayerProjectEntry.FromLayer(layer));
}
