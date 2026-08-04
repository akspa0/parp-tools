using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;
using Xunit;

namespace WowViewer.Core.PM4.Tests;

/// <summary>
/// Pins the MSVT-to-placement-space transform to the ADT ground truth that established it, and
/// checks that the region frame audit can tell frames apart before it is trusted on the corpus.
/// </summary>
/// <remarks>
/// The transform here is not a convention chosen for convenience — it is the one that reproduces
/// real MDDF/MODF positions from the paired <c>_obj0.adt</c>. The literal numbers below are lifted
/// from <c>development_01_00.pm4</c> and <c>development_1_0_obj0.adt</c>, so if the transform drifts
/// these fail with a concrete, checkable discrepancy rather than a vague expectation.
/// </remarks>
public class Pm4PlacementSpaceTests
{
    // development_01_00.pm4 MSVT bounds, and the MDDF positions of the three doodads placed on the
    // same tile by development_1_0_obj0.adt. Measured 2026-08-03.
    private const float TentsMsvtMinX = 41.0f;
    private const float TentsMsvtMaxX = 52.9f;
    private const float TentsMsvtMinY = 778.9f;
    private const float TentsMsvtMaxY = 790.6f;

    private const float TentsAdtMinX = 17015.6f;
    private const float TentsAdtMaxX = 17024.4f;
    private const float TentsAdtMinY = 16277.7f;
    private const float TentsAdtMaxY = 16286.4f;

    [Fact]
    public void Pm4LocalToAdtPlacement_PutsTheTentsWhereTheAdtPlacesThem()
    {
        Vector3 low = Pm4CoordinateService.Pm4LocalToAdtPlacement(new Vector3(TentsMsvtMinX, TentsMsvtMinY, 72.5f));
        Vector3 high = Pm4CoordinateService.Pm4LocalToAdtPlacement(new Vector3(TentsMsvtMaxX, TentsMsvtMaxY, 73.7f));

        // Subtracting from the origin reverses the ordering, so the MSVT minimum becomes the maximum.
        float minX = MathF.Min(low.X, high.X);
        float maxX = MathF.Max(low.X, high.X);
        float minY = MathF.Min(low.Y, high.Y);
        float maxY = MathF.Max(low.Y, high.Y);

        // The ADT placements are three doodads standing inside the PM4 footprint, so the PM4 span
        // must contain them rather than equal them.
        Assert.True(minX <= TentsAdtMinX, $"placement minX {minX} should be at or below the ADT minimum {TentsAdtMinX}");
        Assert.True(maxX >= TentsAdtMaxX, $"placement maxX {maxX} should be at or above the ADT maximum {TentsAdtMaxX}");
        Assert.True(minY <= TentsAdtMinY, $"placement minY {minY} should be at or below the ADT minimum {TentsAdtMinY}");
        Assert.True(maxY >= TentsAdtMaxY, $"placement maxY {maxY} should be at or above the ADT maximum {TentsAdtMaxY}");

        // Height passes straight through.
        Assert.Equal(72.5f, low.Z, 3);
    }

    [Fact]
    public void Pm4LocalToAdtPlacement_UnswappedReadingMissesTheAdtPlacementsEntirely()
    {
        // The eliminated alternative: treat MSVT.X as the ADT's own X field. It lands a full tile
        // away on both axes, which is why the corpus scored it at 0.7% against 92.4%.
        Vector3 unswapped = new(
            Pm4CoordinateService.MapOrigin - TentsMsvtMinY,
            Pm4CoordinateService.MapOrigin - TentsMsvtMinX,
            0f);

        Assert.False(
            unswapped.X >= TentsAdtMinX && unswapped.X <= TentsAdtMaxX,
            "the unswapped reading must not land inside the real ADT placement range");
    }

    [Fact]
    public void PlacementTileIndex_RoundTripsThroughTheBandItNames()
    {
        foreach (int tileIndex in new[] { 0, 1, 18, 22, 31, 63 })
        {
            (float min, float max) = Pm4CoordinateService.GetPlacementTileBand(tileIndex);

            Assert.Equal(tileIndex, Pm4CoordinateService.PlacementCoordinateToTileIndex(max - 0.01f));
            Assert.Equal(tileIndex, Pm4CoordinateService.PlacementCoordinateToTileIndex(min + 0.01f));
        }
    }

    [Fact]
    public void IsWithinPlacementTileBounds_AcceptsTheTentsOnTheirOwnTileAndRejectsTheNeighbour()
    {
        Vector3 tents = Pm4CoordinateService.Pm4LocalToAdtPlacement(new Vector3(45f, 784f, 73f));

        // development_01_00 -> first = 1 bounds Y, second = 0 bounds X.
        Assert.True(Pm4CoordinateService.IsWithinPlacementTileBounds(tents, tileFirst: 1, tileSecond: 0));
        Assert.False(Pm4CoordinateService.IsWithinPlacementTileBounds(tents, tileFirst: 0, tileSecond: 0));
        Assert.False(Pm4CoordinateService.IsWithinPlacementTileBounds(tents, tileFirst: 1, tileSecond: 1));
    }

    [Fact]
    public void TryGetObj0PathForPm4_ReturnsNullRatherThanAnUnrelatedTile()
    {
        string directory = Path.Combine(Path.GetTempPath(), "pm4-obj0-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            // A decoy for a different tile. The old behaviour built a padded name that never
            // existed; a caller that then fell back to "any obj0 in the folder" would pick this.
            File.WriteAllBytes(Path.Combine(directory, "development_9_9_obj0.adt"), []);

            string pm4Path = Path.Combine(directory, "development_01_00.pm4");
            Assert.Null(Pm4CoordinateService.TryGetObj0PathForPm4(pm4Path));

            // The real companion uses the unpadded ADT spelling.
            string companion = Path.Combine(directory, "development_1_0_obj0.adt");
            File.WriteAllBytes(companion, []);
            Assert.Equal(companion, Pm4CoordinateService.TryGetObj0PathForPm4(pm4Path));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Theory]
    // The canonical frame, and the three that the corpus sweep actually observed.
    [InlineData(Pm4CoordinateMode.WorldSpace, false, false, false, "WorldSpace/....")]
    [InlineData(Pm4CoordinateMode.TileLocal, false, false, false, "TileLocal/....")]
    [InlineData(Pm4CoordinateMode.TileLocal, false, true, true, "TileLocal/.UV.")]
    [InlineData(Pm4CoordinateMode.WorldSpace, true, false, false, "WorldSpace/S...")]
    public void DescribeFrame_GivesEachDistinctFrameItsOwnToken(
        Pm4CoordinateMode mode, bool swap, bool invertU, bool invertV, string expected)
    {
        string frame = Pm4RegionFrameAuditAnalyzer.DescribeFrame(mode, new Pm4PlanarTransform(swap, invertU, invertV));

        Assert.Equal(expected, frame);
    }

    [Fact]
    public void DescribeFrame_SeparatesEveryFrameTheContractCanEnumerate()
    {
        // Detector power: the audit groups objects by this token, so two genuinely different frames
        // collapsing onto one token would silently report a region as homogeneous when it is not.
        List<string> tokens = [];
        foreach (Pm4CoordinateMode mode in new[] { Pm4CoordinateMode.TileLocal, Pm4CoordinateMode.WorldSpace })
        {
            foreach (Pm4PlanarTransform transform in Pm4PlacementContract.EnumeratePlanarTransforms(mode))
                tokens.Add(Pm4RegionFrameAuditAnalyzer.DescribeFrame(mode, transform));
        }

        Assert.Equal(tokens.Count, tokens.Distinct(StringComparer.Ordinal).Count());
        Assert.Contains(Pm4RegionFrameAuditAnalyzer.CanonicalFrame, tokens);
    }
}
