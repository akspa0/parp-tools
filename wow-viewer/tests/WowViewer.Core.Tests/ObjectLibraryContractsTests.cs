using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

/// <summary>
/// Spec 077 §1.1 object library contract tests. These guard the deterministic
/// identity rules and the field-level invariants before any tool writes real
/// library entries.
/// </summary>
public sealed class ObjectLibraryContractsTests
{
    [Fact]
    public void LibraryId_IsStableAcrossCalls_ForSameNormalizedPath()
    {
        string first = ObjectLibraryEntry.ComputeLibraryId("world/wmo/azeroth/buildings/stormwind/stormwind.wmo");
        string second = ObjectLibraryEntry.ComputeLibraryId("world/wmo/azeroth/buildings/stormwind/stormwind.wmo");

        Assert.Equal(first, second);
        Assert.StartsWith("objlib_", first);
        Assert.Equal(20, first.Length);
    }

    [Fact]
    public void LibraryId_DiffersAcrossDistinctNormalizedPaths()
    {
        string a = ObjectLibraryEntry.ComputeLibraryId("world/wmo/azeroth/stormwind.wmo");
        string b = ObjectLibraryEntry.ComputeLibraryId("world/wmo/azeroth/ironforge.wmo");

        Assert.NotEqual(a, b);
    }

    [Fact]
    public void LibraryId_ReturnsEmpty_ForBlankInput()
    {
        Assert.Equal(string.Empty, ObjectLibraryEntry.ComputeLibraryId(""));
        Assert.Equal(string.Empty, ObjectLibraryEntry.ComputeLibraryId("   "));
    }

    [Fact]
    public void Entry_DefaultsToNotAttempted_AndUnreviewedVisibility()
    {
        ObjectLibraryEntry entry = new()
        {
            LibraryId = "objlib_abc",
            OriginalAssetPath = "World\\wmo\\Azeroth\\Stormwind.wmo",
            NormalizedAssetPath = "world/wmo/azeroth/stormwind.wmo",
            AssetType = ObjectAssetType.Wmo,
        };

        Assert.Equal(ObjectCaptureStatus.NotAttempted, entry.CaptureStatus);
        Assert.Equal(ObjectVisibilityClass.Unknown, entry.VisibilityClass);
        Assert.Equal(ObjectReviewState.Unreviewed, entry.ReviewState);
        Assert.Equal(0, entry.PlacementObservationCount);
        Assert.Null(entry.PreferredVariantId);
        Assert.Empty(entry.SourceBuilds);
        Assert.Empty(entry.SourceMaps);
    }

    [Fact]
    public void VariantId_IsStableAcrossCalls_ForSamePose()
    {
        Vector3 rot = new(0f, 0f, 0f);
        string a = ObjectCaptureVariant.ComputeVariantId("objlib_abc", "3_3_5_12340", ObjectCaptureMode.OrthographicTopdown, rot, 1f);
        string b = ObjectCaptureVariant.ComputeVariantId("objlib_abc", "3_3_5_12340", ObjectCaptureMode.OrthographicTopdown, rot, 1f);

        Assert.Equal(a, b);
        Assert.StartsWith("objvar_", a);
        Assert.Equal(22, a.Length);
    }

    [Fact]
    public void VariantId_DiffersForDistinctPose()
    {
        Vector3 rotA = new(0f, 0f, 0f);
        Vector3 rotB = new(0f, 0f, 1.5707963f);
        string a = ObjectCaptureVariant.ComputeVariantId("objlib_abc", "3_3_5_12340", ObjectCaptureMode.OrthographicTopdown, rotA, 1f);
        string b = ObjectCaptureVariant.ComputeVariantId("objlib_abc", "3_3_5_12340", ObjectCaptureMode.OrthographicTopdown, rotB, 1f);

        Assert.NotEqual(a, b);
    }

    [Fact]
    public void VariantId_DiffersAcrossCaptureModes()
    {
        Vector3 rot = new(0f, 0f, 0f);
        string a = ObjectCaptureVariant.ComputeVariantId("objlib_abc", "3_3_5_12340", ObjectCaptureMode.OrthographicTopdown, rot, 1f);
        string b = ObjectCaptureVariant.ComputeVariantId("objlib_abc", "3_3_5_12340", ObjectCaptureMode.GeometryProjection, rot, 1f);

        Assert.NotEqual(a, b);
    }

    [Fact]
    public void VariantId_ReturnsEmpty_ForBlankLibraryId()
    {
        string id = ObjectCaptureVariant.ComputeVariantId(
            string.Empty,
            "3_3_5_12340",
            ObjectCaptureMode.OrthographicTopdown,
            new Vector3(0f, 0f, 0f),
            1f);

        Assert.Equal(string.Empty, id);
    }

    [Fact]
    public void BoundingBox_ReportsWidthHeight_AndEmptyState()
    {
        ObjectLibraryBoundingBox populated = new(2, 3, 10, 7);
        Assert.Equal(8, populated.Width);
        Assert.Equal(4, populated.Height);
        Assert.False(populated.IsEmpty);

        ObjectLibraryBoundingBox empty = new(5, 5, 5, 5);
        Assert.True(empty.IsEmpty);

        ObjectLibraryBoundingBox inverted = new(10, 10, 2, 2);
        Assert.True(inverted.IsEmpty);
        Assert.Equal(0, inverted.Width);
        Assert.Equal(0, inverted.Height);
    }
}
