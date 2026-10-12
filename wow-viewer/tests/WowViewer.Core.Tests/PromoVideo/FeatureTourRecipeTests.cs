using WowViewer.Core.Runtime.PromoVideo;

namespace WowViewer.Core.Tests.PromoVideo;

public sealed class FeatureTourRecipeTests
{
    [Fact]
    public void Validate_AcceptsOrderedNonOverlappingCallouts()
    {
        FeatureTourRecipe recipe = CreateRecipe(
        [
            new FeatureTourBeat(0, FeatureTourPresentationKind.Callout, "camera-path", "Camera Path", "Loaded client camera path.", 1.5),
            new FeatureTourBeat(2, FeatureTourPresentationKind.Callout, "renderer", "Direct Renderer Capture", "Frames originate in the renderer.", 1.0),
        ]);

        FeatureTourValidationResult result = FeatureTourRecipeValidator.Validate(recipe, cameraPathDurationSeconds: 5);

        Assert.True(result.IsValid, string.Join(Environment.NewLine, result.Errors));
        Assert.Empty(result.Errors);
    }

    [Theory]
    [InlineData(11)]
    [InlineData(61)]
    public void Validate_RejectsCaptureRatesOutsideSupportedRange(int fps)
    {
        FeatureTourRecipe recipe = CreateRecipe([], fps: fps);

        FeatureTourValidationResult result = FeatureTourRecipeValidator.Validate(recipe, cameraPathDurationSeconds: 5);

        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, error => error.Code == "capture-fps-out-of-range");
    }

    [Fact]
    public void Validate_RejectsOutOfOrderOrOverlappingBeats()
    {
        FeatureTourRecipe outOfOrder = CreateRecipe(
        [
            new FeatureTourBeat(2, FeatureTourPresentationKind.Callout, "first", "First", null, 1),
            new FeatureTourBeat(1, FeatureTourPresentationKind.Callout, "second", "Second", null, 1),
        ]);
        FeatureTourRecipe overlap = CreateRecipe(
        [
            new FeatureTourBeat(1, FeatureTourPresentationKind.Callout, "first", "First", null, 2),
            new FeatureTourBeat(2, FeatureTourPresentationKind.Callout, "second", "Second", null, 1),
        ]);

        FeatureTourValidationResult outOfOrderResult = FeatureTourRecipeValidator.Validate(outOfOrder, cameraPathDurationSeconds: 5);
        FeatureTourValidationResult overlapResult = FeatureTourRecipeValidator.Validate(overlap, cameraPathDurationSeconds: 5);

        Assert.Contains(outOfOrderResult.Errors, error => error.Code == "beats-not-strictly-ordered");
        Assert.Contains(overlapResult.Errors, error => error.Code == "beats-overlap");
    }

    [Fact]
    public void Validate_RejectsUnknownPresentationKindAndBeatOutsideCameraPath()
    {
        FeatureTourRecipe recipe = CreateRecipe(
        [
            new FeatureTourBeat(6, (FeatureTourPresentationKind)99, "unknown", "Unknown", null, 1),
        ]);

        FeatureTourValidationResult result = FeatureTourRecipeValidator.Validate(recipe, cameraPathDurationSeconds: 5);

        Assert.Contains(result.Errors, error => error.Code == "beat-kind-unsupported");
        Assert.Contains(result.Errors, error => error.Code == "beat-outside-camera-path");
    }

    [Fact]
    public void BuiltinCameraPathOverview_UsesOnlyPathNameNotItsMachineLocalDirectory()
    {
        FeatureTourRecipe recipe = BuiltinFeatureTourRecipes.CreateCameraPathOverview("D:\\private-client\\Cameras\\FlybyUndead.mdx", fps: 30);

        Assert.Equal("camera-path-overview", recipe.Id);
        Assert.DoesNotContain("private-client", recipe.DisplayName, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("FlybyUndead", recipe.DisplayName, StringComparison.Ordinal);
        Assert.True(FeatureTourRecipeValidator.Validate(recipe, cameraPathDurationSeconds: 30).IsValid);
    }

    [Fact]
    public void BuiltinTaxiRouteOverview_CreatesValidRecipeWithStationCallouts()
    {
        FeatureTourRecipe recipe = BuiltinFeatureTourRecipes.CreateTaxiRouteOverview(
            routeId: 42,
            routeLabel: "[42] Stormwind -> Ironforge",
            fromStation: "Stormwind",
            toStation: "Ironforge",
            fps: 30);

        Assert.Equal("taxi-route-overview", recipe.Id);
        Assert.Contains("Stormwind -> Ironforge", recipe.DisplayName, StringComparison.Ordinal);
        Assert.False(recipe.RequiresWarmPath);
        Assert.Equal(3, recipe.Beats.Count);
        Assert.Contains("Departing: Stormwind", recipe.Beats[0].Title);
        Assert.Contains("Approaching: Ironforge", recipe.Beats[2].Title);

        FeatureTourValidationResult result = FeatureTourRecipeValidator.Validate(recipe, cameraPathDurationSeconds: 15);
        Assert.True(result.IsValid, string.Join(Environment.NewLine, result.Errors));
    }

    private static FeatureTourRecipe CreateRecipe(IReadOnlyList<FeatureTourBeat> beats, int fps = 30)
        => new(
            FeatureTourRecipe.SchemaV1,
            "camera-path-overview",
            "Camera Path Overview",
            "1",
            RequiresWarmPath: true,
            new FeatureTourCaptureSettings(fps, FullFrameWithTourOverlay: true),
            beats);
}
