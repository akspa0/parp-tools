using WowViewer.Core.Runtime.Marketing;

namespace WowViewer.Core.Tests.MarketingCapture;

public sealed class FeatureTourAttemptTests
{
    [Fact]
    public void Advance_ActivatesOrderedCalloutsWithoutAllocatingOnSteadyFrames()
    {
        FeatureTourRecipe recipe = new(
            FeatureTourRecipe.SchemaV1,
            "camera-path-overview",
            "Camera Path Overview",
            "1",
            RequiresWarmPath: true,
            new FeatureTourCaptureSettings(30, FullFrameWithTourOverlay: true),
            [
                new FeatureTourBeat(0, FeatureTourPresentationKind.Callout, "camera-path", "Camera Path", null, 1),
                new FeatureTourBeat(2, FeatureTourPresentationKind.Callout, "renderer", "Renderer", null, 1),
            ]);

        MarketingTourAttemptStartResult start = MarketingTourAttempt.TryStart(recipe, cameraPathDurationSeconds: 4);
        Assert.True(start.IsStarted, start.Error);
        MarketingTourAttempt attempt = Assert.IsType<MarketingTourAttempt>(start.Attempt);

        attempt.Advance(0.5);
        Assert.Equal("camera-path", attempt.ActivePresentation?.FeatureId);

        attempt.Advance(1.5);
        Assert.Null(attempt.ActivePresentation);

        attempt.Advance(2.5);
        Assert.Equal("renderer", attempt.ActivePresentation?.FeatureId);

        // JIT and the transition allocation happen before the measured steady-state loop.
        attempt.Advance(2.5);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int index = 0; index < 1_000; index++)
            attempt.Advance(2.5);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(0, allocated);
    }

    [Fact]
    public void Cancel_PreservesTerminalReasonAndSuppressesPresentation()
    {
        MarketingTourAttemptStartResult start = MarketingTourAttempt.TryStart(
            BuiltinFeatureTourRecipes.CreateCameraPathOverview("FlybyUndead.mdx", fps: 30),
            cameraPathDurationSeconds: 30);
        MarketingTourAttempt attempt = Assert.IsType<MarketingTourAttempt>(start.Attempt);

        attempt.Advance(0.25);
        attempt.Cancel("active-map-changed");
        attempt.Advance(2.5);

        Assert.True(attempt.IsCancelled);
        Assert.Equal("active-map-changed", attempt.TerminalReason);
        Assert.Null(attempt.ActivePresentation);
    }

    [Fact]
    public void TryStart_RejectsInvalidRecipeBeforeItCanChangeViewerState()
    {
        FeatureTourRecipe invalid = new(
            FeatureTourRecipe.SchemaV1,
            "camera-path-overview",
            "Camera Path Overview",
            "1",
            RequiresWarmPath: true,
            new FeatureTourCaptureSettings(1, FullFrameWithTourOverlay: true),
            []);

        MarketingTourAttemptStartResult start = MarketingTourAttempt.TryStart(invalid, cameraPathDurationSeconds: 5);

        Assert.False(start.IsStarted);
        Assert.Null(start.Attempt);
        Assert.Equal("recipe-invalid", start.ErrorCode);
    }
}
