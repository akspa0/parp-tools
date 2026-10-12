using WowViewer.Core.Runtime.PromoVideo;

namespace WowViewer.Core.Tests.PromoVideo;

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

        PromoTourAttemptStartResult start = PromoTourAttempt.TryStart(recipe, cameraPathDurationSeconds: 4);
        Assert.True(start.IsStarted, start.Error);
        PromoTourAttempt attempt = Assert.IsType<PromoTourAttempt>(start.Attempt);

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
        PromoTourAttemptStartResult start = PromoTourAttempt.TryStart(
            BuiltinFeatureTourRecipes.CreateCameraPathOverview("FlybyUndead.mdx", fps: 30),
            cameraPathDurationSeconds: 30);
        PromoTourAttempt attempt = Assert.IsType<PromoTourAttempt>(start.Attempt);

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

        PromoTourAttemptStartResult start = PromoTourAttempt.TryStart(invalid, cameraPathDurationSeconds: 5);

        Assert.False(start.IsStarted);
        Assert.Null(start.Attempt);
        Assert.Equal("recipe-invalid", start.ErrorCode);
    }
}
