using System;
using System.IO;
using WowViewer.Core.Runtime.PromoVideo;
using WoWViewer;
using Xunit;

namespace WowViewer.Core.Tests.PromoVideo;

public sealed class RecordingModelTests
{
    [Fact]
    public void RecordingRequest_DefaultValues_AreSensible()
    {
        var req = new RecordingRequest();

        Assert.Equal(RecordingSourceKind.Manual, req.SourceKind);
        Assert.False(req.IncludeUi);
        Assert.Equal(30, req.Fps);
        Assert.Equal(0, req.ContainerIndex);
        Assert.True(req.AutoStopOnPathComplete);
        Assert.True(req.AutoStopOnRouteArrival);
        Assert.True(req.RestoreUiChromeOnStop);
        Assert.False(req.DetachCameraOnStop);
        Assert.False(req.ExitAfterRecording);
        Assert.Equal(-1, req.TaxiRouteId);
        Assert.Null(req.TourRecipe);
    }

    [Fact]
    public void RecordingRequest_TaxiRouteInitialization_PreservesOptions()
    {
        FeatureTourRecipe recipe = BuiltinFeatureTourRecipes.CreateTaxiRouteOverview(
            routeId: 101,
            routeLabel: "[101] Orgrimmar -> Thunder Bluff",
            fromStation: "Orgrimmar",
            toStation: "Thunder Bluff",
            fps: 60);

        var req = new RecordingRequest
        {
            SourceKind = RecordingSourceKind.TaxiRoute,
            TaxiRouteId = 101,
            Label = "[101] Orgrimmar -> Thunder Bluff",
            IncludeUi = true,
            Fps = 60,
            ContainerIndex = 1,
            OutputPathOverride = "C:\\captures\\tb_flight.mov",
            AutoStopOnRouteArrival = true,
            DetachCameraOnStop = true,
            TourRecipe = recipe,
            ExitAfterRecording = true,
        };

        Assert.Equal(RecordingSourceKind.TaxiRoute, req.SourceKind);
        Assert.Equal(101, req.TaxiRouteId);
        Assert.True(req.IncludeUi);
        Assert.Equal(60, req.Fps);
        Assert.Equal(1, req.ContainerIndex);
        Assert.Equal("C:\\captures\\tb_flight.mov", req.OutputPathOverride);
        Assert.True(req.AutoStopOnRouteArrival);
        Assert.True(req.DetachCameraOnStop);
        Assert.True(req.ExitAfterRecording);
        Assert.NotNull(req.TourRecipe);
        Assert.Equal("taxi-route-overview", req.TourRecipe.Id);
    }

    [Fact]
    public void RecordingSessionSummary_Record_PreservesCompletionData()
    {
        DateTime completedAt = DateTime.UtcNow;
        var summary = new RecordingSessionSummary(
            SourceKind: RecordingSourceKind.TaxiRoute,
            OutputPath: "I:\\captures\\flight.mp4",
            DurationSeconds: 45.2,
            RecordedFrames: 1356,
            Success: true,
            StatusMessage: "Saved video: I:\\captures\\flight.mp4",
            CompletedAtUtc: completedAt);

        Assert.Equal(RecordingSourceKind.TaxiRoute, summary.SourceKind);
        Assert.Equal("I:\\captures\\flight.mp4", summary.OutputPath);
        Assert.Equal(45.2, summary.DurationSeconds);
        Assert.Equal(1356, summary.RecordedFrames);
        Assert.True(summary.Success);
        Assert.Equal("Saved video: I:\\captures\\flight.mp4", summary.StatusMessage);
        Assert.Equal(completedAt, summary.CompletedAtUtc);
    }
}
