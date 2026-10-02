using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using WowViewer.Core.Runtime.PromoVideo;

namespace WoWViewer;

/// <summary>
/// Source kind that initiated or drives the recording session.
/// </summary>
internal enum RecordingSourceKind
{
    Manual = 0,
    CameraPath = 1,
    TaxiRoute = 2,
    Automation = 3,
}

/// <summary>
/// Parameters defining a video recording request.
/// </summary>
internal sealed class RecordingRequest
{
    public RecordingSourceKind SourceKind { get; init; } = RecordingSourceKind.Manual;
    public string? Label { get; init; }
    public bool IncludeUi { get; init; }
    public int Fps { get; init; } = 30;
    public int ContainerIndex { get; init; } = 0; // 0 = .mp4, 1 = .mov
    public string? OutputPathOverride { get; init; }
    public FeatureTourRecipe? TourRecipe { get; init; }
    public PromoTourAttempt? TourAttempt { get; init; }
    public double? MaxDurationSeconds { get; init; }
    public bool AutoStopOnPathComplete { get; init; } = true;
    public bool AutoStopOnRouteArrival { get; init; } = true;
    public bool RestoreUiChromeOnStop { get; init; } = true;
    public bool PreviousHideUiChrome { get; init; }
    public bool DetachCameraOnStop { get; init; } = false;
    public bool ExitAfterRecording { get; init; } = false;
    public int TaxiRouteId { get; init; } = -1;
    public string? CameraPathName { get; init; }
    public Action<RecordingSessionSummary>? OnCompleted { get; init; }
    public bool IncludeShowreelOverlay { get; init; } = false;
}

/// <summary>
/// State tracking an in-flight video recording session.
/// </summary>
internal sealed class ActiveRecordingSession
{
    public required RecordingRequest Request { get; init; }
    public required Process EncoderProcess { get; init; }
    public required Stream EncoderInput { get; init; }
    public required StringBuilder EncoderErrorOutput { get; init; }
    public required string OutputPath { get; init; }
    public required bool IncludeUi { get; init; }
    public required int Width { get; init; }
    public required int Height { get; init; }
    public required double FrameIntervalSeconds { get; init; }
    public double FrameAccumulatorSeconds { get; set; }
    public byte[] FrameBuffer { get; set; } = Array.Empty<byte>();
    public double ElapsedSeconds { get; set; }
    public int RecordedFrames { get; set; }
    public bool ApplyArcheologyPlayback { get; init; }
    public bool StartedArcheologyPlayback { get; init; }
    public PromoTourAttempt? TourAttempt { get; set; }
    public bool RestoreUiChromeOnStop { get; init; }
    public bool PreviousHideUiChrome { get; init; }
    public float RouteStartTravelDistance { get; set; }
    public float RouteTotalLength { get; set; }
    public bool HasTraveledSignificantDistance { get; set; }
    public bool HasTour => TourAttempt != null || Request.TourRecipe != null || Request.IncludeShowreelOverlay;
}

/// <summary>
/// Summary of a finished recording session. Safely preserved so UI and callers
/// never dereference null when querying output path or final status.
/// </summary>
internal sealed record RecordingSessionSummary(
    RecordingSourceKind SourceKind,
    string OutputPath,
    double DurationSeconds,
    int RecordedFrames,
    bool Success,
    string StatusMessage,
    DateTime CompletedAtUtc);
