using System.Numerics;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.Json;
using ImGuiNET;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WoWViewer.Capture;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Runtime.Marketing;
using WoWViewer.Terrain.Vlm;

namespace WoWViewer;

public partial class ViewerApp
{
    List<CaptureAutomationService.CameraShotPoint> IViewerAppHost.CameraShotPoints => _captureAutomation._cameraShotPoints;
    Queue<CaptureAutomationService.PendingCaptureRequest> IViewerAppHost.CaptureQueue => _captureAutomation._captureQueue;
    ref CaptureAutomationService.PendingCaptureRequest? IViewerAppHost.ActiveCaptureRequest => ref _captureAutomation._activeCaptureRequest;
    ref int IViewerAppHost.TaxiRideCameraRouteId => ref _captureAutomation._taxiRideCameraRouteId;
    ref WorldScene? IViewerAppHost.TaxiRideCameraScene => ref _captureAutomation._taxiRideCameraScene;
    ref CaptureAutomationService.TaxiRideCameraMode IViewerAppHost.TaxiRideCameraMode => ref _captureAutomation._taxiRideCameraMode;
    ref float IViewerAppHost.TaxiRideChaseDistance => ref _captureAutomation._taxiRideChaseDistance;
    ref float IViewerAppHost.TaxiRideChaseHeight => ref _captureAutomation._taxiRideChaseHeight;
    ref float IViewerAppHost.TaxiRideCockpitHeight => ref _captureAutomation._taxiRideCockpitHeight;
    ref float IViewerAppHost.TaxiRideFreeLookYawOffset => ref _captureAutomation._taxiRideFreeLookYawOffset;
    ref float IViewerAppHost.TaxiRideFreeLookPitchOffset => ref _captureAutomation._taxiRideFreeLookPitchOffset;
    ref bool IViewerAppHost.TaxiRideCameraPoseInitialized => ref _captureAutomation._taxiRideCameraPoseInitialized;
    ref long IViewerAppHost.LastTaxiRideCameraTick => ref _captureAutomation._lastTaxiRideCameraTick;
    ref CaptureAutomationService.ActiveVideoRecording? IViewerAppHost.ActiveVideoRecording => ref _captureAutomation._activeVideoRecording;
    void IViewerAppHost.DrawCaptureAutomationContent() => _captureAutomation.DrawCaptureAutomationContent();
    void IViewerAppHost.QueueCurrentCameraCapture(bool includeUi, bool exitAfterCapture, int captureAfterFrames, bool allowWindowCloseOnCapture) => _captureAutomation.QueueCurrentCameraCapture(includeUi, exitAfterCapture, captureAfterFrames, allowWindowCloseOnCapture);
    void IViewerAppHost.EnqueueShotCapture(CaptureAutomationService.CameraShotPoint shot, bool includeUi, bool exitAfterCapture, CaptureAutomationService.CaptureQueueOptions? options) => _captureAutomation.EnqueueShotCapture(shot, includeUi, exitAfterCapture, options);
    void IViewerAppHost.EnqueueShotCapture(CaptureAutomationService.CameraShotPoint shot, bool includeUi, bool exitAfterCapture) => _captureAutomation.EnqueueShotCapture(shot, includeUi, exitAfterCapture);
    void IViewerAppHost.GenerateMkHarvestViewerValidationObjectArtifacts(string datasetRoot, string withObjectsOutputDirectory, string noObjectsOutputDirectory, string objectsOnlyOutputDirectory) => _captureAutomation.GenerateMkHarvestViewerValidationObjectArtifacts(datasetRoot, withObjectsOutputDirectory, noObjectsOutputDirectory, objectsOnlyOutputDirectory);
    bool IViewerAppHost.TryStartCurrentViewVideoRecording(bool includeUi, string? label) => _captureAutomation.TryStartCurrentViewVideoRecording(includeUi, label);
    void IViewerAppHost.StopVideoRecording(string? statusOverride) => _captureAutomation.StopVideoRecording(statusOverride);
    void IViewerAppHost.StopTaxiRideCamera(string? statusMessage) => _captureAutomation.StopTaxiRideCamera(statusMessage);
    string IViewerAppHost.GetCurrentCaptureMapName() => _captureAutomation.GetCurrentCaptureMapName();
    string IViewerAppHost.GetCurrentCaptureBuildVersion() => _captureAutomation.GetCurrentCaptureBuildVersion();
}
