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
using WowViewer.Core.Runtime.PromoVideo;
using WoWViewer.Terrain.Vlm;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// TaxiPanelService: members moved from ViewerApp_CaptureAutomation.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class TaxiPanelService
{
    private float _taxiRideLookAhead = 28f;
    private bool _taxiAutoStopOnRouteArrival = true;
    private bool _taxiRecordWithFeatureTour = false;
    private bool _taxiResetTravelOnRecordStart = true;

    private bool TryStartTaxiRideVideoCapture()
    {
        if (_worldScene == null || _worldScene.TaxiActors.SelectedTaxiRouteId < 0)
        {
            _statusMessage = "Select a taxi route before starting ride capture.";
            return false;
        }

        if (!TryAttachTaxiRideCameraToSelectedRoute())
            return false;

        int routeId = _taxiRideCameraRouteId;
        string routeLabel = _taxiAndAreaPoi.GetTaxiRouteDisplayLabel(routeId);

        if (_taxiResetTravelOnRecordStart)
        {
            _worldScene.TaxiActors.ResetTaxiRouteTravel(routeId);
        }

        FeatureTourRecipe? recipe = null;
        if (_taxiRecordWithFeatureTour)
        {
            var route = _worldScene.TaxiActors.GetTaxiRoute(routeId);
            string? fromStation = route != null ? _worldScene.TaxiActors.GetTaxiNode(route.FromNodeId)?.Name : null;
            string? toStation = route != null ? _worldScene.TaxiActors.GetTaxiNode(route.ToNodeId)?.Name : null;
            recipe = BuiltinFeatureTourRecipes.CreateTaxiRouteOverview(routeId, routeLabel, fromStation, toStation, _videoCaptureFps);
        }

        var request = new RecordingRequest
        {
            SourceKind = RecordingSourceKind.TaxiRoute,
            TaxiRouteId = routeId,
            Label = routeLabel,
            IncludeUi = _videoCaptureIncludeUi,
            Fps = _videoCaptureFps,
            ContainerIndex = _host.VideoCaptureContainerIndex,
            AutoStopOnRouteArrival = _taxiAutoStopOnRouteArrival,
            DetachCameraOnStop = false,
            TourRecipe = recipe,
            IncludeShowreelOverlay = _taxiRecordWithFeatureTour || _host.ShowreelOverlay.Config.EnableOverlay,
        };

        return _recordingCoordinator.TryStartRecording(request, out _);
    }

    private bool TryAttachTaxiRideCameraToSelectedRoute()
    {
        if (_worldScene == null || _worldScene.TaxiActors.SelectedTaxiRouteId < 0)
        {
            _statusMessage = "Select a taxi route before enabling the ride camera.";
            return false;
        }

        // A camera path and a taxi ride both own the camera. Cancel any
        // pending path warmup/playback before attaching the ride route.
        StopCameraPathPlayback();
        _worldScene.TaxiActors.ShowTaxi = true;
        _worldScene.TaxiActors.ShowTaxiActors = true;
        _taxiRideCameraRouteId = _worldScene.TaxiActors.SelectedTaxiRouteId;
        _taxiRideCameraScene = _worldScene;
        _worldScene.TaxiActors.ActiveTaxiRideRouteId = _taxiRideCameraRouteId;
        _taxiRideCameraEnabled = true;
        _taxiRideFreeLookYawOffset = 0f;
        _taxiRideFreeLookPitchOffset = 0f;
        _taxiRideCameraPoseInitialized = false;
        _lastTaxiRideCameraTick = Stopwatch.GetTimestamp();
        _statusMessage = $"Ride camera attached to {_taxiAndAreaPoi.GetTaxiRouteDisplayLabel(_taxiRideCameraRouteId)}.";
        return true;
    }

    internal bool AttachTaxiRideCamera(int routeId)
    {
        if (_worldScene == null)
            return false;

        _worldScene.TaxiActors.SelectedTaxiRouteId = routeId;
        _worldScene.TaxiActors.SelectedTaxiNodeId = -1;
        return TryAttachTaxiRideCameraToSelectedRoute();
    }

    internal void StopTaxiRideCameraPublic(string? statusMessage = null)
    {
        StopTaxiRideCamera(statusMessage);
    }
}
