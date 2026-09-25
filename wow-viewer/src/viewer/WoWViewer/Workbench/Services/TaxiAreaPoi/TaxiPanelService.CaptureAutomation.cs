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
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// TaxiPanelService: members moved from ViewerApp_CaptureAutomation.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class TaxiPanelService
{
    private float _taxiRideLookAhead = 28f;

    private bool TryStartTaxiRideVideoCapture()
    {
        if (_worldScene == null || _worldScene.SelectedTaxiRouteId < 0)
        {
            _statusMessage = "Select a taxi route before starting ride capture.";
            return false;
        }

        if (!TryAttachTaxiRideCameraToSelectedRoute())
            return false;

        return TryStartCurrentViewVideoRecording(_videoCaptureIncludeUi, _taxiAndAreaPoi.GetTaxiRouteDisplayLabel(_taxiRideCameraRouteId));
    }

    private bool TryAttachTaxiRideCameraToSelectedRoute()
    {
        if (_worldScene == null || _worldScene.SelectedTaxiRouteId < 0)
        {
            _statusMessage = "Select a taxi route before enabling the ride camera.";
            return false;
        }

        // A camera path and a taxi ride both own the camera. Cancel any
        // pending path warmup/playback before attaching the ride route.
        StopCameraPathPlayback();
        _worldScene.ShowTaxi = true;
        _worldScene.ShowTaxiActors = true;
        _taxiRideCameraRouteId = _worldScene.SelectedTaxiRouteId;
        _taxiRideCameraScene = _worldScene;
        _worldScene.ActiveTaxiRideRouteId = _taxiRideCameraRouteId;
        _taxiRideCameraEnabled = true;
        _taxiRideFreeLookYawOffset = 0f;
        _taxiRideFreeLookPitchOffset = 0f;
        _taxiRideCameraPoseInitialized = false;
        _lastTaxiRideCameraTick = Stopwatch.GetTimestamp();
        _statusMessage = $"Ride camera attached to {_taxiAndAreaPoi.GetTaxiRouteDisplayLabel(_taxiRideCameraRouteId)}.";
        return true;
    }
}
