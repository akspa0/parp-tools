using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.IO.Mdx;
using WoWViewer.DataSources;
using WoWViewer.Export;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Catalog;
using WoWViewer.Capture;
using WoWViewer.Population;
using WoWViewer.Terrain;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using WowViewer.Core.Mdx;
using System.ComponentModel;
using System.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using static WoWViewer.ViewerApp;
using static WoWViewer.CaptureAutomationService;

namespace WoWViewer;

// TaxiPanelService host bridge. Bridged member types are declared across ViewerApp partial files, so this
// file carries those files' using directives; the moved members keep their original usings.
internal sealed partial class TaxiPanelService
{
    // Host bridge (same names as the former ViewerApp members).
    private ref ActiveVideoRecording? _activeVideoRecording => ref _host.ActiveVideoRecording;
    private DataSourceSessionService _dataSourceSession => _host.DataSourceSession;
    private ref long _lastTaxiRideCameraTick => ref _host.LastTaxiRideCameraTick;
    private StandaloneModelLoaderService _modelLoader => _host.ModelLoader;
    private ref string _selectedObjectInfo => ref _host.SelectedObjectInfo;
    private ref string _statusMessage => ref _host.StatusMessage;
    private ref string _taxiActorModelOverrideInput => ref _host.TaxiActorModelOverrideInput;
    private ref int _taxiActorModelOverrideInputRouteId => ref _host.TaxiActorModelOverrideInputRouteId;
    private ref int _taxiActorModelOverrideTargetRouteId => ref _host.TaxiActorModelOverrideTargetRouteId;
    private TaxiAndAreaPoiSelectionService _taxiAndAreaPoi => _host.TaxiAndAreaPoi;
    private ref bool _taxiRideCameraEnabled => ref _host.TaxiRideCameraEnabled;
    private ref TaxiRideCameraMode _taxiRideCameraMode => ref _host.TaxiRideCameraMode;
    private ref bool _taxiRideCameraPoseInitialized => ref _host.TaxiRideCameraPoseInitialized;
    private ref int _taxiRideCameraRouteId => ref _host.TaxiRideCameraRouteId;
    private ref WorldScene? _taxiRideCameraScene => ref _host.TaxiRideCameraScene;
    private ref float _taxiRideChaseDistance => ref _host.TaxiRideChaseDistance;
    private ref float _taxiRideChaseHeight => ref _host.TaxiRideChaseHeight;
    private ref float _taxiRideCockpitHeight => ref _host.TaxiRideCockpitHeight;
    private ref float _taxiRideFreeLookPitchOffset => ref _host.TaxiRideFreeLookPitchOffset;
    private ref float _taxiRideFreeLookYawOffset => ref _host.TaxiRideFreeLookYawOffset;
    private ref int _videoCaptureFps => ref _host.VideoCaptureFps;
    private ref bool _videoCaptureIncludeUi => ref _host.VideoCaptureIncludeUi;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private void CopyTextToClipboard(string text, string description) => _host.CopyTextToClipboard(text, description);
    private void StopCameraPathPlayback() => _host.StopCameraPathPlayback();
    private void StopTaxiRideCamera(string? statusMessage = null) => _host.StopTaxiRideCamera(statusMessage);
    private void StopVideoRecording(string? statusOverride = null) => _host.StopVideoRecording(statusOverride);
    private bool TryGetSelectedBrowserModelPath(out string assetPath) => _host.TryGetSelectedBrowserModelPath(out assetPath);
    private bool TryStartCurrentViewVideoRecording(bool includeUi, string? label = null) => _host.TryStartCurrentViewVideoRecording(includeUi, label);
}
