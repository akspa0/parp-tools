using System.Numerics;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;
using WowViewer.Core.M2;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using Silk.NET.Input;
using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using WoWViewer.DataSources;
using WoWViewer.Export;
using WoWViewer.Logging;
using WoWViewer.Catalog;
using WoWViewer.Capture;
using WoWViewer.Population;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using System.ComponentModel;
using System.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using static WoWViewer.ViewerApp;
using static WoWViewer.CaptureAutomationService;

namespace WoWViewer;

// CameraPathsService host bridge. Bridged member types are declared across ViewerApp partial files, so this
// file carries those files' using directives; the moved members keep their original usings.
internal sealed partial class CameraPathsService
{
    // Host bridge (same names as the former ViewerApp members).
    private ref PendingCaptureRequest? _activeCaptureRequest => ref _host.ActiveCaptureRequest;
    private ref int _activeUtilitiesTabIndex => ref _host.ActiveUtilitiesTabIndex;
    private ref ActiveVideoRecording? _activeVideoRecording => ref _host.ActiveVideoRecording;
    private ref Camera _camera => ref _host.Camera;
    private Queue<PendingCaptureRequest> _captureQueue => _host.CaptureQueue;
    private ref string? _dbcBuild => ref _host.DbcBuild;
    private ref DBCD.Providers.IDBCProvider? _dbcProvider => ref _host.DbcProvider;
    private ref string? _dbdDir => ref _host.DbdDir;
    private ref float _fovDegrees => ref _host.FovDegrees;
    private ref bool _hideUiChrome => ref _host.HideUiChrome;
    private StandaloneModelLoaderService _modelLoader => _host.ModelLoader;
    private NavigatorPanelService _navigatorPanel => _host.NavigatorPanel;
    private ref bool _showCameraPathWindow => ref _host.ShowCameraPathWindow;
    private ref bool _showCaptureAutomationWindow => ref _host.ShowCaptureAutomationWindow;
    private ref string _statusMessage => ref _host.StatusMessage;
    private ref bool _taxiRideCameraEnabled => ref _host.TaxiRideCameraEnabled;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref bool _useTabUi => ref _host.UseTabUi;
    private ref int _videoCaptureFps => ref _host.VideoCaptureFps;
    private ref bool _videoCaptureIncludeUi => ref _host.VideoCaptureIncludeUi;
    private WorkbenchPanelsService _workbenchPanels => _host.WorkbenchPanels;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private void DrawCaptureAutomationContent() => _host.DrawCaptureAutomationContent();
    private void EnqueueShotCapture(CameraShotPoint shot, bool includeUi, bool exitAfterCapture = false) => _host.EnqueueShotCapture(shot, includeUi, exitAfterCapture);
    private void EnqueueShotCapture(CameraShotPoint shot, bool includeUi, bool exitAfterCapture, CaptureQueueOptions? options) => _host.EnqueueShotCapture(shot, includeUi, exitAfterCapture, options);
    private ViewerKeyContext GetActiveKeyContext() => _host.GetActiveKeyContext();
    private string GetCurrentCaptureBuildVersion() => _host.GetCurrentCaptureBuildVersion();
    private string GetCurrentCaptureMapName() => _host.GetCurrentCaptureMapName();
    private void StopTaxiRideCamera(string? statusMessage = null) => _host.StopTaxiRideCamera(statusMessage);
    private void StopVideoRecording(string? statusOverride = null) => _host.StopVideoRecording(statusOverride);
    private bool TryStartCurrentViewVideoRecording(bool includeUi, string? label = null) => _host.TryStartCurrentViewVideoRecording(includeUi, label);
}
