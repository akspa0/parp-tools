using System.Globalization;
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
using System.ComponentModel;
using System.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// StartupAutomationService host bridge. Bridged member types are declared across ViewerApp partial files, so this
// file carries those files' using directives; the moved members keep their original usings.
internal sealed partial class StartupAutomationService
{
    // Host bridge (same names as the former ViewerApp members).
    private List<CameraShotPoint> _cameraShotPoints => _host.CameraShotPoints;
    private ref string _captureOutputDir => ref _host.CaptureOutputDir;
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private DataSourceSessionService _dataSourceSession => _host.DataSourceSession;
    private DatasetExportDialogsService _datasetExportDialogs => _host.DatasetExportDialogs;
    private ref int _mkHarvestViewerValidationQueued => ref _host.MkHarvestViewerValidationQueued;
    private StandaloneModelLoaderService _modelLoader => _host.ModelLoader;
    private ref MkHarvestViewerValidationCapturePlan? _pendingMkHarvestViewerValidationCapturePlan => ref _host.PendingMkHarvestViewerValidationCapturePlan;
    private ref string _statusMessage => ref _host.StatusMessage;
    private ref IWindow _window => ref _host.Window;
    private void EnqueueShotCapture(CameraShotPoint shot, bool includeUi, bool exitAfterCapture = false) => _host.EnqueueShotCapture(shot, includeUi, exitAfterCapture);
    private void EnqueueShotCapture(CameraShotPoint shot, bool includeUi, bool exitAfterCapture, CaptureQueueOptions? options) => _host.EnqueueShotCapture(shot, includeUi, exitAfterCapture, options);
    private void GenerateMkHarvestViewerValidationObjectArtifacts(string datasetRoot, string withObjectsOutputDirectory, string noObjectsOutputDirectory, string objectsOnlyOutputDirectory) => _host.GenerateMkHarvestViewerValidationObjectArtifacts(datasetRoot, withObjectsOutputDirectory, noObjectsOutputDirectory, objectsOnlyOutputDirectory);
    private void QueueCurrentCameraCapture(bool includeUi, bool exitAfterCapture = false, int captureAfterFrames = 1, bool allowWindowCloseOnCapture = false) => _host.QueueCurrentCameraCapture(includeUi, exitAfterCapture, captureAfterFrames, allowWindowCloseOnCapture);
    private void StitchMkHarvestViewerValidationOutputs(string mapName, string outputDirectory, string noLiquidsOutputDirectory, string noObjectsOutputDirectory, string objectsOnlyOutputDirectory, int requestedResolution) => _host.StitchMkHarvestViewerValidationOutputs(mapName, outputDirectory, noLiquidsOutputDirectory, noObjectsOutputDirectory, objectsOnlyOutputDirectory, requestedResolution);
}
