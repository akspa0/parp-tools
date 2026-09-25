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
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.M2;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WowViewer.Core.IO.Maps;
using WoWViewer.Terrain.Vlm;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using static WoWViewer.ViewerApp;
using static WoWViewer.ThemesService;
using static WoWViewer.Pm4WorkbenchService;

namespace WoWViewer;

/// <summary>
/// Viewer settings persistence: loading and saving viewer_settings.json, known-good client paths, and the persisted settings/override DTOs.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed class ViewerSettingsService
{
    private readonly IViewerAppHost _host;

    internal ViewerSettingsService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref int _activeBottomTabIndex => ref _host.ActiveBottomTabIndex;
    private ref string _activeDatasetVersionRoot => ref _host.ActiveDatasetVersionRoot;
    private ref WorkbenchTab _activeTopTab => ref _host.ActiveTopTab;
    private ref int _activeUtilitiesTabIndex => ref _host.ActiveUtilitiesTabIndex;
    private ref bool _archeologyApplyToNextCapture => ref _host.ArcheologyApplyToNextCapture;
    private ref bool _archeologyApplyToVideoRecording => ref _host.ArcheologyApplyToVideoRecording;
    private ref int _archeologyMaxUniqueId => ref _host.ArcheologyMaxUniqueId;
    private ref int _archeologyMinUniqueId => ref _host.ArcheologyMinUniqueId;
    private ref bool _archeologyPlaybackLoop => ref _host.ArcheologyPlaybackLoop;
    private ref float _archeologyPlaybackSpeed => ref _host.ArcheologyPlaybackSpeed;
    private ref int _archeologyScopeIndex => ref _host.ArcheologyScopeIndex;
    private ref float _bottomDrawerHeight => ref _host.BottomDrawerHeight;
    private ref float _cameraSpeed => ref _host.CameraSpeed;
    private ref string _captureOutputDir => ref _host.CaptureOutputDir;
    private List<WoWViewer.Terrain.ClientBuildOption> _clientBuildOptions => _host.ClientBuildOptions;
    private ref string _datasetCatalogRoot => ref _host.DatasetCatalogRoot;
    private ref float _defaultFogEnd => ref _host.DefaultFogEnd;
    private ref float _defaultFogStart => ref _host.DefaultFogStart;
    private ref bool _enableMultisample => ref _host.EnableMultisample;
    private ref bool _enableTerrainBackfaceCulling => ref _host.EnableTerrainBackfaceCulling;
    private ref bool _forceApplyShellPanelLayout => ref _host.ForceApplyShellPanelLayout;
    private ref float _fovDegrees => ref _host.FovDegrees;
    private ref bool _hasExplicitWmoMliqRotationOverride => ref _host.HasExplicitWmoMliqRotationOverride;
    private ref List<KnownGoodClientPath> _knownGoodClientPaths => ref _host.KnownGoodClientPaths;
    private ref string _lastGameFolderPath => ref _host.LastGameFolderPath;
    private ref string _lastLooseOverlayPath => ref _host.LastLooseOverlayPath;
    private ref float _leftSidebarWidth => ref _host.LeftSidebarWidth;
    private ref Vector2 _minimapPanOffset => ref _host.MinimapPanOffset;
    private ref float _minimapZoom => ref _host.MinimapZoom;
    private ref bool _openForgetKnownGoodClientConfirm => ref _host.OpenForgetKnownGoodClientConfirm;
    private ref string? _pendingForgetKnownGoodClientDisplayName => ref _host.PendingForgetKnownGoodClientDisplayName;
    private ref string? _pendingForgetKnownGoodClientPath => ref _host.PendingForgetKnownGoodClientPath;
    private HashSet<ShellPanelId> _pendingShellPanelLayoutRestore => _host.PendingShellPanelLayoutRestore;
    private ref Vector3 _pm4SavedOverlayRotationDegrees => ref _host.Pm4SavedOverlayRotationDegrees;
    private ref Vector3 _pm4SavedOverlayScale => ref _host.Pm4SavedOverlayScale;
    private ref Vector3 _pm4SavedOverlayTranslation => ref _host.Pm4SavedOverlayTranslation;
    private ref Dictionary<string, Pm4WmoMatchEntry> _pm4WmoMatchEntries => ref _host.Pm4WmoMatchEntries;
    private ref Pm4WmoMatchStore? _pm4WmoMatchStore => ref _host.Pm4WmoMatchStore;
    private ref float _rightSidebarWidth => ref _host.RightSidebarWidth;
    private ref int _savedDetailedAdtTileCountOverride => ref _host.SavedDetailedAdtTileCountOverride;
    private Dictionary<string, SavedObjectPathFilterMap> _savedObjectPathFiltersByMap => _host.SavedObjectPathFiltersByMap;
    private Dictionary<string, SavedPm4ObjectMatchSelection> _savedPm4ObjectMatches => _host.SavedPm4ObjectMatches;
    private Dictionary<ShellPanelId, SavedShellPanelLayout> _savedShellPanelLayouts => _host.SavedShellPanelLayouts;
    private Dictionary<string, Dictionary<int, string>> _savedTaxiActorModelOverridesByMap => _host.SavedTaxiActorModelOverridesByMap;
    private ref int _selectedBuildOptionIndex => ref _host.SelectedBuildOptionIndex;
    private ref string _selectedDatasetVersionRoot => ref _host.SelectedDatasetVersionRoot;
    private ref bool _showLeftSidebar => ref _host.ShowLeftSidebar;
    private ref bool _showMinimapWindow => ref _host.ShowMinimapWindow;
    private ref bool _showRightSidebar => ref _host.ShowRightSidebar;
    private ref bool _showWorkspaceBarsPanel => ref _host.ShowWorkspaceBarsPanel;
    private TerrainWeakSignalRestoreService _terrainWeakSignalRestore => _host.TerrainWeakSignalRestore;
    private ref bool _terrainWeakSignalRestoreAllLoadedTiles => ref _host.TerrainWeakSignalRestoreAllLoadedTiles;
    private ref float _terrainWeakSignalRestoreCandidateMaxHeight => ref _host.TerrainWeakSignalRestoreCandidateMaxHeight;
    private ref float _terrainWeakSignalRestoreCandidateMinHeight => ref _host.TerrainWeakSignalRestoreCandidateMinHeight;
    private ref bool _terrainWeakSignalRestoreEnabled => ref _host.TerrainWeakSignalRestoreEnabled;
    private ref float _terrainWeakSignalRestoreManualFactor => ref _host.TerrainWeakSignalRestoreManualFactor;
    private ref bool _terrainWeakSignalRestoreUseAutoFactor => ref _host.TerrainWeakSignalRestoreUseAutoFactor;
    private ref bool _terrainWeakSignalRestoreUseTextureSubdivisions => ref _host.TerrainWeakSignalRestoreUseTextureSubdivisions;
    private ref TextureFilteringMode _textureFilteringMode => ref _host.TextureFilteringMode;
    private ref float _uiFontScale => ref _host.UiFontScale;
    private ref UiThemeKind _uiTheme => ref _host.UiTheme;
    private ref bool _useDockspaceUi => ref _host.UseDockspaceUi;
    private ref bool _useTabUi => ref _host.UseTabUi;
    private ref int _videoCaptureContainerIndex => ref _host.VideoCaptureContainerIndex;
    private ref int _videoCaptureFps => ref _host.VideoCaptureFps;
    private ref bool _videoCaptureIncludeUi => ref _host.VideoCaptureIncludeUi;
    private ref string _videoEncoderExecutable => ref _host.VideoEncoderExecutable;
    private void ApplySavedPm4AlignmentToScene() => _host.ApplySavedPm4AlignmentToScene();
    private int FindBuildOptionIndex(string? buildVersion) => _host.FindBuildOptionIndex(buildVersion);
    private void NormalizeWorkbenchStateAfterLoad() => _host.NormalizeWorkbenchStateAfterLoad();
    private void RefreshClientBuildOptions() => _host.RefreshClientBuildOptions();
    private void RefreshDatasetCatalog() => _host.RefreshDatasetCatalog();

    private static readonly string ViewerSettingsPath = Path.Combine(SettingsDir, "viewer_settings.json");
    private const int CurrentShellPanelLayoutVersion = 4;
    private const int CurrentWorkbenchNavigationVersion = 4;

    internal void QueueForgetKnownGoodClientPath(KnownGoodClientPath knownClient)
    {
        _pendingForgetKnownGoodClientPath = knownClient.Path;
        _pendingForgetKnownGoodClientDisplayName = knownClient.Name;
        _openForgetKnownGoodClientConfirm = true;
    }

    internal void ClearPendingForgetKnownGoodClientPath()
    {
        _pendingForgetKnownGoodClientPath = null;
        _pendingForgetKnownGoodClientDisplayName = null;
    }

    internal void LoadViewerSettings()
    {
        try
        {
            RefreshClientBuildOptions();

            if (!File.Exists(ViewerSettingsPath))
            {
                _hasExplicitWmoMliqRotationOverride = false;
                WmoRenderer.MliqRotationQuarterTurns = 0;
                RefreshDatasetCatalog();
                return;
            }

            string json = File.ReadAllText(ViewerSettingsPath);
            var settings = JsonSerializer.Deserialize<ViewerSettings>(json);
            if (settings == null)
                return;

            _uiTheme = Enum.IsDefined(typeof(UiThemeKind), settings.UiTheme)
                ? (UiThemeKind)settings.UiTheme
                : UiThemeKind.ModernSlate;

            int savedWmoMliqRotation = ((settings.WmoMliqRotationQuarterTurns % 4) + 4) % 4;
            if (settings.HasExplicitWmoMliqRotationOverride)
            {
                _hasExplicitWmoMliqRotationOverride = true;
                WmoRenderer.MliqRotationQuarterTurns = savedWmoMliqRotation;
            }
            else if (savedWmoMliqRotation == 3)
            {
                _hasExplicitWmoMliqRotationOverride = false;
                WmoRenderer.MliqRotationQuarterTurns = 0;
                ViewerLog.Important(ViewerLog.Category.Wmo,
                    "[ViewerSettings] Migrated legacy WMO MLIQ 270° default to neutral override; WMO liquid rotation is now resolved from the asset version path.");
            }
            else
            {
                _hasExplicitWmoMliqRotationOverride = savedWmoMliqRotation != 0;
                WmoRenderer.MliqRotationQuarterTurns = savedWmoMliqRotation;
            }

            _lastGameFolderPath = settings.LastGameFolderPath ?? "";
            _lastLooseOverlayPath = settings.LastLooseOverlayPath ?? "";
            _datasetCatalogRoot = string.IsNullOrWhiteSpace(settings.LastDatasetCatalogRoot)
                ? _datasetCatalogRoot
                : settings.LastDatasetCatalogRoot;
            _selectedDatasetVersionRoot = settings.LastDatasetVersionRoot ?? string.Empty;
            _activeDatasetVersionRoot = settings.LastActiveDatasetVersionRoot ?? string.Empty;
            RefreshDatasetCatalog();
            _knownGoodClientPaths = NormalizeKnownGoodClientPaths(settings.KnownGoodClientPaths);
            _selectedBuildOptionIndex = FindBuildOptionIndex(settings.LastSelectedBuildVersion);
            _textureFilteringMode = Enum.IsDefined(typeof(TextureFilteringMode), settings.TextureFilteringMode)
                ? (TextureFilteringMode)settings.TextureFilteringMode
                : TextureFilteringMode.Trilinear;
            _enableMultisample = settings.EnableMultisample;
            _enableTerrainBackfaceCulling = settings.EnableTerrainBackfaceCulling;
            RenderQualitySettings.EnableTerrainBackfaceCulling = _enableTerrainBackfaceCulling;
            _defaultFogStart = float.IsFinite(settings.DefaultFogStart)
                ? Math.Clamp(settings.DefaultFogStart, 0f, 5000f)
                : 200f;
            _defaultFogEnd = float.IsFinite(settings.DefaultFogEnd)
                ? Math.Clamp(settings.DefaultFogEnd, 100f, 6000f)
                : 1500f;
            _uiFontScale = float.IsFinite(settings.UiFontScale) && settings.UiFontScale > 0.5f
                ? Math.Clamp(settings.UiFontScale, 0.75f, 2.5f)
                : 1.0f;
            if (ShellLayoutService.HasImGuiContext())
            {
                ImGui.GetIO().FontGlobalScale = _uiFontScale;
            }
            _cameraSpeed = float.IsFinite(settings.CameraSpeed)
                ? Math.Clamp(settings.CameraSpeed, 1f, 500f)
                : 50f;
            _fovDegrees = float.IsFinite(settings.FovDegrees)
                ? Math.Clamp(settings.FovDegrees, 20f, 90f)
                : 45f;
            _showMinimapWindow = settings.ShowMinimapWindow;
            _useDockspaceUi = settings.ShellPanelLayoutVersion < CurrentShellPanelLayoutVersion
                ? true
                : settings.UseDockspaceUi;

            // 069 Phase 6: sticky archeology + tab system persistence
            _archeologyMinUniqueId = settings.ArcheologyMinUniqueId;
            _archeologyMaxUniqueId = settings.ArcheologyMaxUniqueId;
            _archeologyScopeIndex = settings.ArcheologyScopeIndex;
            _archeologyPlaybackSpeed = float.IsFinite(settings.ArcheologyPlaybackSpeed)
                ? Math.Clamp(settings.ArcheologyPlaybackSpeed, 1f, 5000f)
                : 50f;
            _archeologyPlaybackLoop = settings.ArcheologyPlaybackLoop;
            _archeologyApplyToNextCapture = settings.ArcheologyApplyToNextCapture;
            _archeologyApplyToVideoRecording = settings.ArcheologyApplyToVideoRecording;
            _useTabUi = settings.UseTabUi;
            if (Enum.IsDefined(typeof(WorkbenchTab), settings.ActiveTopTab))
                _activeTopTab = (WorkbenchTab)settings.ActiveTopTab;
            else
                _activeTopTab = WorkbenchTab.Quick;
            _activeBottomTabIndex = Math.Max(0, settings.ActiveBottomTab);
            if (_activeTopTab == WorkbenchTab.Editor
                && settings.WorkbenchNavigationVersion < CurrentWorkbenchNavigationVersion)
            {
                // Spec 231: pre-231 Editor page indices remap onto the 4-page IA.
                _activeBottomTabIndex = Workbench.Pages.EditorWorkbenchPages.MigrateLegacyEditorPageIndex(_activeBottomTabIndex);
            }
            _activeUtilitiesTabIndex = _activeTopTab == WorkbenchTab.Utilities
                ? _activeBottomTabIndex
                : 0;
            if (_useTabUi)
                NormalizeWorkbenchStateAfterLoad();
            _showLeftSidebar = settings.ShowLeftSidebar;
            _showRightSidebar = settings.ShowRightSidebar;
            _showWorkspaceBarsPanel = settings.ShowWorkspaceBarsPanel;
            _terrainWeakSignalRestoreEnabled = false;
            _terrainWeakSignalRestoreAllLoadedTiles = false;
            _terrainWeakSignalRestoreUseTextureSubdivisions = true;
            _terrainWeakSignalRestoreUseAutoFactor = settings.EnableWeakSignalTerrainRestoreAutoFactor;
            _terrainWeakSignalRestoreManualFactor = float.IsFinite(settings.WeakSignalTerrainRestoreManualFactor)
                ? Math.Clamp(settings.WeakSignalTerrainRestoreManualFactor, 1f, TerrainWeakSignalRestoreMaxFactor)
                : 16f;
            _terrainWeakSignalRestoreCandidateMinHeight = float.IsFinite(settings.WeakSignalTerrainRestoreCandidateMinHeight)
                ? TerrainWeakSignalRestoreService.ClampTerrainWeakSignalRestoreZ(settings.WeakSignalTerrainRestoreCandidateMinHeight)
                : TerrainWeakSignalRestoreDefaultMinZ;
            _terrainWeakSignalRestoreCandidateMaxHeight = float.IsFinite(settings.WeakSignalTerrainRestoreCandidateMaxHeight)
                ? TerrainWeakSignalRestoreService.ClampTerrainWeakSignalRestoreZ(settings.WeakSignalTerrainRestoreCandidateMaxHeight)
                : TerrainWeakSignalRestoreDefaultMaxZ;
            _terrainWeakSignalRestore.GetTerrainWeakSignalRestoreCandidateRange(out _terrainWeakSignalRestoreCandidateMinHeight, out _terrainWeakSignalRestoreCandidateMaxHeight);
            _leftSidebarWidth = float.IsFinite(settings.LeftSidebarWidth)
                ? settings.LeftSidebarWidth
                : DefaultSidebarWidth;
            _rightSidebarWidth = float.IsFinite(settings.RightSidebarWidth)
                ? settings.RightSidebarWidth
                : DefaultRightSidebarWidth;
            _bottomDrawerHeight = float.IsFinite(settings.BottomDrawerHeight)
                ? settings.BottomDrawerHeight
                : DefaultBottomDrawerHeight;
            _minimapZoom = float.IsFinite(settings.MinimapZoom)
                ? Math.Clamp(settings.MinimapZoom, 1f, 32f)
                : 4f;
            _minimapPanOffset = new Vector2(
                float.IsFinite(settings.MinimapPanOffsetX) ? settings.MinimapPanOffsetX : 0f,
                float.IsFinite(settings.MinimapPanOffsetY) ? settings.MinimapPanOffsetY : 0f);
            _captureOutputDir = string.IsNullOrWhiteSpace(settings.CaptureOutputDir)
                ? Path.Combine(OutputDir, "captures")
                : settings.CaptureOutputDir;
            _videoEncoderExecutable = string.IsNullOrWhiteSpace(settings.VideoEncoderExecutable)
                ? "ffmpeg"
                : settings.VideoEncoderExecutable;
            _videoCaptureFps = Math.Clamp(settings.VideoCaptureFps, 12, 60);
            _videoCaptureIncludeUi = settings.VideoCaptureIncludeUi;
            _videoCaptureContainerIndex = Math.Clamp(settings.VideoCaptureContainerIndex, 0, 1);
            _savedDetailedAdtTileCountOverride = Math.Clamp(settings.DetailedAdtTileCountOverride, 0, Terrain.TerrainManager.MaxManualDetailedTileCount);
            _pm4SavedOverlayTranslation = new Vector3(settings.Pm4TranslationX, settings.Pm4TranslationY, settings.Pm4TranslationZ);
            _pm4SavedOverlayRotationDegrees = new Vector3(settings.Pm4RotationX, settings.Pm4RotationY, settings.Pm4RotationZ);
            _pm4SavedOverlayScale = new Vector3(settings.Pm4ScaleX, settings.Pm4ScaleY, settings.Pm4ScaleZ);
            if (MathF.Abs(_pm4SavedOverlayScale.X) < 0.0001f ||
                MathF.Abs(_pm4SavedOverlayScale.Y) < 0.0001f ||
                MathF.Abs(_pm4SavedOverlayScale.Z) < 0.0001f)
            {
                _pm4SavedOverlayScale = Vector3.One;
            }

            // Migrate the short-lived MirrorX default workaround back to neutral scale
            // now that PM4 tile-local coordinates are remapped at conversion time.
            bool isLegacyMirrorX = MathF.Abs(_pm4SavedOverlayScale.X + 1f) < 0.0001f
                && MathF.Abs(_pm4SavedOverlayScale.Y - 1f) < 0.0001f
                && MathF.Abs(_pm4SavedOverlayScale.Z - 1f) < 0.0001f;
            if (isLegacyMirrorX
                && _pm4SavedOverlayTranslation.LengthSquared() < 0.0001f
                && _pm4SavedOverlayRotationDegrees.LengthSquared() < 0.0001f)
            {
                _pm4SavedOverlayScale = Vector3.One;
            }
            if (_pm4SavedOverlayRotationDegrees == Vector3.Zero && MathF.Abs(settings.Pm4YawDegrees) > 0.001f)
                _pm4SavedOverlayRotationDegrees = new Vector3(0f, 0f, settings.Pm4YawDegrees);

            // Load PM4 WMO match store
            _pm4WmoMatchStore = new Pm4WmoMatchStore(AppContext.BaseDirectory);
            _pm4WmoMatchEntries = _pm4WmoMatchStore.Load();

                        _savedTaxiActorModelOverridesByMap.Clear();
                        if (settings.TaxiActorModelOverrides != null)
                        {
                            foreach (SavedTaxiActorOverride savedOverride in settings.TaxiActorModelOverrides)
                            {
                                if (savedOverride == null
                                    || string.IsNullOrWhiteSpace(savedOverride.MapName)
                                    || savedOverride.RouteId < 0
                                    || string.IsNullOrWhiteSpace(savedOverride.ModelPath))
                                {
                                    continue;
                                }

                                if (!_savedTaxiActorModelOverridesByMap.TryGetValue(savedOverride.MapName, out Dictionary<int, string>? overridesByRoute))
                                {
                                    overridesByRoute = new Dictionary<int, string>();
                                    _savedTaxiActorModelOverridesByMap[savedOverride.MapName] = overridesByRoute;
                                }

                                overridesByRoute[savedOverride.RouteId] = savedOverride.ModelPath.Trim().Replace('/', '\\');
                            }
                        }

                        _savedPm4ObjectMatches.Clear();
                        if (settings.Pm4ObjectMatchSelections != null)
                        {
                            foreach (SavedPm4ObjectMatchSelection selection in settings.Pm4ObjectMatchSelections)
                            {
                                if (selection == null
                                    || string.IsNullOrWhiteSpace(selection.MapName)
                                    || string.IsNullOrWhiteSpace(selection.PlacementKind)
                                    || string.IsNullOrWhiteSpace(selection.ModelPath)
                                    || selection.ObjectPartId < 0)
                                {
                                    continue;
                                }

                                string key = BuildSavedPm4ObjectMatchKey(selection.MapName, selection.TileX, selection.TileY, selection.Ck24, selection.ObjectPartId);
                                _savedPm4ObjectMatches[key] = selection;
                            }
                        }

                        _savedObjectPathFiltersByMap.Clear();
                        if (settings.ObjectPathFilters != null)
                        {
                            foreach (SavedObjectPathFilterMap savedMap in settings.ObjectPathFilters)
                            {
                                if (string.IsNullOrWhiteSpace(savedMap.MapName))
                                    continue;

                                List<SavedObjectPathFilterEntry> savedEntries = savedMap.Filters
                                    .Where(entry => !string.IsNullOrWhiteSpace(entry.PathPrefix) && (entry.AppliesToWmo || entry.AppliesToMdx))
                                    .Select(entry => new SavedObjectPathFilterEntry
                                    {
                                        PathPrefix = entry.PathPrefix.Trim().Replace('/', '\\').Trim('\\'),
                                        AppliesToWmo = entry.AppliesToWmo,
                                        AppliesToMdx = entry.AppliesToMdx,
                                    })
                                    .Where(entry => !string.IsNullOrWhiteSpace(entry.PathPrefix))
                                    .OrderBy(entry => entry.PathPrefix, StringComparer.OrdinalIgnoreCase)
                                    .ToList();

                                if (savedEntries.Count == 0 && savedMap.Enabled)
                                    continue;

                                _savedObjectPathFiltersByMap[savedMap.MapName] = new SavedObjectPathFilterMap
                                {
                                    MapName = savedMap.MapName,
                                    Enabled = savedMap.Enabled,
                                    Filters = savedEntries,
                                };
                            }
                        }

                        _savedShellPanelLayouts.Clear();
                        _pendingShellPanelLayoutRestore.Clear();
                        _forceApplyShellPanelLayout = settings.ShellPanelLayoutVersion != CurrentShellPanelLayoutVersion;
                        if (!_forceApplyShellPanelLayout && settings.ShellPanelLayouts != null)
                        {
                            foreach (SavedShellPanelLayout savedLayout in settings.ShellPanelLayouts)
                            {
                                if (!Enum.IsDefined(typeof(ShellPanelId), savedLayout.PanelId))
                                    continue;

                                if (!float.IsFinite(savedLayout.NormalizedX)
                                    || !float.IsFinite(savedLayout.NormalizedY)
                                    || !float.IsFinite(savedLayout.NormalizedWidth)
                                    || !float.IsFinite(savedLayout.NormalizedHeight))
                                {
                                    continue;
                                }

                                var panelId = (ShellPanelId)savedLayout.PanelId;
                                _savedShellPanelLayouts[panelId] = new SavedShellPanelLayout
                                {
                                    PanelId = savedLayout.PanelId,
                                    NormalizedX = Math.Clamp(savedLayout.NormalizedX, 0f, 0.95f),
                                    NormalizedY = Math.Clamp(savedLayout.NormalizedY, 0f, 0.95f),
                                    NormalizedWidth = Math.Clamp(savedLayout.NormalizedWidth, 0.12f, 1f),
                                    NormalizedHeight = Math.Clamp(savedLayout.NormalizedHeight, 0.12f, 1f),
                                };
                                _pendingShellPanelLayoutRestore.Add(panelId);
                            }
                        }

            ApplySavedPm4AlignmentToScene();
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[ViewerSettings] Failed to load settings: {ex.Message}");
        }
    }

    internal void SaveViewerSettings()
    {
        try
        {
            Directory.CreateDirectory(SettingsDir);

            var settings = new ViewerSettings
            {
                UiTheme = (int)_uiTheme,
                WmoMliqRotationQuarterTurns = WmoRenderer.MliqRotationQuarterTurns,
                HasExplicitWmoMliqRotationOverride = _hasExplicitWmoMliqRotationOverride,
                LastGameFolderPath = _lastGameFolderPath,
                LastLooseOverlayPath = _lastLooseOverlayPath,
                LastDatasetCatalogRoot = _datasetCatalogRoot,
                LastDatasetVersionRoot = string.IsNullOrWhiteSpace(_selectedDatasetVersionRoot)
                    ? null
                    : _selectedDatasetVersionRoot,
                LastActiveDatasetVersionRoot = string.IsNullOrWhiteSpace(_activeDatasetVersionRoot)
                    ? null
                    : _activeDatasetVersionRoot,
                LastSelectedBuildVersion = _clientBuildOptions.Count > 0
                    ? _clientBuildOptions[Math.Clamp(_selectedBuildOptionIndex, 0, _clientBuildOptions.Count - 1)].BuildVersion
                    : null,
                TextureFilteringMode = (int)_textureFilteringMode,
                EnableMultisample = _enableMultisample,
                EnableTerrainBackfaceCulling = _enableTerrainBackfaceCulling,
                DefaultFogStart = _defaultFogStart,
                DefaultFogEnd = _defaultFogEnd,
                CameraSpeed = _cameraSpeed,
                FovDegrees = _fovDegrees,
                KnownGoodClientPaths = _knownGoodClientPaths,
                UiFontScale = _uiFontScale,
                ShowMinimapWindow = _showMinimapWindow,
                UseDockspaceUi = _useDockspaceUi,
                ShowLeftSidebar = _showLeftSidebar,
                ShowRightSidebar = _showRightSidebar,
                ShowWorkspaceBarsPanel = _showWorkspaceBarsPanel,
                ShowBottomDrawer = false,
                EnableWeakSignalTerrainRestore = false,
                EnableWeakSignalTerrainRestoreAllLoadedTiles = false,
                EnableWeakSignalTerrainRestoreUseChunkMode = false,
                EnableWeakSignalTerrainRestoreUseTextureSubdivisions = true,
                EnableWeakSignalTerrainRestoreAutoFactor = _terrainWeakSignalRestoreUseAutoFactor,
                EnableWeakSignalTerrainRestoreUseShadowHeuristic = false,
                WeakSignalTerrainRestoreManualFactor = _terrainWeakSignalRestoreManualFactor,
                WeakSignalTerrainRestoreCandidateMinHeight = _terrainWeakSignalRestoreCandidateMinHeight,
                WeakSignalTerrainRestoreCandidateMaxHeight = _terrainWeakSignalRestoreCandidateMaxHeight,
                LeftSidebarWidth = _leftSidebarWidth,
                RightSidebarWidth = _rightSidebarWidth,
                BottomDrawerHeight = _bottomDrawerHeight,
                MinimapZoom = _minimapZoom,
                MinimapPanOffsetX = _minimapPanOffset.X,
                MinimapPanOffsetY = _minimapPanOffset.Y,
                CaptureOutputDir = _captureOutputDir,
                VideoEncoderExecutable = _videoEncoderExecutable,
                VideoCaptureFps = _videoCaptureFps,
                VideoCaptureIncludeUi = _videoCaptureIncludeUi,
                VideoCaptureContainerIndex = _videoCaptureContainerIndex,
                DetailedAdtTileCountOverride = _savedDetailedAdtTileCountOverride,
                Pm4TranslationX = _pm4SavedOverlayTranslation.X,
                Pm4TranslationY = _pm4SavedOverlayTranslation.Y,
                Pm4TranslationZ = _pm4SavedOverlayTranslation.Z,
                Pm4RotationX = _pm4SavedOverlayRotationDegrees.X,
                Pm4RotationY = _pm4SavedOverlayRotationDegrees.Y,
                Pm4RotationZ = _pm4SavedOverlayRotationDegrees.Z,
                Pm4ScaleX = _pm4SavedOverlayScale.X,
                Pm4ScaleY = _pm4SavedOverlayScale.Y,
                Pm4ScaleZ = _pm4SavedOverlayScale.Z,
                Pm4YawDegrees = _pm4SavedOverlayRotationDegrees.Z,
                TaxiActorModelOverrides = _savedTaxiActorModelOverridesByMap
                    .OrderBy(entry => entry.Key, StringComparer.OrdinalIgnoreCase)
                    .SelectMany(entry => entry.Value
                        .OrderBy(routeEntry => routeEntry.Key)
                        .Select(routeEntry => new SavedTaxiActorOverride
                        {
                            MapName = entry.Key,
                            RouteId = routeEntry.Key,
                            ModelPath = routeEntry.Value
                        }))
                    .ToList(),
                Pm4ObjectMatchSelections = _savedPm4ObjectMatches.Values
                    .OrderBy(selection => selection.MapName, StringComparer.OrdinalIgnoreCase)
                    .ThenBy(selection => selection.TileX)
                    .ThenBy(selection => selection.TileY)
                    .ThenBy(selection => selection.Ck24)
                    .ThenBy(selection => selection.ObjectPartId)
                    .ToList(),
                ObjectPathFilters = _savedObjectPathFiltersByMap.Values
                    .OrderBy(entry => entry.MapName, StringComparer.OrdinalIgnoreCase)
                    .Select(entry => new SavedObjectPathFilterMap
                    {
                        MapName = entry.MapName,
                        Enabled = entry.Enabled,
                        Filters = entry.Filters
                            .OrderBy(filter => filter.PathPrefix, StringComparer.OrdinalIgnoreCase)
                            .Select(filter => new SavedObjectPathFilterEntry
                            {
                                PathPrefix = filter.PathPrefix,
                                AppliesToWmo = filter.AppliesToWmo,
                                AppliesToMdx = filter.AppliesToMdx,
                            })
                            .ToList(),
                    })
                    .ToList(),
                ShellPanelLayouts = _savedShellPanelLayouts.Values
                    .OrderBy(layout => layout.PanelId)
                    .Select(layout => new SavedShellPanelLayout
                    {
                        PanelId = layout.PanelId,
                        NormalizedX = layout.NormalizedX,
                        NormalizedY = layout.NormalizedY,
                        NormalizedWidth = layout.NormalizedWidth,
                        NormalizedHeight = layout.NormalizedHeight,
                    })
                    .ToList(),
                ArcheologyMinUniqueId = _archeologyMinUniqueId,
                ArcheologyMaxUniqueId = _archeologyMaxUniqueId,
                ArcheologyScopeIndex = _archeologyScopeIndex,
                ArcheologyPlaybackSpeed = _archeologyPlaybackSpeed,
                ArcheologyPlaybackLoop = _archeologyPlaybackLoop,
                ArcheologyApplyToNextCapture = _archeologyApplyToNextCapture,
                ArcheologyApplyToVideoRecording = _archeologyApplyToVideoRecording,
                UseTabUi = _useTabUi,
                WorkbenchNavigationVersion = CurrentWorkbenchNavigationVersion,
                ActiveTopTab = (int)_activeTopTab,
                ActiveBottomTab = _activeBottomTabIndex
            };

            string json = JsonSerializer.Serialize(settings, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            File.WriteAllText(ViewerSettingsPath, json);
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[ViewerSettings] Failed to save settings: {ex.Message}");
        }
    }

    private static List<KnownGoodClientPath> NormalizeKnownGoodClientPaths(List<KnownGoodClientPath>? knownGoodClientPaths)
    {
        if (knownGoodClientPaths == null || knownGoodClientPaths.Count == 0)
            return new List<KnownGoodClientPath>();

        var normalizedEntries = new List<KnownGoodClientPath>();
        var seenPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var entry in knownGoodClientPaths)
        {
            if (entry == null || string.IsNullOrWhiteSpace(entry.Path))
                continue;

            string normalizedPath;
            try
            {
                normalizedPath = Path.GetFullPath(entry.Path);
            }
            catch
            {
                continue;
            }

            if (!seenPaths.Add(normalizedPath))
                continue;

            string name = string.IsNullOrWhiteSpace(entry.Name)
                ? Path.GetFileName(Path.TrimEndingDirectorySeparator(normalizedPath))
                : entry.Name.Trim();

            normalizedEntries.Add(new KnownGoodClientPath
            {
                Name = name,
                Path = normalizedPath,
                BuildVersion = string.IsNullOrWhiteSpace(entry.BuildVersion) ? null : entry.BuildVersion.Trim()
            });
        }

        return normalizedEntries
            .OrderBy(entry => entry.Name, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private sealed class ViewerSettings
    {
        public int UiTheme { get; set; } = (int)UiThemeKind.ModernSlate;
        public float UiFontScale { get; set; } = 1.0f;
        public int WmoMliqRotationQuarterTurns { get; set; }
        public bool HasExplicitWmoMliqRotationOverride { get; set; }
        public string? LastGameFolderPath { get; set; }
        public string? LastLooseOverlayPath { get; set; }
        public string? LastDatasetCatalogRoot { get; set; }
        public string? LastDatasetVersionRoot { get; set; }
        public string? LastActiveDatasetVersionRoot { get; set; }
        public string? LastSelectedBuildVersion { get; set; }
        public int TextureFilteringMode { get; set; } = (int)Rendering.TextureFilteringMode.Trilinear;
        public bool EnableMultisample { get; set; } = true;
        public bool EnableTerrainBackfaceCulling { get; set; } = true;
        public List<KnownGoodClientPath> KnownGoodClientPaths { get; set; } = new();
        public bool ShowMinimapWindow { get; set; } = true;
        public bool UseDockspaceUi { get; set; }
        public bool ShowLeftSidebar { get; set; } = true;
        public bool ShowRightSidebar { get; set; } = true;
        public bool ShowWorkspaceBarsPanel { get; set; } = true;
        public bool ShowBottomDrawer { get; set; } = true;
        public bool EnableWeakSignalTerrainRestore { get; set; }
        public bool EnableWeakSignalTerrainRestoreAllLoadedTiles { get; set; } = true;
        public bool EnableWeakSignalTerrainRestoreUseChunkMode { get; set; }
        public bool EnableWeakSignalTerrainRestoreUseTextureSubdivisions { get; set; } = true;
        public bool EnableWeakSignalTerrainRestoreAutoFactor { get; set; } = true;
        public bool EnableWeakSignalTerrainRestoreUseShadowHeuristic { get; set; }
        public float WeakSignalTerrainRestoreManualFactor { get; set; } = 16f;
        public float WeakSignalTerrainRestoreCandidateMinHeight { get; set; } = TerrainWeakSignalRestoreDefaultMinZ;
        public float WeakSignalTerrainRestoreCandidateMaxHeight { get; set; } = TerrainWeakSignalRestoreDefaultMaxZ;
        public int ShellPanelLayoutVersion { get; set; } = CurrentShellPanelLayoutVersion;
        public float LeftSidebarWidth { get; set; } = DefaultSidebarWidth;
        public float RightSidebarWidth { get; set; } = DefaultRightSidebarWidth;
        public float BottomDrawerHeight { get; set; } = DefaultBottomDrawerHeight;
        public float MinimapZoom { get; set; } = 4f;
        public float MinimapPanOffsetX { get; set; }
        public float MinimapPanOffsetY { get; set; }
        public string CaptureOutputDir { get; set; } = Path.Combine(OutputDir, "captures");
        public string VideoEncoderExecutable { get; set; } = "ffmpeg";
        public int VideoCaptureFps { get; set; } = 30;
        public bool VideoCaptureIncludeUi { get; set; }
        public int VideoCaptureContainerIndex { get; set; }
        public int DetailedAdtTileCountOverride { get; set; }
        public float Pm4TranslationX { get; set; }
        public float Pm4TranslationY { get; set; }
        public float Pm4TranslationZ { get; set; }
        public float Pm4RotationX { get; set; }
        public float Pm4RotationY { get; set; }
        public float Pm4RotationZ { get; set; }
        public float Pm4ScaleX { get; set; } = 1f;
        public float Pm4ScaleY { get; set; } = 1f;
        public float Pm4ScaleZ { get; set; } = 1f;
        public float Pm4YawDegrees { get; set; }
        public List<SavedTaxiActorOverride> TaxiActorModelOverrides { get; set; } = new();
        public List<SavedPm4ObjectMatchSelection> Pm4ObjectMatchSelections { get; set; } = new();
        public List<SavedObjectPathFilterMap> ObjectPathFilters { get; set; } = new();
        public List<SavedShellPanelLayout> ShellPanelLayouts { get; set; } = new();

        // 069 Phase 6: sticky archeology settings
        public int ArcheologyMinUniqueId { get; set; } = -1;
        public int ArcheologyMaxUniqueId { get; set; } = -1;
        public int ArcheologyScopeIndex { get; set; }

        // 069 Phase 7: archeology playback + capture integration
        public float ArcheologyPlaybackSpeed { get; set; } = 50f;
        public bool ArcheologyPlaybackLoop { get; set; }
        public bool ArcheologyApplyToNextCapture { get; set; }
        public bool ArcheologyApplyToVideoRecording { get; set; }

        // 069 tab system persistence
        public bool UseTabUi { get; set; } = true;
        public int WorkbenchNavigationVersion { get; set; }
        public int ActiveTopTab { get; set; }
        public int ActiveBottomTab { get; set; }

        // Global fog defaults
        public float DefaultFogStart { get; set; } = 200f;
        public float DefaultFogEnd { get; set; } = 1500f;

        // Camera defaults
        public float CameraSpeed { get; set; } = 50f;
        public float FovDegrees { get; set; } = 45f;
    }

    private sealed class SavedTaxiActorOverride
    {
        public string MapName { get; set; } = "";
        public int RouteId { get; set; }
        public string ModelPath { get; set; } = "";
    }
}
