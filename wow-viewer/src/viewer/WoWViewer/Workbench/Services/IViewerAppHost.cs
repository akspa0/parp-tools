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
using WowViewer.Core.Mdx;
using System;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Passes;
using WoWViewer.UI;
using System.Globalization;
using WowViewer.Core.IO.Casc;
using System.ComponentModel;
using System.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

/// <summary>
/// The ViewerApp state and behaviour that extracted services may use (Epic 251 U-01).
/// Implemented explicitly by <see cref="ViewerApp"/>. Mutable fields are exposed as ref-returning
/// properties so moved ImGui code can keep passing them by <c>ref</c>. Add a member here, not a
/// reference to the god class, when a service needs more.
/// </summary>
internal interface IViewerAppHost
{
    ref IDataSource? DataSource { get; }
    ref string? LoadedFilePath { get; }
    ref string MapConvertLkMapDir { get; }
    ref string MapConvertOutputDir { get; }
    ref string MapConvertProjectSourceKey { get; }
    ref string MapConvertSourcePath { get; }
    ref string ProjectOutputRootDir { get; }
    ref bool ShowMapConverterDialog { get; }
    ref bool ShowWmoConverterDialog { get; }
    ref string WmoConvertSourcePath { get; }
    string GetProjectOutputRootDirectory();
    void HandleProjectOutputRootChanged();
    void LoadWdtTerrain(string wdtPath);
    ref string MkHarvestDatasetRoot { get; }
    ref int MkHarvestViewerValidationCompleted { get; }
    ref int MkHarvestViewerValidationFailed { get; }
    ref int MkHarvestViewerValidationQueued { get; }
    ref MkHarvestViewerValidationCapturePlan? PendingMkHarvestViewerValidationCapturePlan { get; }
    ref bool ShowTerrainTextureTransferDialog { get; }
    ref bool ShowVlmExportDialog { get; }
    ref string TerrainTransferOutputDir { get; }
    ref string TerrainTransferSourceDir { get; }
    ref string TerrainTransferTargetDir { get; }
    ref string VlmClientPath { get; }
    ref string VlmMapName { get; }
    ref string VlmOutputDir { get; }
    ref VlmTerrainManager? VlmTerrainManager { get; }
    void LoadVlmProject(string projectRoot);
    void StitchMkHarvestViewerValidationOutputs(string mapName, string outputDirectory, string noLiquidsOutputDirectory, string noObjectsOutputDirectory, string objectsOnlyOutputDirectory, int requestedResolution);
    HashSet<(int tileX, int tileY, int chunkX, int chunkY)> SelectedChunks { get; }
    ref WowViewer.Core.Runtime.World.Terrain.Stratigraphy.StratigraphyAnchorMode StratigraphyAnchorMode { get; }
    ref bool StratigraphyPolarityInverted { get; }
    ref bool StratigraphyPreserveNegativeFloor { get; }
    ref bool StratigraphyUnhideDevMeshes { get; }
    ref bool StratigraphyUseNeighborAutoFit { get; }
    ref bool StratigraphyUseWdlMagnetization { get; }
    ref float StratigraphyWdlMagnetizationStrength { get; }
    ref TerrainManager? TerrainManager { get; }
    ref TerrainTileScope TerrainTileScope { get; }
    ref bool TerrainWeakSignalRestoreAllLoadedTiles { get; }
    ref float TerrainWeakSignalRestoreCandidateMaxHeight { get; }
    ref float TerrainWeakSignalRestoreCandidateMinHeight { get; }
    ref bool TerrainWeakSignalRestoreEnabled { get; }
    ref float TerrainWeakSignalRestoreManualFactor { get; }
    ref string TerrainWeakSignalRestoreStatus { get; }
    ref bool TerrainWeakSignalRestoreUseAutoFactor { get; }
    ref bool TerrainWeakSignalRestoreUseTextureSubdivisions { get; }
    ref WdlPreviewCacheService? WdlPreviewCacheService { get; }
    (int tileX, int tileY) GetCameraTile();
    string? GetCurrentSessionMapName();
    IReadOnlyList<(int tileX, int tileY)> GetTileScopeList(TerrainTileScope scope);
    ref TerrainTileScope MapGlbScope { get; }
    ref Md5TranslateIndex? Md5Index { get; }
    ref bool ShowAlphaFolderImportScope { get; }
    ref bool ShowHeightmapFolderImportScope { get; }
    ref bool ShowMccvFolderImportScope { get; }
    ref string StatusMessage { get; }
    ref TerrainExportKind TerrainExportKind { get; }
    ref TerrainImportKind TerrainImportKind { get; }
    ref int TerrainTileRangeEndX { get; }
    ref int TerrainTileRangeEndY { get; }
    ref int TerrainTileRangeStartX { get; }
    ref int TerrainTileRangeStartY { get; }
    TerrainWeakSignalRestoreService TerrainWeakSignalRestore { get; }
    ref Camera Camera { get; }
    ref ChunkClipboard? ChunkClipboard { get; }
    ref (int tileX, int tileY, int chunkX, int chunkY)? ChunkClipboardCopiedKey { get; }
    ref (int tileX, int tileY, int chunkX, int chunkY)? ChunkClipboardLockedTargetKey { get; }
    ref ChunkClipboardSet? ChunkClipboardSet { get; }
    ref bool ChunkClipboardShowOverlay { get; }
    ref string ChunkClipboardStatus { get; }
    ref bool ChunkToolEnabled { get; }
    TerrainTileIoService TerrainTileIo { get; }
    string EnsureEditorProjectOutputDirectory(bool forceNew = false);
    string GetEditorProjectName(string? fallbackName = null);
    string? GetEditorProjectSourceKey();
    bool TryPickTerrainChunkUnderMouse(TerrainRenderer renderer, out TerrainRenderer.TerrainChunkInfo info);
    ref string EditorProjectOutputDir { get; }
    ref string SelectedPlacementSaveStatus { get; }
    ref string? SelectedPlacementSaveTargetPath { get; }
    ref WorldScene? WorldScene { get; }
    void RefreshSelectedWorldObjectInfo();
    ref int CurrentMapId { get; }
    ref bool SqlForceStreamRefresh { get; }
    ref SqlWorldPopulationService? SqlPopulationService { get; }
    void DrawToolbarPopupButton(string label, string summary, string popupId, Action drawContent);
    void ExportAnimationStateJson(IAnimationController animator, int currentSeq, string currentSeqName, float seqStart, float seqEnd);
    ref string? LastVirtualPath { get; }
    Dictionary<string, Dictionary<int, string>> SavedTaxiActorModelOverridesByMap { get; }
    ref int SelectedAreaPoiId { get; }
    ref int SelectedObjectIndex { get; }
    ref string SelectedObjectInfo { get; }
    ref string SelectedObjectType { get; }
    ref string TaxiActorModelOverrideInput { get; }
    ref int TaxiActorModelOverrideInputRouteId { get; }
    ref int TaxiActorModelOverrideTargetRouteId { get; }
    void SaveViewerSettings();
    bool TryGetSelectedBrowserModelPath(out string assetPath);
    ref EditorWorkspaceTask EditorWorkspaceTask { get; }
    ref float FovDegrees { get; }
    ref GL Gl { get; }
    ref Pm4ObjectMatchObject? HoveredPm4ObjectMatch { get; }
    ref int HoveredPm4ObjectMatchCacheMaxMatches { get; }
    ref (int tileX, int tileY, uint ck24, int objectPart)? HoveredPm4ObjectMatchKey { get; }
    ref float LastMouseX { get; }
    ref float LastMouseY { get; }
    ref int Pm4ObjectMatchMaxMatchesPerObject { get; }
    ref SceneClusterSelector3D? SceneClusterSelector3D { get; }
    ref SceneCursorRenderer? SceneCursorRenderer { get; }
    TaxiAndAreaPoiSelectionService TaxiAndAreaPoi { get; }
    ref VisualInvestigationMode VisualInvestigationMode { get; }
    ref WorkspaceMode WorkspaceMode { get; }
    bool CanSceneConsumeMouse(float x, float y);
    void ClearSelectedWlLiquidBody(bool clearListIsolation);
    float GetSceneFarPlane();
    bool IsSceneMouseCaptureBlocked(float x, float y);
    void SelectTerrainChunkFromClick(TerrainRenderer.TerrainChunkInfo info);
    void SetSelectedWlLiquidBody(WlLiquidBody body, bool isolateInList, bool focusInspectWorkspace, string? statusMessage = null);
    bool ShouldShowHoveredAssetInfoForInvestigation(HoveredAssetInfo info);
    bool TogglePm4ObjectCollectionMembership((int tileX, int tileY, uint ck24, int objectPart) key, bool reportStatus, bool removeIfPresent = true);
    bool TryFindWlLiquidBodyByKey(string bodyKey, out WlLiquidBody? body);
    bool TryGetSceneViewportRect(out float x, out float y, out float width, out float height);
    bool TryRaycastTerrain(TerrainRenderer renderer, Vector3 rayOrigin, Vector3 rayDir, float maxDistance, out TerrainRenderer.TerrainChunkInfo info);
    bool TryRaycastTerrain(TerrainRenderer renderer, Vector3 rayOrigin, Vector3 rayDir, float maxDistance, out TerrainRenderer.TerrainChunkInfo info, out Vector3 hitPoint);
    bool TryResolveHoveredWlLiquidBody(HoveredAssetInfo hoveredInfo, out WlLiquidBody? body);
    ref FixedBottomDrawerTab ActiveBottomDrawerTab { get; }
    ref float BottomDrawerHeight { get; }
    ref Vector2 DockspaceHostPosition { get; }
    ref Vector2 DockspaceHostSize { get; }
    ref bool ForceApplyShellPanelLayout { get; }
    ref bool FullscreenMinimap { get; }
    ref bool HideUiChrome { get; }
    ref ImGuiController ImGui { get; }
    ref float LeftSidebarWidth { get; }
    ref string ModelInfo { get; }
    ref ShellPanelId? PendingFocusedShellPanel { get; }
    ref FixedBottomDrawerTab? PendingRightSidebarSection { get; }
    HashSet<ShellPanelId> PendingShellPanelLayoutRestore { get; }
    ref float RightSidebarWidth { get; }
    Dictionary<ShellPanelId, SavedShellPanelLayout> SavedShellPanelLayouts { get; }
    ref bool ShowLeftSidebar { get; }
    ref bool ShowMinimapWindow { get; }
    ref bool ShowModelInfo { get; }
    ref bool ShowRightSidebar { get; }
    ref bool ShowWorkspaceBarsPanel { get; }
    ref bool UseDockspaceUi { get; }
    ref bool UseTabUi { get; }
    ref IWindow Window { get; }
    float ClampFixedSidebarWidth(float width, bool isLeftSidebar, float displayWidth);
    float GetTopChromeHeight();
    void SetEditorWorkspaceTask(EditorWorkspaceTask task);
    ref List<MapDefinition> DiscoveredMaps { get; }
    ref Vector3? PendingWorldSpawnOverride { get; }
    ref MapDefinition? SelectedMapForPreview { get; }
    ref Vector2? SelectedSpawnTile { get; }
    ref bool ShowWdlPreview { get; }
    ref WdlPreviewRenderer? WdlPreviewRenderer { get; }
    void LoadFileFromDataSource(string virtualPath);
    void LoadMapAtDefaultSpawn(MapDefinition map);
    string? ResolveMapWdtPath(string mapDirectory);
    ref AreaTableService? AreaTableService { get; }
    ref WowViewer.Core.World.AreaLookupResult? CurrentAreaLookup { get; }
    ref string CurrentAreaName { get; }
    ref ISceneRenderer? Renderer { get; }
    HashSet<string> ReportedAreaDiagnostics { get; }
    ref bool AutoFrameModelOnLoad { get; }
    ref string? DbcBuild { get; }
    ref float LastWorldSceneCameraPitch { get; }
    ref Vector3 LastWorldSceneCameraPosition { get; }
    ref float LastWorldSceneCameraYaw { get; }
    ref string? LastWorldSceneWdtPath { get; }
    ref string? LoadedFileName { get; }
    ref M2StaticRenderModel? LoadedM2Runtime { get; }
    ref MdxFile? LoadedMdx { get; }
    ref WmoV14ToV17Converter.WmoV14Data? LoadedWmo { get; }
    ref Rendering.LoadingScreen? LoadingScreen { get; }
    HashSet<string> LoggedStandaloneMissingSkinPaths { get; }
    SqlSpawnStreamingService SqlSpawnStreaming { get; }
    Dictionary<string, string?> StandaloneSkinPathCache { get; }
    ref ReplaceableTextureResolver? TexResolver { get; }
    WdlPreviewService WdlPreview { get; }
    void FrameCurrentModel();
    string? TryGetLoadedLocalWdtPath();
    ref DBCD.Providers.IDBCProvider? DbcProvider { get; }
    ref string? DbdDir { get; }
    ref float DefaultFogEnd { get; }
    ref float DefaultFogStart { get; }
    ref MinimapRenderer? MinimapRenderer { get; }
    StandaloneModelLoaderService ModelLoader { get; }
    ref int SavedDetailedAdtTileCountOverride { get; }
    Dictionary<string, SavedObjectPathFilterMap> SavedObjectPathFiltersByMap { get; }
    List<TerrainHiddenTileCandidate> TerrainAnalysisHiddenCandidates { get; }
    ref int TerrainAnalysisHiddenSelectedIndex { get; }
    ref string TerrainAnalysisHiddenStatus { get; }
    ref (int tileX, int tileY)? TerrainAnalysisPreviewCompareTile { get; }
    ref float? TerrainAnalysisPreviewSimilarity { get; }
    void ApplyLayoutObjectPreviewModeToScene();
    void ApplySavedPm4AlignmentToScene();
    void InvalidatePm4DerivedReports();
    bool FullLoadMode { get; set; }
    ref bool AutoOpenWorldMapsPanel { get; }
    ref AssetCatalogView? CatalogView { get; }
    ref string ExtensionFilter { get; }
    ref List<string> FilteredFiles { get; }
    ref string LastGameFolderPath { get; }
    ref string LastLooseOverlayPath { get; }
    ref string SearchFilter { get; }
    ref int SelectedFileIndex { get; }
    WorldLoaderService WorldLoader { get; }
    M2CameraPathDocument CameraPath { get; }
    ref Terrain.BoundingBoxRenderer? EditorOverlayBb { get; }
    ref int LastMcnkOverlayChunkCount { get; }
    ref int LastMcnkWeakCornerCount { get; }
    ref McnkOverlayFlags McnkOverlayFlags { get; }
    ShellLayoutService ShellLayout { get; }
    ref bool ShowCameraPathOverlay { get; }
    ref bool ShowMcnkFlagOverlay { get; }
    ref bool ShowMcnkWeakCorners { get; }
    DataSourceSessionService DataSourceSession { get; }
    Dictionary<(int tileX, int tileY), WowViewer.Core.Runtime.World.Terrain.Stratigraphy.StratigraphyTileAnalysis> StratigraphyTileAnalyses { get; }
    ConverterDialogsService ConverterDialogs { get; }
    PlacementEditService PlacementEditing { get; }
    ref bool TaxiRideCameraEnabled { get; }
    ref bool WlLayerListIsolationEnabled { get; }
    ref string WlLayerSelectedBodyKey { get; }
    void DrawTerrainChunkInvestigationPanel(bool defaultOpen);
    void DrawVisualInvestigationToolbox(bool showWorldObjectRangeControls);
    void OpenPm4Workbench(Pm4WorkbenchTab tab);
    bool ShouldIncludeWlBodyInUiList(WlLiquidBody body);
    bool IsWlListIsolationActive { get; }
    ref int ActiveBottomTabIndex { get; }
    ref string ActiveDatasetVersionRoot { get; }
    ref WorkbenchTab ActiveTopTab { get; }
    ref int ActiveUtilitiesTabIndex { get; }
    ref bool ArcheologyApplyToNextCapture { get; }
    ref bool ArcheologyApplyToVideoRecording { get; }
    ref int ArcheologyMaxUniqueId { get; }
    ref int ArcheologyMinUniqueId { get; }
    ref bool ArcheologyPlaybackLoop { get; }
    ref float ArcheologyPlaybackSpeed { get; }
    ref int ArcheologyScopeIndex { get; }
    ref float CameraSpeed { get; }
    ref string CaptureOutputDir { get; }
    List<WoWViewer.Terrain.ClientBuildOption> ClientBuildOptions { get; }
    ref string DatasetCatalogRoot { get; }
    ref bool EnableMultisample { get; }
    ref bool EnableTerrainBackfaceCulling { get; }
    ref bool HasExplicitWmoMliqRotationOverride { get; }
    ref List<KnownGoodClientPath> KnownGoodClientPaths { get; }
    ref Vector2 MinimapPanOffset { get; }
    ref float MinimapZoom { get; }
    ref bool OpenForgetKnownGoodClientConfirm { get; }
    ref string? PendingForgetKnownGoodClientDisplayName { get; }
    ref string? PendingForgetKnownGoodClientPath { get; }
    ref Vector3 Pm4SavedOverlayRotationDegrees { get; }
    ref Vector3 Pm4SavedOverlayScale { get; }
    ref Vector3 Pm4SavedOverlayTranslation { get; }
    ref Dictionary<string, Pm4WmoMatchEntry> Pm4WmoMatchEntries { get; }
    ref Pm4WmoMatchStore? Pm4WmoMatchStore { get; }
    Dictionary<string, SavedPm4ObjectMatchSelection> SavedPm4ObjectMatches { get; }
    ref int SelectedBuildOptionIndex { get; }
    ref string SelectedDatasetVersionRoot { get; }
    ref TextureFilteringMode TextureFilteringMode { get; }
    ref float UiFontScale { get; }
    ref UiThemeKind UiTheme { get; }
    ref int VideoCaptureContainerIndex { get; }
    ref int VideoCaptureFps { get; }
    ref bool VideoCaptureIncludeUi { get; }
    ref string VideoEncoderExecutable { get; }
    int FindBuildOptionIndex(string? buildVersion);
    void NormalizeWorkbenchStateAfterLoad();
    void RefreshClientBuildOptions();
    void RefreshDatasetCatalog();
    ProjectOutputService ProjectOutput { get; }
    ref string FolderInputBuf { get; }
    ref bool PendingKnownGoodClientAttachLooseFolder { get; }
    ref string? PendingKnownGoodClientBuildVersion { get; }
    ref string? PendingKnownGoodClientPath { get; }
    ViewerSettingsService Settings { get; }
    ref bool ShowBuildSelectionDialog { get; }
    ref bool ShowFolderInput { get; }
    ref bool ShowListfileInput { get; }
    ref bool ShowRosettaDatastoreDialog { get; }
    CascAhdrSourceService CascAhdrSource { get; }
    ClientDialogsService ClientDialogs { get; }
    ref UtilitiesBottomTab? LegacyUtilityPage { get; }
    ref bool ShowFileBrowser { get; }
    ref bool ShowLogViewer { get; }
    ref bool ShowPerfWindow { get; }
    ref bool ShowSettingsWindow { get; }
    ref bool ShowSynthesizedMinimapExportDialog { get; }
    ref bool WantExportGlb { get; }
    ref bool WantExportGlbCollision { get; }
    ref bool WantExportMapGlbTiles { get; }
    ref bool WantOpenFile { get; }
    ref bool WantSelectDatasetCatalogRoot { get; }
    ref bool WantTerrainExport { get; }
    ref bool WantTerrainImport { get; }
    ref bool WorkbenchOpen { get; }
    ViewerKeyContext GetActiveKeyContext();
    void OpenCapturePanelTab(CapturePanelTab tab);
    void OpenWorkbenchTab(WorkbenchTab topTab, int bottomIndex = -1);
    void OpenWorkbenchTab(ModelBottomTab tab);
    void OpenWorkbenchTab(WorldBottomTab tab);
    void OpenWorkbenchTab(ToolsBottomTab tab);
    void OpenWorkbenchTab(UtilitiesBottomTab tab);
    void PrepareSynthesizedMinimapExportDialogInputs();
    void ResetCamera();
    void SetWorkspaceMode(WorkspaceMode mode);
    ref int ActivePm4TabIndex { get; }
    ref ActiveVideoRecording? ActiveVideoRecording { get; }
    ref long LastTaxiRideCameraTick { get; }
    ref TaxiRideCameraMode TaxiRideCameraMode { get; }
    ref bool TaxiRideCameraPoseInitialized { get; }
    ref int TaxiRideCameraRouteId { get; }
    ref WorldScene? TaxiRideCameraScene { get; }
    ref float TaxiRideChaseDistance { get; }
    ref float TaxiRideChaseHeight { get; }
    ref float TaxiRideCockpitHeight { get; }
    ref float TaxiRideFreeLookPitchOffset { get; }
    ref float TaxiRideFreeLookYawOffset { get; }
    void CopyTextToClipboard(string text, string description);
    void StopCameraPathPlayback();
    void StopTaxiRideCamera(string? statusMessage = null);
    void StopVideoRecording(string? statusOverride = null);
    bool TryStartCurrentViewVideoRecording(bool includeUi, string? label = null);
    // HOST-IFACE-END
}
