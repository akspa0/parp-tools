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
    // HOST-IFACE-END
}
