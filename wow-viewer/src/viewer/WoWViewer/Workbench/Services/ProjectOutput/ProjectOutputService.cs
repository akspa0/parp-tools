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
using static WoWViewer.ViewerApp;

namespace WoWViewer;

/// <summary>
/// Project output paths: the project output root, per-editor-project folders, timestamped output folders and path-segment sanitising.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed class ProjectOutputService
{
    private readonly IViewerAppHost _host;

    internal ProjectOutputService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ConverterDialogsService _converterDialogs => _host.ConverterDialogs;
    private DataSourceSessionService _dataSourceSession => _host.DataSourceSession;
    private ref string _editorProjectOutputDir => ref _host.EditorProjectOutputDir;
    private ref string? _lastWorldSceneWdtPath => ref _host.LastWorldSceneWdtPath;
    private ref string? _loadedFilePath => ref _host.LoadedFilePath;
    private ref string _mapConvertOutputDir => ref _host.MapConvertOutputDir;
    private ref string _mapConvertProjectSourceKey => ref _host.MapConvertProjectSourceKey;
    private ref string _mapConvertSourcePath => ref _host.MapConvertSourcePath;
    private PlacementEditService _placementEditing => _host.PlacementEditing;
    private ref string _projectOutputRootDir => ref _host.ProjectOutputRootDir;
    private ref string _selectedPlacementSaveStatus => ref _host.SelectedPlacementSaveStatus;

    private string _editorProjectSourceKey = string.Empty;

    internal string GetProjectOutputRootDirectory()
    {
        if (string.IsNullOrWhiteSpace(_projectOutputRootDir))
            _projectOutputRootDir = ProjectsDir;

        return Path.GetFullPath(_projectOutputRootDir);
    }

    internal void HandleProjectOutputRootChanged()
    {
        _editorProjectOutputDir = string.Empty;
        _editorProjectSourceKey = string.Empty;
        _mapConvertOutputDir = string.Empty;
        _mapConvertProjectSourceKey = string.Empty;
        _placementEditing.RefreshProjectManagedPlacementTargets();

        if (!string.IsNullOrWhiteSpace(_mapConvertSourcePath))
            _converterDialogs.EnsureMapConverterProjectOutputDirectory(forceNew: false);
    }

    internal static string SanitizeProjectPathSegment(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "project";

        char[] invalid = Path.GetInvalidFileNameChars();
        var builder = new StringBuilder(value.Trim().Length);
        foreach (char c in value.Trim())
        {
            builder.Append(Array.IndexOf(invalid, c) >= 0 || char.IsControl(c)
                ? '_'
                : char.IsWhiteSpace(c) ? '_' : c);
        }

        string sanitized = builder.ToString().Trim('.', ' ');
        return string.IsNullOrWhiteSpace(sanitized) ? "project" : sanitized;
    }

    internal static string CreateTimestampedProjectOutputDirectory(string rootDirectory, string projectName)
    {
        string safeProjectName = SanitizeProjectPathSegment(projectName);
        string projectRoot = Path.Combine(rootDirectory, safeProjectName);
        Directory.CreateDirectory(projectRoot);

        string timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
        string candidate = Path.Combine(projectRoot, timestamp);
        int suffix = 1;
        while (Directory.Exists(candidate))
        {
            candidate = Path.Combine(projectRoot, $"{timestamp}_{suffix:D2}");
            suffix++;
        }

        return candidate;
    }

    internal string GetEditorProjectName(string? fallbackName = null)
    {
        if (!string.IsNullOrWhiteSpace(_dataSourceSession.GetCurrentSessionMapName()))
            return SanitizeProjectPathSegment(_dataSourceSession.GetCurrentSessionMapName()!);

        string? wdtPath = _dataSourceSession.TryGetLoadedLocalWdtPath();
        if (!string.IsNullOrWhiteSpace(wdtPath))
            return SanitizeProjectPathSegment(Path.GetFileNameWithoutExtension(wdtPath));

        if (!string.IsNullOrWhiteSpace(_lastWorldSceneWdtPath) && File.Exists(_lastWorldSceneWdtPath))
            return SanitizeProjectPathSegment(Path.GetFileNameWithoutExtension(_lastWorldSceneWdtPath));

        if (!string.IsNullOrWhiteSpace(fallbackName))
            return SanitizeProjectPathSegment(fallbackName);

        return "project";
    }

    internal string? GetEditorProjectSourceKey()
    {
        string? wdtPath = _dataSourceSession.TryGetLoadedLocalWdtPath();
        if (!string.IsNullOrWhiteSpace(wdtPath))
            return Path.GetFullPath(wdtPath);

        if (!string.IsNullOrWhiteSpace(_lastWorldSceneWdtPath) && File.Exists(_lastWorldSceneWdtPath))
            return Path.GetFullPath(_lastWorldSceneWdtPath);

        if (!string.IsNullOrWhiteSpace(_loadedFilePath) && File.Exists(_loadedFilePath))
            return Path.GetFullPath(_loadedFilePath);

        string? currentMapName = _dataSourceSession.GetCurrentSessionMapName();
        return string.IsNullOrWhiteSpace(currentMapName) ? null : $"map:{currentMapName}";
    }

    internal string EnsureEditorProjectOutputDirectory(bool forceNew = false)
    {
        string sourceKey = GetEditorProjectSourceKey() ?? $"editor:{GetEditorProjectName()}";
        if (!forceNew
            && !string.IsNullOrWhiteSpace(_editorProjectOutputDir)
            && string.Equals(_editorProjectSourceKey, sourceKey, StringComparison.OrdinalIgnoreCase))
        {
            return _editorProjectOutputDir;
        }

        _editorProjectSourceKey = sourceKey;
        _editorProjectOutputDir = CreateTimestampedProjectOutputDirectory(GetProjectOutputRootDirectory(), GetEditorProjectName());
        return _editorProjectOutputDir;
    }

    internal string DescribeEditorProjectOutputDirectory()
    {
        if (!string.IsNullOrWhiteSpace(_editorProjectOutputDir))
            return _editorProjectOutputDir;

        return Path.Combine(GetProjectOutputRootDirectory(), GetEditorProjectName(), "<timestamp>");
    }

    internal void StartNewEditorProjectOutputDirectory()
    {
        _editorProjectOutputDir = EnsureEditorProjectOutputDirectory(forceNew: true);
        _placementEditing.RefreshProjectManagedPlacementTargets();
        _selectedPlacementSaveStatus = $"Created new project output folder: {_editorProjectOutputDir}";
    }
}
