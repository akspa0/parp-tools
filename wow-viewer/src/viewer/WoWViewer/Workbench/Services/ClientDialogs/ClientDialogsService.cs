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
/// Client selection dialogs: game-folder / known-good client pickers and loose-folder attachment prompts.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed partial class ClientDialogsService
{
    private readonly IViewerAppHost _host;

    internal ClientDialogsService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private List<WoWViewer.Terrain.ClientBuildOption> _clientBuildOptions => _host.ClientBuildOptions;
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private DataSourceSessionService _dataSourceSession => _host.DataSourceSession;
    private ref string? _dbcBuild => ref _host.DbcBuild;
    private ref string _folderInputBuf => ref _host.FolderInputBuf;
    private ref List<KnownGoodClientPath> _knownGoodClientPaths => ref _host.KnownGoodClientPaths;
    private ref bool _pendingKnownGoodClientAttachLooseFolder => ref _host.PendingKnownGoodClientAttachLooseFolder;
    private ref string? _pendingKnownGoodClientBuildVersion => ref _host.PendingKnownGoodClientBuildVersion;
    private ref string? _pendingKnownGoodClientPath => ref _host.PendingKnownGoodClientPath;
    private ref int _selectedBuildOptionIndex => ref _host.SelectedBuildOptionIndex;
    private ViewerSettingsService _settings => _host.Settings;
    private ref bool _showBuildSelectionDialog => ref _host.ShowBuildSelectionDialog;
    private ref bool _showFolderInput => ref _host.ShowFolderInput;
    private ref bool _showListfileInput => ref _host.ShowListfileInput;
    private ref bool _showRosettaDatastoreDialog => ref _host.ShowRosettaDatastoreDialog;
    private ref string _statusMessage => ref _host.StatusMessage;
    private WorldLoaderService _worldLoader => _host.WorldLoader;

    private static readonly WoWViewer.Terrain.ClientBuildOption[] FallbackClientBuildOptions =
    {
        new("Alpha (0.x) - 0.5.3.3368", "0.5.3.3368"),
        new("Alpha (0.x) - 0.7.0.3694", "0.7.0.3694"),
        new("Alpha (0.x) - 0.8.0.3734", "0.8.0.3734"),
        new("Alpha (0.x) - 0.9.0.3807", "0.9.0.3807"),
        new("Alpha (0.x) - 0.9.1.3810", "0.9.1.3810"),
        new("Alpha (0.x) - 0.10.3892", "0.10.3892"),
        new("Burning Crusade (2.x) - 2.4.3.8606", "2.4.3.8606"),
        new("Wrath (3.x) - 3.0.1.8303", "3.0.1.8303"),
        new("Wrath (3.x) - 3.3.5.12340", "3.3.5.12340"),
        new("Cataclysm (4.x) - 4.0.0.11927", "4.0.0.11927"),
        new("Cataclysm (4.x) - 4.0.1.12304", "4.0.1.12304")
    };
    private string? _pendingGameFolderPath;
    private string _buildSelectionFilter = "";
    private string? _buildSelectionHint;
}
