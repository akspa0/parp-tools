using ImGuiNET;
using WoWViewer.Terrain;
using WoWViewer.UI;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

internal sealed partial class MainMenuBarService
{
    private readonly TerrainFileOperationsService _terrainFiles;
    private readonly ModelWorldFileOperationsService _modelWorldFiles;
    public const string AboutPopupId = "About WoWViewer";
    public const string ForgetBasePopupId = "Confirm Forget Known-Good Base";
    public const string KeyboardShortcutsPopupId = "Keyboard Shortcuts";
    public const string UiReferencePopupId = "UI Reference";

    public void DrawTerrainFileOperationsContent() => _terrainFiles.DrawContent();
    public void DrawModelExportContent() => _modelWorldFiles.DrawModelExportContent();

    /// <summary>Inject into EditorConvertersPage to replace the legacy converter wrapper.</summary>
    public void DrawConvertersContent()
    {
        if (SharedUiWidgets.SectionHeader("Map Converter", defaultOpen: true, id: "EditorMapConverter"))
        {
            ImGui.TextWrapped("Convert maps using the existing converter service and its output manifest.");
            if (SharedUiWidgets.ActionButton("Launch Map Converter"))
            {
                PrepareMapConverterDialogInputs();
                _showMapConverterDialog = true;
            }
        }
        if (SharedUiWidgets.SectionHeader("WMO Converter", defaultOpen: true, id: "EditorWmoConverter"))
        {
            ImGui.TextWrapped("Convert WMO files using the existing converter service.");
            if (SharedUiWidgets.ActionButton("Launch WMO Converter"))
            {
                PrepareWmoConverterDialogInputs();
                _showWmoConverterDialog = true;
            }
        }
    }

    public void DrawWorldExportContent()
    {
        _modelWorldFiles.DrawWorldExportContent();
        if (SharedUiWidgets.SectionHeader("DAT Conversion", defaultOpen: false, id: "WorldDatExport"))
        {
            MapExportFormats.DrawCheckboxes("world_destination");
            if (SharedUiWidgets.ActionButton("Export Loaded DAT Map...", _terrainManager?.Adapter is AhdrTerrainAdapter && MapExportFormats.Any))
                RunDatExport(_cascAhdrSource.ExportLoadedDatMap);
            if (SharedUiWidgets.ActionButton("Export Nearby Tiles as DAT v26 (experimental)", _terrainManager?.Adapter is StandardTerrainAdapter && _dataSource != null))
                RunDatExport(() => _cascAhdrSource.ExportNearbyTilesAsDatV26(radius: 2));
            ImGui.TextDisabled("Uses the existing DAT exporter and its output/manifest status; runtime confirmation is operator-owned.");
        }
    }

    // The existing provider is a submenu: call inside a manager-owned world settings menu.
    public void DrawDatHeightScaleMenuContent() => _cascAhdrSource.DrawAhdrHeightScaleMenu();
    public void RequestAbout() => _openAboutPopup = true;
    public void RequestKeyboardShortcuts() => _showKeyboardShortcutsWindow = true;
    public void RequestUiReference() => _showUiReferenceWindow = true;

    /// <summary>Drain at the manager popup root. IDs are content keys, not shell types.</summary>
    public bool TryTakePopupRequest(out string popupId)
    {
        if (_openForgetKnownGoodClientConfirm)
        {
            _openForgetKnownGoodClientConfirm = false;
            popupId = ForgetBasePopupId;
        }
        else if (_openAboutPopup)
        {
            _openAboutPopup = false;
            popupId = AboutPopupId;
        }
        else if (_showKeyboardShortcutsWindow)
        {
            _showKeyboardShortcutsWindow = false;
            popupId = KeyboardShortcutsPopupId;
        }
        else if (_showUiReferenceWindow)
        {
            _showUiReferenceWindow = false;
            popupId = UiReferencePopupId;
        }
        else { popupId = ""; return false; }
        return true;
    }

    /// <summary>Draw inside the manager-owned popup. True requests closing it.</summary>
    public bool DrawPopupContent(string popupId)
    {
        switch (popupId)
        {
            case AboutPopupId:
                DrawAboutContent();
                return SharedUiWidgets.ActionButton("Close##about");
            case KeyboardShortcutsPopupId:
                DrawKeyboardShortcutsContent();
                return SharedUiWidgets.ActionButton("Close##shortcuts");
            case ForgetBasePopupId:
                string name = string.IsNullOrWhiteSpace(_pendingForgetKnownGoodClientDisplayName)
                    ? "this saved base" : _pendingForgetKnownGoodClientDisplayName!;
                ImGui.TextWrapped($"Remove saved base '{name}'?");
                if (!string.IsNullOrWhiteSpace(_pendingForgetKnownGoodClientPath)) ImGui.TextWrapped(_pendingForgetKnownGoodClientPath);
                if (SharedUiWidgets.ActionButton("Remove##saved_base"))
                {
                    if (!string.IsNullOrWhiteSpace(_pendingForgetKnownGoodClientPath))
                        _clientDialogs.ForgetKnownGoodClientPath(_pendingForgetKnownGoodClientPath);
                    OnPopupClosed(popupId);
                    return true;
                }
                ImGui.SameLine();
                if (SharedUiWidgets.ActionButton("Cancel##saved_base")) { OnPopupClosed(popupId); return true; }
                return false;
            default:
                throw new ArgumentOutOfRangeException(nameof(popupId), popupId, "Unknown main-menu popup content.");
        }
    }

    /// <summary>Also call on manager title-bar close and escape paths.</summary>
    public void OnPopupClosed(string popupId)
    {
        if (popupId == ForgetBasePopupId) _settings.ClearPendingForgetKnownGoodClientPath();
    }

    private static void DrawAboutContent()
    {
        ImGui.TextUnformatted(ViewerProductName);
        ImGui.TextDisabled($"Version {GetViewerDisplayVersion()}");
        ImGui.Spacing();
        ImGui.TextWrapped("World/model viewer and debugging surface for WoW Alpha, Wrath, and early Cataclysm data.");
        ImGui.Spacing();
        ImGui.TextWrapped("Author: github.com/akspa0/parp-tools");
        ImGui.TextWrapped("Discord: discord.gg/6YdUksuKuU");
        ImGui.Spacing();
        ImGui.TextUnformatted("In memory of Hayven Games");
        ImGui.TextWrapped("An inspiration for this project and a friend, whose short films explored World of Warcraft's secrets and little-known details through game footage.");
        ImGui.Spacing();
        ImGui.Separator();
        ImGui.TextWrapped("Thanks to...");
        ImGui.TextWrapped("Marlamin, schlumpf, Dovah, Pirate the Explorer, fean, implave, IS4, Mjollna, Adspartan (Noggit), and Skarn (Noggit-Red).");
        ImGui.TextWrapped("Special thanks to Kruithne for wow.export, whose tools and format research have been invaluable to the community.");
        ImGui.TextWrapped("Without the WoW Exploration community, this project would not exist. Everyone named here contributed inspiration to this project in some way.");
        ImGui.TextDisabled("This tooling is about restoration, not touching up or polishing what we recover - it is the instrument for restoring what already exists. Noggit and Noggit-Red remain the preferred editors for fine-tuning the results this library and tooling produce.");
        ImGui.Spacing();
        ImGui.TextWrapped("Special thanks to WoWdev.wiki, Exploration Reboot, The Alpha Project, and everyone in the Pre-Alpha Restoration Project discord!");
        ImGui.Spacing();
    }

    private void LoadPickedFile(string path)
    {
        try
        {
            if (!File.Exists(path)) throw new FileNotFoundException("Selected file no longer exists.", path);
            _lastLooseOverlayPath = Path.GetDirectoryName(path);
            _statusMessage = "";
            _modelLoader.LoadFileFromDisk(path);
            // The loader catches failures and may defer world loading. Preserve its result;
            // never overwrite it with an unconditional 'Loaded' or claim async completion.
            string result = _statusMessage;
            _statusMessage = string.IsNullOrWhiteSpace(result) ? $"Load requested: {path}" : $"{result} — {path}";
        }
        catch (Exception ex) { _statusMessage = $"Open failed — {path}: {ex.Message}"; }
    }

    private void RunFolderAction(string path, Action action)
    {
        try
        {
            if (!Directory.Exists(path)) throw new DirectoryNotFoundException(path);
            _statusMessage = "";
            action();
            string result = _statusMessage;
            _statusMessage = string.IsNullOrWhiteSpace(result) ? $"Folder action requested: {path}" : $"{result} — {path}";
        }
        catch (Exception ex) { _statusMessage = $"Folder action failed — {path}: {ex.Message}"; }
    }

    private void RunDatExport(Action action)
    {
        try
        {
            _statusMessage = "Select DAT output; export completion will be reported by the existing exporter.";
            action();
            if (_statusMessage.StartsWith("DAT v26 export: 0 tiles", StringComparison.Ordinal))
                _statusMessage = $"DAT export failed: no artifacts written. {_statusMessage}";
        }
        catch (Exception ex) { _statusMessage = $"DAT export failed: {ex.Message}"; }
    }

    private void HandleFileOperationRequests()
    {
        if (_wantTerrainExport)
        {
            _wantTerrainExport = false;
            var kind = _terrainExportKind;
            _terrainExportKind = TerrainExportKind.None;
            _terrainFiles.RequestExport(kind);
        }
        if (_wantTerrainImport)
        {
            _wantTerrainImport = false;
            var kind = _terrainImportKind;
            _terrainImportKind = TerrainImportKind.None;
            _terrainFiles.RequestImport(kind);
        }
        if (_wantExportGlbCollision)
        {
            _wantExportGlbCollision = false;
            if (_loadedFilePath != null && _renderer != null) _modelWorldFiles.RequestModelExport(true);
            else _modelWorldFiles.RequestWorldExport(TerrainTileScope.CurrentTile, true);
        }
        if (_wantExportGlb)
        {
            _wantExportGlb = false;
            if (_loadedFilePath != null && _renderer != null) _modelWorldFiles.RequestModelExport(false);
            else _modelWorldFiles.RequestWorldExport(TerrainTileScope.CurrentTile, false);
        }
        if (_wantExportMapGlbTiles)
        {
            _wantExportMapGlbTiles = false;
            _modelWorldFiles.RequestWorldExport(_mapGlbScope, false);
        }
    }
}
