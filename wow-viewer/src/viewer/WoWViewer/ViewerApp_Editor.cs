using System.Numerics;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.Editor;
using WowViewer.Core.Editor.Bridge;
using WowViewer.Core.Editor.Eras;
using WowViewer.Core.Editor.Logging;
using WowViewer.Core.Editor.Operations;
using WowViewer.Core.Editor.Plugins;
using WowViewer.Core.Editor.Session;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Reconciliation;
using WowViewer.Core.PM4.Services;
using WoWViewer.Logging;
using WoWViewer.Terrain;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;

using WowViewer.Core.IO.Terrain;

namespace WoWViewer;

/// <summary>
/// The Editor destination (Spec 166) and the PM4/Museum reconciliation surface (Spec 176 Phases 3–4).
/// This is a thin host shell: the plugin host, bridge, session, and reconciliation engine all live in
/// core. The viewer only adapts live scene state into the bridge snapshot and draws the plugin surface.
/// </summary>
public partial class ViewerApp
{
    private EditorHost? _editorHost;
    private EditorSession? _editorSession;
    private EditorSceneReaderAdapter? _editorSceneReader;
    private EditorLogAdapter? _editorLog;
    private string _reconciliationPm4Path = string.Empty;
    private string _reconciliationMuseumPath = string.Empty;

    private Workbench.Pages.EditorWorkbenchPages? _editorPages;

    /// <summary>
    /// Spec 228 pattern: the shell keeps one aggregate field for the Spec 231
    /// Editor workbench pages and delegates; the pages receive a narrow
    /// <see cref="Workbench.Pages.ViewerAppContext"/> and never reach back into
    /// god-class internals beyond the explicit draw contract.
    /// </summary>
    private Workbench.Pages.EditorWorkbenchPages EnsureEditorPages()
    {
        if (_editorPages == null)
            _editorPages = new Workbench.Pages.EditorWorkbenchPages(new Workbench.Pages.ViewerAppContext(this));
        return _editorPages;
    }

    void Workbench.Pages.IEditorPageHost.DrawTasksAndWorkspace() => _editorPanels.DrawArchaeologyEditorTasksSubTab();
    void Workbench.Pages.IEditorPageHost.DrawObjectLibrary() => _editorPanels.DrawRosettaObjectLibrarySubTab();
    void Workbench.Pages.IEditorPageHost.DrawPopulation() => _sqlSpawnStreaming.DrawPopulationSubTabContent();
    void Workbench.Pages.IEditorPageHost.DrawTerrainLab() => _terrainControlsPanel.DrawTerrainLabSubTab();
    void Workbench.Pages.IEditorPageHost.DrawImportsAndExports() => _editorPanels.DrawArchaeologyEditorImportsSubTab();
    void Workbench.Pages.IEditorPageHost.DrawConverters() => _workbenchPanels.DrawConvertersSubTabContent();
    void Workbench.Pages.IEditorPageHost.DrawPm4Exports() => _pm4Workbench.DrawPm4ExportCommandSet();
    void Workbench.Pages.IEditorPageHost.DrawPm4Workbench() => _pm4Workbench.DrawPm4WorkbenchInspector();
    void Workbench.Pages.IEditorPageHost.DrawNewMapCreator() => _newMapCreatorService.Draw(_editorPanels.LoadGeneratedNewMap);

    private readonly Workbench.Services.NewMapCreatorService _newMapCreatorService = new();

    private void EnsureEditorHost()
    {
        if (_editorHost != null)
            return;

        _editorLog = new EditorLogAdapter();
        EditorBuildVersion build = EditorBuildVersion.TryParse(_dbcBuild, out EditorBuildVersion parsed)
            ? parsed
            : EditorBuildVersion.Parse("0.0.0");

        _editorHost = new EditorHost(build, _editorLog);
        _editorHost.Register(new ReferenceEditorPlugin());
        _editorHost.Register(new TerrainTemplateEditorPlugin());
        _editorHost.Register(new ChunkManipulatorEditorPlugin());

        _editorSceneReader = new EditorSceneReaderAdapter(this);
        _editorSession = new EditorSession(new EditorApplierAdapter(this), _editorLog, _editorProjectOutputDir);
    }

    /// <summary>Adapts the live scene into the editor bridge snapshot.</summary>
    private sealed class EditorSceneReaderAdapter : IEditorSceneReader
    {
        private readonly ViewerApp _app;

        public EditorSceneReaderAdapter(ViewerApp app) => _app = app;

        public EditorSceneSnapshot Capture()
        {
            var selection = new List<EditorSelectionEntry>();
            if (_app._worldScene?.SelectedInstance is ObjectInstance selected)
            {
                selection.Add(new EditorSelectionEntry(
                    _app._worldScene.SelectedObjectType == Terrain.ObjectType.Wmo ? EditorSelectionKind.WorldModel : EditorSelectionKind.Model,
                    _app._captureAutomation.GetCurrentCaptureMapName(),
                    selected.TileX,
                    selected.TileY,
                    selected.PlacementEntryIndex,
                    selected.UniqueId,
                    selected.ModelPath,
                    selected.PlacementPosition));
            }

            return new EditorSceneSnapshot(
                _app._captureAutomation.GetCurrentCaptureMapName(),
                new EditorCamera(_app._camera.Position, _app._camera.Forward, Vector3.UnitZ),
                [],
                selection);
        }
    }

    /// <summary>Applies an editor operation to the live scene and source file.</summary>
    private sealed class EditorApplierAdapter : IEditorOperationApplier
    {
        private readonly ViewerApp _app;

        public EditorApplierAdapter(ViewerApp app) => _app = app;

        public void Apply(EditorOperation operation)
        {
            switch (operation)
            {
                case ReconciliationApplyOperation reconciliation:
                    _app._archaeologyPanel.MaterializeReconciliationOutput(reconciliation);
                    break;
                case PlacementMoveOperation move:
                    _app._worldScene?.TryUpdateSelectedPlacementPosition(move.NewPosition, out _);
                    break;
                default:
                    break;
            }
        }
    }

    /// <summary>Routes editor log lines to the viewer's existing log surface.</summary>
    private sealed class EditorLogAdapter : IEditorLog
    {
        public void Trace(string message) => ViewerLog.Trace($"[Editor] {message}");
        public void Info(string message) => ViewerLog.Info(ViewerLog.Category.General, $"[Editor] {message}");
        public void Warn(string message) => ViewerLog.Important(ViewerLog.Category.General, $"[Editor] {message}");
        public void Error(string message, Exception? exception = null)
            => ViewerLog.Error(ViewerLog.Category.General, $"[Editor] {message}{(exception is null ? "" : $" | {exception.Message}")}");
    }
}
