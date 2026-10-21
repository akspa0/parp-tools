using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Globalization;
using ImGuiNET;
using WoWViewer.Logging;
using WoWViewer.Terrain;
using WoWViewer.Workbench;
using MslkEntry = WowViewer.Core.PM4.Models.Pm4MslkEntry;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// Inspection exports and preservation of existing saved overlay state. Matching is offline.
internal sealed partial class Pm4WorkbenchService
{

    /// <summary>
    /// Single export command set, hosted by Viewer > PM4 > Selection.
    /// </summary>
    internal void DrawPm4ExportCommandSet()
    {
        if (_worldScene == null)
        {
            ImGui.TextDisabled("Load a world with PM4 data to export.");
            return;
        }

        if (ImGui.Button("Export Objects JSON..."))
            RunPm4Export(ExportPm4ObjectsJson);
        if (ImGui.Button("Export OBJ Set..."))
            RunPm4Export(ExportPm4ObjectsObjSet);
        if (ImGui.Button("Export LLM Evidence Bundle..."))
            RunPm4Export(ExportPm4LlmEvidenceBundle);
        if (ImGui.Button("Export Visible Overlay Report"))
            RunPm4Export(ExportPm4OverlayReport);
    }

    // Include picker/output-root setup failures in the same reporting contract as writer failures.
    private void RunPm4Export(Action export)
    {
        try
        {
            Directory.CreateDirectory(ExportDir);
            export();
        }
        catch (Exception ex)
        {
            _statusMessage = $"PM4 export failed: {ex.Message}";
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 Export] Export setup failed: {ex}");
        }
    }

    internal void ExportPm4ObjectsJson()
    {
        if (_worldScene == null)
            return;

        if (_pm4JsonExportRunning)
        {
            _statusMessage = "A PM4 JSON export is already running.";
            return;
        }

        string defaultName = $"pm4_objects_{DateTime.Now:yyyyMMdd_HHmmss}.json";
        ImGuiPathPicker.Instance.Open(
            "Save PM4 Objects JSON",
            ImGuiPathPickerMode.SaveFile,
            ExportDir,
            ".json",
            picked =>
            {
                if (string.IsNullOrWhiteSpace(picked))
                    return;

                // Spec 231 D6: the interchange JSON includes full geometry, which walks
                // every loaded PM4 file — run it off the render thread with a re-entrancy
                // guard so the viewer stays responsive during the dump.
                WorldScene scene = _worldScene;
                _pm4JsonExportRunning = true;
                _statusMessage = "Exporting PM4 objects JSON… (running in background; the viewer stays responsive)";
                _ = Task.Run(() =>
                {
                    try
                    {
                        string json = scene.Pm4Overlay.BuildPm4OverlayInterchangeJson(includeGeometry: true);
                        File.WriteAllText(picked, json, Encoding.UTF8);
                        _statusMessage = $"Exported PM4 objects JSON: {picked}";
                    }
                    catch (Exception ex)
                    {
                        _statusMessage = $"PM4 JSON export failed: {ex.Message}";
                        ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 Export] JSON export failed: {ex}");
                    }
                    finally
                    {
                        _pm4JsonExportRunning = false;
                    }
                });
            },
            defaultName);
    }

    /// <summary>True while a background PM4 OBJ export is in flight; guards re-entrancy.</summary>
    private bool _pm4ObjExportRunning;

    /// <summary>
    /// True while a background PM4 objects JSON dump is in flight.
    /// </summary>
    private bool _pm4JsonExportRunning;

    internal void ExportPm4ObjectsObjSet()
    {
        if (_worldScene == null)
            return;

        if (_pm4ObjExportRunning)
        {
            _statusMessage = "PM4 OBJ export is already running.";
            return;
        }

        Directory.CreateDirectory(ExportDir);
        ImGuiPathPicker.Instance.Open(
            "Choose a folder for PM4 OBJ export",
            pickFolder: true,
            initialPath: ExportDir,
            filterExtension: null,
            picked =>
            {
                if (string.IsNullOrWhiteSpace(picked))
                    return;

                // The export re-reads and re-decodes every PM4 file on the map and writes one
                // OBJ per object. Run it off the render thread — synchronously it stalled the
                // frame loop for the whole export and the window appeared frozen. The MPQ data
                // source is safe for concurrent reads: terrain streaming already reads it from
                // background workers (MaxConcurrentMpqReads semaphore in TerrainManager).
                WorldScene scene = _worldScene;
                _pm4ObjExportRunning = true;
                _statusMessage = "Exporting PM4 OBJ set… (running in background; the viewer stays responsive)";
                _ = Task.Run(() =>
                {
                    try
                    {
                        Pm4OfflineObjExportSummary summary = scene.Pm4Overlay.ExportPm4ObjectsAsObjDirectory(picked);
                        _statusMessage =
                            $"Exported PM4 OBJ set: {summary.ExportedObjectCount} objects across {summary.ExportedTileCount} tiles to {summary.OutputDirectory} (manifest: {summary.ManifestPath}).";
                        ViewerLog.Info(ViewerLog.Category.Terrain,
                            $"[PM4 Export] OBJ set finished: {summary.ExportedObjectCount} objects, {summary.ExportedTileCount} tiles -> {summary.OutputDirectory}");
                    }
                    catch (Exception ex)
                    {
                        _statusMessage = $"PM4 OBJ export failed: {ex.Message}";
                        ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 Export] OBJ export failed: {ex}");
                    }
                    finally
                    {
                        _pm4ObjExportRunning = false;
                    }
                });
            });
    }

    internal void InvalidatePm4DerivedReports()
    {
        _hoveredPm4ObjectMatch = null;
        _hoveredPm4ObjectMatchKey = null;
        _hoveredPm4ObjectMatchCacheMaxMatches = -1;
        _pm4OutlineCache = null;
    }

    private void FocusCameraOnBounds(Vector3 boundsMin, Vector3 boundsMax)
    {
        Vector3 center = (boundsMin + boundsMax) * 0.5f;
        Vector3 extent = Vector3.Max(boundsMax - boundsMin, new Vector3(1f, 1f, 1f));
        float distance = MathF.Max(extent.Length() * 1.35f, 80f);

        _camera.Position = center + new Vector3(distance, 0f, MathF.Max(extent.Z * 0.6f, 30f));
        _camera.Yaw = 180f;
        _camera.Pitch = -18f;
    }

    internal void ApplySavedPm4AlignmentToScene()
    {
        if (_worldScene == null)
            return;

        _worldScene.Pm4Overlay.Pm4OverlayTranslation = _pm4SavedOverlayTranslation;
        _worldScene.Pm4Overlay.Pm4OverlayRotationDegrees = _pm4SavedOverlayRotationDegrees;
        _worldScene.Pm4Overlay.Pm4OverlayScale = _pm4SavedOverlayScale;
        InvalidatePm4DerivedReports();
    }

}
