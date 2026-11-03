using System;
using System.IO;
using System.Numerics;
using ImGuiNET;
using WoWViewer.Logging;
using WoWViewer.Terrain;
using WoWViewer.UI;

namespace WoWViewer;

/// <summary>
/// Authoritative consolidated Export & Report Workbench (Spec 290, AGENTS.md §10/§11).
/// Unifies PM4 collision meshes, world models, terrain geometry, and ML datasets into one clean right-dock panel.
/// </summary>
internal sealed class ExportWorkbenchService
{
    private readonly IViewerAppHost _host;
    private string _exportStatusMessage = "Ready.";

    internal ExportWorkbenchService(IViewerAppHost host)
    {
        _host = host ?? throw new ArgumentNullException(nameof(host));
    }

    /// <summary>
    /// Renders the unified Export & Report Workbench content inside the right dock.
    /// </summary>
    public void DrawExportPanelContent()
    {
        ImGui.TextUnformatted("Export & Report Workbench");
        ImGui.TextDisabled("Authoritative consolidated hub for PM4 models, terrain, world placements, and dataset exports.");
        ImGui.Separator();

        // 1. PM4 Navigation & Collision Exports
        if (SharedUiWidgets.SectionHeader("PM4 Navigation & Collision", defaultOpen: true, id: "ExportPm4Section"))
        {
            DrawPm4ExportSection();
        }

        // 2. World & Terrain Geometry
        if (SharedUiWidgets.SectionHeader("World & Terrain Geometry", defaultOpen: false, id: "ExportWorldSection"))
        {
            DrawWorldExportSection();
        }

        // 3. Model & Asset Mesh Exports
        if (SharedUiWidgets.SectionHeader("Model & Asset Meshes", defaultOpen: false, id: "ExportModelSection"))
        {
            DrawModelExportSection();
        }

        // 4. Archaeology & ML Datasets
        if (SharedUiWidgets.SectionHeader("Archaeology & ML Datasets", defaultOpen: false, id: "ExportDatasetsSection"))
        {
            DrawDatasetExportSection();
        }

        ImGui.Spacing();
        ImGui.Separator();
        ImGui.TextDisabled($"Status: {_exportStatusMessage}");
    }

    private void DrawPm4ExportSection()
    {
        var worldScene = _host.WorldScene;
        if (worldScene == null || !worldScene.Pm4Overlay.ShowPm4Overlay)
        {
            ImGui.TextDisabled("Load a world with PM4 overlay enabled to export navigation data.");
            return;
        }

        var overlay = worldScene.Pm4Overlay;
        int loadedFiles = overlay.Pm4LoadedFiles;
        int loadedObjects = overlay.Pm4ObjectCount;

        ImGui.TextDisabled($"Loaded PM4 data: {loadedFiles} files, {loadedObjects} objects across memory.");

        if (SharedUiWidgets.ActionButton("Export Objects JSON...##Pm4Json"))
        {
            _host.Pm4Workbench.RunPm4Export(_host.Pm4Workbench.ExportPm4ObjectsJson);
            _exportStatusMessage = "Started PM4 objects JSON export.";
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Export all loaded PM4 objects across tiles into an authoritative JSON structure.");

        if (SharedUiWidgets.ActionButton("Export OBJ Set...##Pm4Obj"))
        {
            _host.Pm4Workbench.RunPm4Export(_host.Pm4Workbench.ExportPm4ObjectsObjSet);
            _exportStatusMessage = "Started PM4 OBJ set export.";
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Export complete tile OBJ models and individual discrete object OBJ files with manifest.");

        if (SharedUiWidgets.ActionButton("Export LLM Evidence Bundle...##Pm4Llm"))
        {
            _host.Pm4Workbench.RunPm4Export(_host.Pm4Workbench.ExportPm4LlmEvidenceBundle);
            _exportStatusMessage = "Started PM4 LLM evidence bundle export.";
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Export research evidence bundle containing JSON, Markdown analysis, and topological SVG maps.");

        if (SharedUiWidgets.ActionButton("Export Visible Overlay Report##Pm4Report"))
        {
            _host.Pm4Workbench.RunPm4Export(_host.Pm4Workbench.ExportPm4OverlayReport);
            _exportStatusMessage = "Exported visible PM4 overlay report to output folder.";
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Write a human-readable diagnostic report of all visible PM4 surfaces, classes, and heights.");

        if (SharedUiWidgets.ActionButton("Save Merged Placements to Zarr v60##Pm4Zarr"))
        {
            _host.Pm4Workbench.SaveMergedPlacementsToZarr();
            _exportStatusMessage = "Saved merged placement archive to Zarr v60.";
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Archive resident ADT and magnetized PM4 placements into an authentic Zarr v60 store.");

        if (SharedUiWidgets.ActionButton("Patch & Export Monolithic ADTs##Pm4ExportAdt"))
        {
            var res = _host.Pm4DoodadDuplication.SavePatchedAdts();
            _exportStatusMessage = $"Saved {res.tilesWritten} patched ADTs to {res.outDir} ({res.placementsAdded} placements).";
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Patch modified ADTs with curated placements (preserving terrain/textures via AdtPlacementEditor, or creating monolithic 3.3.5 ADTs for unauthored tiles).");

        string activeMap = !string.IsNullOrEmpty(_host.LoadedFileName)
            ? Path.GetFileNameWithoutExtension(_host.LoadedFileName)
            : "development";

        if (SharedUiWidgets.ActionButton("Save Taught Pairings Project##ExportSavePairings"))
        {
            string path = _host.Pm4DoodadDuplication.SaveTaughtProject(activeMap);
            _exportStatusMessage = $"Saved {_host.Pm4DoodadDuplication.TaughtPairings.Count} taught pairings to {path}";
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip($"Saves the current taught pairings to 'output/projects/pm4_pairings/{activeMap}.pairings.json' and syncs the global curation library.");

        if (SharedUiWidgets.ActionButton("Export Taught Pairings Manifest##ExportPairingsManifest"))
        {
            string path = _host.Pm4DoodadDuplication.ExportTaughtPairingsManifest(activeMap);
            _exportStatusMessage = $"Exported taught pairing manifest to {path}";
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Exports an authoritative JSON composition manifest of all taught shape rules and footprint statistics.");

        // If an object is actively selected in the scene, provide quick single-object details and copy
        if (overlay.SelectedPm4ObjectKey.HasValue)
        {
            var selKey = overlay.SelectedPm4ObjectKey.Value;
            ImGui.Spacing();
            ImGui.TextUnformatted($"Selected PM4 Object: Tile ({selKey.tileX}, {selKey.tileY}) Ck24 0x{selKey.ck24:X6} Part {selKey.objectPart}");

            if (overlay.TryGetSelectedPm4ObjectDebugInfo(out var debugInfo))
            {
                ImGui.TextDisabled($"Part {debugInfo.ObjectPartId}: {debugInfo.SurfaceCount} surfaces, dominant class 0x{debugInfo.DominantGroupKey:X2}");

                if (SharedUiWidgets.ActionButton("Copy Selected JSON##Pm4CopyJson"))
                {
                    string json = System.Text.Json.JsonSerializer.Serialize(new
                    {
                        tile_x = selKey.tileX,
                        tile_y = selKey.tileY,
                        ck24 = $"0x{selKey.ck24:X6}",
                        part_id = selKey.objectPart,
                        surface_count = debugInfo.SurfaceCount,
                        dominant_class = $"0x{debugInfo.DominantGroupKey:X2}",
                        dominant_mscn_ref = debugInfo.DominantMscnRefIndex,
                        bounds_min = new[] { debugInfo.BoundsMin.X, debugInfo.BoundsMin.Y, debugInfo.BoundsMin.Z },
                        bounds_max = new[] { debugInfo.BoundsMax.X, debugInfo.BoundsMax.Y, debugInfo.BoundsMax.Z }
                    }, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
                    ImGui.SetClipboardText(json);
                    _exportStatusMessage = "Copied selected PM4 object JSON to clipboard.";
                }
            }
        }
    }

    private void DrawWorldExportSection()
    {
        _host.MainMenuBar.DrawWorldExportContent();
        _host.MainMenuBar.DrawTerrainFileOperationsContent();
    }

    private void DrawModelExportSection()
    {
        _host.MainMenuBar.DrawModelExportContent();
    }

    private void DrawDatasetExportSection()
    {
        if (SharedUiWidgets.ActionButton("Synthesized Minimap Export Dialog...##DsMinimap"))
        {
            _host.DrawSynthesizedMinimapExportContent(showCloseButton: false);
            _exportStatusMessage = "Opened Synthesized Minimap export dialog.";
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Export synthetic minimap residual models and hillshades across maps.");

        if (SharedUiWidgets.ActionButton("Open VLM / ML Dataset Exporter...##DsVlm"))
        {
            _host.DatasetExportDialogs.DrawVlmExportDialog();
            _exportStatusMessage = "Opened VLM dataset export dialog.";
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Export paired training datasets for terrain ML and visual elevation refinement.");
    }
}
