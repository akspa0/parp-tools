using System;
using System.Collections.Generic;
using System.Numerics;
using ImGuiNET;
using WoWViewer.Logging;
using WoWViewer.Terrain;
using WoWViewer.Terrain.Pm4;
using WoWViewer.Terrain.Pm4.Services;
using WoWViewer.UI;
using WowViewer.Core.Runtime.World;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;

namespace WoWViewer.Terrain.Pm4.UI;

/// <summary>
/// Modernized PM4 Sidebar Workbench UI service (Spec 288 Phase 4, AGENTS.md §11).
/// Uses SharedUiWidgets to provide an interactive color legend, surface class matrix,
/// and streamlined object inspection and magnetization tools.
/// </summary>
internal sealed class Pm4WorkbenchSidebarService
{
    public static readonly Pm4WorkbenchSidebarService Instance = new();

    private static readonly (byte Value, string Label, string Tooltip)[] SurfaceClassEntries =
    [
        (0x03, "0x03  Doodad surfaces (M2)", "184,356 surfaces. 100.0% carry NO placement height — authoritative M2 doodad marker."),
        (0x10, "0x10  Lowest floors (WMO)", "87,243 surfaces. Sits lowest inside its object (0.291 of Z extent). 95.3% Z-dominant."),
        (0x11, "0x11  Floors (WMO)", "13,493 surfaces. 0.402 relative height in object, 95.2% Z-dominant."),
        (0x12, "0x12  Floors (common)", "161,359 surfaces, most common placed class. 0.474 height, 91.8% Z-dominant."),
        (0x13, "0x13  Floors (upper)", "71,293 surfaces. 0.427 height in object, 93.3% Z-dominant."),
        (0x14, "0x14  Highest / angled", "105 surfaces. Highest in object (0.624) and non-horizontal surfaces."),
        (0x15, "0x15  Lowest (rare)", "243 surfaces. Lowest of all (0.161) and almost perfectly flat.")
    ];

    private Pm4WorkbenchSidebarService() { }

    /// <summary>
    /// Renders the Primary Overlay and Display Layers section.
    /// </summary>
    public void DrawOverlaySettings(
        Pm4OverlayScene overlay,
        string loosePm4InputPath,
        Action requestLooseFile,
        Action exportReport)
    {
        bool showPm4Overlay = overlay.ShowPm4Overlay;
        if (ImGui.Checkbox("PM4 Overlay", ref showPm4Overlay))
            overlay.ShowPm4Overlay = showPm4Overlay;

        ImGui.SameLine();
        if (SharedUiWidgets.ActionButton("Reload PM4"))
            overlay.ReloadPm4Overlay();

        ImGui.SameLine();
        if (SharedUiWidgets.ActionButton("Export Report"))
            exportReport();

        if (SharedUiWidgets.SectionHeader("Loose PM4 / PD4 Loader", defaultOpen: false, id: "Pm4LooseLoader"))
        {
            ImGui.TextDisabled("Load any .pm4 or .pd4 file directly into the overlay.");
            if (SharedUiWidgets.ActionButton("Open Loose PM4 / PD4 File..."))
                requestLooseFile();
            if (!string.IsNullOrWhiteSpace(loosePm4InputPath))
                ImGui.TextDisabled($"Last loaded: {System.IO.Path.GetFileName(loosePm4InputPath)}");
        }

        if (SharedUiWidgets.SectionHeader("Display & Geometry Layers", defaultOpen: true, id: "Pm4DisplayLayers"))
        {
            bool showPm4Solid = overlay.ShowPm4SolidOverlay;
            if (ImGui.Checkbox("PM4 Solid Fill", ref showPm4Solid))
                overlay.ShowPm4SolidOverlay = showPm4Solid;

            ImGui.SameLine();
            bool pm4IgnoreDepth = overlay.Pm4OverlayIgnoreDepth;
            if (ImGui.Checkbox("PM4 X-Ray", ref pm4IgnoreDepth))
                overlay.Pm4OverlayIgnoreDepth = pm4IgnoreDepth;

            bool showPm4Bounds = overlay.ShowPm4ObjectBounds;
            if (ImGui.Checkbox("PM4 Bounds", ref showPm4Bounds))
                overlay.ShowPm4ObjectBounds = showPm4Bounds;

            ImGui.SameLine();
            bool showPm4Ck24Bounds = overlay.ShowPm4Ck24Bounds;
            if (ImGui.Checkbox("PM4 CK24 Bounds", ref showPm4Ck24Bounds))
                overlay.ShowPm4Ck24Bounds = showPm4Ck24Bounds;

            bool showPathWalls = overlay.Pm4ShowPathWalls;
            if (ImGui.Checkbox("Show MSPV/MSPI walls", ref showPathWalls))
            {
                overlay.Pm4ShowPathWalls = showPathWalls;
                overlay.ReloadPm4Overlay();
            }
            if (ImGui.IsItemHovered())
            {
                ImGui.SetTooltip(
                    "MSLK path windows drawn as vertical wall quads.\n" +
                    "MSUR is the walkable floors; MSPV/MSPI are the vertical walls/steps between them.");
            }
            if (overlay.Pm4ShowPathWalls && overlay.Pm4LoadAttempted)
            {
                ImGui.SameLine();
                ImGui.TextDisabled($"({overlay.Pm4WallFaceCount} faces)");
            }

            bool showPm4Refs = overlay.ShowPm4PositionRefs;
            if (ImGui.Checkbox("PM4 MPRL Refs (terrain contacts)", ref showPm4Refs))
                overlay.ShowPm4PositionRefs = showPm4Refs;

            ImGui.SameLine();
            bool showPm4Centroids = overlay.ShowPm4ObjectCentroids;
            if (ImGui.Checkbox("PM4 Centroids", ref showPm4Centroids))
                overlay.ShowPm4ObjectCentroids = showPm4Centroids;

            bool gateTerrain = overlay.Pm4GateToLoadedTerrain;
            if (ImGui.Checkbox("Gate PM4 to loaded terrain tiles", ref gateTerrain))
                overlay.Pm4GateToLoadedTerrain = gateTerrain;
            if (ImGui.IsItemHovered())
            {
                ImGui.SetTooltip("When checked, only draws PM4 tiles whose terrain tile is resident.");
            }

            bool showRecovered = overlay.ShowPm4GeneratedPlacements;
            if (ImGui.Checkbox("Recovered placements (sidecar)", ref showRecovered))
                overlay.ShowPm4GeneratedPlacements = showRecovered;
            if (showRecovered)
            {
                ImGui.SameLine();
                bool terrainlessOnly = overlay.Pm4GeneratedPlacementsTerrainlessOnly;
                if (ImGui.Checkbox("Terrain-less tiles only", ref terrainlessOnly))
                    overlay.Pm4GeneratedPlacementsTerrainlessOnly = terrainlessOnly;
            }

            bool showPlacementZ = overlay.ShowPm4PlacementZPlane;
            if (ImGui.Checkbox("Placement Z markers (MSUR.0x1C)", ref showPlacementZ))
                overlay.ShowPm4PlacementZPlane = showPlacementZ;
            if (showPlacementZ)
            {
                ImGui.SameLine();
                bool allZ = overlay.ShowPm4PlacementZForAllObjects;
                if (ImGui.Checkbox("all objects##Pm4PlacementZAll", ref allZ))
                    overlay.ShowPm4PlacementZForAllObjects = allZ;
            }
        }

        if (SharedUiWidgets.SectionHeader("Nodes & Wireframe Styling", defaultOpen: false, id: "Pm4NodeStyling"))
        {
            bool showPm4Mscn = overlay.ShowPm4MscnNodes;
            if (ImGui.Checkbox("MSCN Nodes (cyan, boundary chain)", ref showPm4Mscn))
                overlay.ShowPm4MscnNodes = showPm4Mscn;

            bool showPm4Mspv = overlay.ShowPm4MspvNodes;
            if (ImGui.Checkbox("MSPV Nodes (magenta, MSLK wall vertices)", ref showPm4Mspv))
                overlay.ShowPm4MspvNodes = showPm4Mspv;

            bool renderNodesAsCubes = overlay.Pm4RenderNodesAsCubes;
            if (ImGui.Checkbox("Nodes as Solid Cubes", ref renderNodesAsCubes))
                overlay.Pm4RenderNodesAsCubes = renderNodesAsCubes;

            ImGui.SameLine();
            bool pm4FlipAllObjY = overlay.Pm4FlipAllObjectsY;
            if (ImGui.Checkbox("Mirror PM4 N/S", ref pm4FlipAllObjY))
                overlay.Pm4FlipAllObjectsY = pm4FlipAllObjY;

            float mscnSize = overlay.Pm4MscnCubeSize;
            ImGui.SetNextItemWidth(110f);
            if (ImGui.SliderFloat("MSCN size", ref mscnSize, 0.2f, 4f))
                overlay.Pm4MscnCubeSize = mscnSize;
            ImGui.SameLine();
            float mspvSize = overlay.Pm4MspvCubeSize;
            ImGui.SetNextItemWidth(110f);
            if (ImGui.SliderFloat("MSPV size", ref mspvSize, 0.2f, 4f))
                overlay.Pm4MspvCubeSize = mspvSize;

            float mscnAlpha = overlay.Pm4MscnCubeAlpha;
            ImGui.SetNextItemWidth(110f);
            if (ImGui.SliderFloat("MSCN alpha", ref mscnAlpha, 0.1f, 1f))
                overlay.Pm4MscnCubeAlpha = mscnAlpha;
            ImGui.SameLine();
            float mspvAlpha = overlay.Pm4MspvCubeAlpha;
            ImGui.SetNextItemWidth(110f);
            if (ImGui.SliderFloat("MSPV alpha", ref mspvAlpha, 0.1f, 1f))
                overlay.Pm4MspvCubeAlpha = mspvAlpha;

            float lineWidth = overlay.Pm4WireframeLineWidth;
            ImGui.SetNextItemWidth(140f);
            if (ImGui.SliderFloat("Wireframe width", ref lineWidth, 1f, 8f))
                overlay.Pm4WireframeLineWidth = lineWidth;

            float renderDistance = overlay.Pm4RenderDistance;
            ImGui.SetNextItemWidth(140f);
            if (ImGui.SliderFloat("PM4 draw distance", ref renderDistance, Pm4OverlayScene.Pm4MinRenderDistance, Pm4OverlayScene.Pm4MaxRenderDistance, "%.0f"))
                overlay.Pm4RenderDistance = renderDistance;
        }
    }

    /// <summary>
    /// Renders the Interactive Color Palette & Visual Legend (T016).
    /// </summary>
    public void DrawPaletteAndLegend(Pm4OverlayScene overlay, Func<Pm4OverlayColorMode, string> getModeLabel)
    {
        if (!SharedUiWidgets.SectionHeader("Color Palette & Visual Legend", defaultOpen: true, id: "Pm4PaletteSection"))
            return;

        Pm4OverlayColorMode colorMode = overlay.Pm4ColorMode;
        if (ImGui.BeginCombo("Color Palette", getModeLabel(colorMode)))
        {
            foreach (Pm4OverlayColorMode mode in Enum.GetValues<Pm4OverlayColorMode>())
            {
                bool isSelected = mode == colorMode;
                if (ImGui.Selectable(getModeLabel(mode), isSelected))
                    overlay.Pm4ColorMode = mode;
                if (isSelected)
                    ImGui.SetItemDefaultFocus();
            }
            ImGui.EndCombo();
        }

        Pm4ColorLegendInfo legend = overlay.GetPm4ColorLegend(24);
        if (!string.IsNullOrWhiteSpace(legend.Description))
            ImGui.TextDisabled(legend.Description);

        if (legend.Entries.Count == 0)
        {
            ImGui.TextDisabled("No visible PM4 objects matching the active palette.");
            return;
        }

        ImGui.Spacing();
        for (int i = 0; i < legend.Entries.Count; i++)
        {
            Pm4ColorLegendEntry entry = legend.Entries[i];

            ImGui.ColorButton(
                $"##Pm4LegendSwatch_{i}",
                new Vector4(entry.Color, 1f),
                ImGuiColorEditFlags.NoTooltip | ImGuiColorEditFlags.NoDragDrop,
                new Vector2(16f, 16f));

            ImGui.SameLine();
            if (entry.IsSelected)
            {
                ImGui.TextColored(new Vector4(1f, 1f, 0.35f, 1f), $"{entry.Label}  [selected]");
            }
            else if (legend.IsContinuous)
            {
                ImGui.TextUnformatted(entry.Label);
            }
            else
            {
                ImGui.TextUnformatted($"{entry.Label}");
                ImGui.SameLine();
                ImGui.TextDisabled($"[{entry.ObjectCount}]");
            }
        }

        if (legend.IsTruncated)
            ImGui.TextDisabled($"Showing {legend.Entries.Count} of {legend.TotalEntryCount} legend entries.");
    }

    /// <summary>
    /// Renders the Surface Class Filter Matrix (T017).
    /// </summary>
    public void DrawSurfaceClassFilterMatrix(Pm4OverlayScene overlay)
    {
        if (!SharedUiWidgets.SectionHeader("Surface Class Filter Matrix (MSUR.0x00)", defaultOpen: true, id: "Pm4ClassMatrixSection"))
            return;

        ImGui.TextDisabled("Quick Filters:");
        if (SharedUiWidgets.ActionButton("Show All##Pm4ShowAllClasses"))
        {
            for (int i = 0; i < SurfaceClassEntries.Length; i++)
                overlay.SetPm4SurfaceClassVisible(SurfaceClassEntries[i].Value, true);
        }
        ImGui.SameLine();
        if (SharedUiWidgets.ActionButton("Doodads Only (0x03)##Pm4DoodadOnlyClass"))
        {
            for (int i = 0; i < SurfaceClassEntries.Length; i++)
                overlay.SetPm4SurfaceClassVisible(SurfaceClassEntries[i].Value, SurfaceClassEntries[i].Value == 0x03);
        }
        ImGui.SameLine();
        if (SharedUiWidgets.ActionButton("Floors Only##Pm4FloorsOnlyClass"))
        {
            for (int i = 0; i < SurfaceClassEntries.Length; i++)
                overlay.SetPm4SurfaceClassVisible(SurfaceClassEntries[i].Value, SurfaceClassEntries[i].Value != 0x03);
        }

        ImGui.Spacing();
        for (int i = 0; i < SurfaceClassEntries.Length; i++)
        {
            var (cls, label, tooltip) = SurfaceClassEntries[i];
            bool isVisible = overlay.IsPm4SurfaceClassVisible(cls);
            if (ImGui.Checkbox($"{label}##Pm4Class_{cls:X2}", ref isVisible))
                overlay.SetPm4SurfaceClassVisible(cls, isVisible);

            if (ImGui.IsItemHovered())
            {
                ImGui.BeginTooltip();
                ImGui.TextUnformatted(tooltip);
                ImGui.TextUnformatted("Filters by object's dominant surface class.");
                ImGui.EndTooltip();
            }
        }

        bool splitCk24Conn = overlay.Pm4SplitCk24ByConnectivity;
        if (ImGui.Checkbox("Split CK24 by Connectivity", ref splitCk24Conn))
        {
            overlay.Pm4SplitCk24ByConnectivity = splitCk24Conn;
            overlay.ReloadPm4Overlay();
        }

        bool splitCk24Mscn = overlay.Pm4SplitCk24ByMscnRef;
        if (ImGui.Checkbox("Split CK24 by MSUR.0x18 (mslk_first_index)", ref splitCk24Mscn))
        {
            overlay.Pm4SplitCk24ByMscnRef = splitCk24Mscn;
            overlay.ReloadPm4Overlay();
        }
    }

    /// <summary>
    /// Renders the Live PM4 Alignment & Model Duplication section (T018).
    /// </summary>
    public void DrawAlignmentAndDuplication(
        Pm4OverlayScene overlay,
        IViewerAppHost host,
        WorldScene? worldScene)
    {
        if (!SharedUiWidgets.SectionHeader("Live PM4 Alignment & Magnetization", defaultOpen: true, id: "Pm4AlignmentWidget"))
            return;

        ImGui.TextWrapped(
            "Magnetize 2021 reconstructive ADT object placements to authoritative PM4 collision hulls on-the-fly, " +
            "using PM4 bounding hulls, centroids, and MSUR.0x1C placement Z as the ground truth template.");
        ImGui.Spacing();

        if (worldScene == null)
            return;

        int activeTileX = 0;
        int activeTileY = 0;
        bool hasTile = false;
        ObjectInstance? selectedObj = null;

        if (worldScene.Selection.SelectedInstance.HasValue)
        {
            selectedObj = worldScene.Selection.SelectedInstance.Value;
            if (selectedObj.Value.HasTileCoordinate)
            {
                activeTileX = selectedObj.Value.TileX;
                activeTileY = selectedObj.Value.TileY;
                hasTile = true;
            }
        }

        if (!hasTile && host.TerrainInspection.TryGetPinnedTerrainChunkInspectionTarget(out var pinnedChunk, out _))
        {
            activeTileX = pinnedChunk.TileX;
            activeTileY = pinnedChunk.TileY;
            hasTile = true;
        }

        if (selectedObj.HasValue)
        {
            var inst = selectedObj.Value;
            ImGui.TextColored(new Vector4(0.45f, 0.85f, 0.45f, 1f), $"Selected Object: {inst.ModelName} (#{inst.UniqueId})");
            if (SharedUiWidgets.ActionButton("Magnetize Selected Object to PM4 Hull"))
            {
                host.Pm4DoodadDuplication.MagnetizePlacementToPm4(inst);
            }
            ImGui.SameLine();
            if (SharedUiWidgets.ActionButton("Fill PM4 Doodad Hulls with Model"))
            {
                host.Pm4DoodadDuplication.DuplicateModelToUnmatchedPm4Doodads(inst, inst.TileX, inst.TileY);
            }
        }
        else
        {
            ImGui.TextDisabled("Select an M2 object in the scene to magnetize or duplicate it.");
        }

        if (hasTile)
        {
            if (SharedUiWidgets.ActionButton($"Magnetize All Placements on Tile ({activeTileX}, {activeTileY})"))
            {
                host.Pm4DoodadDuplication.MagnetizeTilePlacementsToPm4(activeTileX, activeTileY);
            }
        }

        if (host.Pm4DoodadDuplication.CanUndo)
        {
            ImGui.SameLine();
            if (SharedUiWidgets.ActionButton("Undo PM4 Operation"))
            {
                host.Pm4DoodadDuplication.UndoLastOperation();
            }
        }
    }
}

