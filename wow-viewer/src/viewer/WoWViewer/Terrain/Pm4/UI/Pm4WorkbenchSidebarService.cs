using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using ImGuiNET;
using WoWViewer;
using WoWViewer.Logging;
using WoWViewer.Terrain;
using WoWViewer.Terrain.Pm4;
using WoWViewer.Terrain.Pm4.Services;
using WoWViewer.UI;
using WoWViewer.Workbench.Services.Pm4Workbench;
using WoWViewer.Workbench.Services.WorkbenchPanels;
using WowViewer.Core.PM4.Footprint;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.Maps;
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
    private readonly ManualCurationWorkbenchService _manualCuration = new();
    private readonly Pm4ShapeMatchWorkbenchService _shapeMatchWorkbench = new();
    private readonly Pm4ShapeLibraryPanelService _shapeLibraryPanel = new();
    private readonly Pm4ShapeMatchReviewPanelService _shapeReviewPanel = new();
    private readonly Pm4ShapeMatchBatchPanelService _shapeBatchPanel = new();

    public Pm4ShapeMatchWorkbenchService ShapeMatchWorkbench => _shapeMatchWorkbench;

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
        Action exportReport,
        Action? openExports = null)
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
        if (openExports != null)
        {
            ImGui.SameLine();
            if (SharedUiWidgets.ActionButton("Open Exports..."))
                openExports();
        }

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
            if (ImGui.Checkbox("PM4 Object Z Bounds", ref showPm4Ck24Bounds))
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

            bool showExtrudedWalls = overlay.ShowPm4ExtrudedWalls;
            if (ImGui.Checkbox("PM4 Extruded 3D Footprint Walls", ref showExtrudedWalls))
                overlay.ShowPm4ExtrudedWalls = showExtrudedWalls;
            if (ImGui.IsItemHovered())
            {
                ImGui.SetTooltip(
                    "Reconstructs synthetic vertical walls by extruding the 2D perimeter of PM4 top surfaces\n" +
                    "down to MPRL terrain contact elevations or MSUR.0x1C Placement-Z.");
            }

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
            WoWViewer.UI.SharedUiWidgets.SetNextItemWidth(110f);
            if (ImGui.SliderFloat("MSCN size", ref mscnSize, 0.2f, 4f))
                overlay.Pm4MscnCubeSize = mscnSize;
            ImGui.SameLine();
            float mspvSize = overlay.Pm4MspvCubeSize;
            WoWViewer.UI.SharedUiWidgets.SetNextItemWidth(110f);
            if (ImGui.SliderFloat("MSPV size", ref mspvSize, 0.2f, 4f))
                overlay.Pm4MspvCubeSize = mspvSize;

            float mscnAlpha = overlay.Pm4MscnCubeAlpha;
            WoWViewer.UI.SharedUiWidgets.SetNextItemWidth(110f);
            if (ImGui.SliderFloat("MSCN alpha", ref mscnAlpha, 0.1f, 1f))
                overlay.Pm4MscnCubeAlpha = mscnAlpha;
            ImGui.SameLine();
            float mspvAlpha = overlay.Pm4MspvCubeAlpha;
            WoWViewer.UI.SharedUiWidgets.SetNextItemWidth(110f);
            if (ImGui.SliderFloat("MSPV alpha", ref mspvAlpha, 0.1f, 1f))
                overlay.Pm4MspvCubeAlpha = mspvAlpha;

            float lineWidth = overlay.Pm4WireframeLineWidth;
            WoWViewer.UI.SharedUiWidgets.SetNextItemWidth(140f);
            if (ImGui.SliderFloat("Wireframe width", ref lineWidth, 1f, 8f))
                overlay.Pm4WireframeLineWidth = lineWidth;

            float renderDistance = overlay.Pm4RenderDistance;
            WoWViewer.UI.SharedUiWidgets.SetNextItemWidth(140f);
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
        if (ImGui.Checkbox("Split Object Z by Connectivity", ref splitCk24Conn))
        {
            overlay.Pm4SplitCk24ByConnectivity = splitCk24Conn;
            overlay.ReloadPm4Overlay();
        }

        bool splitCk24Mscn = overlay.Pm4SplitCk24ByMscnRef;
        if (ImGui.Checkbox("Split Object Z by MSUR.0x18 (mslk_first_index)", ref splitCk24Mscn))
        {
            overlay.Pm4SplitCk24ByMscnRef = splitCk24Mscn;
            overlay.ReloadPm4Overlay();
        }
    }

    private List<Pm4DirectMatchResult> _activeMatches = new();
    private uint _activeMatchCk24;
    private (int tileX, int tileY, uint ck24, int objectPart)? _cachedTaughtMatchKey;
    private int _cachedTaughtMatchRuleCount = -1;
    private TaughtPairing? _cachedTaughtMatch;
    private float _cachedTaughtScale;
    private Vector3 _cachedTaughtPos;
    private Vector3 _cachedTaughtRot;

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
            "Magnetize reconstructive ADT object placements to authoritative PM4 collision hulls, " +
            "pair models to hulls via Key+Click human teaching, and automate footprint placement.");
        ImGui.Spacing();

        if (worldScene == null)
            return;

        int totalLoadedTiles = overlay._pm4TileObjects.Count;
        int taughtCount = host.Pm4DoodadDuplication.TaughtPairings.Count;
        int totalMdx = 0;
        int totalWmo = 0;
        IPm4OverlayHost pm4Host = worldScene;
        foreach (var l in pm4Host.TileMdxInstances.Values) totalMdx += l.Count;
        foreach (var l in pm4Host.TileWmoInstances.Values) totalWmo += l.Count;
        int totalPlacements = totalMdx + totalWmo;

        if (worldScene != null)
        {
            host.Pm4MapWidePropagation.FlushPendingScenePlacements(worldScene);
        }

        ImGui.TextDisabled($"Loaded Tiles: {totalLoadedTiles}  |  Placed Models: {totalPlacements}  |  Taught Rules: {taughtCount}");
        ImGui.Spacing();

        var prop = host.Pm4MapWidePropagation;
        if (prop.IsRunning)
        {
            ImGui.Spacing();
            ImGui.TextColored(UiTokens.Brass, $"Map Propagation Active: {prop.Phase}");
            ImGui.ProgressBar(prop.ProgressFraction, new Vector2(-1f, 18f), $"{prop.ProcessedTiles}/{prop.TotalTiles} tiles ({prop.ProgressFraction:P0})");
            ImGui.TextDisabled($"Placements Added: {prop.PlacementsAdded} | Current: {prop.CurrentTileName}");
            if (SharedUiWidgets.ActionButton("Cancel Propagation##pm4_cancel_prop"))
            {
                prop.Cancel();
            }
            ImGui.Spacing();
        }
        else
        {
            if (SharedUiWidgets.ActionButton("✦ Harvest Pairings from Placed Models##pm4_auto_harvest"))
            {
                host.Pm4DoodadDuplication.HarvestPairingsFromPlacedObjects(host);
            }
            if (ImGui.IsItemHovered())
            {
                ImGui.SetTooltip("Scans all placed models across loaded/donor tiles, correlates them with matching PM4 navmesh hulls, and auto-generates verified taught pairing rules.");
            }

            if (taughtCount > 0)
            {
                ImGui.SameLine();
                string buttonText = prop.TotalTiles > 0
                    ? $"Apply All Rules Across Map ({prop.TotalTiles} tiles)##pm4_propagate_all"
                    : $"Apply All Rules Across Map##pm4_propagate_all";
                if (SharedUiWidgets.ActionButton(buttonText))
                {
                    prop.StartPropagationAndExport();
                }
                if (ImGui.IsItemHovered())
                {
                    ImGui.SetTooltip("Populates matching unoccupied PM4 hulls across all map tiles (including unopened ones) using taught rules, fills missing data from donor layers, and exports patched LK ADTs.");
                }
            }

            if (!string.IsNullOrWhiteSpace(prop.LastSummaryMessage))
            {
                ImGui.Spacing();
                ImGui.TextColored(UiTokens.Success, prop.LastSummaryMessage);
            }
        }
        ImGui.Spacing();
        ImGui.Separator();
        ImGui.Spacing();

        string mapName = ManualCurationWorkbenchService.NormalizeMapName(host.LoadedFileName);
        _manualCuration.ApplyProjectToScene(mapName, worldScene);

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

        (int tileX, int tileY, uint ck24, int objectPart)? selectedPm4Key = overlay.SelectedPm4ObjectKey;
        Pm4OverlayObject? selectedPm4Obj = null;
        int pm4TileX = 0;
        int pm4TileY = 0;

        if (selectedPm4Key.HasValue)
        {
            pm4TileX = selectedPm4Key.Value.tileX;
            pm4TileY = selectedPm4Key.Value.tileY;
            if (!hasTile)
            {
                activeTileX = pm4TileX;
                activeTileY = pm4TileY;
                hasTile = true;
            }

            if (!overlay._pm4ObjectLookup.TryGetValue(selectedPm4Key.Value, out selectedPm4Obj))
            {
                if (overlay._pm4TileObjects.TryGetValue((pm4TileX, pm4TileY), out var list))
                {
                    selectedPm4Obj = list.FirstOrDefault(o => o.Ck24 == selectedPm4Key.Value.ck24 && o.ObjectPartId == selectedPm4Key.Value.objectPart)
                                  ?? list.FirstOrDefault(o => o.Ck24 == selectedPm4Key.Value.ck24);
                }
            }
        }

        if (!hasTile && host.TerrainInspection.TryGetPinnedTerrainChunkInspectionTarget(out var pinnedChunk, out _))
        {
            activeTileX = pinnedChunk.TileX;
            activeTileY = pinnedChunk.TileY;
            hasTile = true;
        }

        // Automatic cross-tile PM4 shape identification:
        // Whenever a PM4 hull is selected, instantly find similar/same shapes across all loaded tiles.
        IReadOnlyList<SimilarPm4HullInfo>? similarHulls = null;
        int distinctTiles = 0;
        int unoccupiedHulls = 0;
        if (selectedPm4Obj != null)
        {
            similarHulls = host.Pm4DoodadDuplication.Curation.FindSimilarHullsAcrossTiles(
                selectedPm4Obj, pm4TileX, pm4TileY, worldScene);
            distinctTiles = similarHulls.Select(h => (h.TileX, h.TileY)).Distinct().Count();
            unoccupiedHulls = similarHulls.Count(h => !h.IsOccupied);
        }

        // Branch 1: Both PM4 Hull AND Scene Model are selected (Human-in-the-loop pairing & teaching)
        if (selectedObj.HasValue && selectedPm4Obj != null)
        {
            var inst = selectedObj.Value;
            Vector3 pm4Centroid = Vector3.Transform(selectedPm4Obj.Center, selectedPm4Obj.BaseTransform);
            float dist = Vector3.Distance(inst.PlacementPosition, pm4Centroid);

            // Log / ensure curated relationship
            host.Pm4DoodadDuplication.Curation.TryDetectSpatialRelationship(
                selectedPm4Obj, pm4TileX, pm4TileY, worldScene, out var curatedRel, inst);

            ImGui.TextColored(UiTokens.Brass, "✦ Active Human Pairing & Curation:");
            ImGui.BulletText($"Model: {inst.ModelName} (#{inst.UniqueId})");
            float hullZ = selectedPm4Obj.Ck24 != 0 ? BitConverter.UInt32BitsToSingle(selectedPm4Obj.Ck24 << 8) : 0f;
            ImGui.BulletText($"PM4 Hull: [{pm4TileX}_{pm4TileY}] Obj #{selectedPm4Obj.ObjectPartId} (Z={hullZ:F1}yd, Lnk={selectedPm4Obj.LinkGroupObjectId}) — {dist:F1} yds away");
            if (similarHulls != null)
            {
                ImGui.BulletText($"Cross-Tile Identified: {similarHulls.Count} matching 3D shapes across {distinctTiles} loaded tiles ({unoccupiedHulls} unoccupied)");
            }
            ImGui.Spacing();

            if (SharedUiWidgets.ActionButton("Align Model to this PM4 Hull (Centroid, Height & Yaw)"))
            {
                host.Pm4DoodadDuplication.AlignPlacementToPm4Hull(inst, selectedPm4Obj, pm4TileX, pm4TileY, fitScale: false, out _);
            }

            ImGui.TextDisabled($"Yaw: {inst.PlacementRotation.Z:F1}°  |  Z: {inst.PlacementPosition.Z:F3}");
            ImGui.SameLine();
            if (SharedUiWidgets.ActionButton("Rotate +90°##pair_rot_p90"))
            {
                host.Pm4DoodadDuplication.RotatePlacement(inst, 90f);
            }
            ImGui.SameLine();
            if (SharedUiWidgets.ActionButton("Rotate -90°##pair_rot_m90"))
            {
                host.Pm4DoodadDuplication.RotatePlacement(inst, -90f);
            }
            ImGui.SameLine();
            if (SharedUiWidgets.ActionButton("Flip 180°##pair_rot_180"))
            {
                host.Pm4DoodadDuplication.RotatePlacement(inst, 180f);
            }
            ImGui.TextDisabled("Nudge Z:");
            ImGui.SameLine();
            if (SharedUiWidgets.ActionButton("-1 yd##nudge_zm1")) host.Pm4DoodadDuplication.NudgePlacementZ(inst, -1.0f);
            ImGui.SameLine();
            if (SharedUiWidgets.ActionButton("-0.1##nudge_zm01")) host.Pm4DoodadDuplication.NudgePlacementZ(inst, -0.1f);
            ImGui.SameLine();
            if (SharedUiWidgets.ActionButton("+0.1##nudge_zp01")) host.Pm4DoodadDuplication.NudgePlacementZ(inst, 0.1f);
            ImGui.SameLine();
            if (SharedUiWidgets.ActionButton("+1 yd##nudge_zp1")) host.Pm4DoodadDuplication.NudgePlacementZ(inst, 1.0f);

            // The zero-friction primary automation action:
            string verifyBtnText = $"✓ Verify & Propagate 3D Shape Across All Loaded Tiles ({unoccupiedHulls} Hulls)##verify_propagate_paired";
            if (curatedRel != null && SharedUiWidgets.ActionButton(verifyBtnText, unoccupiedHulls > 0))
            {
                host.Pm4DoodadDuplication.Curation.VerifyAndPropagateAcrossAllTiles(curatedRel, worldScene!, out _);
            }

            if (SharedUiWidgets.ActionButton("Teach & Save Pairing Rule Only"))
            {
                host.Pm4DoodadDuplication.TeachPairing(selectedPm4Obj, inst, pm4TileX, pm4TileY);
            }
            ImGui.SameLine();
            if (SharedUiWidgets.ActionButton("Export PM4 Hull to GLB##pair_export_glb"))
            {
                host.Pm4Workbench.ExportSelectedPm4ObjectGlb();
            }

            _manualCuration.DrawPlacementEditor(mapName, host, worldScene, inst);
            _manualCuration.DrawAssociationEditor(
                mapName,
                inst,
                new Pm4HullIdentity(pm4TileX, pm4TileY, selectedPm4Obj.Ck24,
                    selectedPm4Obj.ObjectPartId, selectedPm4Obj.LinkGroupObjectId),
                host.CatalogView?.SelectedEntry,
                pm4Centroid);

            if (SharedUiWidgets.ActionButton("Clear Selection (Next Pair)##clear_pair_selection"))
            {
                worldScene.Selection.ClearSelection();
                worldScene.Pm4Overlay.ClearPm4ObjectSelection();
            }
        }
        // Branch 2: Only PM4 Hull is selected
        else if (selectedPm4Obj != null)
        {
            Vector3 pm4Centroid = Vector3.Transform(selectedPm4Obj.Center, selectedPm4Obj.BaseTransform);
            float hullOnlyZ = selectedPm4Obj.Ck24 != 0 ? BitConverter.UInt32BitsToSingle(selectedPm4Obj.Ck24 << 8) : 0f;
            ImGui.TextColored(UiTokens.Brass, $"Selected PM4 Hull: [{pm4TileX}_{pm4TileY}] Obj #{selectedPm4Obj.ObjectPartId} (Z={hullOnlyZ:F1}yd, Lnk={selectedPm4Obj.LinkGroupObjectId})");
            ImGui.TextDisabled($"Centroid: ({pm4Centroid.X:F1}, {pm4Centroid.Y:F1}, {pm4Centroid.Z:F1}) | Group: 0x{selectedPm4Obj.DominantGroupKey:X2} | Tris: {selectedPm4Obj.Triangles.Count}");
            if (similarHulls != null && similarHulls.Count > 0)
            {
                ImGui.TextColored(UiTokens.Success, $"Cross-Tile Identified: {similarHulls.Count} matching 3D shapes across {distinctTiles} loaded tiles ({unoccupiedHulls} unoccupied)");
            }
            ImGui.Spacing();

            if (SharedUiWidgets.ActionButton("Export Selected PM4 to GLB##pm4_export_sel_glb"))
            {
                host.Pm4Workbench.ExportSelectedPm4ObjectGlb();
            }
            ImGui.SameLine();
            if (SharedUiWidgets.ActionButton("Export Tile PM4 to GLB##pm4_export_tile_glb"))
            {
                host.Pm4Workbench.ExportActiveTilePm4Glb();
            }
            ImGui.Spacing();

            // Automatic spatial relationship detection with resident ADT models
            bool hasSpatialRel = host.Pm4DoodadDuplication.Curation.TryDetectSpatialRelationship(
                selectedPm4Obj, pm4TileX, pm4TileY, worldScene, out var spatialRel);

            if (hasSpatialRel && spatialRel != null)
            {
                ImGui.TextColored(UiTokens.Brass, spatialRel.Verified ? "✓ Verified Spatial Relationship:" : "✦ Spatial Relationship Detected & Logged:");
                ImGui.BulletText($"Model: {spatialRel.ModelName} ({spatialRel.AssetKind})");
                ImGui.BulletText($"Fit: {spatialRel.FitConfidence:P0} (delta: {spatialRel.PositionOffset.Length():F1} yds, yaw: {spatialRel.YawOffset:F1}°)");

                string propBtnText = spatialRel.Verified
                    ? $"Propagate 3D Shape Across All Loaded Tiles ({unoccupiedHulls} Hulls)##propagate_all_rel"
                    : $"✓ Verify & Propagate 3D Shape Across All Tiles ({unoccupiedHulls} Hulls)##verify_propagate_rel";

                if (SharedUiWidgets.ActionButton(propBtnText, unoccupiedHulls > 0))
                {
                    host.Pm4DoodadDuplication.Curation.VerifyAndPropagateAcrossAllTiles(spatialRel, worldScene!, out _);
                }
                ImGui.Spacing();
            }

            var currKey = (pm4TileX, pm4TileY, selectedPm4Obj.Ck24, selectedPm4Obj.ObjectPartId);
            int ruleCount = host.Pm4DoodadDuplication.TaughtPairings.Count;
            if (_cachedTaughtMatchKey != currKey || _cachedTaughtMatchRuleCount != ruleCount)
            {
                _cachedTaughtMatchKey = currKey;
                _cachedTaughtMatchRuleCount = ruleCount;
                _cachedTaughtMatch = host.Pm4DoodadDuplication.FindMatchingTaughtRule(
                    selectedPm4Obj, pm4TileX, pm4TileY,
                    out _cachedTaughtScale, out _cachedTaughtPos, out _cachedTaughtRot);
            }

            var taughtMatch = _cachedTaughtMatch;
            float taughtScale = _cachedTaughtScale;
            Vector3 taughtPos = _cachedTaughtPos;
            Vector3 taughtRot = _cachedTaughtRot;
            if (taughtMatch != null)
            {
                ImGui.TextColored(UiTokens.Success, $"Taught 3D Shape Match: {taughtMatch.ModelName} ({taughtMatch.AssetKind})");
                if (SharedUiWidgets.ActionButton($"Spawn & Propagate 3D Shape Across All Tiles ({unoccupiedHulls} Hulls)##propagate_taught"))
                {
                    host.Pm4DoodadDuplication.InstantiateTaughtPairingIntoHull(selectedPm4Obj, pm4TileX, pm4TileY, taughtMatch, taughtScale, taughtPos, taughtRot);
                    host.Pm4DoodadDuplication.ApplyTaughtPairingAcrossAllTiles(taughtMatch);
                }
                if (SharedUiWidgets.ActionButton($"Spawn into this Hull Only##spawn_taught_single"))
                {
                    host.Pm4DoodadDuplication.InstantiateTaughtPairingIntoHull(selectedPm4Obj, pm4TileX, pm4TileY, taughtMatch, taughtScale, taughtPos, taughtRot);
                }
                ImGui.Spacing();
            }

            // 1-Click pairing with nearby resident scene models
            var nearbyModels = FindNearbyResidentModels(worldScene, pm4TileX, pm4TileY, pm4Centroid, maxRadius: 100f, maxResults: 3);
            if (nearbyModels.Count > 0)
            {
                ImGui.Spacing();
                ImGui.TextUnformatted("Nearby Scene Models:");
                for (int m = 0; m < nearbyModels.Count; m++)
                {
                    var (nm, distToPm4) = nearbyModels[m];
                    if (SharedUiWidgets.ActionButton($"Pair with {nm.ModelName} (#{nm.UniqueId}) — {distToPm4:F1} yds##pair_near_{m}"))
                    {
                        worldScene.Selection.SelectSceneObjectByUniqueId(nm.UniqueId);
                    }
                }
            }

            if (SharedUiWidgets.ActionButton("Clear Hull Selection##clear_hull_sel"))
            {
                overlay.ClearPm4ObjectSelection();
            }

            ImGui.Spacing();
            ImGui.TextWrapped("💡 Tip: Click an object in the viewport to pair & align it with this PM4 hull, or match resident models below.");
            ImGui.Spacing();

            if (SharedUiWidgets.ActionButton("Match this PM4 Footprint to Resident Models"))
            {
                MatchPm4Footprint(overlay, host, worldScene, selectedPm4Obj, pm4TileX, pm4TileY);
            }

            if (_activeMatches.Count > 0 && _activeMatchCk24 == selectedPm4Obj.Ck24)
            {
                ImGui.Spacing();
                ImGui.TextColored(UiTokens.Success, $"Candidate Matches ({_activeMatches.Count}):");
                for (int i = 0; i < _activeMatches.Count; i++)
                {
                    var match = _activeMatches[i];
                    bool isFromDb = Pm4PriorKnowledgeDatabase.Instance.TryGetAsset(match.Candidate.ModelPath, out _);
                    string srcTag = isFromDb ? "[Prior DB]" : "[Resident]";
                    ImGui.BulletText($"{match.Candidate.ModelName} {srcTag} — Score: {match.Score:P0} (Scale {match.SolvedScale:F2}, Yaw {match.SolvedRotation.Z:F1}°)");

                    if (SharedUiWidgets.ActionButton($"Verify & Propagate 3D Shape Across All Tiles ({unoccupiedHulls} Hulls)##propagate_cand_{i}", unoccupiedHulls > 0))
                    {
                        var shape3D = selectedPm4Obj.Shape3D;
                        bool isWmo = Pm4DoodadDuplicationService.IsWmoAsset(match.Candidate.ModelPath, match.Candidate.AssetKind);
                        string assetKind = isWmo ? "WMO" : Pm4DoodadDuplicationService.ResolveAssetKind(match.Candidate.ModelPath, match.Candidate.AssetKind);
                        float actualScale = isWmo ? 1.0f : match.SolvedScale;

                        var rel = new CuratedSpatialRelationship
                        {
                            ModelName = match.Candidate.ModelName,
                            ModelPath = match.Candidate.ModelPath,
                            AssetKind = assetKind,
                            Ck24 = selectedPm4Obj.Ck24,
                            DominantGroupKey = selectedPm4Obj.DominantGroupKey,
                            Length = match.Candidate.Descriptor.Length,
                            Width = match.Candidate.Descriptor.Width,
                            Height = match.Candidate.Descriptor.Height,
                            SolvedScale = actualScale,
                            YawOffset = match.SolvedRotation.Z - selectedPm4Obj.BaseRotationRadians,
                            PositionOffset = match.SolvedPosition - pm4Centroid,
                            FitConfidence = (float)match.Score,
                            DetectedUtc = DateTime.UtcNow,
                            Verified = true,
                            VerticalSliceRadii = shape3D.VerticalSliceRadii,
                            InertiaTensor = shape3D.InertiaTensor,
                            NormalDistribution = shape3D.NormalDistribution
                        };

                        host.Pm4DoodadDuplication.SpawnCandidateIntoPm4Hull(selectedPm4Obj, pm4TileX, pm4TileY, match, out _);
                        host.Pm4DoodadDuplication.Curation.VerifyAndPropagateAcrossAllTiles(rel, worldScene!, out _);
                    }

                    if (SharedUiWidgets.ActionButton($"Spawn into this Hull Only##spawn_{i}"))
                    {
                        host.Pm4DoodadDuplication.SpawnCandidateIntoPm4Hull(selectedPm4Obj, pm4TileX, pm4TileY, match, out _);
                    }

                }
            }

            _manualCuration.DrawAssociationEditor(
                mapName,
                null,
                new Pm4HullIdentity(pm4TileX, pm4TileY, selectedPm4Obj.Ck24,
                    selectedPm4Obj.ObjectPartId, selectedPm4Obj.LinkGroupObjectId),
                host.CatalogView?.SelectedEntry,
                pm4Centroid);
        }
        // Branch 3: Only Scene Object is selected
        else if (selectedObj.HasValue)
        {
            var inst = selectedObj.Value;
            ImGui.TextColored(UiTokens.Success, $"Selected Object: {inst.ModelName} (#{inst.UniqueId})");
            ImGui.TextDisabled($"Pos: ({inst.PlacementPosition.X:F1}, {inst.PlacementPosition.Y:F1}, {inst.PlacementPosition.Z:F1}) | Tile: ({inst.TileX}, {inst.TileY})");
            ImGui.Spacing();

            ImGui.TextWrapped("💡 Tip: Click or Ctrl+Click a PM4 collision hull in the viewport to pair it with this model.");
            ImGui.Spacing();

            if (SharedUiWidgets.ActionButton("Magnetize Selected Object to Nearest PM4 Hull"))
            {
                host.Pm4DoodadDuplication.MagnetizePlacementToPm4(inst);
            }

            ImGui.TextDisabled($"Yaw: {inst.PlacementRotation.Z:F1}°  |  Z: {inst.PlacementPosition.Z:F3}");
            ImGui.SameLine();
            if (SharedUiWidgets.ActionButton("Rotate +90°##sel_rot_p90"))
            {
                host.Pm4DoodadDuplication.RotatePlacement(inst, 90f);
            }
            ImGui.SameLine();
            if (SharedUiWidgets.ActionButton("Rotate -90°##sel_rot_m90"))
            {
                host.Pm4DoodadDuplication.RotatePlacement(inst, -90f);
            }
            ImGui.SameLine();
            if (SharedUiWidgets.ActionButton("Flip 180°##sel_rot_180"))
            {
                host.Pm4DoodadDuplication.RotatePlacement(inst, 180f);
            }

            if (SharedUiWidgets.ActionButton("Fill Unmatched PM4 Doodad Hulls with this Model"))
            {
                host.Pm4DoodadDuplication.DuplicateModelToUnmatchedPm4Doodads(inst, inst.TileX, inst.TileY);
            }

            if (SharedUiWidgets.ActionButton("Match Nearest PM4 Footprint to Resident Models"))
            {
                MatchSelectedObjectFootprint(overlay, host, worldScene, inst);
            }

            // 1-Click pairing with nearby PM4 collision hulls
            var nearbyHulls = FindNearbyPm4Hulls(overlay, inst.TileX, inst.TileY, inst.PlacementPosition, maxRadius: 100f, maxResults: 3);
            if (nearbyHulls.Count > 0)
            {
                ImGui.Spacing();
                ImGui.TextUnformatted("Nearby PM4 Collision Hulls:");
                for (int h = 0; h < nearbyHulls.Count; h++)
                {
                    var (nh, tx, ty, distToHull) = nearbyHulls[h];
                    float nearZ = nh.Ck24 != 0 ? BitConverter.UInt32BitsToSingle(nh.Ck24 << 8) : 0f;
                    if (SharedUiWidgets.ActionButton($"Pair with PM4 Hull [{tx}_{ty}] Obj #{nh.ObjectPartId} (Z={nearZ:F1}yd) — {distToHull:F1} yds##pair_near_hull_{h}"))
                    {
                        overlay.SelectPm4Object((tx, ty, nh.Ck24, nh.ObjectPartId));
                    }
                    ImGui.SameLine();
                    if (SharedUiWidgets.ActionButton($"Snap & Align##align_quick_{h}"))
                    {
                        overlay.SelectPm4Object((tx, ty, nh.Ck24, nh.ObjectPartId));
                        host.Pm4DoodadDuplication.AlignPlacementToPm4Hull(inst, nh, tx, ty, fitScale: false, out _);
                    }
                }
            }

            if (SharedUiWidgets.ActionButton("Clear Model Selection##clear_model_sel"))
            {
                worldScene.Selection.ClearSelection();
            }

            _manualCuration.DrawPlacementEditor(mapName, host, worldScene, inst);
            _manualCuration.DrawAssociationEditor(mapName, inst, null);

            // Same Neighborhood / Family Models from Prior Knowledge Database
            string family = Pm4PriorKnowledgeDatabase.DetectFamily(inst.ModelPath);
            var familyAssets = Pm4PriorKnowledgeDatabase.Instance.GetNeighborhoodAssets(inst.ModelPath);
            if (familyAssets.Count > 0)
            {
                ImGui.Spacing();
                if (ImGui.TreeNodeEx($"Same Neighborhood / Family Models ({family}: {familyAssets.Count})###family_models", ImGuiTreeNodeFlags.DefaultOpen))
                {
                    for (int f = 0; f < Math.Min(familyAssets.Count, 8); f++)
                    {
                        var fa = familyAssets[f];
                        if (string.Equals(fa.ModelPath, inst.ModelPath, StringComparison.OrdinalIgnoreCase))
                            continue;

                        ImGui.BulletText($"{fa.ModelName} ({fa.AssetKind})");
                    }
                    ImGui.TreePop();
                }
            }

            if (_activeMatches.Count > 0)
            {
                ImGui.Spacing();
                ImGui.TextColored(UiTokens.Success, $"Candidate Matches ({_activeMatches.Count}):");
                for (int i = 0; i < _activeMatches.Count; i++)
                {
                    var match = _activeMatches[i];
                    bool isFromDb = Pm4PriorKnowledgeDatabase.Instance.TryGetAsset(match.Candidate.ModelPath, out _);
                    string srcTag = isFromDb ? "[Prior DB]" : "[Resident]";
                    ImGui.BulletText($"{match.Candidate.ModelName} {srcTag} — Score: {match.Score:P0} (Scale {match.SolvedScale:F2}, Yaw {match.SolvedRotation.Z:F1}°)");
                }
            }
        }
        // Branch 4: Neither selected
        else
        {
            ImGui.TextDisabled("No object or PM4 collision hull selected.");
            ImGui.Spacing();
            ImGui.TextWrapped("💡 Human Teaching Workflow: Click a PM4 collision hull in the viewport, then Ctrl+Click a scene model (or vice-versa) to pair, align, and teach.");
        }

        ImGui.Spacing();
        if (SharedUiWidgets.ActionButton("Patch & Export Monolithic ADTs to Output Folder..."))
        {
            host.Pm4DoodadDuplication.SavePatchedAdts();
        }

        string activeMap = !string.IsNullOrEmpty(host.LoadedFileName)
            ? Path.GetFileNameWithoutExtension(host.LoadedFileName)
            : "development";
        bool projectExists = Pm4PairingProjectStore.Exists(activeMap);
        taughtCount = host.Pm4DoodadDuplication.TaughtPairings.Count;

        if (taughtCount > 0 || projectExists)
        {
            ImGui.Spacing();
            if (ImGui.TreeNodeEx($"Taught Shape Library & Project ({taughtCount} rules)##Pm4TaughtRules", ImGuiTreeNodeFlags.DefaultOpen))
            {
                ImGui.TextDisabled($"Active Map: '{activeMap}' | {taughtCount} rule(s) in memory");
                string projectRelPath = $"output/projects/pm4_pairings/{activeMap}.pairings.json";
                if (projectExists)
                {
                    ImGui.TextColored(UiTokens.Success, $"● Saved project: {projectRelPath}");
                }
                else
                {
                    ImGui.TextColored(UiTokens.BrassDim, $"○ No map project file (Save writes to {projectRelPath})");
                }

                ImGui.Spacing();
                if (SharedUiWidgets.ActionButton("Save Project##Pm4SaveProject"))
                {
                    host.Pm4DoodadDuplication.SaveTaughtProject(activeMap);
                }
                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip($"Saves the current taught pairings to '{projectRelPath}', updates manifest, and syncs global curation library.");

                ImGui.SameLine();
                if (SharedUiWidgets.ActionButton("Load Project##Pm4LoadProject"))
                {
                    host.Pm4DoodadDuplication.LoadTaughtProject(activeMap, out _);
                }
                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip($"Reloads saved pairings from '{projectRelPath}' (or global library if missing).");

                ImGui.SameLine();
                if (SharedUiWidgets.ActionButton("Export Manifest##Pm4ExportManifest"))
                {
                    host.Pm4DoodadDuplication.ExportTaughtPairingsManifest(activeMap);
                }
                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip("Exports an authoritative JSON composition manifest containing metrics and footprint statistics.");

                if (!string.IsNullOrEmpty(host.Pm4DoodadDuplication.LastOperationMessage))
                {
                    ImGui.Spacing();
                    ImGui.TextColored(UiTokens.Success, host.Pm4DoodadDuplication.LastOperationMessage);
                }

                if (taughtCount > 0)
                {
                    ImGui.Spacing();
                    if (SharedUiWidgets.ActionButton($"Apply All Taught Rules Across Whole Map##pm4_apply_all_rules_project"))
                    {
                        host.Pm4MapWidePropagation.StartPropagationAndExport();
                    }
                    if (ImGui.IsItemHovered())
                    {
                        ImGui.SetTooltip("Asynchronously applies all taught pairing rules across ALL map tiles (including unopened ones) and exports patched LK ADTs.");
                    }
                    ImGui.SameLine();
                    if (SharedUiWidgets.ActionButton($"Apply to Loaded Tiles Only ({overlay._pm4TileObjects.Count})##pm4_propagate_loaded_only"))
                    {
                        host.Pm4DoodadDuplication.ApplyAllTaughtPairingsAcrossAllTiles();
                    }

                    ImGui.Spacing();
                    for (int i = 0; i < host.Pm4DoodadDuplication.TaughtPairings.Count; i++)
                    {
                        var rule = host.Pm4DoodadDuplication.TaughtPairings[i];
                        ImGui.BulletText($"{rule.ModelName} ({rule.AssetKind}) — {rule.Length:F1}x{rule.Width:F1}x{rule.Height:F1} yds");
                        if (SharedUiWidgets.ActionButton($"Apply to All Tiles##rule_apply_{i}"))
                        {
                            host.Pm4DoodadDuplication.ApplyTaughtPairingAcrossAllTiles(rule);
                        }
                        ImGui.SameLine();
                        if (SharedUiWidgets.ActionButton($"Delete##rule_del_{i}"))
                        {
                            host.Pm4DoodadDuplication.DeleteTaughtPairing(rule);
                            break;
                        }
                    }
                }
                ImGui.TreePop();
            }
        }

        ImGui.Spacing();
        ImGui.Separator();
        ImGui.Spacing();

        if (hasTile)
        {
            if (SharedUiWidgets.ActionButton($"Magnetize All Placements on Tile ({activeTileX}, {activeTileY})"))
            {
                host.Pm4DoodadDuplication.MagnetizeTilePlacementsToPm4(activeTileX, activeTileY);
            }
        }

        if (host.Pm4DoodadDuplication.CanUndo)
        {
            if (SharedUiWidgets.ActionButton("Undo Last PM4 Operation"))
            {
                host.Pm4DoodadDuplication.UndoLastOperation();
            }
        }

        if (!string.IsNullOrWhiteSpace(host.Pm4DoodadDuplication.LastOperationMessage))
        {
            ImGui.Spacing();
            SharedUiWidgets.StatusReadout("PM4 Status", host.Pm4DoodadDuplication.LastOperationMessage);
        }

        // Spec 295: Visual PM4 Shape Library, Calibrated Proposal Preview, Human-in-the-Loop Review Queue, and Batch Analysis
        ImGui.Spacing();
        ImGui.Separator();
        _shapeLibraryPanel.Draw(_shapeMatchWorkbench, worldScene);
        _shapeReviewPanel.Draw(
            _shapeMatchWorkbench,
            _manualCuration,
            overlay,
            host,
            worldScene,
            selectedPm4Obj,
            activeTileX,
            activeTileY,
            activeMap);
        _shapeBatchPanel.Draw(
            _shapeMatchWorkbench,
            _manualCuration,
            worldScene,
            activeMap);
    }

    private void MatchPm4Footprint(
        Pm4OverlayScene overlay,
        IViewerAppHost host,
        WorldScene worldScene,
        Pm4OverlayObject pm4Obj,
        int pm4TileX,
        int pm4TileY)
    {
        var surfaceVerts = new List<Vector3>(pm4Obj.Triangles.Count * 3);
        for (int t = 0; t < pm4Obj.Triangles.Count; t++)
        {
            surfaceVerts.Add(Vector3.Transform(pm4Obj.Triangles[t].A, pm4Obj.BaseTransform));
            surfaceVerts.Add(Vector3.Transform(pm4Obj.Triangles[t].B, pm4Obj.BaseTransform));
            surfaceVerts.Add(Vector3.Transform(pm4Obj.Triangles[t].C, pm4Obj.BaseTransform));
        }

        overlay._pm4TilePositionRefs.TryGetValue((pm4TileX, pm4TileY), out var mprlList);
        float? placementZ = pm4Obj.Ck24 != 0 ? BitConverter.UInt32BitsToSingle(pm4Obj.Ck24 << 8) : null;
        WorldScene.TransformBounds(pm4Obj.BoundsMin, pm4Obj.BoundsMax, pm4Obj.BaseTransform, out Vector3 worldMin, out _);

        Pm4FootprintPrism prism = Pm4FootprintExtruder.ExtrudeFootprint(surfaceVerts, mprlList, placementZ, worldMin.Z);

        var candidates = new List<Pm4ModelCandidate>();
        var seenModels = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // 1. Explicitly include selected scene object if one is currently selected in inspector
        if (worldScene.Selection.SelectedInstance is { } selectedObjInst)
        {
            if (seenModels.Add(selectedObjInst.ModelPath))
            {
                Vector3 min = selectedObjInst.SelectionBoundsResolved && selectedObjInst.SelectionLocalBoundsMax != Vector3.Zero ? selectedObjInst.SelectionLocalBoundsMin : selectedObjInst.LocalBoundsMin;
                Vector3 max = selectedObjInst.SelectionBoundsResolved && selectedObjInst.SelectionLocalBoundsMax != Vector3.Zero ? selectedObjInst.SelectionLocalBoundsMax : selectedObjInst.LocalBoundsMax;
                candidates.Add(Pm4ModelCandidate.Create(selectedObjInst.ModelPath, selectedObjInst.AssetKind, min, max));
            }
        }

        // 2. Models resident in the scene/tile
        IPm4OverlayHost pm4Host = worldScene;
        foreach (var list in pm4Host.TileMdxInstances.Values)
        {
            foreach (var placed in list)
            {
                if (seenModels.Add(placed.ModelPath))
                {
                    Vector3 min = placed.SelectionBoundsResolved && placed.SelectionLocalBoundsMax != Vector3.Zero ? placed.SelectionLocalBoundsMin : placed.LocalBoundsMin;
                    Vector3 max = placed.SelectionBoundsResolved && placed.SelectionLocalBoundsMax != Vector3.Zero ? placed.SelectionLocalBoundsMax : placed.LocalBoundsMax;
                    candidates.Add(Pm4ModelCandidate.Create(placed.ModelPath, "M2", min, max));
                }
            }
        }
        foreach (var list in pm4Host.TileWmoInstances.Values)
        {
            foreach (var placed in list)
            {
                if (seenModels.Add(placed.ModelPath))
                {
                    Vector3 min = placed.SelectionBoundsResolved && placed.SelectionLocalBoundsMax != Vector3.Zero ? placed.SelectionLocalBoundsMin : placed.LocalBoundsMin;
                    Vector3 max = placed.SelectionBoundsResolved && placed.SelectionLocalBoundsMax != Vector3.Zero ? placed.SelectionLocalBoundsMax : placed.LocalBoundsMax;
                    candidates.Add(Pm4ModelCandidate.Create(placed.ModelPath, "WMO", min, max));
                }
            }
        }

        // 3. Pre-populated Prior Knowledge Database candidates (curated, taught, and whole-map harvest)
        string? targetFamily = worldScene.Selection.SelectedInstance is { } selInst ? Pm4PriorKnowledgeDatabase.DetectFamily(selInst.ModelPath) : null;
        var dbCandidates = Pm4PriorKnowledgeDatabase.Instance.GetDatabaseCandidates(targetFamily);
        for (int i = 0; i < dbCandidates.Count; i++)
        {
            var dbCand = dbCandidates[i];
            if (seenModels.Add(dbCand.ModelPath))
            {
                candidates.Add(dbCand);
            }
        }

        if (candidates.Count == 0)
        {
            string noModelsMsg = "No candidate models available in scene or prior database.";
            host.StatusMessage = noModelsMsg;
            host.Pm4DoodadDuplication.LastOperationMessage = noModelsMsg;
            _activeMatches.Clear();
            return;
        }

        var rawMatches = Pm4DirectModelMatcher.MatchPrism(prism, candidates, maxResults: 8);
        _activeMatches = rawMatches.Where(m => m.Score >= 0.40 && m.SolvedScale <= 3.0f).Take(4).ToList();
        _activeMatchCk24 = pm4Obj.Ck24;

        if (_activeMatches.Count > 0)
        {
            var top = _activeMatches[0];
            string msg = $"✓ Found {_activeMatches.Count} match candidates. Best: {top.Candidate.ModelName} (Score: {top.Score:P0}, Scale: {top.SolvedScale:F2}, Yaw: {top.SolvedRotation.Z:F1}°).";
            host.StatusMessage = msg;
            host.Pm4DoodadDuplication.LastOperationMessage = msg;
        }
        else
        {
            string msg = "No matching resident models found for this PM4 footprint.";
            host.StatusMessage = msg;
            host.Pm4DoodadDuplication.LastOperationMessage = msg;
        }
    }

    private void MatchSelectedObjectFootprint(
        Pm4OverlayScene overlay,
        IViewerAppHost host,
        WorldScene worldScene,
        ObjectInstance inst)
    {
        Pm4OverlayObject? bestMatch = null;
        int bestTx = inst.TileX;
        int bestTy = inst.TileY;
        float bestDist = 45.0f;

        foreach (var (tileCoord, objList) in overlay._pm4TileObjects)
        {
            for (int i = 0; i < objList.Count; i++)
            {
                var obj = objList[i];
                Vector3 centroid = Vector3.Transform(obj.Center, obj.BaseTransform);
                float dist = Vector3.Distance(centroid, inst.PlacementPosition);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestMatch = obj;
                    bestTx = tileCoord.tileX;
                    bestTy = tileCoord.tileY;
                }
            }
        }

        if (bestMatch == null)
        {
            string msg = "No PM4 collision hull found near selected object.";
            host.StatusMessage = msg;
            host.Pm4DoodadDuplication.LastOperationMessage = msg;
            _activeMatches.Clear();
            return;
        }

        MatchPm4Footprint(overlay, host, worldScene, bestMatch, bestTx, bestTy);
    }

    private static List<(ObjectInstance Model, float Distance)> FindNearbyResidentModels(
        WorldScene worldScene, int tileX, int tileY, Vector3 origin, float maxRadius = 100f, int maxResults = 3)
    {
        var results = new List<(ObjectInstance Model, float Distance)>();
        IPm4OverlayHost pm4Host = worldScene;

        void ScanList(List<ObjectInstance>? list)
        {
            if (list == null) return;
            for (int i = 0; i < list.Count; i++)
            {
                var inst = list[i];
                float d = Vector3.Distance(inst.PlacementPosition, origin);
                if (d <= maxRadius)
                    results.Add((inst, d));
            }
        }

        if (pm4Host.TileWmoInstances.TryGetValue((tileX, tileY), out var wmoList)) ScanList(wmoList);
        if (pm4Host.TileMdxInstances.TryGetValue((tileX, tileY), out var mdxList)) ScanList(mdxList);

        if (results.Count == 0)
        {
            foreach (var pair in pm4Host.TileWmoInstances)
                if (Math.Abs(pair.Key.Item1 - tileX) <= 1 && Math.Abs(pair.Key.Item2 - tileY) <= 1) ScanList(pair.Value);
            foreach (var pair in pm4Host.TileMdxInstances)
                if (Math.Abs(pair.Key.Item1 - tileX) <= 1 && Math.Abs(pair.Key.Item2 - tileY) <= 1) ScanList(pair.Value);
        }

        results.Sort((a, b) => a.Item2.CompareTo(b.Item2));
        if (results.Count > maxResults)
            results.RemoveRange(maxResults, results.Count - maxResults);

        return results;
    }

    private static List<(Pm4OverlayObject Hull, int TileX, int TileY, float Distance)> FindNearbyPm4Hulls(
        Pm4OverlayScene overlay, int tileX, int tileY, Vector3 origin, float maxRadius = 100f, int maxResults = 3)
    {
        var results = new List<(Pm4OverlayObject Hull, int TileX, int TileY, float Distance)>();

        void ScanList(int tx, int ty, List<Pm4OverlayObject>? list)
        {
            if (list == null) return;
            for (int i = 0; i < list.Count; i++)
            {
                var hull = list[i];
                Vector3 center = Vector3.Transform(hull.Center, hull.BaseTransform);
                float d = Vector3.Distance(center, origin);
                if (d <= maxRadius)
                    results.Add((hull, tx, ty, d));
            }
        }

        if (overlay._pm4TileObjects.TryGetValue((tileX, tileY), out var hullList)) ScanList(tileX, tileY, hullList);

        if (results.Count == 0)
        {
            foreach (var pair in overlay._pm4TileObjects)
                if (Math.Abs(pair.Key.Item1 - tileX) <= 1 && Math.Abs(pair.Key.Item2 - tileY) <= 1) ScanList(pair.Key.Item1, pair.Key.Item2, pair.Value);
        }

        results.Sort((a, b) => a.Item4.CompareTo(b.Item4));
        if (results.Count > maxResults)
            results.RemoveRange(maxResults, results.Count - maxResults);

        return results;
    }
}
