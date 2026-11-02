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
using WowViewer.Core.PM4.Footprint;
using WowViewer.Core.PM4.Matching;
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

    private List<Pm4DirectMatchResult> _activeMatches = new();
    private uint _activeMatchCk24;

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

        // Branch 1: Both PM4 Hull AND Scene Model are selected (Human-in-the-loop pairing & teaching)
        if (selectedObj.HasValue && selectedPm4Obj != null)
        {
            var inst = selectedObj.Value;
            Vector3 pm4Centroid = Vector3.Transform(selectedPm4Obj.Center, selectedPm4Obj.BaseTransform);
            float dist = Vector3.Distance(inst.PlacementPosition, pm4Centroid);

            ImGui.TextColored(UiTokens.Brass, "Active Pairing (Human Teaching Mode):");
            ImGui.BulletText($"Model: {inst.ModelName} (#{inst.UniqueId})");
            ImGui.BulletText($"PM4 Hull: 0x{selectedPm4Obj.Ck24:X6} (part {selectedPm4Obj.ObjectPartId}, group 0x{selectedPm4Obj.DominantGroupKey:X2}) — {dist:F1} yds away");
            ImGui.Spacing();

            if (SharedUiWidgets.ActionButton("Align Model to this PM4 Hull (Centroid, Height & Yaw)"))
            {
                host.Pm4DoodadDuplication.AlignPlacementToPm4Hull(inst, selectedPm4Obj, pm4TileX, pm4TileY, fitScale: false, out _);
            }

            if (SharedUiWidgets.ActionButton("Teach & Save Pairing for Automation"))
            {
                host.Pm4DoodadDuplication.TeachPairing(selectedPm4Obj, inst, pm4TileX, pm4TileY);
            }

            var matchingTaught = host.Pm4DoodadDuplication.TaughtPairings.FirstOrDefault(t =>
                (t.Ck24 != 0 && t.Ck24 == selectedPm4Obj.Ck24) ||
                (t.Ck24 == 0 && selectedPm4Obj.Ck24 == 0 && t.DominantGroupKey == selectedPm4Obj.DominantGroupKey));

            if (matchingTaught != null && hasTile)
            {
                if (SharedUiWidgets.ActionButton($"Apply Taught Rule to All Matching Hulls on Tile ({activeTileX}, {activeTileY})"))
                {
                    host.Pm4DoodadDuplication.ApplyTaughtPairingToTile(matchingTaught, activeTileX, activeTileY);
                }
            }
        }
        // Branch 2: Only PM4 Hull is selected
        else if (selectedPm4Obj != null)
        {
            Vector3 pm4Centroid = Vector3.Transform(selectedPm4Obj.Center, selectedPm4Obj.BaseTransform);
            ImGui.TextColored(UiTokens.Brass, $"Selected PM4 Hull: 0x{selectedPm4Obj.Ck24:X6} (part {selectedPm4Obj.ObjectPartId})");
            ImGui.TextDisabled($"Centroid: ({pm4Centroid.X:F1}, {pm4Centroid.Y:F1}, {pm4Centroid.Z:F1}) | Group: 0x{selectedPm4Obj.DominantGroupKey:X2} | Tris: {selectedPm4Obj.Triangles.Count}");
            ImGui.Spacing();

            ImGui.TextWrapped("💡 Tip: Ctrl+Click an object in the viewport to pair & align it with this PM4 hull, or match resident models below.");
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
                    ImGui.BulletText($"{match.Candidate.ModelName} — Score: {match.Score:P0} (Scale {match.SolvedScale:F2}, Yaw {match.SolvedRotation.Z:F1}°)");
                    if (SharedUiWidgets.ActionButton($"Spawn {match.Candidate.ModelName} into Hull##spawn_{i}"))
                    {
                        host.Pm4DoodadDuplication.SpawnCandidateIntoPm4Hull(selectedPm4Obj, pm4TileX, pm4TileY, match, out _);
                    }
                }
            }
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

            if (SharedUiWidgets.ActionButton("Fill Unmatched PM4 Doodad Hulls with this Model"))
            {
                host.Pm4DoodadDuplication.DuplicateModelToUnmatchedPm4Doodads(inst, inst.TileX, inst.TileY);
            }

            if (SharedUiWidgets.ActionButton("Match Nearest PM4 Footprint to Resident Models"))
            {
                MatchSelectedObjectFootprint(overlay, host, worldScene, inst);
            }

            if (_activeMatches.Count > 0)
            {
                ImGui.Spacing();
                ImGui.TextColored(UiTokens.Success, $"Candidate Matches ({_activeMatches.Count}):");
                for (int i = 0; i < _activeMatches.Count; i++)
                {
                    var match = _activeMatches[i];
                    ImGui.BulletText($"{match.Candidate.ModelName} — Score: {match.Score:P0} (Scale {match.SolvedScale:F2}, Yaw {match.SolvedRotation.Z:F1}°)");
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

        if (candidates.Count == 0)
        {
            string noModelsMsg = "No candidate models resident in scene to match against.";
            host.StatusMessage = noModelsMsg;
            host.Pm4DoodadDuplication.LastOperationMessage = noModelsMsg;
            _activeMatches.Clear();
            return;
        }

        _activeMatches = Pm4DirectModelMatcher.MatchPrism(prism, candidates, maxResults: 4);
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
}
