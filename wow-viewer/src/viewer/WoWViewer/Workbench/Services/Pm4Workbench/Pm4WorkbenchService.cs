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

/// <summary>
/// PM4 workbench UI: the workbench inspector, overlay and selection tabs, scene facts and outliner, perf panel.
/// Partial class containing PM4 alignment and viewer utility windows.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed partial class Pm4WorkbenchService
{
    private readonly IViewerAppHost _host;

    internal Pm4WorkbenchService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge: see Pm4WorkbenchService.Host.cs.

    private string _loosePm4InputPath = @"C:\WoW4-data\unk_pm4s\World\Maps\unk\unk1_00_00.pm4";

    internal void OpenPm4Workbench(Pm4WorkbenchTab tab)
    {
        _shellLayout.FocusShellPanel(ShellPanelId.Pm4Workbench);
        _pendingPm4WorkbenchTab = tab;
        _activeBottomDrawerTab = FixedBottomDrawerTab.Pm4;
        _pendingRightSidebarSection = FixedBottomDrawerTab.Pm4;
        _showRightSidebar = true;
        if (_workspaceMode == WorkspaceMode.Editor)
            SetEditorWorkspaceTask(EditorWorkspaceTask.Pm4Evidence);
    }

    internal void DrawPm4WorkbenchInspector()
    {
        if (_worldScene == null)
        {
            ImGui.TextDisabled("PM4 workbench becomes available once a world scene is loaded.");
            return;
        }

        ImGui.TextDisabled("Hover stays lightweight. Click a PM4 object to inspect its matches, graph, and correlation here.");
        ImGui.SetNextItemOpen(true, ImGuiCond.Once);
        DrawPm4GlossarySummary();

        if (!ImGui.BeginTabBar("##Pm4WorkbenchTabs"))
            return;

        ImGuiTabItemFlags overlayFlags = _pendingPm4WorkbenchTab == Pm4WorkbenchTab.Overlay
            ? ImGuiTabItemFlags.SetSelected
            : ImGuiTabItemFlags.None;
        bool overlayTabOpen = true;
        if (ImGui.BeginTabItem("Overlay", ref overlayTabOpen, overlayFlags))
        {
            DrawPm4OverlayWorkbenchContent();
            ImGui.EndTabItem();
        }

        ImGuiTabItemFlags selectionFlags = _pendingPm4WorkbenchTab == Pm4WorkbenchTab.Selection
            ? ImGuiTabItemFlags.SetSelected
            : ImGuiTabItemFlags.None;
        bool selectionTabOpen = true;
        if (ImGui.BeginTabItem("Selection", ref selectionTabOpen, selectionFlags))
        {
            DrawPm4SelectionWorkbenchContent();
            ImGui.EndTabItem();
        }

        ImGuiTabItemFlags correlationFlags = _pendingPm4WorkbenchTab == Pm4WorkbenchTab.Correlation
            ? ImGuiTabItemFlags.SetSelected
            : ImGuiTabItemFlags.None;
        bool correlationTabOpen = true;
        if (ImGui.BeginTabItem("Correlation", ref correlationTabOpen, correlationFlags))
        {
            // Spec 231 D2: the workbench correlation copy was a strict subset of the
            // canonical correlation page; both now draw the single survivor.
            DrawPm4WmoCorrelationContent();
            ImGui.EndTabItem();
        }

        _pendingPm4WorkbenchTab = null;
        ImGui.EndTabBar();
    }

    private void DrawPm4GlossarySummary()
    {
        if (!ImGui.CollapsingHeader("PM4 Glossary / Evidence"))
            return;

        ImGui.TextWrapped("PM4/PD4 are server-side navigation-mesh files (no renderable geometry): walkable surfaces, their adjacency, and what blocks movement. The cyan (MSCN) and magenta (MSPV) cubes you can toggle below are the visible projection of this data. Authoritative readings: docs/architecture/pm4-chunk-semantics.md (2026-06-09) and docs/wowdev-wiki/pm4-pd4-draft.md (2026-08-24, measured).");
        ImGui.BulletText("MSUR — 32-byte surface record; a polygon fan over MSVI[0x14 .. 0x14+vertex count at 0x01]. 0x04 = normal; 0x10 = plane distance (NOT a Y-up height); 0x18 = start of the surface's MSLK adjacency window (length at 0x02) — measured 100.0000% against MSLK; the old 'indexes MSCN' reading is eliminated; 0x1C = IEEE float, see CK24 below.");
        ImGui.BulletText("MSLK — the undirected surface-adjacency graph of the navmesh (98.76% reciprocity). RefIndex names a NEIGHBOURING surface, not the owner. ~47% of links carry an MSPI/MSPV vertical wall quad; the rest are open passage. Viewer TypeFlags: 0x03 = M2 top (walkable), 0x10 = interior floor (walkable), 0x12 = exterior solid (structural wall).");
        ImGui.BulletText("MSCN — 3D connector/boundary node positions in the world frame (NOT normals, NOT per-surface indexed from MSUR). Objects share nodes (64.4%): node reuse is where objects meet. No in-file index consumer is known; the consumer is likely external. Visible as the cyan cubes.");
        ImGui.BulletText("MSPV — wall-quad vertices on the blocked subset of MSLK links (0 of 598,790 path windows are Z-dominant = walls; MSUR normals are 91.7% Z-dominant = floors). Visible as the magenta cubes.");
        ImGui.BulletText("MSVI / MSVT — mesh-index stream into mesh vertex positions: the actual 3D positions of every vertex of every polygon. Walked as MSUR.MsviFirstIndex..MsviFirstIndex+IndexCount -> MSVI -> MSVT. Polygon sizes are not triangles (77% quads in the measured corpus).");
        ImGui.BulletText("MPRL — per-tile position reference: a 3D position + heading, used for spawn anchors. MPRL is the only permuted-axis chunk in the file.");
        ImGui.BulletText("MSHD — 32-byte file header. 0x00/0x08 = clamped world-unit spans on the two axes (not counts); 0x04 == 1 marks a tile with no surfaces; 0x0C-0x1C are zero in all 502 measured files (reserved). The viewer's 'MSHD Region' grouping value (Field04) is a tile-level hint only — not a proven per-object grouping semantic.");
        ImGui.BulletText("CK24: legacy viewer alias for MSUR field 0x1C. MEASURED 2026-08: it is an IEEE-754 float holding the Z coordinate of the ADT placement that produced the object (93.58% bit-exact vs MODF/MDDF). The old 'type' byte is the float's exponent band (why a tile shows only ~4 'types'); grouping by it collides for objects at equal height. 0x1C == 0.0f is the unattributed bucket (vertically stretched geometry), not an object id.");
        ImGui.BulletText("part / ObjectPartId: viewer-generated split id. WoWViewer assigns it during the current overlay build after CK24 grouping, dominant MSLK grouping, optional split, then optional connectivity split. It is not a raw PM4 field.");
        ImGui.BulletText("MSLK Group: dominant MSLK.GroupObjectId seen in the current viewer object. Strong grouping hint, not final proof of identity.");
        ImGui.BulletText("Linked MPRL refs: position-reference rows attached to the current viewer object or its dominant link family. Used as placement evidence.");
        ImGui.BulletText("Group / Attr / MscnRef: dominant MSUR values across the selected viewer object. NOTE: 'MscnRef' is a legacy alias for MSUR.0x18, now measured to be the surface's MSLK adjacency window start — not an MSCN index. Debugging aid, not authoritative.");
        ImGui.BulletText("PM4 Graph: the viewer's current decomposition of the selected object, not a literal raw node graph stored in PM4. MSLK is an adjacency graph of surfaces, not a graph of 'navmesh nodes'.");
        ImGui.BulletText("Match uid: nearby MODF/MDDF placement candidate id. It is not a PM4-native object id. Per-object placement is (placement.X, placement.Y, MSUR.0x1C): X/Y come from the joined MODF/MDDF record, 0x1C is the authored Z and the join key.");
        ImGui.BulletText("cyan:magenta ratio: per-object topology fingerprint. ~0 magenta = disjoint decoration (no MSLK links between surfaces). ~1:1 = connected WMO. >1:1 = contiguous M2 with a dense connection graph. Used by the spec 050/052 matcher as a pre-filter.");
    }

    internal void DrawPm4OverlayWorkbenchContent()
    {
        if (_worldScene == null)
            return;

        bool showPm4Overlay = _worldScene.Pm4Overlay.ShowPm4Overlay;
        if (ImGui.Checkbox("PM4 Overlay", ref showPm4Overlay))
            _worldScene.Pm4Overlay.ShowPm4Overlay = showPm4Overlay;

        ImGui.SameLine();
        if (ImGui.Button("Reload PM4"))
            _worldScene.Pm4Overlay.ReloadPm4Overlay();

        ImGui.SameLine();
        if (ImGui.Button("Save Overlay Align"))
            SaveCurrentPm4Alignment();

        ImGui.Separator();
        ImGui.TextDisabled("Direct PM4/PD4 File Loader (ignores map name and naming rules)");
        ImGui.SetNextItemWidth(340f);
        ImGui.InputText("##LoosePm4InputPath", ref _loosePm4InputPath, 512);
        ImGui.SameLine();
        if (ImGui.Button("Load Loose File"))
        {
            if (!string.IsNullOrWhiteSpace(_loosePm4InputPath) && System.IO.File.Exists(_loosePm4InputPath))
            {
                if (_worldScene.Pm4Overlay.LoadLoosePm4File(_loosePm4InputPath))
                    _statusMessage = $"Loaded loose PM4/PD4 file: {System.IO.Path.GetFileName(_loosePm4InputPath)}";
                else
                    _statusMessage = $"Failed to decode loose PM4/PD4 file: {_loosePm4InputPath}";
            }
            else
            {
                _statusMessage = $"Loose PM4/PD4 file not found: {_loosePm4InputPath}";
            }
        }
        ImGui.Separator();

        bool showPm4Solid = _worldScene.Pm4Overlay.ShowPm4SolidOverlay;
        if (ImGui.Checkbox("PM4 Solid Fill", ref showPm4Solid))
            _worldScene.Pm4Overlay.ShowPm4SolidOverlay = showPm4Solid;

        ImGui.SameLine();
        bool pm4IgnoreDepth = _worldScene.Pm4Overlay.Pm4OverlayIgnoreDepth;
        if (ImGui.Checkbox("PM4 X-Ray", ref pm4IgnoreDepth))
            _worldScene.Pm4Overlay.Pm4OverlayIgnoreDepth = pm4IgnoreDepth;

        ImGui.SameLine();
        bool showPm4Bounds = _worldScene.Pm4Overlay.ShowPm4ObjectBounds;
        if (ImGui.Checkbox("PM4 Bounds", ref showPm4Bounds))
            _worldScene.Pm4Overlay.ShowPm4ObjectBounds = showPm4Bounds;

        ImGui.SameLine();
        bool showPm4Ck24Bounds = _worldScene.Pm4Overlay.ShowPm4Ck24Bounds;
        if (ImGui.Checkbox("PM4 CK24 Bounds", ref showPm4Ck24Bounds))
            _worldScene.Pm4Overlay.ShowPm4Ck24Bounds = showPm4Ck24Bounds;

        bool showRecovered = _worldScene.Pm4Overlay.ShowPm4GeneratedPlacements;
        if (ImGui.Checkbox("Recovered placements", ref showRecovered))
            _worldScene.Pm4Overlay.ShowPm4GeneratedPlacements = showRecovered;
        if (ImGui.IsItemHovered())
        {
            ImGui.BeginTooltip();
            ImGui.TextUnformatted("Placements rebuilt from PM4 geometry, for tiles that have none.");
            ImGui.TextUnformatted("Position and size are derived; the asset name is a ranked guess.");
            ImGui.TextUnformatted("Green = tight shape match, amber = loose.");
            int count = _worldScene.Pm4Overlay.Pm4GeneratedPlacementCount;
            ImGui.TextUnformatted(count > 0
                ? $"{count} loaded."
                : "None loaded - run: pm4 generate-placements");
            ImGui.EndTooltip();
        }

        if (showRecovered)
        {
            ImGui.SameLine();
            bool terrainlessOnly = _worldScene.Pm4Overlay.Pm4GeneratedPlacementsTerrainlessOnly;
            if (ImGui.Checkbox("Terrain-less tiles only", ref terrainlessOnly))
                _worldScene.Pm4Overlay.Pm4GeneratedPlacementsTerrainlessOnly = terrainlessOnly;
            if (ImGui.IsItemHovered())
            {
                ImGui.BeginTooltip();
                ImGui.TextUnformatted("Tiles that still have an ADT already draw their real placements.");
                ImGui.TextUnformatted("Turn this off to compare recovered boxes against those.");
                ImGui.EndTooltip();
            }
        }

        bool showPlacementZ = _worldScene.Pm4Overlay.ShowPm4PlacementZPlane;
        if (ImGui.Checkbox("Placement Z markers", ref showPlacementZ))
            _worldScene.Pm4Overlay.ShowPm4PlacementZPlane = showPlacementZ;
        if (ImGui.IsItemHovered())
        {
            ImGui.BeginTooltip();
            ImGui.TextUnformatted("Marker at the SELECTED object's placement height.");
            ImGui.TextUnformatted("It should sit at the base of that object.");
            ImGui.TextUnformatted("Only the height comes from the data; position is the object's centre.");
            ImGui.EndTooltip();
        }

        if (showPlacementZ)
        {
            ImGui.SameLine();
            bool allZ = _worldScene.Pm4Overlay.ShowPm4PlacementZForAllObjects;
            if (ImGui.Checkbox("all objects##Pm4PlacementZAll", ref allZ))
                _worldScene.Pm4Overlay.ShowPm4PlacementZForAllObjects = allZ;
            if (ImGui.IsItemHovered())
            {
                ImGui.BeginTooltip();
                ImGui.TextUnformatted("Draws one for every placed object. Expect a field of cubes.");
                ImGui.EndTooltip();
            }
        }

        bool showPm4Refs = _worldScene.Pm4Overlay.ShowPm4PositionRefs;
        if (ImGui.Checkbox("PM4 MPRL Refs", ref showPm4Refs))
            _worldScene.Pm4Overlay.ShowPm4PositionRefs = showPm4Refs;

        ImGui.SameLine();
        bool showPm4Centroids = _worldScene.Pm4Overlay.ShowPm4ObjectCentroids;
        if (ImGui.Checkbox("PM4 Centroids", ref showPm4Centroids))
            _worldScene.Pm4Overlay.ShowPm4ObjectCentroids = showPm4Centroids;

        bool showPm4Mscn = _worldScene.Pm4Overlay.ShowPm4MscnNodes;
        if (ImGui.Checkbox("MSCN Nodes (cyan, per-surface connector anchor)", ref showPm4Mscn))
            _worldScene.Pm4Overlay.ShowPm4MscnNodes = showPm4Mscn;
        ImGui.SameLine();
        bool showPm4Mspv = _worldScene.Pm4Overlay.ShowPm4MspvNodes;
        if (ImGui.Checkbox("MSPV Nodes (magenta, per-link path vertex)", ref showPm4Mspv))
            _worldScene.Pm4Overlay.ShowPm4MspvNodes = showPm4Mspv;

        bool renderNodesAsCubes = _worldScene.Pm4Overlay.Pm4RenderNodesAsCubes;
        if (ImGui.Checkbox("Nodes as Solid Cubes", ref renderNodesAsCubes))
            _worldScene.Pm4Overlay.Pm4RenderNodesAsCubes = renderNodesAsCubes;

        float mscnSize = _worldScene.Pm4Overlay.Pm4MscnCubeSize;
        ImGui.SetNextItemWidth(100f);
        if (ImGui.SliderFloat("MSCN size", ref mscnSize, 0.2f, 4f))
            _worldScene.Pm4Overlay.Pm4MscnCubeSize = mscnSize;
        ImGui.SameLine();
        float mspvSize = _worldScene.Pm4Overlay.Pm4MspvCubeSize;
        ImGui.SetNextItemWidth(100f);
        if (ImGui.SliderFloat("MSPV size", ref mspvSize, 0.2f, 4f))
            _worldScene.Pm4Overlay.Pm4MspvCubeSize = mspvSize;

        ImGui.SameLine();
        float mscnAlpha = _worldScene.Pm4Overlay.Pm4MscnCubeAlpha;
        ImGui.SetNextItemWidth(100f);
        if (ImGui.SliderFloat("MSCN α", ref mscnAlpha, 0.1f, 1f))
            _worldScene.Pm4Overlay.Pm4MscnCubeAlpha = mscnAlpha;
        ImGui.SameLine();
        float mspvAlpha = _worldScene.Pm4Overlay.Pm4MspvCubeAlpha;
        ImGui.SetNextItemWidth(100f);
        if (ImGui.SliderFloat("MSPV α", ref mspvAlpha, 0.1f, 1f))
            _worldScene.Pm4Overlay.Pm4MspvCubeAlpha = mspvAlpha;

        ImGui.SameLine();
        float lineWidth = _worldScene.Pm4Overlay.Pm4WireframeLineWidth;
        ImGui.SetNextItemWidth(120f);
        if (ImGui.SliderFloat("Wire width", ref lineWidth, 1f, 8f))
            _worldScene.Pm4Overlay.Pm4WireframeLineWidth = lineWidth;

        ImGui.SameLine();
        bool pm4FlipAllObjY = _worldScene.Pm4Overlay.Pm4FlipAllObjectsY;
        if (ImGui.Checkbox("Mirror PM4 N/S", ref pm4FlipAllObjY))
            _worldScene.Pm4Overlay.Pm4FlipAllObjectsY = pm4FlipAllObjY;

        ImGui.SameLine();
        if (ImGui.Button("Export Report"))
            ExportPm4OverlayReport();

        // Surface classes, keyed by MSUR._0x00. Measured 2026-08-24 over 309 files with
        // `pm4 surface-class`; labels carry what each value actually separates. The old
        // "CK24 0x40 / 0x80 / Other" checkboxes filtered by Ck24Type, which is the EXPONENT BAND of
        // the placement-Z float - i.e. they filtered by height octave - so they are gone.
        ImGui.TextDisabled("Surface class (MSUR._0x00)");
        foreach ((byte cls, string label) in Pm4SurfaceClassLabels)
        {
            bool visible = _worldScene.Pm4Overlay.IsPm4SurfaceClassVisible(cls);
            if (ImGui.Checkbox(label, ref visible))
                _worldScene.Pm4Overlay.SetPm4SurfaceClassVisible(cls, visible);
            if (ImGui.IsItemHovered())
            {
                // TextUnformatted, not Text: these strings carry '%' from measured figures and
                // ImGui.Text would read it as a printf conversion specifier.
                ImGui.BeginTooltip();
                ImGui.TextUnformatted(Pm4SurfaceClassTooltips[cls]);
                ImGui.TextUnformatted("Filters by the object's dominant class; cannot isolate one class within an object.");
                ImGui.EndTooltip();
            }
        }

        Pm4OverlayColorMode colorMode = _worldScene.Pm4Overlay.Pm4ColorMode;
        if (ImGui.BeginCombo("PM4 Color", GetPm4ColorModeLabel(colorMode)))
        {
            foreach (Pm4OverlayColorMode mode in Enum.GetValues<Pm4OverlayColorMode>())
            {
                bool isSelected = mode == colorMode;
                if (ImGui.Selectable(GetPm4ColorModeLabel(mode), isSelected))
                    _worldScene.Pm4Overlay.Pm4ColorMode = mode;
                if (isSelected)
                    ImGui.SetItemDefaultFocus();
            }

            ImGui.EndCombo();
        }

        bool splitCk24Connectivity = _worldScene.Pm4Overlay.Pm4SplitCk24ByConnectivity;
        if (ImGui.Checkbox("Split CK24 by Connectivity", ref splitCk24Connectivity))
        {
            _worldScene.Pm4Overlay.Pm4SplitCk24ByConnectivity = splitCk24Connectivity;
            _worldScene.Pm4Overlay.ReloadPm4Overlay();
        }

        bool splitCk24ByMscnRef = _worldScene.Pm4Overlay.Pm4SplitCk24ByMscnRef;
        if (ImGui.Checkbox("Split CK24 by MscnRef", ref splitCk24ByMscnRef))
        {
            _worldScene.Pm4Overlay.Pm4SplitCk24ByMscnRef = splitCk24ByMscnRef;
            _worldScene.Pm4Overlay.ReloadPm4Overlay();
        }

        bool showPathWalls = _worldScene.Pm4Overlay.Pm4ShowPathWalls;
        if (ImGui.Checkbox("Show MSPV/MSPI walls", ref showPathWalls))
        {
            _worldScene.Pm4Overlay.Pm4ShowPathWalls = showPathWalls;
            _worldScene.Pm4Overlay.ReloadPm4Overlay();
        }

        if (ImGui.IsItemHovered())
        {
            ImGui.SetTooltip(
                "MSLK path windows drawn as vertical faces.\n" +
                "Measured over the 616-file development corpus: 98% of windows are exactly 4\n" +
                "indices, 99.6% coplanar, and none of 598,790 faces has Z as its dominant normal.\n" +
                "MSUR is the floors; this is the walls between them.");
        }

        if (_worldScene.Pm4Overlay.Pm4ShowPathWalls && _worldScene.Pm4Overlay.Pm4LoadAttempted)
            ImGui.TextDisabled($"  wall faces: {_worldScene.Pm4Overlay.Pm4WallFaceCount}");

        if (_worldScene.Pm4Overlay.IsPm4Loading)
            ImGui.TextColored(new Vector4(1.0f, 0.85f, 0.35f, 1.0f), $"PM4 loading... {_worldScene.Pm4Overlay.Pm4Status}");
        else if (_worldScene.Pm4Overlay.Pm4LoadAttempted)
            ImGui.TextDisabled($"PM4: {_worldScene.Pm4Overlay.Pm4LoadedFiles}/{_worldScene.Pm4Overlay.Pm4TotalFiles} files, {_worldScene.Pm4Overlay.Pm4VisibleObjectCount}/{_worldScene.Pm4Overlay.Pm4ObjectCount} objects, {_worldScene.Pm4Overlay.Pm4VisibleLineCount}/{_worldScene.Pm4Overlay.Pm4LineCount} lines, {_worldScene.Pm4Overlay.Pm4VisibleTriangleCount}/{_worldScene.Pm4Overlay.Pm4TriangleCount} tris, {_worldScene.Pm4Overlay.Pm4VisiblePositionRefCount}/{_worldScene.Pm4Overlay.Pm4PositionRefCount} refs");
        else
            ImGui.TextDisabled("Toggle PM4 Overlay to lazy-load navmesh debug data.");

        if (_worldScene.Pm4Overlay.Pm4LoadAttempted)
        {
            int totalMsur = _worldScene.Pm4Overlay.Pm4TotalMsurCount;
            int shortIdx = _worldScene.Pm4Overlay.Pm4DroppedShortIndexCount;
            int oorMsvi = _worldScene.Pm4Overlay.Pm4DroppedOutOfRangeMsviCount;
            int emptyComp = _worldScene.Pm4Overlay.Pm4DroppedEmptyComponentCount;
            int longEdge = _worldScene.Pm4Overlay.Pm4RejectedLongEdges;
            int keptSurfaces = totalMsur - shortIdx - oorMsvi - emptyComp;
            ImGui.TextColored(new Vector4(1.0f, 0.85f, 0.35f, 1.0f),
                $"MSUR: {totalMsur} raw | kept: {keptSurfaces} | dropped: short-index={shortIdx}, out-of-range={oorMsvi}, empty={emptyComp}, long-edge-lines={longEdge}");
            ImGui.TextDisabled($"Status: {_worldScene.Pm4Overlay.Pm4Status}");
        }

        ImGui.TextDisabled($"Overlay Align: T=({_worldScene.Pm4Overlay.Pm4OverlayTranslation.X:F2}, {_worldScene.Pm4Overlay.Pm4OverlayTranslation.Y:F2}, {_worldScene.Pm4Overlay.Pm4OverlayTranslation.Z:F2}) Rot=({_worldScene.Pm4Overlay.Pm4OverlayRotationDegrees.X:F2}, {_worldScene.Pm4Overlay.Pm4OverlayRotationDegrees.Y:F2}, {_worldScene.Pm4Overlay.Pm4OverlayRotationDegrees.Z:F2})° S=({_worldScene.Pm4Overlay.Pm4OverlayScale.X:F3}, {_worldScene.Pm4Overlay.Pm4OverlayScale.Y:F3}, {_worldScene.Pm4Overlay.Pm4OverlayScale.Z:F3})");

        DrawPm4ColorLegend("WorkbenchOverlay");
    }

    internal void DrawPm4SelectionWorkbenchContent()
    {
        if (_worldScene == null)
            return;

        if (!_worldScene.Pm4Overlay.HasSelectedPm4Object || !_worldScene.Pm4Overlay.SelectedPm4ObjectKey.HasValue)
        {
            ImGui.TextDisabled("No PM4 object selected. Left-click PM4 geometry to inspect one object at a time.");
            DrawPm4ObjectCollectionSummary("WorkbenchSelection");
            if (ImGui.Button("Open Data I/O"))
                OpenWorkbenchTab(WorkbenchTab.Editor, 2); // Spec 231 D1: exports live on the Data I/O page
            return;
        }

        int requestedMatches = _pm4ObjectMatchMaxMatchesPerObject;
        ImGui.SetNextItemWidth(130f);
        if (ImGui.SliderInt("Top Matches", ref requestedMatches, 3, 5))
            _pm4ObjectMatchMaxMatchesPerObject = Math.Clamp(requestedMatches, 3, 5);

        ImGui.SameLine();
        if (ImGui.Button("Open Advanced Align"))
            _activePm4TabIndex = (int)Pm4BottomTab.Alignment;

        ImGui.SameLine();
        if (ImGui.Button("Save Overlay Align"))
            SaveCurrentPm4Alignment();

        if (ImGui.CollapsingHeader("Selected PM4", ImGuiTreeNodeFlags.DefaultOpen))
        {
            var selectedPm4 = _worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value;
            ImGui.Text($"tile ({selectedPm4.tileX}, {selectedPm4.tileY}) CK24=0x{selectedPm4.ck24:X6} part={selectedPm4.objectPart}");
            ImGui.TextDisabled("part = viewer-generated split id from the current overlay build, not a raw PM4 field.");

            if (_worldScene.Pm4Overlay.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo debugInfo))
            {
                // MSUR._0x1C is the producing placement's Z as an IEEE-754 float, not a packed key
                // (bit-exact for 93.58% of objects; see docs/wowdev-wiki/pm4-pd4-draft.md). Show it
                // as a height, and show the resolved placement where the object library can name it.
                // The stored Ck24 is the top 24 bits, so this reconstruction carries ~0.003%
                // relative error and is displayed as approximate.
                float placementZ = BitConverter.UInt32BitsToSingle(debugInfo.Ck24 << 8);
                ImGui.TextColored(new Vector4(0.55f, 0.85f, 0.55f, 1f),
                    $"Placement Z ~= {placementZ:F3}   (MSUR._0x1C as float; 'CK24' is a slice of it)");
                ImGui.TextDisabled($"Surfaces={debugInfo.SurfaceCount}  bounds Z {debugInfo.BoundsMin.Z:F2}..{debugInfo.BoundsMax.Z:F2}  centre ({debugInfo.Center.X:F1}, {debugInfo.Center.Y:F1}, {debugInfo.Center.Z:F1})");
                ImGui.TextDisabled($"Raw slice 0x{debugInfo.Ck24:X6} (exponent band 0x{debugInfo.Ck24Type:X2} - NOT a type)");
                ImGui.TextDisabled($"MSHD F00={debugInfo.MshdField00} Region={debugInfo.MshdRegionId} F08={debugInfo.MshdField08}");
                ImGui.TextDisabled($"Group=0x{debugInfo.DominantGroupKey:X2} Attr=0x{debugInfo.DominantAttributeMask:X2} MscnRef={debugInfo.DominantMscnRefIndex} AvgH={debugInfo.AverageSurfaceHeight:F2}");
                ImGui.TextDisabled($"MSLKGroup=0x{debugInfo.LinkGroupObjectId:X8} Linked MPRL refs={debugInfo.LinkedPositionRefCount}");
                if (debugInfo.DistinctTypeFlags != 0)
                {
                    var tf = new List<string>();
                    for (int bit = 1; bit < 32; bit++)
                        if ((debugInfo.DistinctTypeFlags & (1u << bit)) != 0)
                            tf.Add(bit switch { 0x03 => "m2-top", 0x10 => "floor-int", 0x12 => "ext-solid", _ => $"0x{bit:X2}" });
                    byte gk = debugInfo.DominantGroupKey;
                    bool match = (debugInfo.DistinctTypeFlags & (1u << gk)) != 0;
                    string gkl = gk switch { 0x03 => "m2-surf", 0x10 => "floor-int", 0x12 => "ext-solid", 0x13 => "portal-int", _ => $"0x{gk:X2}" };
                    ImGui.TextDisabled($"GroupKey={gkl} TypeFlags: {string.Join(", ", tf)} {(match ? "MATCH" : "MISMATCH")}");
                }
                else
                {
                    byte gk = debugInfo.DominantGroupKey;
                    string gkl = gk switch { 0x03 => "m2-surf", 0x10 => "floor-int", 0x12 => "ext-solid", 0x13 => "portal-int", _ => $"0x{gk:X2}" };
                    ImGui.TextDisabled($"GroupKey={gkl} TypeFlags: none");
                }
            }

            ImGui.TextDisabled($"Tile layer align: T=({_worldScene.Pm4Overlay.SelectedPm4Ck24LayerTranslation.X:F2}, {_worldScene.Pm4Overlay.SelectedPm4Ck24LayerTranslation.Y:F2}, {_worldScene.Pm4Overlay.SelectedPm4Ck24LayerTranslation.Z:F2}) Rot=({_worldScene.Pm4Overlay.SelectedPm4Ck24LayerRotationDegrees.X:F2}, {_worldScene.Pm4Overlay.SelectedPm4Ck24LayerRotationDegrees.Y:F2}, {_worldScene.Pm4Overlay.SelectedPm4Ck24LayerRotationDegrees.Z:F2})° S=({_worldScene.Pm4Overlay.SelectedPm4Ck24LayerScale.X:F3}, {_worldScene.Pm4Overlay.SelectedPm4Ck24LayerScale.Y:F3}, {_worldScene.Pm4Overlay.SelectedPm4Ck24LayerScale.Z:F3})");
            ImGui.TextDisabled($"Object align: T=({_worldScene.Pm4Overlay.SelectedPm4ObjectTranslation.X:F2}, {_worldScene.Pm4Overlay.SelectedPm4ObjectTranslation.Y:F2}, {_worldScene.Pm4Overlay.SelectedPm4ObjectTranslation.Z:F2}) Rot=({_worldScene.Pm4Overlay.SelectedPm4ObjectRotationDegrees.X:F2}, {_worldScene.Pm4Overlay.SelectedPm4ObjectRotationDegrees.Y:F2}, {_worldScene.Pm4Overlay.SelectedPm4ObjectRotationDegrees.Z:F2})° S=({_worldScene.Pm4Overlay.SelectedPm4ObjectScale.X:F3}, {_worldScene.Pm4Overlay.SelectedPm4ObjectScale.Y:F3}, {_worldScene.Pm4Overlay.SelectedPm4ObjectScale.Z:F3})");

            if (ImGui.Button("Clear PM4 Selection"))
                _worldScene.Pm4Overlay.ClearPm4ObjectSelection();
            ImGui.SameLine();
            if (ImGui.Button("Open Data I/O"))
                OpenWorkbenchTab(WorkbenchTab.Editor, 2); // Spec 231 D1: exports live on the Data I/O page

            // Inline WMO match button when a WMO-type object is selected
            if (debugInfo.Ck24Type is 0x42 or 0x43)
            {
                ImGui.Separator();
                ImGui.TextColored(new Vector4(0.8f, 0.9f, 1f, 1f), "WMO Detection");

                string? clientRoot = _dataSourceSession.GetActiveGamePath();
                if (!string.IsNullOrWhiteSpace(clientRoot) && _dataSourceSession.GetCurrentSessionMapName() != null)
                {
                    string matchKey = Pm4WmoGroupMatchService.GetMatchKey(
                        _dataSourceSession.GetCurrentSessionMapName()!, debugInfo.TileX, debugInfo.TileY, debugInfo.Ck24);

                    if (_pm4WmoMatchStore == null)
                        _pm4WmoMatchStore = new Pm4WmoMatchStore(AppContext.BaseDirectory);
                    if (_pm4WmoMatchEntries.Count == 0)
                        _pm4WmoMatchEntries = _pm4WmoMatchStore.Load();

                    bool hasSaved = _pm4WmoMatchEntries.TryGetValue(matchKey, out var saved);
                    if (hasSaved && saved != null)
                    {
                        ImGui.TextColored(new Vector4(0.95f, 0.85f, 0.35f, 1f), $"Matched: {saved.ModelName}");
                        ImGui.TextDisabled(saved.WmoPath);
                        ImGui.SameLine();
                        if (ImGui.SmallButton("Clear"))
                        {
                            _pm4WmoMatchEntries.Remove(matchKey);
                            _pm4WmoMatchStore.Save(_pm4WmoMatchEntries);
                            _pm4WmoGroupMatchResult = null;
                        }
                    }
                    else
                    {
                        ImGui.TextDisabled("No saved WMO match for this CK24.");
                    }

                    ImGui.Spacing();
                    if (ImGui.Button("Find WMO Match", new Vector2(180f, 28f)))
                    {
                        var clusters = _worldScene.Pm4Overlay.GetPm4SurfaceGroupClusters(
                            debugInfo.TileX, debugInfo.TileY, debugInfo.Ck24);
                        _pm4WmoGroupMatchResult = Pm4WmoGroupMatchService.MatchFromPlacement(
                            clientRoot, _dataSourceSession.GetCurrentSessionMapName()!,
                            debugInfo.TileX, debugInfo.TileY, debugInfo.Ck24,
                            debugInfo.BoundsMin, debugInfo.BoundsMax, clusters);
                        _pm4WmoMatchStatus = _pm4WmoGroupMatchResult.ErrorMessage ?? "";
                    }
                    ImGui.SameLine();
                    if (ImGui.Button("Shape Search", new Vector2(120f, 28f)))
                    {
                        // Mine the map's own placement ADTs first (split _obj0 and monolithic LK
                        // root files alike — the Museum corpus is hand-made ground truth about
                        // what fits), then fall back to raw client WMO geometry.
                        var adtMatches = _dataSourceSession.GetCurrentSessionMapName() is { Length: > 0 } shapeMapName
                            ? Pm4WmoGroupMatchService.SearchAdtPlacementsByShape(
                                clientRoot, shapeMapName, debugInfo.BoundsMin, debugInfo.BoundsMax)
                            : Array.Empty<Pm4WmoFallbackCandidate>();
                        var clientMatches = Pm4WmoGroupMatchService.SearchWmoByShape(
                            clientRoot, debugInfo.BoundsMin, debugInfo.BoundsMax);
                        var merged = adtMatches
                            .Concat(clientMatches)
                            .GroupBy(c => c.ModelPath, StringComparer.OrdinalIgnoreCase)
                            .Select(g => g.First())
                            .ToList();

                        _pm4WmoGroupMatchResult = _pm4WmoGroupMatchResult != null
                            ? new Pm4WmoMatchResult(
                                _pm4WmoGroupMatchResult.HasAdtData,
                                _pm4WmoGroupMatchResult.Placements,
                                merged)
                            : new Pm4WmoMatchResult(false, Array.Empty<Pm4WmoPlacementResult>(), merged);
                    }

                    if (!string.IsNullOrWhiteSpace(_pm4WmoMatchStatus))
                        ImGui.TextColored(new Vector4(1f, 0.7f, 0.3f, 1f), _pm4WmoMatchStatus);

                    // Show quick match results summary inline
                    if (_pm4WmoGroupMatchResult != null)
                    {
                        int placements = _pm4WmoGroupMatchResult.Placements.Count;
                        int fallbacks = _pm4WmoGroupMatchResult.FallbackCandidates.Count;
                        ImGui.TextDisabled($"Placements: {placements}  Fallback candidates: {fallbacks}");
                    }
                }
                else
                {
                    ImGui.TextDisabled("Load a game folder to enable WMO matching.");
                }
            }
        }

        DrawSelectedPm4RegionSummary("WorkbenchSelectedRegion");

        if (ImGui.CollapsingHeader("Match details"))
            DrawPm4WmoGroupMatchDetail();
    }

    private void DrawPm4WmoGroupMatchDetail()
    {
        if (_pm4WmoGroupMatchResult == null)
        {
            ImGui.TextDisabled("No match results. Click 'Find WMO Match' or 'Shape Search' above.");
            return;
        }

        int tileX = _worldScene?.Pm4Overlay.SelectedPm4ObjectKey?.tileX ?? 0;
        int tileY = _worldScene?.Pm4Overlay.SelectedPm4ObjectKey?.tileY ?? 0;
        uint ck24 = _worldScene?.Pm4Overlay.SelectedPm4ObjectKey?.ck24 ?? 0;
        var clusters = _worldScene?.Pm4Overlay.GetPm4SurfaceGroupClusters(tileX, tileY, ck24) ?? Array.Empty<Pm4SurfaceGroupCluster>();
        string? mapName = _dataSourceSession.GetCurrentSessionMapName();
        string matchKey = mapName != null
            ? Pm4WmoGroupMatchService.GetMatchKey(mapName, tileX, tileY, ck24)
            : "";
        Pm4WmoMatchEntry? savedEntry = null;
        bool hasSavedMatch = !string.IsNullOrWhiteSpace(matchKey)
            && _pm4WmoMatchEntries.TryGetValue(matchKey, out savedEntry);

        // Surface clusters summary
        if (clusters.Count > 0)
        {
            ImGui.Spacing();
            ImGui.Text($"PM4 Groups ({clusters.Count} group(s)):");
            ImGui.Separator();
            for (int ci = 0; ci < clusters.Count; ci++)
            {
                var cluster = clusters[ci];
                string gkLabel = cluster.GroupKey switch
                {
                    0x03 => "M2 surf",
                    0x10 => "Floor",
                    0x12 => "Exterior",
                    0x13 => "Portal",
                    _ => $"0x{cluster.GroupKey:X2}"
                };
                ImGui.TextDisabled($"  GroupKey {gkLabel}: {cluster.SurfaceCount} surfaces, bounds=({cluster.BoundsMin.X:F1},{cluster.BoundsMin.Y:F1},{cluster.BoundsMin.Z:F1})..({cluster.BoundsMax.X:F1},{cluster.BoundsMax.Y:F1},{cluster.BoundsMax.Z:F1})");
            }
            ImGui.Spacing();
        }

        // ADT placements
        if (_pm4WmoGroupMatchResult.Placements.Count == 0 && !_pm4WmoGroupMatchResult.HasAdtData)
        {
            ImGui.TextDisabled("No ADT placement data found.");
        }
        else if (_pm4WmoGroupMatchResult.Placements.Count == 0)
        {
            ImGui.TextDisabled("ADT data found but no overlapping WMO placements.");
        }

        foreach (var placement in _pm4WmoGroupMatchResult.Placements)
        {
            string headerLabel = $"{placement.ModelName} [uid={placement.UniqueId}]  ({placement.WmoGroupCount} groups)";
            bool isSaved = savedEntry != null
                && string.Equals(savedEntry.WmoPath, placement.ModelPath, StringComparison.OrdinalIgnoreCase);

            if (isSaved)
                ImGui.PushStyleColor(ImGuiCol.Text, new Vector4(0.95f, 0.85f, 0.35f, 1f));

            if (ImGui.TreeNodeEx($"##Placement_{placement.UniqueId}", ImGuiTreeNodeFlags.DefaultOpen, headerLabel))
            {
                if (isSaved)
                    ImGui.PopStyleColor();

                ImGui.TextDisabled($"Path: {placement.ModelPath}");
                ImGui.TextDisabled($"Position: ({placement.PlacementPosition.X:F1}, {placement.PlacementPosition.Y:F1}, {placement.PlacementPosition.Z:F1})");
                ImGui.TextDisabled($"World bounds: ({placement.WorldBoundsMin.X:F1},{placement.WorldBoundsMin.Y:F1},{placement.WorldBoundsMin.Z:F1}) .. ({placement.WorldBoundsMax.X:F1},{placement.WorldBoundsMax.Y:F1},{placement.WorldBoundsMax.Z:F1})");
                ImGui.TextDisabled($"WMO overall bounds: ({placement.WmoBoundsMin.X:F1},{placement.WmoBoundsMin.Y:F1},{placement.WmoBoundsMin.Z:F1}) .. ({placement.WmoBoundsMax.X:F1},{placement.WmoBoundsMax.Y:F1},{placement.WmoBoundsMax.Z:F1})");

                // Confirm match button
                if (!isSaved)
                {
                    if (ImGui.SmallButton($"Confirm Match##{placement.UniqueId}"))
                    {
                        _pm4WmoMatchEntries[matchKey] = new Pm4WmoMatchEntry
                        {
                            MapName = mapName,
                            TileX = tileX,
                            TileY = tileY,
                            Ck24 = ck24,
                            WmoPath = placement.ModelPath,
                            ModelName = placement.ModelName,
                            Source = "manual",
                        };
                        _pm4WmoMatchStore.Save(_pm4WmoMatchEntries);
                        _pm4WmoMatchStatus = $"Saved match: {placement.ModelName}";
                    }
                }

                // Group match table
                if (placement.GroupMatches.Count > 0)
                {
                    ImGui.Spacing();
                    ImGui.Text("Group overlap (Jaccard):");

                    if (ImGui.BeginTable("##GroupMatchTable", 7,
                        ImGuiTableFlags.BordersV | ImGuiTableFlags.BordersOuterH | ImGuiTableFlags.RowBg))
                    {
                        ImGui.TableSetupColumn("PM4 GK", ImGuiTableColumnFlags.WidthFixed, 60f);
                        ImGui.TableSetupColumn("WMO Grp", ImGuiTableColumnFlags.WidthFixed, 50f);
                        ImGui.TableSetupColumn("Flags", ImGuiTableColumnFlags.WidthFixed, 50f);
                        ImGui.TableSetupColumn("Overlap", ImGuiTableColumnFlags.WidthFixed, 60f);
                        ImGui.TableSetupColumn("PM4 Surfaces", ImGuiTableColumnFlags.WidthFixed, 80f);
                        ImGui.TableSetupColumn("WMO Bounds");
                        ImGui.TableSetupColumn("PM4 Bounds");
                        ImGui.TableHeadersRow();

                        foreach (var match in placement.GroupMatches)
                        {
                            ImGui.TableNextRow();
                            ImGui.TableNextColumn();
                            string gkLabel = match.Pm4GroupKey switch
                            {
                                0x03 => "M2",
                                0x10 => "Floor",
                                0x12 => "Ext",
                                0x13 => "Portal",
                                _ => $"0x{match.Pm4GroupKey:X2}"
                            };
                            ImGui.TextDisabled(gkLabel);

                            ImGui.TableNextColumn();
                            ImGui.TextDisabled($"#{match.WmoGroupIndex}");

                            ImGui.TableNextColumn();
                            ImGui.TextDisabled($"0x{match.WmoGroupFlags:X}");

                            ImGui.TableNextColumn();
                            float overlap = match.JaccardOverlap;
                            Vector4 color = overlap >= 0.8f
                                ? new Vector4(0.3f, 1f, 0.3f, 1f)
                                : overlap >= 0.4f
                                    ? new Vector4(1f, 0.85f, 0.3f, 1f)
                                    : new Vector4(1f, 0.5f, 0.5f, 1f);
                            ImGui.TextColored(color, $"{overlap:P1}");

                            ImGui.TableNextColumn();
                            ImGui.TextDisabled($"{match.Pm4SurfaceCount}");

                            ImGui.TableNextColumn();
                            ImGui.TextDisabled($"({match.WmoBoundsMin.X:F0},{match.WmoBoundsMin.Y:F0},{match.WmoBoundsMin.Z:F0})..({match.WmoBoundsMax.X:F0},{match.WmoBoundsMax.Y:F0},{match.WmoBoundsMax.Z:F0})");

                            ImGui.TableNextColumn();
                            ImGui.TextDisabled($"({match.Pm4BoundsMin.X:F0},{match.Pm4BoundsMin.Y:F0},{match.Pm4BoundsMin.Z:F0})..({match.Pm4BoundsMax.X:F0},{match.Pm4BoundsMax.Y:F0},{match.Pm4BoundsMax.Z:F0})");
                        }

                        ImGui.EndTable();
                    }
                }

                ImGui.TreePop();
            }
            else if (isSaved)
            {
                ImGui.PopStyleColor();
            }
        }

        // Fallback shape search candidates
        if (_pm4WmoGroupMatchResult.FallbackCandidates.Count > 0)
        {
            ImGui.Spacing();
            if (ImGui.TreeNodeEx("Fallback Shape Candidates", ImGuiTreeNodeFlags.DefaultOpen))
            {
                if (ImGui.BeginTable("##FallbackTable", 5,
                    ImGuiTableFlags.BordersV | ImGuiTableFlags.BordersOuterH | ImGuiTableFlags.RowBg))
                {
                    ImGui.TableSetupColumn("WMO", ImGuiTableColumnFlags.WidthStretch);
                    ImGui.TableSetupColumn("Volume", ImGuiTableColumnFlags.WidthFixed, 55f);
                    ImGui.TableSetupColumn("Footprint", ImGuiTableColumnFlags.WidthFixed, 60f);
                    ImGui.TableSetupColumn("Span", ImGuiTableColumnFlags.WidthFixed, 50f);
                    ImGui.TableSetupColumn("Score", ImGuiTableColumnFlags.WidthFixed, 55f);
                    ImGui.TableHeadersRow();

                    foreach (var fb in _pm4WmoGroupMatchResult.FallbackCandidates)
                    {
                        ImGui.TableNextRow();
                        ImGui.TableNextColumn();
                        ImGui.TextDisabled(fb.ModelName);

                        ImGui.TableNextColumn();
                        ImGui.TextDisabled($"{fb.VolumeRatio:P0}");

                        ImGui.TableNextColumn();
                        ImGui.TextDisabled($"{fb.FootprintRatio:P0}");

                        ImGui.TableNextColumn();
                        ImGui.TextDisabled($"{fb.SpanRatio:P0}");

                        ImGui.TableNextColumn();
                        float score = fb.CombinedScore;
                        Vector4 sc = score >= 0.7f
                            ? new Vector4(0.3f, 1f, 0.3f, 1f)
                            : score >= 0.4f
                                ? new Vector4(1f, 0.85f, 0.3f, 1f)
                                : new Vector4(1f, 0.5f, 0.5f, 1f);
                        ImGui.TextColored(sc, $"{score:P1}");
                    }

                    ImGui.EndTable();
                }

                ImGui.TreePop();
            }
        }
    }


    internal void DrawPerfWindow()
    {
        // 069 Phase 16: wrapper keeps legacy floating-window behavior.
        // Workbench sub-tab uses DrawPerfContent directly.
        ImGui.SetNextWindowSize(new Vector2(360, 0), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("Perf", ref _showPerfWindow, ImGuiWindowFlags.AlwaysAutoResize))
        {
            ImGui.End();
            return;
        }
        DrawPerfContent();
        ImGui.End();
    }

    /// <summary>
    /// Utilities &gt; Perf. Frame timing over time is the primary content here: this is where anyone
    /// chasing a stutter looks first. Memory/GC and asset counters stay on Runtime Stats so the two
    /// pages do not duplicate each other.
    /// </summary>
    internal void DrawPerfContent()
    {
        // Frame history first, and outside the terrain guard: frame timing is meaningful whenever a
        // world is loaded, not only when a terrain renderer exists.
        DrawFrameHistoryContent();

        var terrainRenderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (terrainRenderer == null)
        {
            if (_worldScene == null)
                ImGui.TextDisabled("Load a world to see frame timing and terrain stats.");
            return;
        }

        ImGui.Separator();
        ImGui.Text($"Chunks: {terrainRenderer.ChunksRendered} rendered, {terrainRenderer.ChunksCulled} culled");
        ImGui.TextDisabled("Chunk counts are for the last terrain Render() call.");
    }

    // -- PM4 outliner: Region -> Tile -> Object -------------------------------
    private string _pm4OutlinerFilter = string.Empty;
    private bool _pm4OutlinerNamedOnly;
    private IReadOnlyList<Pm4OutlineRegion>? _pm4OutlineCache;

    /// <summary>
    /// Hierarchical PM4 scene graph, grouped by MSHD region.
    /// </summary>
    /// <remarks>
    /// Objects are labelled by the placed asset that produced them where that resolves, which is
    /// possible because MSUR._0x1C is the producing placement's Z rather than an opaque key.
    /// Unresolved objects are still listed, labelled by value - an outliner that hides what it
    /// cannot name is not an inventory.
    /// </remarks>
    private Pm4SceneFacts? _pm4SceneFacts;

    /// <summary>
    /// Live PM4 measurements for the loaded scene, so the corpus findings are checkable here.
    /// </summary>
    private void DrawPm4SceneFacts()
    {
        if (_worldScene == null)
        {
            ImGui.TextDisabled("Load a world to measure PM4 objects.");
            return;
        }

        if (ImGui.Button("Measure loaded PM4##Pm4Facts"))
            _pm4SceneFacts = _worldScene.Pm4Overlay.BuildPm4SceneFacts();
        ImGui.SameLine();
        ImGui.TextDisabled("counts every visible PM4 object; respects the class filters above");

        if (_pm4SceneFacts is not { } f)
            return;

        ImGui.Separator();
        ImGui.TextUnformatted($"objects {f.Objects}    with a placement height {f.ObjectsWithHeight}    without {f.ObjectsWithoutHeight}");
        ImGui.TextUnformatted($"resolved to a placed asset: {f.ObjectsResolvedToAsset}");
        ImGui.Separator();

        // The cross-check: MSUR._0x00 == 0x03 and MSUR._0x1C == 0 are unrelated fields that pick out
        // the same objects. Corpus-wide they disagree once in 1,929.
        if (f.ClassHeightDisagreements == 0)
        {
            ImGui.TextColored(new Vector4(0.45f, 0.9f, 0.5f, 1f),
                "class and height agree on every object");
        }
        else
        {
            ImGui.TextColored(new Vector4(1.0f, 0.7f, 0.25f, 1f),
                $"{f.ClassHeightDisagreements} object(s) where class and height DISAGREE");
            if (ImGui.IsItemHovered())
            {
                ImGui.BeginTooltip();
                ImGui.TextUnformatted("Class 0x03 should never carry a placement height, and every");
                ImGui.TextUnformatted("other class should always carry one. Corpus-wide these two");
                ImGui.TextUnformatted("unrelated fields disagree on 1 object in 1,929.");
                ImGui.TextUnformatted("Anything counted here is worth inspecting directly.");
                ImGui.EndTooltip();
            }
        }

        ImGui.Separator();
        ImGui.TextDisabled("surface class      with height   without");
        foreach (Pm4SceneClassFact c in f.Classes)
        {
            bool expectedDoodad = c.SurfaceClass == 0x03;
            bool clean = expectedDoodad ? c.WithHeight == 0 : c.WithoutHeight == 0;
            Vector4 colour = clean
                ? new Vector4(0.72f, 0.76f, 0.84f, 1f)
                : new Vector4(1.0f, 0.7f, 0.25f, 1f);
            SceneHoverAndPickService.TextColoredUnformatted(colour, $"  0x{c.SurfaceClass:X2}{(expectedDoodad ? " (doodad)" : "")}".PadRight(20)
                + $"{c.WithHeight,10}   {c.WithoutHeight,7}");
        }
    }

    internal void DrawPm4Outliner()
    {
        if (_worldScene == null)
        {
            ImGui.TextDisabled("Load a world to use the PM4 outliner.");
            return;
        }

        ImGui.SetNextItemWidth(240f);
        ImGui.InputText("Filter##Pm4Outliner", ref _pm4OutlinerFilter, 128);
        ImGui.SameLine();
        ImGui.Checkbox("Named only##Pm4Outliner", ref _pm4OutlinerNamedOnly);
        ImGui.SameLine();
        if (ImGui.Button("Refresh##Pm4Outliner"))
            _pm4OutlineCache = null;

        // Built on demand and cached: the asset resolve is O(objects x WMO instances), which is far
        // too much to repeat every frame. Cleared by InvalidatePm4DerivedReports when the PM4 data
        // behind it changes.
        _pm4OutlineCache ??= _worldScene.Pm4Overlay.BuildPm4Outline();
        IReadOnlyList<Pm4OutlineRegion> regions = _pm4OutlineCache;

        if (ImGui.CollapsingHeader("Scene measurements##Pm4Facts"))
            DrawPm4SceneFacts();
        ImGui.Separator();
        int totalObjects = regions.Sum(static r => r.ObjectCount);
        int totalNamed = regions.Sum(static r => r.NamedObjectCount);
        ImGui.TextDisabled($"{regions.Count} regions | {totalObjects} objects | {totalNamed} resolved to an asset");
        ImGui.Separator();

        string filter = _pm4OutlinerFilter.Trim();
        bool hasFilter = filter.Length > 0;

        if (!ImGui.BeginChild("##Pm4OutlinerTree", Vector2.Zero, true))
        {
            ImGui.EndChild();
            return;
        }

        foreach (Pm4OutlineRegion region in regions)
        {
            if (!ImGui.TreeNodeEx(
                    $"Region {region.RegionId}##pm4region{region.RegionId}",
                    ImGuiTreeNodeFlags.SpanAvailWidth))
            {
                continue;
            }

            ImGui.SameLine();
            ImGui.TextDisabled($"  {region.ObjectCount} objects / {region.Tiles.Count} tiles / {region.NamedObjectCount} named");

            foreach (Pm4OutlineTile tile in region.Tiles)
            {
                var shown = tile.Objects
                    .Where(o => !_pm4OutlinerNamedOnly || o.AssetName is not null)
                    .Where(o => !hasFilter
                        || (o.AssetName?.Contains(filter, StringComparison.OrdinalIgnoreCase) ?? false)
                        || $"0x{o.Ck24:X6}".Contains(filter, StringComparison.OrdinalIgnoreCase))
                    .ToList();

                if (shown.Count == 0)
                    continue;

                if (!ImGui.TreeNodeEx(
                        $"Tile ({tile.TileX}, {tile.TileY})##pm4tile{region.RegionId}_{tile.TileX}_{tile.TileY}",
                        ImGuiTreeNodeFlags.SpanAvailWidth))
                {
                    continue;
                }

                ImGui.SameLine();
                ImGui.TextDisabled($"  {shown.Count} objects");

                foreach (Pm4OutlineObject obj in shown)
                {
                    string label = obj.AssetName is not null
                        ? $"{System.IO.Path.GetFileName(obj.AssetName)}  #{obj.UniqueId}"
                        : $"(unresolved) z={obj.PlacementZ:F3}  0x{obj.Ck24:X6}";

                    bool selected = _worldScene.Pm4Overlay.SelectedPm4ObjectKey.HasValue
                        && _worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value.Equals(obj.Key);

                    if (ImGui.Selectable(
                            $"{label}##pm4obj{region.RegionId}_{obj.Key.tileX}_{obj.Key.tileY}_{obj.Ck24}_{obj.Key.objectPart}",
                            selected))
                    {
                        _worldScene.Pm4Overlay.SelectPm4Object(obj.Key);
                    }

                    if (ImGui.IsItemHovered())
                    {
                        ImGui.BeginTooltip();
                        ImGui.Text(obj.AssetName ?? "(no asset resolved)");
                        ImGui.TextDisabled($"placement Z = {obj.PlacementZ:F4}   surfaces = {obj.SurfaceCount}");
                        ImGui.TextDisabled($"CK24 slice 0x{obj.Ck24:X6}  type 0x{obj.Ck24Type:X2}  part {obj.Key.objectPart}");
                        if (obj.MatchDelta.HasValue)
                            ImGui.TextDisabled($"asset match delta = {obj.MatchDelta.Value:F4}");
                        ImGui.EndTooltip();
                    }

                    if (ImGui.IsItemHovered() && ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                        FocusCameraOnBounds(obj.BoundsMin, obj.BoundsMax);
                }

                ImGui.TreePop();
            }

            ImGui.TreePop();
        }

        ImGui.EndChild();
    }
}
