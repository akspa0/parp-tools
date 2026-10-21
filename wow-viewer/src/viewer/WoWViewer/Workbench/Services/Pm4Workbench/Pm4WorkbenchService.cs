using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Globalization;
using ImGuiNET;
using WoWViewer.Logging;
using WoWViewer.Terrain;
using WoWViewer.Workbench;
using WoWViewer.UI;
using MslkEntry = WowViewer.Core.PM4.Models.Pm4MslkEntry;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

/// <summary>
/// PM4 inspection: overlay, selection, info, outliner, object groups, and inspection exports.
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

    private enum InspectionPage { Overlay, Selection, Info, Outliner, ObjectGroups }
    private static readonly string[] InspectionPageLabels = ["Overlay", "Selection", "Info", "Outliner", "Object Groups"];

    private string _loosePm4InputPath = string.Empty;

    /// <summary>Opens the shared path picker for a loose PM4/PD4 file (no typed path).</summary>
    private void RequestLoosePm4File()
    {
        ImGuiPathPicker.Instance.Open(
            "Select Loose PM4 / PD4 File",
            pickFolder: false,
            initialPath: string.IsNullOrWhiteSpace(_loosePm4InputPath) ? null : _loosePm4InputPath,
            filterExtension: ".pm4;.pd4",
            path =>
            {
                try
                {
                    if (!System.IO.File.Exists(path))
                        throw new System.IO.FileNotFoundException("Selected file no longer exists.", path);
                    if (_worldScene.Pm4Overlay.LoadLoosePm4File(path))
                    {
                        _loosePm4InputPath = path;
                        _statusMessage = $"Loaded loose PM4/PD4 file: {System.IO.Path.GetFileName(path)}";
                    }
                    else
                    {
                        _statusMessage = $"Failed to decode loose PM4/PD4 file: {path}";
                    }
                }
                catch (Exception ex)
                {
                    _statusMessage = $"Open PM4/PD4 failed - {path}: {ex.Message}";
                }
            });
    }

    internal void OpenPm4Workbench(Pm4WorkbenchTab tab)
    {
        _host.UiManager.OpenViewer(ViewerPanel.PM4);
        _pendingPm4WorkbenchTab = tab;
    }

    internal void DrawPm4WorkbenchInspector()
    {
        if (_worldScene == null)
        {
            ImGui.TextDisabled("PM4 workbench becomes available once a world scene is loaded.");
            return;
        }

        ImGui.TextDisabled("Hover stays lightweight. Click a PM4 object to inspect it here.");
        ImGui.SetNextItemOpen(true, ImGuiCond.Once);
        DrawPm4GlossarySummary();

        if (_pendingPm4WorkbenchTab is { } pending)
        {
            _activePm4TabIndex = pending switch
            {
                Pm4WorkbenchTab.Overlay => (int)InspectionPage.Overlay,
                _ => (int)InspectionPage.Selection,
            };
            _pendingPm4WorkbenchTab = null;
        }

        string[] labels = InspectionPageLabels;
        _activePm4TabIndex = Math.Clamp(_activePm4TabIndex, 0, labels.Length - 1);
        if (!ImGui.BeginTabBar("##Pm4WorkbenchTabs"))
            return;

        for (int index = 0; index < labels.Length; index++)
        {
            ImGuiTabItemFlags flags = _activePm4TabIndex == index
                ? ImGuiTabItemFlags.SetSelected
                : ImGuiTabItemFlags.None;
            bool open = true;
            if (ImGui.BeginTabItem(labels[index], ref open, flags))
            {
                _activePm4TabIndex = index;
                DrawPm4TabContent((InspectionPage)index);
                ImGui.EndTabItem();
            }
        }

        ImGui.EndTabBar();
    }

    private void DrawPm4TabContent(InspectionPage tab)
    {
        switch (tab)
        {
            case InspectionPage.Overlay:
                DrawPm4OverlayWorkbenchContent();
                break;
            case InspectionPage.Selection:
                DrawPm4SelectionWorkbenchContent();
                break;
            case InspectionPage.Info:
                DrawPm4InfoPanelContent();
                break;
            case InspectionPage.Outliner:
                DrawPm4Outliner();
                break;
            case InspectionPage.ObjectGroups:
                DrawPm4ObjectGroups();
                break;
        }
    }

    private void DrawPm4GlossarySummary()
    {
        if (!ImGui.CollapsingHeader("PM4 Glossary / Evidence"))
            return;

        ImGui.TextWrapped("PM4/PD4 are server-side navigation-mesh files: walkable surfaces, their adjacency, and what blocks movement. The cyan (MSCN) and magenta (MSPV) cubes are the visible projection of this data. The single authoritative reading is docs/research/pm4-format-reference.md.");
        ImGui.BulletText("MSUR — 32-byte surface record. 0x01 = MSVI index count; 0x02 = MSLK window count; 0x04/0x08/0x0C = normal; 0x10 = plane distance; 0x14 = MSVI window start; 0x18 = MSLK window start; 0x1C = IEEE-754 float, the producing placement's Z.");
        ImGui.BulletText("MSLK — the surface-adjacency graph. msur_index names a neighbouring surface in the tile addressed by 0x0C. 0x01 is a zero-based source-edge ordinal. 0x04 is the object-group id and equals MPRL.0x04. A non-empty MSPI window is a vertical wall quad; an empty one is open passage.");
        ImGui.BulletText("MSCN — ordered connector/boundary node positions. Objects share nodes; nothing in the file indexes MSCN. Visible as the cyan cubes.");
        ImGui.BulletText("MSPV / MSPI — wall-quad vertices and indices on the blocked subset of MSLK links. Visible as the magenta cubes.");
        ImGui.BulletText("MSVI / MSVT — mesh indices and vertex positions. A surface is a polygon fan over MSVI[MSUR.0x14 .. + MSUR.0x01) into MSVT.");
        ImGui.BulletText("MPRL — terrain-contact points. 0x00 is a per-tile region value; 0x04 is the object-group id; position is the only permuted chunk (height is Y).");
        ImGui.BulletText("MSHD — 32-byte header. 0x00/0x08 are clamped world-unit spans; 0x04 == 1 marks a tile with no surfaces; 0x0C–0x1C are zero.");
        ImGui.BulletText("part / ObjectPartId — viewer-generated split id from the current overlay build, not a raw PM4 field.");
        ImGui.BulletText("MSLK Group — dominant MSLK.0x04 in the current viewer object. A grouping hint, not proof of identity.");
        ImGui.BulletText("PM4 Graph — the viewer's decomposition of the selected object, not a literal raw node graph stored in PM4.");
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


        ImGui.Separator();
        ImGui.TextDisabled("Direct PM4/PD4 File Loader (ignores map name and naming rules)");
        if (SharedUiWidgets.ActionButton("Open Loose PM4 / PD4 File..."))
            RequestLoosePm4File();
        if (!string.IsNullOrWhiteSpace(_loosePm4InputPath))
            ImGui.TextDisabled($"Last loaded: {System.IO.Path.GetFileName(_loosePm4InputPath)}");
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

        DrawPm4ColorLegend("WorkbenchOverlay");
    }

    internal void DrawPm4SelectionWorkbenchContent()
    {
        if (_worldScene == null)
            return;

        if (SharedUiWidgets.SectionHeader("PM4 Exports", defaultOpen: false, id: "Pm4InspectionExports"))
            DrawPm4ExportCommandSet();

        if (!_worldScene.Pm4Overlay.HasSelectedPm4Object || !_worldScene.Pm4Overlay.SelectedPm4ObjectKey.HasValue)
        {
            ImGui.TextDisabled("No PM4 object selected. Left-click PM4 geometry to inspect one object at a time.");
            DrawPm4ObjectCollectionSummary("WorkbenchSelection");
            return;
        }

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

            if (ImGui.Button("Clear PM4 Selection"))
                _worldScene.Pm4Overlay.ClearPm4ObjectSelection();
        }

        DrawSelectedPm4RegionSummary("WorkbenchSelectedRegion");
    }

    // -- PM4 outliner: Region -> Tile -> Object -------------------------------
    private string _pm4OutlinerFilter = string.Empty;
    private bool _pm4OutlinerNamedOnly;
    private IReadOnlyList<Pm4OutlineRegion>? _pm4OutlineCache;

    // -- PM4 object groups: MPRL Unk04 (= MSLK.GroupObjectId) per selected tile --
    private string _pm4ObjectGroupFilter = string.Empty;
    private bool _pm4ObjectGroupConsistentOnly;
    private string? _pm4ObjectGroupSourcePath;
    private IReadOnlyList<WowViewer.Core.PM4.Research.Pm4ObjectGroupEntry>? _pm4ObjectGroupCache;

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

    /// <summary>
    /// Lists the MPRL object groups (<c>Unk04</c> = <c>MSLK.GroupObjectId</c>) of the selected PM4
    /// tile, with point/surface counts, ground-surface class, nearest authored model and a
    /// class-consistency flag. Clicking a group focuses the camera on its bounds.
    /// </summary>
    internal void DrawPm4ObjectGroups()
    {
        if (_worldScene == null)
        {
            ImGui.TextDisabled("Load a world to inspect PM4 object groups.");
            return;
        }

        string? sourcePath = _worldScene.Pm4Overlay.GetSelectedPm4SourcePath();
        if (string.IsNullOrWhiteSpace(sourcePath))
        {
            ImGui.TextDisabled("Select a PM4 object in the scene or outliner to inspect its object groups.");
            return;
        }

        ImGui.TextDisabled($"Source: {System.IO.Path.GetFileName(sourcePath)}");
        ImGui.SameLine();
        if (ImGui.Button("Refresh##Pm4ObjectGroups"))
            _pm4ObjectGroupCache = null;

        if (_pm4ObjectGroupCache is null || !string.Equals(_pm4ObjectGroupSourcePath, sourcePath, StringComparison.OrdinalIgnoreCase))
        {
            _pm4ObjectGroupSourcePath = sourcePath;
            _pm4ObjectGroupCache = _worldScene.Pm4Overlay.BuildPm4ObjectGroups(sourcePath);
        }
        IReadOnlyList<WowViewer.Core.PM4.Research.Pm4ObjectGroupEntry> groups = _pm4ObjectGroupCache;

        ImGui.SetNextItemWidth(240f);
        ImGui.InputText("Filter##Pm4ObjectGroups", ref _pm4ObjectGroupFilter, 128);
        ImGui.SameLine();
        ImGui.Checkbox("Consistent only##Pm4ObjectGroups", ref _pm4ObjectGroupConsistentOnly);

        int consistent = groups.Count(static g => g.Consistent);
        ImGui.TextDisabled($"{groups.Count} groups | {consistent} consistent | {groups.Sum(static g => g.PointCount)} points");
        ImGui.Separator();

        if (!ImGui.BeginChild("##Pm4ObjectGroupsList", Vector2.Zero, true))
        {
            ImGui.EndChild();
            return;
        }

        string filter = _pm4ObjectGroupFilter.Trim();
        bool hasFilter = filter.Length > 0;
        foreach (var group in groups)
        {
            if (_pm4ObjectGroupConsistentOnly && !group.Consistent)
                continue;
            if (hasFilter
                && !group.GroupObjectId.ToString().Contains(filter, StringComparison.OrdinalIgnoreCase)
                && !(group.NearestModel?.Contains(filter, StringComparison.OrdinalIgnoreCase) ?? false))
            {
                continue;
            }

            string model = group.NearestModel is null ? "(no model)" : System.IO.Path.GetFileName(group.NearestModel);
            string label = $"#{group.GroupObjectId}  {group.PointCount} pts / {group.SurfaceCount} surf  {group.SurfaceClass} -> {model} @ {group.NearestDistance:F1}";
            if (ImGui.Selectable($"{label}##pm4group{group.GroupObjectId}"))
                FocusCameraOnBounds(group.BoundsMin, group.BoundsMax);

            if (ImGui.IsItemHovered())
            {
                ImGui.BeginTooltip();
                ImGui.Text(group.NearestModel ?? "(no nearest model)");
                ImGui.TextDisabled($"consistent = {group.Consistent}  surface class = {group.SurfaceClass}");
                ImGui.TextDisabled($"centroid = ({group.Centroid.X:F2}, {group.Centroid.Y:F2}, {group.Centroid.Z:F2})");
                ImGui.EndTooltip();
            }
        }

        ImGui.EndChild();
    }

    /// <summary>
    /// Saves the current PM4 selection as a human-curated object → model match. Called from the
    /// object library and from the PM4 Selection tab.
    /// </summary>
}
