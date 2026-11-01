using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Globalization;
using ImGuiNET;
using WoWViewer.Logging;
using WoWViewer.Terrain;
using WoWViewer.Workbench;
using WoWViewer.UI;
using WoWViewer.Terrain.Pm4.UI;
using MslkEntry = WowViewer.Core.PM4.Models.Pm4MslkEntry;
using static WoWViewer.ViewerApp;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;

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
        DrawPm4GlossarySummary();

        _pendingPm4WorkbenchTab = null;

        // 1. Core PM4 Overlay & Geometry Layer Controls
        DrawPm4OverlayWorkbenchContent();

        // 2. Live PM4 Alignment & Magnetization Widget
        DrawPm4AlignmentWidget();

        // 3. Selected PM4 Object Details & Exports
        if (SharedUiWidgets.SectionHeader("Selected PM4 Object Details & Exports", defaultOpen: true, id: "Pm4SelectedInspector"))
        {
            DrawPm4SelectionWorkbenchContent();
        }

        // 4. PM4 Outliner & Hierarchy
        if (SharedUiWidgets.SectionHeader("PM4 Outliner & Hierarchy", defaultOpen: false, id: "Pm4OutlinerWidget"))
        {
            DrawPm4Outliner();
        }

        // 5. PM4 Object Groups
        if (SharedUiWidgets.SectionHeader("PM4 Object Groups", defaultOpen: false, id: "Pm4ObjectGroupsWidget"))
        {
            DrawPm4ObjectGroups();
        }
    }

    private void DrawPm4AlignmentWidget()
    {
        if (_worldScene != null)
            Pm4WorkbenchSidebarService.Instance.DrawAlignmentAndDuplication(_worldScene.Pm4Overlay, _host, _worldScene);
    }

    private void DrawPm4GlossarySummary()
    {
        if (!SharedUiWidgets.SectionHeader("PM4 Glossary & Format Reference", defaultOpen: false, id: "Pm4Glossary"))
            return;

        ImGui.PushTextWrapPos(0f);
        ImGui.TextWrapped(
            "PM4 (per-ADT tile) and PD4 (per-WMO) are server-side navigation-mesh files. " +
            "Authoritative reference: docs/research/pm4-format-reference.md.");
        ImGui.Spacing();

        if (ImGui.TreeNodeEx("Core Surface & Mesh Chunks (MSUR / MSVT / MSVI)##Pm4GlossarySurfaces", ImGuiTreeNodeFlags.DefaultOpen))
        {
            ImGui.TextColored(new Vector4(0.85f, 0.75f, 0.45f, 1f), "MSUR (32 B) - Walkable floor polygon record:");
            ImGui.TextWrapped(
                "  * 0x00 (surface_class): 0x03 = doodad/M2 surface (100% have 0x1C == 0); 0x10..0x15 = placed WMO/object surfaces stratified by relative height.\n" +
                "  * 0x01 (msvi_index_count) & 0x14 (MSVI_first_index): polygon fan MSVI[0x14 .. +0x01) into MSVT.\n" +
                "  * 0x02 (mslk_index_count) & 0x18 (mslk_first_index): contiguous adjacency window MSLK[0x18 .. +0x02).\n" +
                "  * 0x04..0x0C (normal) & 0x10 (plane_distance): plane equation (-dot(normal, v0) in raw MSVT space).\n" +
                "  * 0x1C (placement_z): IEEE-754 float equal to the producing MODF/MDDF placement Position.Z (0.0f = unattributed/merged doodad bucket).");
            ImGui.Spacing();
            ImGui.TextColored(new Vector4(0.85f, 0.75f, 0.45f, 1f), "MSVT (12 B) & MSVI (4 B) - Surface mesh:");
            ImGui.TextWrapped("Surface mesh vertices (C3Vector, YXZ axis order) and uint32 polygon-fan vertex indices.");
            ImGui.TreePop();
        }

        if (ImGui.TreeNodeEx("Adjacency, Wall Quads & Contours (MSLK / MSPV / MSPI / MSCN)##Pm4GlossaryLinks", ImGuiTreeNodeFlags.DefaultOpen))
        {
            ImGui.TextColored(new Vector4(0.85f, 0.75f, 0.45f, 1f), "MSLK (20 B) - Directed surface-adjacency edge:");
            ImGui.TextWrapped(
                "  * 0x00 (flags): bit 0 (0x01, 0x11) = open passage across a shared edge (MSPI_first_index == -1); other values carry a vertical wall window in MSPI/MSPV. Bit 4 marks same-tile-only links.\n" +
                "  * 0x01 (source_edge_index): 0-based source polygon edge ordinal e -> (e, (e + 1) mod count).\n" +
                "  * 0x04 (object_group_id): spatial/object group ID; equals MPRL.0x04 (98.97% match).\n" +
                "  * 0x08..0x0B (MSPI_first_index int24 + MSPI_index_count uint8): window into MSPI.\n" +
                "  * 0x0C (target_tile): packed tile address 0xFFFF0000 | (tileY << 8) | tileX.\n" +
                "  * 0x10 (msur_index): target surface index inside target_tile (100% reciprocal on same-tile links).");
            ImGui.Spacing();
            ImGui.TextColored(new Vector4(0.85f, 0.75f, 0.45f, 1f), "MSPV (12 B) & MSPI (4 B) - Vertical wall quads:");
            ImGui.TextWrapped("Vertical wall-quad vertices (magenta cubes) and indices on blocked/step MSLK edges (98% are 4-index coplanar vertical quads).");
            ImGui.Spacing();
            ImGui.TextColored(new Vector4(0.85f, 0.75f, 0.45f, 1f), "MSCN (12 B) - Boundary contour chains:");
            ImGui.TextWrapped("Ordered boundary/contour node chains (cyan cubes). C3Vector positions (not normals); nothing in the file indexes MSCN directly.");
            ImGui.TreePop();
        }

        if (ImGui.TreeNodeEx("Terrain Contact, Header & Destructibles (MPRL / MPRR / MSHD / MDB*)##Pm4GlossaryTerrain", ImGuiTreeNodeFlags.DefaultOpen))
        {
            ImGui.TextColored(new Vector4(0.85f, 0.75f, 0.45f, 1f), "MPRL (24 B) - Terrain contact points:");
            ImGui.TextWrapped("Terrain-intersection contact points: 0x00 = per-tile region ID; 0x04 = object_group_id (= MSLK.0x04); position = permuted C3Vector with height in Y; 0x14/0x16 = point counter and flags (0xFFFF / 0x3FFF sentinel).");
            ImGui.Spacing();
            ImGui.TextColored(new Vector4(0.85f, 0.75f, 0.45f, 1f), "MPRR (4 B) - Quadtree spatial index:");
            ImGui.TextWrapped("Quadtree spatial index over MPRL in a 768x768 tile grid. Runs have length 4k + 3 (min bound, max bound, MPRL indices, 0xFFFF terminator).");
            ImGui.Spacing();
            ImGui.TextColored(new Vector4(0.85f, 0.75f, 0.45f, 1f), "MSHD (32 B) - Tile Header:");
            ImGui.TextWrapped("Header: 0x00 & 0x08 are clamped world-unit spans (<= 534); 0x04 == 1 marks a tile with no surfaces; 0x0C..0x1C are zero.");
            ImGui.Spacing();
            ImGui.TextColored(new Vector4(0.85f, 0.75f, 0.45f, 1f), "MDBH / MDBI / MDBF / MDOS / MDSF - Destructibles:");
            ImGui.TextWrapped("Destructible-building table and per-surface destruction state links (MDSF maps msur_index -> mdos_index).");
            ImGui.TreePop();
        }

        if (ImGui.TreeNodeEx("Coordinates & Viewer Terms (CK24 / Part / MscnRef)##Pm4GlossaryViewer", ImGuiTreeNodeFlags.DefaultOpen))
        {
            ImGui.TextColored(new Vector4(0.85f, 0.75f, 0.45f, 1f), "Coordinates:");
            ImGui.TextWrapped("MSVT/MSPV/MSCN use YXZ: world = (17066.666 - x, 17066.666 - y, z). MPRL uses ZXY with height in Y: world = (17066.666 - z, 17066.666 - x, y).");
            ImGui.Spacing();
            ImGui.TextColored(new Vector4(0.85f, 0.75f, 0.45f, 1f), "CK24 (0xXXXXXX):");
            ImGui.TextWrapped("Viewer grouping key formed from the top 24 bits of MSUR.0x1C (placement_z float). Its high byte ('Ck24Type', e.g. 0x40..0x43) is the IEEE-754 exponent band of the height, NOT an object type enum.");
            ImGui.Spacing();
            ImGui.TextColored(new Vector4(0.85f, 0.75f, 0.45f, 1f), "part / ObjectPartId:");
            ImGui.TextWrapped("Viewer-generated connected-component split ID within a tile + CK24 bucket, not a raw PM4 field.");
            ImGui.Spacing();
            ImGui.TextColored(new Vector4(0.85f, 0.75f, 0.45f, 1f), "MscnRef (legacy label):");
            ImGui.TextWrapped("Older viewer code called MSUR.0x18 'MscnRef'; measured format analysis proved MSUR.0x18 is mslk_first_index (start of the surface's MSLK window).");
            ImGui.TreePop();
        }
        ImGui.PopTextWrapPos();
    }

    internal void DrawPm4OverlayWorkbenchContent()
    {
        if (_worldScene == null)
            return;

        Pm4WorkbenchSidebarService.Instance.DrawOverlaySettings(
            _worldScene.Pm4Overlay,
            _loosePm4InputPath,
            RequestLoosePm4File,
            ExportPm4OverlayReport);

        Pm4WorkbenchSidebarService.Instance.DrawPaletteAndLegend(
            _worldScene.Pm4Overlay,
            GetPm4ColorModeLabel);

        Pm4WorkbenchSidebarService.Instance.DrawSurfaceClassFilterMatrix(
            _worldScene.Pm4Overlay);

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
                    $"Placement Z ~= {placementZ:F3}   (MSUR.0x1C as float; CK24 is its top 24 bits)");
                ImGui.TextDisabled($"Surfaces={debugInfo.SurfaceCount}  bounds Z {debugInfo.BoundsMin.Z:F2}..{debugInfo.BoundsMax.Z:F2}  centre ({debugInfo.Center.X:F1}, {debugInfo.Center.Y:F1}, {debugInfo.Center.Z:F1})");
                ImGui.TextDisabled($"Raw slice 0x{debugInfo.Ck24:X6} (exponent band 0x{debugInfo.Ck24Type:X2} - NOT a type)");
                ImGui.TextDisabled($"MSHD F00={debugInfo.MshdField00} Region(0x04)={debugInfo.MshdRegionId} F08={debugInfo.MshdField08}");
                ImGui.TextDisabled($"SurfaceClass(0x00)=0x{debugInfo.DominantGroupKey:X2} Attr=0x{debugInfo.DominantAttributeMask:X2} MslkFirstIdx(0x18)={debugInfo.DominantMscnRefIndex} AvgH={debugInfo.AverageSurfaceHeight:F2}");
                ImGui.TextDisabled($"MSLK Group(0x04)=0x{debugInfo.LinkGroupObjectId:X8} Linked MPRL refs={debugInfo.LinkedPositionRefCount}");
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

            if (_worldScene.Pm4Overlay.TryGetSelectedPm4ObjectResearchInfo(out var ri) && ri.MslkRawEntries.Count > 0)
            {
                ImGui.Separator();
                ImGui.TextDisabled("MSLK entries ($ri.MslkRawEntries.Count):");
                foreach (string line in ri.MslkRawEntries.Take(5))
                    ImGui.TextUnformatted(line);
                if (ri.MslkRawEntries.Count > 5)
                    ImGui.TextDisabled("+ $ri.MslkRawEntries.Count - 5 more");
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
