using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Globalization;
using ImGuiNET;
using WoWViewer.Logging;
using WoWViewer.Terrain;

namespace WoWViewer;

/// <summary>
/// Partial class containing PM4 alignment and viewer utility windows.
/// </summary>
public partial class ViewerApp
{
    private void OpenPm4Workbench(Pm4WorkbenchTab tab)
    {
        FocusShellPanel(ShellPanelId.Pm4Workbench);
        _pendingPm4WorkbenchTab = tab;
        _activeBottomDrawerTab = FixedBottomDrawerTab.Pm4;
        _pendingRightSidebarSection = FixedBottomDrawerTab.Pm4;
        _showRightSidebar = true;
        if (_workspaceMode == WorkspaceMode.Editor)
            SetEditorWorkspaceTask(EditorWorkspaceTask.Pm4Evidence);
    }

    private void DrawPm4WorkbenchInspector()
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
            DrawPm4CorrelationInspectorContent();
            ImGui.EndTabItem();
        }

        _pendingPm4WorkbenchTab = null;
        ImGui.EndTabBar();
    }

    private void DrawPm4GlossarySummary()
    {
        if (!ImGui.CollapsingHeader("PM4 Glossary / Evidence"))
            return;

        ImGui.TextWrapped("The PM4 file carries six interconnected streams that together describe a coarse, type-classified summary of the WMO/M2 walkable + structural topology. The cyan (MSCN) and magenta (MSPV) cubes you can toggle below are the visible projection of this data. See wow-viewer/docs/architecture/pm4-chunk-semantics.md for the full reading.");
        ImGui.BulletText("MSUR — one polygon-fan surface record. IndexCount (>=3 = real polygon), Height (signed plane-distance term, not a Y-up height), MscnRefIndex (indexes MSCN), GroupKey/AttributeMask (local aliases, semantics open).");
        ImGui.BulletText("MSCN — scene-graph connector anchors. One 3D point per entry. Referenced by MSUR.MscnRefIndex. Used by the client as a placement/connector anchor for that surface. Visible as the cyan cubes.");
        ImGui.BulletText("MSLK — a link record. Says: 'surface at RefIndex connects to path-vertex chain MSPI[link.MspiFirstIndex..link.MspiFirstIndex+link.MspiIndexCount], and the connection has TypeFlags in {0x03 walkable M2-top, 0x10 walkable interior floor, 0x12 structural exterior solid}.' Subtype and SystemFlag semantics still open.");
        ImGui.BulletText("MSPV — path-vertex positions. Reached via MSPI[indices] from MSLK. The 3D positions the client uses to draw the actual connection between two surfaces (wall-floor corner, roof ridge, buttress line). Visible as the magenta cubes. Only present when surfaces are connected.");
        ImGui.BulletText("MSVI / MSVT — mesh-index stream into mesh vertex positions. The actual 3D positions of every vertex of every polygon the PM4 file describes. Walked as MSUR.MsviFirstIndex..MsviFirstIndex+IndexCount -> MSVI -> MSVT.");
        ImGui.BulletText("MPRL — per-tile position reference. A 3D position + heading. Used by the client for spawn anchors. Linked from MSLK.RefIndex when the link is an MPRL reference (not all are).");
        ImGui.Separator();
        ImGui.BulletText("MSHD region: promoted from MSHD.Field04. Current research says it behaves like a reusable scene/group bucket across tiles, useful for grouping/coloring but not a packed tile coordinate or proven placement semantic.");
        ImGui.BulletText("CK24: viewer alias for the packed MSUR field at 0x1C. Type = high byte, ObjId = low 16 bits.");
        ImGui.BulletText("part / ObjectPartId: viewer-generated split id. WoWViewer assigns it during the current overlay build after CK24 grouping, dominant MSLK grouping, optional MscnRef split, then optional connectivity split. It is not a raw PM4 field.");
        ImGui.BulletText("MSLK Group: dominant MSLK.GroupObjectId seen in the current viewer object. Strong grouping hint, not final proof of identity.");
        ImGui.BulletText("Linked MPRL refs: position-reference rows attached to the current viewer object or its dominant link family. Used as placement evidence.");
        ImGui.BulletText("Group / Attr / MscnRef: dominant MSUR values across the currently selected viewer object. Useful for debugging, not guaranteed unique or authoritative.");
        ImGui.BulletText("PM4 Graph: the viewer's current decomposition of the selected object, not a literal raw node graph stored in PM4.");
        ImGui.BulletText("Match uid: nearby MODF/MDDF placement candidate id. It is not a PM4-native object id.");
        ImGui.BulletText("cyan:magenta ratio: per-object topology fingerprint. ~0 magenta = disjoint decoration (no MSLK links between surfaces). ~1:1 = connected WMO. >1:1 = contiguous M2 with a dense connection graph. Used by the spec 050/052 matcher as a pre-filter.");
    }

    private void DrawPm4OverlayWorkbenchContent()
    {
        if (_worldScene == null)
            return;

        bool showPm4Overlay = _worldScene.ShowPm4Overlay;
        if (ImGui.Checkbox("PM4 Overlay", ref showPm4Overlay))
            _worldScene.ShowPm4Overlay = showPm4Overlay;

        ImGui.SameLine();
        if (ImGui.Button("Reload PM4"))
            _worldScene.ReloadPm4Overlay();

        ImGui.SameLine();
        if (ImGui.Button("Save Overlay Align"))
            SaveCurrentPm4Alignment();

        bool showPm4Solid = _worldScene.ShowPm4SolidOverlay;
        if (ImGui.Checkbox("PM4 Solid Fill", ref showPm4Solid))
            _worldScene.ShowPm4SolidOverlay = showPm4Solid;

        ImGui.SameLine();
        bool pm4IgnoreDepth = _worldScene.Pm4OverlayIgnoreDepth;
        if (ImGui.Checkbox("PM4 X-Ray", ref pm4IgnoreDepth))
            _worldScene.Pm4OverlayIgnoreDepth = pm4IgnoreDepth;

        ImGui.SameLine();
        bool showPm4Bounds = _worldScene.ShowPm4ObjectBounds;
        if (ImGui.Checkbox("PM4 Bounds", ref showPm4Bounds))
            _worldScene.ShowPm4ObjectBounds = showPm4Bounds;

        ImGui.SameLine();
        bool showPm4Ck24Bounds = _worldScene.ShowPm4Ck24Bounds;
        if (ImGui.Checkbox("PM4 CK24 Bounds", ref showPm4Ck24Bounds))
            _worldScene.ShowPm4Ck24Bounds = showPm4Ck24Bounds;

        bool showPm4Refs = _worldScene.ShowPm4PositionRefs;
        if (ImGui.Checkbox("PM4 MPRL Refs", ref showPm4Refs))
            _worldScene.ShowPm4PositionRefs = showPm4Refs;

        ImGui.SameLine();
        bool showPm4Centroids = _worldScene.ShowPm4ObjectCentroids;
        if (ImGui.Checkbox("PM4 Centroids", ref showPm4Centroids))
            _worldScene.ShowPm4ObjectCentroids = showPm4Centroids;

        bool showPm4Mscn = _worldScene.ShowPm4MscnNodes;
        if (ImGui.Checkbox("MSCN Nodes (cyan, per-surface connector anchor)", ref showPm4Mscn))
            _worldScene.ShowPm4MscnNodes = showPm4Mscn;
        ImGui.SameLine();
        bool showPm4Mspv = _worldScene.ShowPm4MspvNodes;
        if (ImGui.Checkbox("MSPV Nodes (magenta, per-link path vertex)", ref showPm4Mspv))
            _worldScene.ShowPm4MspvNodes = showPm4Mspv;

        bool renderNodesAsCubes = _worldScene.Pm4RenderNodesAsCubes;
        if (ImGui.Checkbox("Nodes as Solid Cubes", ref renderNodesAsCubes))
            _worldScene.Pm4RenderNodesAsCubes = renderNodesAsCubes;

        float mscnSize = _worldScene.Pm4MscnCubeSize;
        ImGui.SetNextItemWidth(100f);
        if (ImGui.SliderFloat("MSCN size", ref mscnSize, 0.2f, 4f))
            _worldScene.Pm4MscnCubeSize = mscnSize;
        ImGui.SameLine();
        float mspvSize = _worldScene.Pm4MspvCubeSize;
        ImGui.SetNextItemWidth(100f);
        if (ImGui.SliderFloat("MSPV size", ref mspvSize, 0.2f, 4f))
            _worldScene.Pm4MspvCubeSize = mspvSize;

        ImGui.SameLine();
        float mscnAlpha = _worldScene.Pm4MscnCubeAlpha;
        ImGui.SetNextItemWidth(100f);
        if (ImGui.SliderFloat("MSCN α", ref mscnAlpha, 0.1f, 1f))
            _worldScene.Pm4MscnCubeAlpha = mscnAlpha;
        ImGui.SameLine();
        float mspvAlpha = _worldScene.Pm4MspvCubeAlpha;
        ImGui.SetNextItemWidth(100f);
        if (ImGui.SliderFloat("MSPV α", ref mspvAlpha, 0.1f, 1f))
            _worldScene.Pm4MspvCubeAlpha = mspvAlpha;

        ImGui.SameLine();
        float lineWidth = _worldScene.Pm4WireframeLineWidth;
        ImGui.SetNextItemWidth(120f);
        if (ImGui.SliderFloat("Wire width", ref lineWidth, 1f, 8f))
            _worldScene.Pm4WireframeLineWidth = lineWidth;

        ImGui.SameLine();
        bool pm4FlipAllObjY = _worldScene.Pm4FlipAllObjectsY;
        if (ImGui.Checkbox("Mirror PM4 N/S", ref pm4FlipAllObjY))
            _worldScene.Pm4FlipAllObjectsY = pm4FlipAllObjY;

        ImGui.SameLine();
        if (ImGui.Button("Export Report"))
            ExportPm4OverlayReport();

        bool showType40 = _worldScene.ShowPm4Type40;
        if (ImGui.Checkbox("CK24 0x40", ref showType40))
            _worldScene.ShowPm4Type40 = showType40;

        ImGui.SameLine();
        bool showType80 = _worldScene.ShowPm4Type80;
        if (ImGui.Checkbox("CK24 0x80", ref showType80))
            _worldScene.ShowPm4Type80 = showType80;

        ImGui.SameLine();
        bool showTypeOther = _worldScene.ShowPm4TypeOther;
        if (ImGui.Checkbox("CK24 Other", ref showTypeOther))
            _worldScene.ShowPm4TypeOther = showTypeOther;

        Pm4OverlayColorMode colorMode = _worldScene.Pm4ColorMode;
        if (ImGui.BeginCombo("PM4 Color", GetPm4ColorModeLabel(colorMode)))
        {
            foreach (Pm4OverlayColorMode mode in Enum.GetValues<Pm4OverlayColorMode>())
            {
                bool isSelected = mode == colorMode;
                if (ImGui.Selectable(GetPm4ColorModeLabel(mode), isSelected))
                    _worldScene.Pm4ColorMode = mode;
                if (isSelected)
                    ImGui.SetItemDefaultFocus();
            }

            ImGui.EndCombo();
        }

        bool splitCk24Connectivity = _worldScene.Pm4SplitCk24ByConnectivity;
        if (ImGui.Checkbox("Split CK24 by Connectivity", ref splitCk24Connectivity))
        {
            _worldScene.Pm4SplitCk24ByConnectivity = splitCk24Connectivity;
            _worldScene.ReloadPm4Overlay();
        }

        bool splitCk24ByMscnRef = _worldScene.Pm4SplitCk24ByMscnRef;
        if (ImGui.Checkbox("Split CK24 by MscnRef", ref splitCk24ByMscnRef))
        {
            _worldScene.Pm4SplitCk24ByMscnRef = splitCk24ByMscnRef;
            _worldScene.ReloadPm4Overlay();
        }

        if (_worldScene.IsPm4Loading)
            ImGui.TextColored(new Vector4(1.0f, 0.85f, 0.35f, 1.0f), $"PM4 loading... {_worldScene.Pm4Status}");
        else if (_worldScene.Pm4LoadAttempted)
            ImGui.TextDisabled($"PM4: {_worldScene.Pm4LoadedFiles}/{_worldScene.Pm4TotalFiles} files, {_worldScene.Pm4VisibleObjectCount}/{_worldScene.Pm4ObjectCount} objects, {_worldScene.Pm4VisibleLineCount}/{_worldScene.Pm4LineCount} lines, {_worldScene.Pm4VisibleTriangleCount}/{_worldScene.Pm4TriangleCount} tris, {_worldScene.Pm4VisiblePositionRefCount}/{_worldScene.Pm4PositionRefCount} refs");
        else
            ImGui.TextDisabled("Toggle PM4 Overlay to lazy-load navmesh debug data.");

        if (_worldScene.Pm4LoadAttempted)
        {
            int totalMsur = _worldScene.Pm4TotalMsurCount;
            int shortIdx = _worldScene.Pm4DroppedShortIndexCount;
            int oorMsvi = _worldScene.Pm4DroppedOutOfRangeMsviCount;
            int emptyComp = _worldScene.Pm4DroppedEmptyComponentCount;
            int longEdge = _worldScene.Pm4RejectedLongEdges;
            int keptSurfaces = totalMsur - shortIdx - oorMsvi - emptyComp;
            ImGui.TextColored(new Vector4(1.0f, 0.85f, 0.35f, 1.0f),
                $"MSUR: {totalMsur} raw | kept: {keptSurfaces} | dropped: short-index={shortIdx}, out-of-range={oorMsvi}, empty={emptyComp}, long-edge-lines={longEdge}");
            ImGui.TextDisabled($"Status: {_worldScene.Pm4Status}");
        }

        ImGui.TextDisabled($"Overlay Align: T=({_worldScene.Pm4OverlayTranslation.X:F2}, {_worldScene.Pm4OverlayTranslation.Y:F2}, {_worldScene.Pm4OverlayTranslation.Z:F2}) Rot=({_worldScene.Pm4OverlayRotationDegrees.X:F2}, {_worldScene.Pm4OverlayRotationDegrees.Y:F2}, {_worldScene.Pm4OverlayRotationDegrees.Z:F2})° S=({_worldScene.Pm4OverlayScale.X:F3}, {_worldScene.Pm4OverlayScale.Y:F3}, {_worldScene.Pm4OverlayScale.Z:F3})");

        DrawPm4ColorLegend("WorkbenchOverlay");
    }

    private void DrawPm4SelectionWorkbenchContent()
    {
        if (_worldScene == null)
            return;

        if (!_worldScene.HasSelectedPm4Object || !_worldScene.SelectedPm4ObjectKey.HasValue)
        {
            ImGui.TextDisabled("No PM4 object selected. Left-click PM4 geometry to inspect one object at a time.");
            DrawPm4ObjectCollectionSummary("WorkbenchSelection");
            if (ImGui.Button("Dump PM4 Objects JSON"))
                ExportPm4ObjectsJson();
            ImGui.SameLine();
            if (ImGui.Button("Export PM4 OBJ Set"))
                ExportPm4ObjectsObjSet();
            ImGui.SameLine();
            if (ImGui.Button("Export PM4 LLM Bundle"))
                ExportPm4LlmEvidenceBundle();
            return;
        }

        int requestedMatches = _pm4ObjectMatchMaxMatchesPerObject;
        ImGui.SetNextItemWidth(130f);
        if (ImGui.SliderInt("Top Matches", ref requestedMatches, 3, 5))
            _pm4ObjectMatchMaxMatchesPerObject = Math.Clamp(requestedMatches, 3, 5);

        ImGui.SameLine();
        if (ImGui.Button("Open Advanced Align"))
            _showPm4AlignmentWindow = true;

        ImGui.SameLine();
        if (ImGui.Button("Save Overlay Align"))
            SaveCurrentPm4Alignment();

        if (ImGui.CollapsingHeader("Selected PM4", ImGuiTreeNodeFlags.DefaultOpen))
        {
            var selectedPm4 = _worldScene.SelectedPm4ObjectKey.Value;
            ImGui.Text($"tile ({selectedPm4.tileX}, {selectedPm4.tileY}) CK24=0x{selectedPm4.ck24:X6} part={selectedPm4.objectPart}");
            ImGui.TextDisabled("part = viewer-generated split id from the current overlay build, not a raw PM4 field.");

            if (_worldScene.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo debugInfo))
            {
                ImGui.TextDisabled($"Type=0x{debugInfo.Ck24Type:X2} ObjId={debugInfo.Ck24ObjectId} Surfaces={debugInfo.SurfaceCount}");
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

            ImGui.TextDisabled($"Tile layer align: T=({_worldScene.SelectedPm4Ck24LayerTranslation.X:F2}, {_worldScene.SelectedPm4Ck24LayerTranslation.Y:F2}, {_worldScene.SelectedPm4Ck24LayerTranslation.Z:F2}) Rot=({_worldScene.SelectedPm4Ck24LayerRotationDegrees.X:F2}, {_worldScene.SelectedPm4Ck24LayerRotationDegrees.Y:F2}, {_worldScene.SelectedPm4Ck24LayerRotationDegrees.Z:F2})° S=({_worldScene.SelectedPm4Ck24LayerScale.X:F3}, {_worldScene.SelectedPm4Ck24LayerScale.Y:F3}, {_worldScene.SelectedPm4Ck24LayerScale.Z:F3})");
            ImGui.TextDisabled($"Object align: T=({_worldScene.SelectedPm4ObjectTranslation.X:F2}, {_worldScene.SelectedPm4ObjectTranslation.Y:F2}, {_worldScene.SelectedPm4ObjectTranslation.Z:F2}) Rot=({_worldScene.SelectedPm4ObjectRotationDegrees.X:F2}, {_worldScene.SelectedPm4ObjectRotationDegrees.Y:F2}, {_worldScene.SelectedPm4ObjectRotationDegrees.Z:F2})° S=({_worldScene.SelectedPm4ObjectScale.X:F3}, {_worldScene.SelectedPm4ObjectScale.Y:F3}, {_worldScene.SelectedPm4ObjectScale.Z:F3})");

            if (ImGui.Button("Clear PM4 Selection"))
                _worldScene.ClearPm4ObjectSelection();
            ImGui.SameLine();
            if (ImGui.Button("Dump PM4 Objects JSON"))
                ExportPm4ObjectsJson();
            ImGui.SameLine();
            if (ImGui.Button("Export PM4 OBJ Set"))
                ExportPm4ObjectsObjSet();
            ImGui.SameLine();
            if (ImGui.Button("Export PM4 LLM Bundle"))
                ExportPm4LlmEvidenceBundle();

            // Inline WMO match button when a WMO-type object is selected
            if (debugInfo.Ck24Type is 0x42 or 0x43)
            {
                ImGui.Separator();
                ImGui.TextColored(new Vector4(0.8f, 0.9f, 1f, 1f), "WMO Detection");

                string? clientRoot = GetActiveGamePath();
                if (!string.IsNullOrWhiteSpace(clientRoot) && GetCurrentSessionMapName() != null)
                {
                    string matchKey = Pm4WmoGroupMatchService.GetMatchKey(
                        GetCurrentSessionMapName()!, debugInfo.TileX, debugInfo.TileY, debugInfo.Ck24);

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
                        var clusters = _worldScene.GetPm4SurfaceGroupClusters(
                            debugInfo.TileX, debugInfo.TileY, debugInfo.Ck24);
                        _pm4WmoGroupMatchResult = Pm4WmoGroupMatchService.MatchFromPlacement(
                            clientRoot, GetCurrentSessionMapName()!,
                            debugInfo.TileX, debugInfo.TileY, debugInfo.Ck24,
                            debugInfo.BoundsMin, debugInfo.BoundsMax, clusters);
                        _pm4WmoMatchStatus = _pm4WmoGroupMatchResult.ErrorMessage ?? "";
                    }
                    ImGui.SameLine();
                    if (ImGui.Button("Shape Search", new Vector2(120f, 28f)))
                    {
                        var fallback = Pm4WmoGroupMatchService.SearchWmoByShape(
                            clientRoot, debugInfo.BoundsMin, debugInfo.BoundsMax);
                        _pm4WmoGroupMatchResult = _pm4WmoGroupMatchResult != null
                            ? new Pm4WmoMatchResult(
                                _pm4WmoGroupMatchResult.HasAdtData,
                                _pm4WmoGroupMatchResult.Placements,
                                fallback)
                            : new Pm4WmoMatchResult(false, Array.Empty<Pm4WmoPlacementResult>(), fallback);
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

        int tileX = _worldScene?.SelectedPm4ObjectKey?.tileX ?? 0;
        int tileY = _worldScene?.SelectedPm4ObjectKey?.tileY ?? 0;
        uint ck24 = _worldScene?.SelectedPm4ObjectKey?.ck24 ?? 0;
        var clusters = _worldScene?.GetPm4SurfaceGroupClusters(tileX, tileY, ck24) ?? Array.Empty<Pm4SurfaceGroupCluster>();
        string? mapName = GetCurrentSessionMapName();
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

    private void DrawPm4CorrelationInspectorContent()
    {
        if (_worldScene == null)
            return;

        EnsurePm4WmoCorrelationReportLoaded();

        int requestedMatches = _pm4WmoCorrelationMaxMatchesPerPlacement;
        ImGui.SetNextItemWidth(90f);
        if (ImGui.InputInt("Max Matches", ref requestedMatches))
        {
            _pm4WmoCorrelationMaxMatchesPerPlacement = Math.Clamp(requestedMatches, 1, 32);
            RefreshPm4WmoCorrelationReport();
        }

        ImGui.SameLine();
        if (ImGui.Button("Refresh"))
            RefreshPm4WmoCorrelationReport();

        ImGui.SameLine();
        if (ImGui.Button("Dump JSON"))
            ExportPm4WmoCorrelationJson();

        ImGui.SameLine();
        if (ImGui.Checkbox("Only Near", ref _pm4WmoCorrelationNearOnly))
        {
            if (_selectedPm4WmoCorrelationPlacementIndex >= 0)
                _selectedPm4WmoCorrelationMatchIndex = 0;
        }

        ImGui.SetNextItemWidth(-1f);
        ImGui.InputTextWithHint("##Pm4WmoCorrelationFilterWorkbench", "Filter model name or path", ref _pm4WmoCorrelationModelFilter, 256);

        if (_pm4WmoCorrelationReport == null)
        {
            ImGui.TextDisabled("No PM4/WMO correlation report is loaded.");
            return;
        }

        Pm4WmoCorrelationReport report = _pm4WmoCorrelationReport;
        ImGui.TextDisabled($"Generated {report.GeneratedAtUtc:yyyy-MM-dd HH:mm:ss} UTC | placements {report.Summary.WmoPlacementCount}, resolved WMO meshes {report.Summary.WmoMeshResolvedCount}, PM4 objects {report.Summary.Pm4ObjectCount}");
        ImGui.TextDisabled($"Candidates {report.Summary.PlacementsWithCandidates}/{report.Summary.WmoPlacementCount}, near {report.Summary.PlacementsWithNearCandidates}, PM4 status: {report.Pm4Status}");

        string filter = _pm4WmoCorrelationModelFilter.Trim();
        var filteredPlacements = report.Placements
            .Select((placement, index) => new { placement, index })
            .Where(entry => !_pm4WmoCorrelationNearOnly || entry.placement.Pm4NearCandidateCount > 0)
            .Where(entry => string.IsNullOrWhiteSpace(filter)
                || entry.placement.ModelName.Contains(filter, StringComparison.OrdinalIgnoreCase)
                || entry.placement.ModelPath.Contains(filter, StringComparison.OrdinalIgnoreCase)
                || entry.placement.ModelKey.Contains(filter, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(entry => entry.placement.Pm4Matches.Count > 0 ? entry.placement.Pm4Matches[0].FootprintOverlapRatio : 0f)
            .ThenBy(entry => entry.placement.ModelName, StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (filteredPlacements.Count == 0)
        {
            ImGui.TextDisabled("No placements matched the current filter.");
            return;
        }

        if (!filteredPlacements.Any(entry => entry.index == _selectedPm4WmoCorrelationPlacementIndex))
        {
            _selectedPm4WmoCorrelationPlacementIndex = filteredPlacements[0].index;
            _selectedPm4WmoCorrelationMatchIndex = 0;
        }

        float leftWidth = MathF.Min(360f, ImGui.GetContentRegionAvail().X * 0.44f);
        if (ImGui.BeginChild("##Pm4WmoPlacementListWorkbench", new Vector2(leftWidth, 360f), true))
        {
            for (int i = 0; i < filteredPlacements.Count; i++)
            {
                var entry = filteredPlacements[i];
                Pm4WmoCorrelationPlacement placement = entry.placement;
                bool selected = entry.index == _selectedPm4WmoCorrelationPlacementIndex;
                string label = $"[{placement.TileX},{placement.TileY}] {placement.ModelName}##Pm4WmoPlacementWorkbench{entry.index}";
                if (ImGui.Selectable(label, selected))
                {
                    _selectedPm4WmoCorrelationPlacementIndex = entry.index;
                    _selectedPm4WmoCorrelationMatchIndex = 0;
                }

                if (placement.Pm4Matches.Count > 0)
                {
                    Pm4WmoCorrelationMatch best = placement.Pm4Matches[0];
                    ImGui.TextDisabled($"best CK24=0x{best.Ck24:X6} part={best.ObjectPartId} overlap={best.FootprintOverlapRatio:F2} dist={best.FootprintDistance:F1}");
                }
                else
                {
                    ImGui.TextDisabled("No PM4 candidates in the current tile neighborhood.");
                }

                ImGui.Separator();
            }
        }
        ImGui.EndChild();

        ImGui.SameLine();

        if (ImGui.BeginChild("##Pm4WmoPlacementDetailsWorkbench", Vector2.Zero, true))
        {
            Pm4WmoCorrelationPlacement placement = report.Placements[_selectedPm4WmoCorrelationPlacementIndex];
            ImGui.TextWrapped($"{placement.ModelName} (tile {placement.TileX},{placement.TileY}, uid {placement.UniqueId})");
            ImGui.TextDisabled(placement.ModelPath);

            if (placement.Pm4Matches.Count > 0)
            {
                Pm4WmoCorrelationMatch selectedMatch = placement.Pm4Matches[Math.Clamp(_selectedPm4WmoCorrelationMatchIndex, 0, placement.Pm4Matches.Count - 1)];

                if (ImGui.Button("Select PM4"))
                    SelectPm4CorrelationMatch(selectedMatch, frameCamera: false);

                ImGui.SameLine();
                if (ImGui.Button("Frame PM4"))
                    SelectPm4CorrelationMatch(selectedMatch, frameCamera: true);

                ImGui.SameLine();
                if (ImGui.Button("Frame Pair"))
                {
                    Vector3 boundsMin = Vector3.Min(placement.WorldBoundsMin, selectedMatch.BoundsMin);
                    Vector3 boundsMax = Vector3.Max(placement.WorldBoundsMax, selectedMatch.BoundsMax);
                    SelectPm4CorrelationMatch(selectedMatch, frameCamera: false);
                    FocusCameraOnBounds(boundsMin, boundsMax);
                }
            }

            ImGui.Separator();
            ImGui.TextDisabled($"Placement pos: ({placement.PlacementPosition.X:F2}, {placement.PlacementPosition.Y:F2}, {placement.PlacementPosition.Z:F2})");
            ImGui.TextDisabled($"World bounds min: ({placement.WorldBoundsMin.X:F2}, {placement.WorldBoundsMin.Y:F2}, {placement.WorldBoundsMin.Z:F2})");
            ImGui.TextDisabled($"World bounds max: ({placement.WorldBoundsMax.X:F2}, {placement.WorldBoundsMax.Y:F2}, {placement.WorldBoundsMax.Z:F2})");

            ImGui.Separator();
            ImGui.Text($"PM4 matches ({placement.Pm4Matches.Count}/{placement.Pm4CandidateCount} shown, near={placement.Pm4NearCandidateCount})");

            for (int matchIndex = 0; matchIndex < placement.Pm4Matches.Count; matchIndex++)
            {
                Pm4WmoCorrelationMatch match = placement.Pm4Matches[matchIndex];
                bool selected = matchIndex == _selectedPm4WmoCorrelationMatchIndex;
                string label = $"CK24 0x{match.Ck24:X6} part {match.ObjectPartId}##Pm4WmoMatchWorkbench{matchIndex}";
                if (ImGui.Selectable(label, selected))
                    _selectedPm4WmoCorrelationMatchIndex = matchIndex;

                ImGui.TextDisabled($"footprint overlap={match.FootprintOverlapRatio:F3} area={match.FootprintAreaRatio:F3} dist={match.FootprintDistance:F2}");
                ImGui.TextDisabled($"planar gap={match.PlanarGap:F2} vertical gap={match.VerticalGap:F2} center={match.CenterDistance:F2}");
                ImGui.Separator();
            }
        }
        ImGui.EndChild();
    }

    private void DrawPerfWindow()
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

    private void DrawPerfContent()
    {
        var terrainRenderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (terrainRenderer == null)
        {
            ImGui.Text("No terrain loaded.");
            return;
        }
        ImGui.Text($"Chunks: {terrainRenderer.ChunksRendered} rendered, {terrainRenderer.ChunksCulled} culled");
        ImGui.TextDisabled("Stats are for the last terrain Render() call.");
    }

    private void DrawPm4AlignmentWindow()
    {
        if (_worldScene == null)
        {
            _showPm4AlignmentWindow = false;
            return;
        }

        ImGui.SetNextWindowSize(new Vector2(430, 0), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("PM4 Alignment", ref _showPm4AlignmentWindow, ImGuiWindowFlags.AlwaysAutoResize))
        {
            ImGui.End();
            return;
        }

        ImGui.TextWrapped("PM4 alignment is now tile-local for the selected CK24 bucket plus object-local for the selected part. Select one PM4 object, then adjust the tile CK24 block or the object block.");
        ImGui.TextDisabled("Global PM4 overlay transforms are no longer edited in this window.");
        ImGui.TextDisabled("Use Overlay > Flip All Obj Y for map-wide Y mirror correction.");

        ImGui.Text("Translation Step:");
        if (ImGui.RadioButton("0.5u", MathF.Abs(_pm4TranslationStepUnits - 0.5f) < 0.001f))
            _pm4TranslationStepUnits = 0.5f;
        ImGui.SameLine();
        if (ImGui.RadioButton("1u", MathF.Abs(_pm4TranslationStepUnits - 1f) < 0.001f))
            _pm4TranslationStepUnits = 1f;
        ImGui.SameLine();
        if (ImGui.RadioButton("10u", MathF.Abs(_pm4TranslationStepUnits - 10f) < 0.001f))
            _pm4TranslationStepUnits = 10f;
        ImGui.SameLine();
        if (ImGui.RadioButton("100u", MathF.Abs(_pm4TranslationStepUnits - 100f) < 0.001f))
            _pm4TranslationStepUnits = 100f;
        ImGui.SameLine();
        if (ImGui.RadioButton("533.333u", MathF.Abs(_pm4TranslationStepUnits - 533.3333f) < 0.01f))
            _pm4TranslationStepUnits = 533.3333f;

        ImGui.Text("Rotation Step:");
        if (ImGui.RadioButton("1 deg", MathF.Abs(_pm4RotationStepDegrees - 1f) < 0.001f))
            _pm4RotationStepDegrees = 1f;
        ImGui.SameLine();
        if (ImGui.RadioButton("5 deg", MathF.Abs(_pm4RotationStepDegrees - 5f) < 0.001f))
            _pm4RotationStepDegrees = 5f;
        ImGui.SameLine();
        if (ImGui.RadioButton("15 deg", MathF.Abs(_pm4RotationStepDegrees - 15f) < 0.001f))
            _pm4RotationStepDegrees = 15f;
        ImGui.SameLine();
        if (ImGui.RadioButton("45 deg", MathF.Abs(_pm4RotationStepDegrees - 45f) < 0.001f))
            _pm4RotationStepDegrees = 45f;
        ImGui.SameLine();
        if (ImGui.RadioButton("90 deg", MathF.Abs(_pm4RotationStepDegrees - 90f) < 0.001f))
            _pm4RotationStepDegrees = 90f;

        ImGui.Text("Scale Step:");
        if (ImGui.RadioButton("0.01", MathF.Abs(_pm4ScaleStepUnits - 0.01f) < 0.0001f))
            _pm4ScaleStepUnits = 0.01f;
        ImGui.SameLine();
        if (ImGui.RadioButton("0.1", MathF.Abs(_pm4ScaleStepUnits - 0.1f) < 0.0001f))
            _pm4ScaleStepUnits = 0.1f;
        ImGui.SameLine();
        if (ImGui.RadioButton("0.25", MathF.Abs(_pm4ScaleStepUnits - 0.25f) < 0.0001f))
            _pm4ScaleStepUnits = 0.25f;
        ImGui.SameLine();
        if (ImGui.RadioButton("1.0", MathF.Abs(_pm4ScaleStepUnits - 1f) < 0.0001f))
            _pm4ScaleStepUnits = 1f;

        ImGui.Separator();

        if (!_worldScene.HasSelectedPm4Object || !_worldScene.SelectedPm4ObjectKey.HasValue)
        {
            ImGui.TextDisabled("No PM4 object selected. Left-click PM4 geometry to pick an object.");
            if (ImGui.Button("Clear PM4 Selection"))
                _worldScene.ClearPm4ObjectSelection();
            ImGui.SameLine();
            if (ImGui.Button("Dump PM4 Objects JSON"))
                ExportPm4ObjectsJson();
            ImGui.SameLine();
            if (ImGui.Button("Export PM4 OBJ Set"))
                ExportPm4ObjectsObjSet();
            ImGui.SameLine();
            if (ImGui.Button("PM4 Object Match"))
            {
                _showPm4ObjectMatchWindow = true;
                EnsurePm4ObjectMatchReportLoaded();
            }
            ImGui.SameLine();
            if (ImGui.Button("Dump PM4/WMO Correlation JSON"))
                ExportPm4WmoCorrelationJson();
            ImGui.SameLine();
            if (ImGui.Button("PM4/WMO Panel"))
            {
                _showPm4WmoCorrelationWindow = true;
                EnsurePm4WmoCorrelationReportLoaded();
            }
            ImGui.End();
            return;
        }

        var selectedPm4 = _worldScene.SelectedPm4ObjectKey.Value;
        uint? selectedLayerCk24 = _worldScene.SelectedPm4RawCk24;
        Vector3 selectedObjectTranslation = _worldScene.SelectedPm4ObjectTranslation;
        Vector3 selectedObjectRotation = _worldScene.SelectedPm4ObjectRotationDegrees;
        Vector3 selectedObjectScale = _worldScene.SelectedPm4ObjectScale;
        Vector3 selectedLayerTranslation = _worldScene.SelectedPm4Ck24LayerTranslation;
        Vector3 selectedLayerRotation = _worldScene.SelectedPm4Ck24LayerRotationDegrees;
        Vector3 selectedLayerScale = _worldScene.SelectedPm4Ck24LayerScale;
        bool translationChanged = false;
        bool rotationChanged = false;
        bool scaleChanged = false;
        bool layerTranslationChanged = false;
        bool layerRotationChanged = false;
        bool layerScaleChanged = false;

        ImGui.Text($"Selected: tile ({selectedPm4.tileX}, {selectedPm4.tileY}) CK24=0x{selectedPm4.ck24:X6} part={selectedPm4.objectPart}");
        if (_worldScene.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo debugInfo))
        {
            ImGui.TextDisabled($"Type=0x{debugInfo.Ck24Type:X2} ObjId={debugInfo.Ck24ObjectId} Surfaces={debugInfo.SurfaceCount}");
            ImGui.TextDisabled($"Group=0x{debugInfo.DominantGroupKey:X2} Attr=0x{debugInfo.DominantAttributeMask:X2} mscnRef={debugInfo.DominantMscnRefIndex} AvgH={debugInfo.AverageSurfaceHeight:F2}");
            ImGui.TextDisabled($"Part={debugInfo.ObjectPartId} MSLKGroup=0x{debugInfo.LinkGroupObjectId:X8}");
            ImGui.TextDisabled($"Linked MPRL refs={debugInfo.LinkedPositionRefCount}");
            if (debugInfo.DistinctTypeFlags != 0)
            {
                List<string> typeFlagLabels = [];
                for (int bit = 1; bit < 32; bit++)
                {
                    if ((debugInfo.DistinctTypeFlags & (1u << bit)) != 0)
                    {
                        string label = bit switch
                        {
                            0x03 => "m2-top(0x03)",
                            0x10 => "interior-floor(0x10)",
                            0x12 => "exterior-solid(0x12)",
                            _ => $"0x{bit:X2}",
                        };
                        typeFlagLabels.Add(label);
                    }
                }
                ImGui.TextDisabled($"TypeFlags: {string.Join(", ", typeFlagLabels)}");
            }
            if (debugInfo.LinkedPositionRefSummary.TotalCount > 0)
            {
                if (debugInfo.LinkedPositionRefSummary.HasNormalHeadings)
                {
                    ImGui.TextDisabled(
                        $"MPRL normal={debugInfo.LinkedPositionRefSummary.NormalCount} term={debugInfo.LinkedPositionRefSummary.TerminatorCount} floors={debugInfo.LinkedPositionRefSummary.FloorMin}..{debugInfo.LinkedPositionRefSummary.FloorMax}");
                    ImGui.TextDisabled(
                        $"MPRL heading={debugInfo.LinkedPositionRefSummary.HeadingMinDegrees:F2}..{debugInfo.LinkedPositionRefSummary.HeadingMaxDegrees:F2} mean={debugInfo.LinkedPositionRefSummary.HeadingMeanDegrees:F2} deg");
                }
                else
                {
                    ImGui.TextDisabled(
                        $"MPRL normal={debugInfo.LinkedPositionRefSummary.NormalCount} term={debugInfo.LinkedPositionRefSummary.TerminatorCount}");
                }
            }
            ImGui.TextDisabled($"Planar: swap={debugInfo.SwapPlanarAxes} invertU={debugInfo.InvertU} invertV={debugInfo.InvertV} windingFlip={debugInfo.InvertsWinding}");
        }

        if (selectedLayerCk24.HasValue && _worldScene.TryGetSelectedPm4Ck24LayerStats(out int layerTileCount, out int layerObjectCount))
            ImGui.TextDisabled($"Tile CK24 0x{selectedLayerCk24.Value:X6} on ({selectedPm4.tileX}, {selectedPm4.tileY}): {layerObjectCount} parts across {layerTileCount} tile");

        if (_worldScene.TryGetSelectedPm4ObjectResearchInfo(out Pm4SelectedObjectResearchInfo researchInfo)
            && ImGui.CollapsingHeader("PM4 Research", ImGuiTreeNodeFlags.DefaultOpen))
        {
            ImGui.TextDisabled($"Source: {Path.GetFileName(researchInfo.SourcePath)}");
            ImGui.TextDisabled($"v{researchInfo.Version} MSLK={researchInfo.MslkCount} MSUR={researchInfo.MsurCount} MSCN={researchInfo.MscnCount} MPRL={researchInfo.MprlCount}");
            ImGui.TextDisabled($"RefIndex mismatches={researchInfo.InvalidRefIndexCount} diagnostics={researchInfo.DiagnosticCount} hypotheses={researchInfo.MatchingCk24HypothesisCount}/{researchInfo.TotalHypothesisCount}");

            if (researchInfo.Diagnostics.Count > 0)
            {
                for (int i = 0; i < researchInfo.Diagnostics.Count; i++)
                    ImGui.TextDisabled($"diag: {researchInfo.Diagnostics[i]}");
            }

            if (researchInfo.TopMatches.Count == 0)
            {
                ImGui.TextDisabled("No raw PM4 hypotheses matched the selected CK24.");
            }
            else
            {
                ImGui.Text("Top raw hypotheses:");
                for (int i = 0; i < researchInfo.TopMatches.Count; i++)
                {
                    Pm4ResearchHypothesisMatch match = researchInfo.TopMatches[i];
                    string headingText = match.MprlHeadingMeanDegrees.HasValue
                        ? $" heading={match.MprlHeadingMeanDegrees.Value:F1} delta={match.HeadingDeltaDegrees?.ToString("F1") ?? "n/a"}"
                        : string.Empty;
                    ImGui.BulletText($"{match.Family}#{match.FamilyObjectIndex} score={match.SimilarityScore:F2} surfaces={match.SurfaceCount} indices={match.TotalIndexCount} mscnRef={match.MscnRefCount} groups={match.GroupKeyCount} linkGroups={match.LinkGroupCount} dominant=0x{match.DominantLinkGroupObjectId:X} mode={match.CoordinateMode} planar=(swap={match.PlanarTransform.SwapPlanarAxes},u={match.PlanarTransform.InvertU},v={match.PlanarTransform.InvertV}) yaw={match.FrameYawDegrees:F1}{headingText} linkedMPRL={match.LinkedMprlRefCount}/{match.LinkedMprlInBoundsCount}");
                }
            }

            if (researchInfo.MshdRawFields != null)
            {
                ImGui.Separator();
                ImGui.TextDisabled(researchInfo.MshdRawFields);
            }

            if (researchInfo.MslkRawEntries.Count > 0)
            {
                ImGui.Separator();
                ImGui.TextDisabled($"MSLK entries for this CK24 ({researchInfo.MslkRawEntries.Count}):");
                for (int i = 0; i < researchInfo.MslkRawEntries.Count && i < 16; i++)
                    ImGui.TextDisabled(researchInfo.MslkRawEntries[i]);
                if (researchInfo.MslkRawEntries.Count > 16)
                    ImGui.TextDisabled($"... and {researchInfo.MslkRawEntries.Count - 16} more");
            }
        }

        ImGui.Separator();
        ImGui.Text("Tile CK24 Translation:");

        if (ImGui.Button("Layer X <<"))
        {
            selectedLayerTranslation.X -= _pm4TranslationStepUnits;
            layerTranslationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer X >>"))
        {
            selectedLayerTranslation.X += _pm4TranslationStepUnits;
            layerTranslationChanged = true;
        }

        if (ImGui.Button("Layer Y <<"))
        {
            selectedLayerTranslation.Y -= _pm4TranslationStepUnits;
            layerTranslationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer Y >>"))
        {
            selectedLayerTranslation.Y += _pm4TranslationStepUnits;
            layerTranslationChanged = true;
        }

        if (ImGui.Button("Layer Z <<"))
        {
            selectedLayerTranslation.Z -= _pm4TranslationStepUnits;
            layerTranslationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer Z >>"))
        {
            selectedLayerTranslation.Z += _pm4TranslationStepUnits;
            layerTranslationChanged = true;
        }

        ImGui.Separator();
        ImGui.Text("Tile CK24 Rotation:");

        if (ImGui.Button("Layer Rot X -"))
        {
            selectedLayerRotation.X -= _pm4RotationStepDegrees;
            layerRotationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer Rot X +"))
        {
            selectedLayerRotation.X += _pm4RotationStepDegrees;
            layerRotationChanged = true;
        }

        if (ImGui.Button("Layer Rot Y -"))
        {
            selectedLayerRotation.Y -= _pm4RotationStepDegrees;
            layerRotationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer Rot Y +"))
        {
            selectedLayerRotation.Y += _pm4RotationStepDegrees;
            layerRotationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer Rot Y +180"))
        {
            selectedLayerRotation.Y += 180f;
            layerRotationChanged = true;
        }

        if (ImGui.Button("Layer Rot Z -"))
        {
            selectedLayerRotation.Z -= _pm4RotationStepDegrees;
            layerRotationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer Rot Z +"))
        {
            selectedLayerRotation.Z += _pm4RotationStepDegrees;
            layerRotationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer Rot Z +180"))
        {
            selectedLayerRotation.Z += 180f;
            layerRotationChanged = true;
        }

        ImGui.Separator();
        ImGui.Text("Tile CK24 Scale:");

        if (ImGui.Button("Layer Sx -"))
        {
            selectedLayerScale.X -= _pm4ScaleStepUnits;
            layerScaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer Sx +"))
        {
            selectedLayerScale.X += _pm4ScaleStepUnits;
            layerScaleChanged = true;
        }

        if (ImGui.Button("Layer Sy -"))
        {
            selectedLayerScale.Y -= _pm4ScaleStepUnits;
            layerScaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer Sy +"))
        {
            selectedLayerScale.Y += _pm4ScaleStepUnits;
            layerScaleChanged = true;
        }

        if (ImGui.Button("Layer Sz -"))
        {
            selectedLayerScale.Z -= _pm4ScaleStepUnits;
            layerScaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer Sz +"))
        {
            selectedLayerScale.Z += _pm4ScaleStepUnits;
            layerScaleChanged = true;
        }

        ImGui.Text("Tile CK24 Axis Flips:");
        if (ImGui.Button("Flip Layer X"))
        {
            selectedLayerScale.X = -selectedLayerScale.X;
            layerScaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Flip Layer Y"))
        {
            selectedLayerScale.Y = -selectedLayerScale.Y;
            layerScaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Flip Layer Z"))
        {
            selectedLayerScale.Z = -selectedLayerScale.Z;
            layerScaleChanged = true;
        }

        ImGui.Text("Tile CK24 Winding:");
        if (ImGui.Button("Wind Tile X"))
        {
            selectedLayerScale.X = ToggleWindingComponent(selectedLayerScale.X);
            layerScaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Wind Tile Y"))
        {
            selectedLayerScale.Y = ToggleWindingComponent(selectedLayerScale.Y);
            layerScaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Wind Tile Z"))
        {
            selectedLayerScale.Z = ToggleWindingComponent(selectedLayerScale.Z);
            layerScaleChanged = true;
        }

        bool pm4TransformChanged = false;

        if (layerTranslationChanged)
        {
            _worldScene.SelectedPm4Ck24LayerTranslation = selectedLayerTranslation;
            pm4TransformChanged = true;
        }
        if (layerRotationChanged)
        {
            _worldScene.SelectedPm4Ck24LayerRotationDegrees = NormalizeRotationDegrees(selectedLayerRotation);
            pm4TransformChanged = true;
        }
        if (layerScaleChanged)
        {
            _worldScene.SelectedPm4Ck24LayerScale = selectedLayerScale;
            pm4TransformChanged = true;
        }

        ImGui.Separator();
        ImGui.Text("Object Translation:");

        if (ImGui.Button("Obj X <<"))
        {
            selectedObjectTranslation.X -= _pm4TranslationStepUnits;
            translationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj X >>"))
        {
            selectedObjectTranslation.X += _pm4TranslationStepUnits;
            translationChanged = true;
        }

        if (ImGui.Button("Obj Y <<"))
        {
            selectedObjectTranslation.Y -= _pm4TranslationStepUnits;
            translationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj Y >>"))
        {
            selectedObjectTranslation.Y += _pm4TranslationStepUnits;
            translationChanged = true;
        }

        if (ImGui.Button("Obj Z <<"))
        {
            selectedObjectTranslation.Z -= _pm4TranslationStepUnits;
            translationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj Z >>"))
        {
            selectedObjectTranslation.Z += _pm4TranslationStepUnits;
            translationChanged = true;
        }

        ImGui.Separator();
        ImGui.Text("Object Rotation:");

        if (ImGui.Button("Obj Rot X -"))
        {
            selectedObjectRotation.X -= _pm4RotationStepDegrees;
            rotationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj Rot X +"))
        {
            selectedObjectRotation.X += _pm4RotationStepDegrees;
            rotationChanged = true;
        }

        if (ImGui.Button("Obj Rot Y -"))
        {
            selectedObjectRotation.Y -= _pm4RotationStepDegrees;
            rotationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj Rot Y +"))
        {
            selectedObjectRotation.Y += _pm4RotationStepDegrees;
            rotationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj Rot Y +180"))
        {
            selectedObjectRotation.Y += 180f;
            rotationChanged = true;
        }

        if (ImGui.Button("Obj Rot Z -"))
        {
            selectedObjectRotation.Z -= _pm4RotationStepDegrees;
            rotationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj Rot Z +"))
        {
            selectedObjectRotation.Z += _pm4RotationStepDegrees;
            rotationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj Rot Z +180"))
        {
            selectedObjectRotation.Z += 180f;
            rotationChanged = true;
        }

        ImGui.Separator();
        ImGui.Text("Object Scale:");

        if (ImGui.Button("Obj Sx -"))
        {
            selectedObjectScale.X -= _pm4ScaleStepUnits;
            scaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj Sx +"))
        {
            selectedObjectScale.X += _pm4ScaleStepUnits;
            scaleChanged = true;
        }

        if (ImGui.Button("Obj Sy -"))
        {
            selectedObjectScale.Y -= _pm4ScaleStepUnits;
            scaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj Sy +"))
        {
            selectedObjectScale.Y += _pm4ScaleStepUnits;
            scaleChanged = true;
        }

        if (ImGui.Button("Obj Sz -"))
        {
            selectedObjectScale.Z -= _pm4ScaleStepUnits;
            scaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj Sz +"))
        {
            selectedObjectScale.Z += _pm4ScaleStepUnits;
            scaleChanged = true;
        }

        ImGui.Text("Object Axis Flips:");
        if (ImGui.Button("Flip Obj X"))
        {
            selectedObjectScale.X = -selectedObjectScale.X;
            scaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Flip Obj Y"))
        {
            selectedObjectScale.Y = -selectedObjectScale.Y;
            scaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Flip Obj Z"))
        {
            selectedObjectScale.Z = -selectedObjectScale.Z;
            scaleChanged = true;
        }

        ImGui.Text("Object Winding:");
        if (ImGui.Button("Wind Obj X"))
        {
            selectedObjectScale.X = ToggleWindingComponent(selectedObjectScale.X);
            scaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Wind Obj Y"))
        {
            selectedObjectScale.Y = ToggleWindingComponent(selectedObjectScale.Y);
            scaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Wind Obj Z"))
        {
            selectedObjectScale.Z = ToggleWindingComponent(selectedObjectScale.Z);
            scaleChanged = true;
        }

        if (translationChanged)
        {
            _worldScene.SelectedPm4ObjectTranslation = selectedObjectTranslation;
            pm4TransformChanged = true;
        }
        if (rotationChanged)
        {
            _worldScene.SelectedPm4ObjectRotationDegrees = NormalizeRotationDegrees(selectedObjectRotation);
            pm4TransformChanged = true;
        }
        if (scaleChanged)
        {
            _worldScene.SelectedPm4ObjectScale = selectedObjectScale;
            pm4TransformChanged = true;
        }

        ImGui.Separator();

        if (ImGui.Button("Reset Layer Move"))
        {
            _worldScene.SelectedPm4Ck24LayerTranslation = Vector3.Zero;
            pm4TransformChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Reset Layer Rot"))
        {
            _worldScene.SelectedPm4Ck24LayerRotationDegrees = Vector3.Zero;
            pm4TransformChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Reset Layer Scale"))
        {
            _worldScene.SelectedPm4Ck24LayerScale = Vector3.One;
            pm4TransformChanged = true;
        }

        if (ImGui.Button("Reset Layer 9DoF"))
        {
            _worldScene.SelectedPm4Ck24LayerTranslation = Vector3.Zero;
            _worldScene.SelectedPm4Ck24LayerRotationDegrees = Vector3.Zero;
            _worldScene.SelectedPm4Ck24LayerScale = Vector3.One;
            pm4TransformChanged = true;
        }

        ImGui.SameLine();
        if (ImGui.Button("Print Layer Alignment") && selectedLayerCk24.HasValue)
        {
            Vector3 t = _worldScene.SelectedPm4Ck24LayerTranslation;
            Vector3 r = _worldScene.SelectedPm4Ck24LayerRotationDegrees;
            Vector3 s = _worldScene.SelectedPm4Ck24LayerScale;
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"[PM4 Tile CK24 Align] tile=({selectedPm4.tileX},{selectedPm4.tileY}) ck24=0x{selectedLayerCk24.Value:X6} T=({t.X:F3},{t.Y:F3},{t.Z:F3}) Rot=({r.X:F3},{r.Y:F3},{r.Z:F3}) Scale=({s.X:F4},{s.Y:F4},{s.Z:F4})");
        }

            ImGui.TextDisabled($"Tile Move: ({_worldScene.SelectedPm4Ck24LayerTranslation.X:F3}, {_worldScene.SelectedPm4Ck24LayerTranslation.Y:F3}, {_worldScene.SelectedPm4Ck24LayerTranslation.Z:F3})");
            ImGui.TextDisabled($"Tile Rot: ({_worldScene.SelectedPm4Ck24LayerRotationDegrees.X:F3}, {_worldScene.SelectedPm4Ck24LayerRotationDegrees.Y:F3}, {_worldScene.SelectedPm4Ck24LayerRotationDegrees.Z:F3}) deg");
            ImGui.TextDisabled($"Tile Scale: ({_worldScene.SelectedPm4Ck24LayerScale.X:F4}, {_worldScene.SelectedPm4Ck24LayerScale.Y:F4}, {_worldScene.SelectedPm4Ck24LayerScale.Z:F4})");

        ImGui.Separator();

        if (ImGui.Button("Reset Obj Move"))
        {
            _worldScene.SelectedPm4ObjectTranslation = Vector3.Zero;
            pm4TransformChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Reset Obj Rot"))
        {
            _worldScene.SelectedPm4ObjectRotationDegrees = Vector3.Zero;
            pm4TransformChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Reset Obj Scale"))
        {
            _worldScene.SelectedPm4ObjectScale = Vector3.One;
            pm4TransformChanged = true;
        }

        if (ImGui.Button("Reset Obj 9DoF"))
        {
            _worldScene.SelectedPm4ObjectTranslation = Vector3.Zero;
            _worldScene.SelectedPm4ObjectRotationDegrees = Vector3.Zero;
            _worldScene.SelectedPm4ObjectScale = Vector3.One;
            pm4TransformChanged = true;
        }

        ImGui.SameLine();
        if (ImGui.Button("Clear PM4 Selection"))
            _worldScene.ClearPm4ObjectSelection();

        if (pm4TransformChanged)
            InvalidatePm4DerivedReports();

        if (ImGui.Button("Dump PM4 Objects JSON"))
            ExportPm4ObjectsJson();
        ImGui.SameLine();
        if (ImGui.Button("Export PM4 OBJ Set"))
            ExportPm4ObjectsObjSet();
        ImGui.SameLine();
        if (ImGui.Button("PM4 Object Match"))
        {
            _showPm4ObjectMatchWindow = true;
            EnsurePm4ObjectMatchReportLoaded();
        }
        ImGui.SameLine();
        if (ImGui.Button("Dump PM4/WMO Correlation JSON"))
            ExportPm4WmoCorrelationJson();
        ImGui.SameLine();
        if (ImGui.Button("PM4/WMO Panel"))
        {
            _showPm4WmoCorrelationWindow = true;
            EnsurePm4WmoCorrelationReportLoaded();
        }
        ImGui.SameLine();
        if (ImGui.Button("Print Obj Alignment"))
        {
            Vector3 t = _worldScene.SelectedPm4ObjectTranslation;
            Vector3 r = _worldScene.SelectedPm4ObjectRotationDegrees;
            Vector3 s = _worldScene.SelectedPm4ObjectScale;
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"[PM4 Obj Align] tile=({selectedPm4.tileX},{selectedPm4.tileY}) ck24=0x{selectedPm4.ck24:X6} part={selectedPm4.objectPart} T=({t.X:F3},{t.Y:F3},{t.Z:F3}) Rot=({r.X:F3},{r.Y:F3},{r.Z:F3}) Scale=({s.X:F4},{s.Y:F4},{s.Z:F4})");
        }

        ImGui.TextDisabled($"Obj Move: ({_worldScene.SelectedPm4ObjectTranslation.X:F3}, {_worldScene.SelectedPm4ObjectTranslation.Y:F3}, {_worldScene.SelectedPm4ObjectTranslation.Z:F3})");
        ImGui.TextDisabled($"Obj Rot: ({_worldScene.SelectedPm4ObjectRotationDegrees.X:F3}, {_worldScene.SelectedPm4ObjectRotationDegrees.Y:F3}, {_worldScene.SelectedPm4ObjectRotationDegrees.Z:F3}) deg");
        ImGui.TextDisabled($"Obj Scale: ({_worldScene.SelectedPm4ObjectScale.X:F4}, {_worldScene.SelectedPm4ObjectScale.Y:F4}, {_worldScene.SelectedPm4ObjectScale.Z:F4})");

        ImGui.End();
    }

    private void DrawPm4WmoCorrelationWindow()
    {
        if (_worldScene == null)
        {
            _showPm4WmoCorrelationWindow = false;
            _pm4WmoCorrelationReport = null;
            return;
        }

        EnsurePm4WmoCorrelationReportLoaded();

        ImGui.SetNextWindowSize(new Vector2(1120, 720), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("PM4/WMO Correlation", ref _showPm4WmoCorrelationWindow))
        {
            ImGui.End();
            return;
        }

        int requestedMatches = _pm4WmoCorrelationMaxMatchesPerPlacement;
        ImGui.SetNextItemWidth(90f);
        if (ImGui.InputInt("Max Matches", ref requestedMatches))
        {
            _pm4WmoCorrelationMaxMatchesPerPlacement = Math.Clamp(requestedMatches, 1, 32);
            RefreshPm4WmoCorrelationReport();
        }

        ImGui.SameLine();
        if (ImGui.Button("Refresh"))
            RefreshPm4WmoCorrelationReport();

        ImGui.SameLine();
        if (ImGui.Button("Dump JSON"))
            ExportPm4WmoCorrelationJson();

        ImGui.SameLine();
        if (ImGui.Checkbox("Only Near", ref _pm4WmoCorrelationNearOnly))
        {
            if (_selectedPm4WmoCorrelationPlacementIndex >= 0)
                _selectedPm4WmoCorrelationMatchIndex = 0;
        }

        ImGui.SameLine();
        ImGui.SetNextItemWidth(260f);
        ImGui.InputTextWithHint("##Pm4WmoCorrelationFilter", "Filter model name or path", ref _pm4WmoCorrelationModelFilter, 256);

        if (_pm4WmoCorrelationReport == null)
        {
            ImGui.TextDisabled("No PM4/WMO correlation report is loaded.");
            ImGui.End();
            return;
        }

        Pm4WmoCorrelationReport report = _pm4WmoCorrelationReport;
        ImGui.TextDisabled(
            $"Generated {report.GeneratedAtUtc:yyyy-MM-dd HH:mm:ss} UTC | placements {report.Summary.WmoPlacementCount}, resolved WMO meshes {report.Summary.WmoMeshResolvedCount}, PM4 objects {report.Summary.Pm4ObjectCount}");
        ImGui.TextDisabled(
            $"Candidates {report.Summary.PlacementsWithCandidates}/{report.Summary.WmoPlacementCount}, near {report.Summary.PlacementsWithNearCandidates}, PM4 status: {report.Pm4Status}");
        ImGui.Separator();

        string filter = _pm4WmoCorrelationModelFilter.Trim();
        var filteredPlacements = report.Placements
            .Select((placement, index) => new { placement, index })
            .Where(entry => !_pm4WmoCorrelationNearOnly || entry.placement.Pm4NearCandidateCount > 0)
            .Where(entry => string.IsNullOrWhiteSpace(filter)
                || entry.placement.ModelName.Contains(filter, StringComparison.OrdinalIgnoreCase)
                || entry.placement.ModelPath.Contains(filter, StringComparison.OrdinalIgnoreCase)
                || entry.placement.ModelKey.Contains(filter, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(entry => entry.placement.Pm4Matches.Count > 0 ? entry.placement.Pm4Matches[0].FootprintOverlapRatio : 0f)
            .ThenBy(entry => entry.placement.ModelName, StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (filteredPlacements.Count == 0)
        {
            ImGui.TextDisabled("No placements matched the current filter.");
            ImGui.End();
            return;
        }

        if (!filteredPlacements.Any(entry => entry.index == _selectedPm4WmoCorrelationPlacementIndex))
        {
            _selectedPm4WmoCorrelationPlacementIndex = filteredPlacements[0].index;
            _selectedPm4WmoCorrelationMatchIndex = 0;
        }

        float leftWidth = MathF.Min(430f, ImGui.GetContentRegionAvail().X * 0.42f);
        if (ImGui.BeginChild("##Pm4WmoPlacementList", new Vector2(leftWidth, 0f), true))
        {
            for (int i = 0; i < filteredPlacements.Count; i++)
            {
                var entry = filteredPlacements[i];
                Pm4WmoCorrelationPlacement placement = entry.placement;
                bool selected = entry.index == _selectedPm4WmoCorrelationPlacementIndex;
                string label = $"[{placement.TileX},{placement.TileY}] {placement.ModelName}##Pm4WmoPlacement{entry.index}";
                if (ImGui.Selectable(label, selected))
                {
                    _selectedPm4WmoCorrelationPlacementIndex = entry.index;
                    _selectedPm4WmoCorrelationMatchIndex = 0;
                }

                ImGui.TextDisabled($"uid={placement.UniqueId} candidates={placement.Pm4CandidateCount} near={placement.Pm4NearCandidateCount}");
                if (placement.Pm4Matches.Count > 0)
                {
                    Pm4WmoCorrelationMatch best = placement.Pm4Matches[0];
                    ImGui.TextDisabled(
                        $"best CK24=0x{best.Ck24:X6} part={best.ObjectPartId} footprint={best.FootprintOverlapRatio:F2} area={best.FootprintAreaRatio:F2} dist={best.FootprintDistance:F1}");
                }
                else
                {
                    ImGui.TextDisabled("No PM4 candidates in the current tile neighborhood.");
                }

                ImGui.Separator();
            }
        }
        ImGui.EndChild();

        ImGui.SameLine();

        if (ImGui.BeginChild("##Pm4WmoPlacementDetails", Vector2.Zero, true))
        {
            Pm4WmoCorrelationPlacement placement = report.Placements[_selectedPm4WmoCorrelationPlacementIndex];
            ImGui.Text($"{placement.ModelName} (tile {placement.TileX},{placement.TileY}, uid {placement.UniqueId})");
            ImGui.TextDisabled(placement.ModelPath);

            if (ImGui.Button("Frame WMO"))
                FocusCameraOnBounds(placement.WorldBoundsMin, placement.WorldBoundsMax);

            if (placement.Pm4Matches.Count > 0)
            {
                Pm4WmoCorrelationMatch selectedMatch = placement.Pm4Matches[Math.Clamp(_selectedPm4WmoCorrelationMatchIndex, 0, placement.Pm4Matches.Count - 1)];

                ImGui.SameLine();
                if (ImGui.Button("Select PM4"))
                    SelectPm4CorrelationMatch(selectedMatch, frameCamera: false);

                ImGui.SameLine();
                if (ImGui.Button("Frame PM4"))
                    SelectPm4CorrelationMatch(selectedMatch, frameCamera: true);

                ImGui.SameLine();
                if (ImGui.Button("Frame Pair"))
                {
                    Vector3 boundsMin = Vector3.Min(placement.WorldBoundsMin, selectedMatch.BoundsMin);
                    Vector3 boundsMax = Vector3.Max(placement.WorldBoundsMax, selectedMatch.BoundsMax);
                    SelectPm4CorrelationMatch(selectedMatch, frameCamera: false);
                    FocusCameraOnBounds(boundsMin, boundsMax);
                }

                ImGui.SameLine();
                if (ImGui.Button("Snap PM4 XY"))
                    AlignPm4CorrelationMatchToPlacement(placement, selectedMatch, includeZ: false);

                ImGui.SameLine();
                if (ImGui.Button("Snap PM4 XYZ"))
                    AlignPm4CorrelationMatchToPlacement(placement, selectedMatch, includeZ: true);
            }

            ImGui.Separator();
            ImGui.TextDisabled($"Placement pos: ({placement.PlacementPosition.X:F2}, {placement.PlacementPosition.Y:F2}, {placement.PlacementPosition.Z:F2})");
            ImGui.TextDisabled($"Placement rot: ({placement.PlacementRotation.X:F2}, {placement.PlacementRotation.Y:F2}, {placement.PlacementRotation.Z:F2}) scale={placement.PlacementScale:F3}");
            ImGui.TextDisabled($"World bounds min: ({placement.WorldBoundsMin.X:F2}, {placement.WorldBoundsMin.Y:F2}, {placement.WorldBoundsMin.Z:F2})");
            ImGui.TextDisabled($"World bounds max: ({placement.WorldBoundsMax.X:F2}, {placement.WorldBoundsMax.Y:F2}, {placement.WorldBoundsMax.Z:F2})");
            if (placement.AdtPlacement.Found)
                ImGui.TextDisabled($"ADT flags=0x{placement.AdtPlacement.Flags:X4}");
            else
                ImGui.TextDisabled("No raw MODF placement metadata was found for this unique id.");

            if (placement.WmoMesh.Available)
            {
                ImGui.TextDisabled(
                    $"WMO v{placement.WmoMesh.Version}: groups={placement.WmoMesh.GroupCount} verts={placement.WmoMesh.VertexCount} tris={placement.WmoMesh.TriangleCount} batches={placement.WmoMesh.BatchCount}");
                ImGui.TextDisabled(
                    $"Footprint samples={placement.WmoMesh.FootprintSampleCount} hull={placement.WmoMesh.WorldFootprintHullPointCount} area={placement.WmoMesh.WorldFootprintArea:F1}");
            }
            else
            {
                ImGui.TextDisabled("WMO mesh summary is unavailable for this placement.");
            }

            ImGui.Separator();
            ImGui.Text($"PM4 matches ({placement.Pm4Matches.Count}/{placement.Pm4CandidateCount} shown, near={placement.Pm4NearCandidateCount})");

            if (placement.Pm4Matches.Count == 0)
            {
                ImGui.TextDisabled("No PM4 candidate objects are available for this placement.");
            }
            else if (ImGui.BeginChild("##Pm4WmoMatchList", Vector2.Zero, false))
            {
                for (int matchIndex = 0; matchIndex < placement.Pm4Matches.Count; matchIndex++)
                {
                    Pm4WmoCorrelationMatch match = placement.Pm4Matches[matchIndex];
                    bool selected = matchIndex == _selectedPm4WmoCorrelationMatchIndex;
                    string label = $"CK24 0x{match.Ck24:X6} part {match.ObjectPartId}##Pm4WmoMatch{matchIndex}";
                    if (ImGui.Selectable(label, selected))
                        _selectedPm4WmoCorrelationMatchIndex = matchIndex;

                    ImGui.TextDisabled(
                        $"tile=({match.TileX},{match.TileY}) type=0x{match.Ck24Type:X2} objId={match.Ck24ObjectId} sameTile={match.SameTile}");
                    ImGui.TextDisabled(
                        $"footprint overlap={match.FootprintOverlapRatio:F3} area={match.FootprintAreaRatio:F3} dist={match.FootprintDistance:F2}");
                    ImGui.TextDisabled(
                        $"planar gap={match.PlanarGap:F2} vertical gap={match.VerticalGap:F2} center={match.CenterDistance:F2} planar overlap={match.PlanarOverlapRatio:F3}");
                    ImGui.TextDisabled(
                        $"surfaces={match.SurfaceCount} linked refs={match.LinkedPositionRefCount} mscnRef={match.DominantMscnRefIndex} avgH={match.AverageSurfaceHeight:F2}");
                    ImGui.Separator();
                }

                ImGui.EndChild();
            }
        }
        ImGui.EndChild();

        ImGui.End();
    }

    private void SaveCurrentPm4Alignment()
    {
        if (_worldScene == null)
            return;

        _pm4SavedOverlayTranslation = _worldScene.Pm4OverlayTranslation;
        _pm4SavedOverlayRotationDegrees = _worldScene.Pm4OverlayRotationDegrees;
        _pm4SavedOverlayScale = _worldScene.Pm4OverlayScale;
        SaveViewerSettings();

        _statusMessage = $"Saved PM4 alignment: T=({_pm4SavedOverlayTranslation.X:F2}, {_pm4SavedOverlayTranslation.Y:F2}, {_pm4SavedOverlayTranslation.Z:F2}) Rot=({_pm4SavedOverlayRotationDegrees.X:F2}, {_pm4SavedOverlayRotationDegrees.Y:F2}, {_pm4SavedOverlayRotationDegrees.Z:F2})° S=({_pm4SavedOverlayScale.X:F3}, {_pm4SavedOverlayScale.Y:F3}, {_pm4SavedOverlayScale.Z:F3})";
    }

    private void ExportPm4ObjectsJson()
    {
        if (_worldScene == null)
            return;

        string defaultName = $"pm4_objects_{DateTime.Now:yyyyMMdd_HHmmss}.json";
        string? picked = ShowSaveFileDialogSTA(
            "Save PM4 Objects JSON",
            "JSON Files (*.json)|*.json|All Files (*.*)|*.*",
            ExportDir,
            defaultName);

        if (string.IsNullOrWhiteSpace(picked))
            return;

        try
        {
            string json = _worldScene.BuildPm4OverlayInterchangeJson(includeGeometry: true);
            File.WriteAllText(picked, json, Encoding.UTF8);
            _statusMessage = $"Exported PM4 objects JSON: {picked}";
        }
        catch (Exception ex)
        {
            _statusMessage = $"PM4 JSON export failed: {ex.Message}";
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 Export] JSON export failed: {ex}");
        }
    }

    private void ExportPm4ObjectsObjSet()
    {
        if (_worldScene == null)
            return;

        Directory.CreateDirectory(ExportDir);
        string? picked = ShowFolderDialogSTA(
            "Choose a folder for PM4 OBJ export",
            ExportDir,
            showNewFolderButton: true);

        if (string.IsNullOrWhiteSpace(picked))
            return;

        try
        {
            Pm4OfflineObjExportSummary summary = _worldScene.ExportPm4ObjectsAsObjDirectory(picked);
            _statusMessage =
                $"Exported PM4 OBJ set: {summary.ExportedObjectCount} objects across {summary.ExportedTileCount} tiles to {summary.OutputDirectory} (manifest: {summary.ManifestPath}).";
        }
        catch (Exception ex)
        {
            _statusMessage = $"PM4 OBJ export failed: {ex.Message}";
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 Export] OBJ export failed: {ex}");
        }
    }

    private void ExportPm4WmoCorrelationJson()
    {
        if (_worldScene == null)
            return;

        string defaultName = $"pm4_wmo_correlation_{DateTime.Now:yyyyMMdd_HHmmss}.json";
        string? picked = ShowSaveFileDialogSTA(
            "Save PM4/WMO Correlation JSON",
            "JSON Files (*.json)|*.json|All Files (*.*)|*.*",
            ExportDir,
            defaultName);

        if (string.IsNullOrWhiteSpace(picked))
            return;

        try
        {
            string json = _worldScene.BuildPm4WmoPlacementCorrelationJson();
            File.WriteAllText(picked, json, Encoding.UTF8);
            _statusMessage = $"Exported PM4/WMO correlation JSON: {picked}";
        }
        catch (Exception ex)
        {
            _statusMessage = $"PM4/WMO correlation export failed: {ex.Message}";
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 Correlation] JSON export failed: {ex}");
        }
    }

    private void InvalidatePm4DerivedReports()
    {
        _pm4ObjectMatchReport = null;
        _selectedPm4ObjectMatch = null;
        _selectedPm4ObjectMatchKey = null;
        _selectedPm4ObjectMatchCacheMaxMatches = -1;
        _hoveredPm4ObjectMatch = null;
        _hoveredPm4ObjectMatchKey = null;
        _hoveredPm4ObjectMatchCacheMaxMatches = -1;
        _pm4WmoCorrelationReport = null;
    }

    private void EnsurePm4WmoCorrelationReportLoaded()
    {
        if (_pm4WmoCorrelationReport == null)
            RefreshPm4WmoCorrelationReport();
    }

    private void RefreshPm4WmoCorrelationReport()
    {
        if (_worldScene == null)
            return;

        try
        {
            _pm4WmoCorrelationReport = _worldScene.BuildPm4WmoPlacementCorrelationReport(_pm4WmoCorrelationMaxMatchesPerPlacement);
            if (_pm4WmoCorrelationReport.Placements.Count == 0)
            {
                _selectedPm4WmoCorrelationPlacementIndex = -1;
                _selectedPm4WmoCorrelationMatchIndex = 0;
            }
            else if (_selectedPm4WmoCorrelationPlacementIndex < 0 || _selectedPm4WmoCorrelationPlacementIndex >= _pm4WmoCorrelationReport.Placements.Count)
            {
                _selectedPm4WmoCorrelationPlacementIndex = 0;
                _selectedPm4WmoCorrelationMatchIndex = 0;
            }

            _statusMessage = $"Refreshed PM4/WMO correlation report ({_pm4WmoCorrelationReport.Summary.WmoPlacementCount} placements).";
        }
        catch (Exception ex)
        {
            _pm4WmoCorrelationReport = null;
            _statusMessage = $"PM4/WMO correlation refresh failed: {ex.Message}";
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 Correlation] Report refresh failed: {ex}");
        }
    }

    private void EnsurePm4ObjectMatchReportLoaded()
    {
        if (_pm4ObjectMatchReport == null)
            RefreshPm4ObjectMatchReport();
    }

    private void RefreshPm4ObjectMatchReport()
    {
        if (_worldScene == null)
            return;

        try
        {
            _pm4ObjectMatchReport = _worldScene.BuildPm4ObjectMatchReport(_pm4ObjectMatchMaxMatchesPerObject);
            if (_pm4ObjectMatchReport.Objects.Count == 0)
            {
                _selectedPm4ObjectMatchObjectIndex = -1;
                _selectedPm4ObjectMatchCandidateIndex = 0;
            }
            else if (!TryGetSelectedPm4ObjectMatch(out _)
                && (_selectedPm4ObjectMatchObjectIndex < 0 || _selectedPm4ObjectMatchObjectIndex >= _pm4ObjectMatchReport.Objects.Count))
            {
                _selectedPm4ObjectMatchObjectIndex = 0;
                _selectedPm4ObjectMatchCandidateIndex = 0;
            }

            _statusMessage = $"Refreshed PM4 object match report ({_pm4ObjectMatchReport.Summary.Pm4ObjectCount} PM4 objects).";
        }
        catch (Exception ex)
        {
            _pm4ObjectMatchReport = null;
            _statusMessage = $"PM4 object match refresh failed: {ex.Message}";
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 Object Match] Report refresh failed: {ex}");
        }
    }

    private void DrawPm4ObjectMatchWindow()
    {
        if (_worldScene == null)
        {
            _showPm4ObjectMatchWindow = false;
            _pm4ObjectMatchReport = null;
            return;
        }

        EnsurePm4ObjectMatchReportLoaded();

        ImGui.SetNextWindowSize(new Vector2(1220, 760), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("PM4 Object Match", ref _showPm4ObjectMatchWindow))
        {
            ImGui.End();
            return;
        }

        int requestedMatches = _pm4ObjectMatchMaxMatchesPerObject;
        ImGui.SetNextItemWidth(110f);
        if (ImGui.SliderInt("Top Matches", ref requestedMatches, 3, 5))
        {
            _pm4ObjectMatchMaxMatchesPerObject = Math.Clamp(requestedMatches, 3, 5);
            RefreshPm4ObjectMatchReport();
        }

        ImGui.SameLine();
        if (ImGui.Button("Refresh"))
            RefreshPm4ObjectMatchReport();

        if (_pm4ObjectMatchReport == null)
        {
            ImGui.TextDisabled("No PM4 object match report is loaded.");
            ImGui.End();
            return;
        }

        Pm4ObjectMatchReport report = _pm4ObjectMatchReport;
        ImGui.TextDisabled(
            $"Generated {report.GeneratedAtUtc:yyyy-MM-dd HH:mm:ss} UTC | PM4 objects {report.Summary.Pm4ObjectCount}, WMO placements {report.Summary.WmoPlacementCount}, M2 placements {report.Summary.M2PlacementCount}");
        ImGui.TextDisabled(
            $"Objects with candidates {report.Summary.ObjectsWithCandidates}/{report.Summary.Pm4ObjectCount}, near {report.Summary.ObjectsWithNearCandidates}, status: {report.Pm4Status}");
        ImGui.TextDisabled("Ranking keeps WMO-mesh priority for non-zero families, but zero/root PM4 objects with linked refs now prefer M2 anchors before the usual tile/anchor/planar fit checks.");
        ImGui.Separator();

        if (!_worldScene.HasSelectedPm4Object)
        {
            ImGui.TextDisabled("Select a PM4 object in the scene to see its top suggested matches.");
            ImGui.End();
            return;
        }

        if (!TryGetSelectedPm4ObjectMatch(out Pm4ObjectMatchObject objectMatch))
        {
            ImGui.TextDisabled("The selected PM4 object is not present in the current match report. Refresh and try again.");
            ImGui.End();
            return;
        }

        DrawPm4SelectedObjectMatchSuggestions("WindowPm4Match", compact: false);

        ImGui.End();
    }

    private bool TryGetSelectedPm4ObjectMatch(out Pm4ObjectMatchObject objectMatch)
    {
        objectMatch = null!;

        if (_worldScene == null || !_worldScene.SelectedPm4ObjectKey.HasValue)
            return false;

        var selectedKey = _worldScene.SelectedPm4ObjectKey.Value;
        if (_selectedPm4ObjectMatch != null
            && _selectedPm4ObjectMatchKey.HasValue
            && _selectedPm4ObjectMatchKey.Value == selectedKey
            && _selectedPm4ObjectMatchCacheMaxMatches == _pm4ObjectMatchMaxMatchesPerObject)
        {
            objectMatch = _selectedPm4ObjectMatch;
            return true;
        }

        if (!_worldScene.TryBuildSelectedPm4ObjectMatch(_pm4ObjectMatchMaxMatchesPerObject, out Pm4ObjectMatchObject selectedMatch))
            return false;

        _selectedPm4ObjectMatch = selectedMatch;
        _selectedPm4ObjectMatchKey = selectedKey;
        _selectedPm4ObjectMatchCacheMaxMatches = _pm4ObjectMatchMaxMatchesPerObject;
        objectMatch = selectedMatch;

        if (_pm4ObjectMatchReport == null)
            return true;

        for (int index = 0; index < _pm4ObjectMatchReport.Objects.Count; index++)
        {
            Pm4ObjectMatchObject candidate = _pm4ObjectMatchReport.Objects[index];
            if (candidate.TileX != selectedKey.tileX
                || candidate.TileY != selectedKey.tileY
                || candidate.Ck24 != selectedKey.ck24
                || candidate.ObjectPartId != selectedKey.objectPart)
            {
                continue;
            }

            _selectedPm4ObjectMatchObjectIndex = index;
            if (_selectedPm4ObjectMatchCandidateIndex < 0 || _selectedPm4ObjectMatchCandidateIndex >= candidate.Candidates.Count)
                _selectedPm4ObjectMatchCandidateIndex = 0;
            return true;
        }

        _selectedPm4ObjectMatchObjectIndex = -1;
        return true;
    }

    private void DrawPm4SelectedObjectMatchSuggestions(string idSuffix, bool compact)
    {
        if (_worldScene == null || !_worldScene.HasSelectedPm4Object)
        {
            ImGui.TextDisabled("Select a PM4 object in the scene to see suggested matches.");
            return;
        }

        if (!TryGetSelectedPm4ObjectMatch(out Pm4ObjectMatchObject objectMatch))
        {
            ImGui.TextDisabled("No PM4 match report entry is available for the current selection.");
            return;
        }

        int shownCandidateCount = Math.Min(objectMatch.Candidates.Count, Math.Clamp(_pm4ObjectMatchMaxMatchesPerObject, 3, 5));
        bool hasSaved = TryGetSavedPm4ObjectMatch(objectMatch, out SavedPm4ObjectMatchSelection? savedSelection);

        ImGui.Text($"CK24 0x{objectMatch.Ck24:X6} part {objectMatch.ObjectPartId} (tile {objectMatch.TileX},{objectMatch.TileY})");
        ImGui.TextDisabled($"wmo={objectMatch.WmoCandidateCount} m2={objectMatch.M2CandidateCount} near={objectMatch.NearCandidateCount} refs={objectMatch.LinkedPositionRefCount} mslk=0x{objectMatch.LinkGroupObjectId:X8}");

        if (hasSaved && savedSelection != null)
            ImGui.TextColored(new Vector4(0.95f, 0.85f, 0.35f, 1f), $"Saved: {savedSelection.PlacementKind} uid={savedSelection.PlacementUniqueId} {savedSelection.ModelName} [{savedSelection.EvidenceSource}]");
        else
            ImGui.TextDisabled("Saved: none");

        if (!compact)
        {
            if (ImGui.Button($"Frame PM4##{idSuffix}"))
                SelectPm4ObjectMatchObject(objectMatch, frameCamera: true);

            ImGui.SameLine();
            if (ImGui.Button($"Jump To Alignment##{idSuffix}"))
                SelectPm4ObjectMatchObject(objectMatch, frameCamera: false);

            ImGui.SameLine();
            if (ImGui.Button($"Clear Saved Choice##{idSuffix}"))
                ClearSavedPm4ObjectMatch(objectMatch);
        }
        else if (ImGui.SmallButton($"Clear Saved Choice##{idSuffix}"))
        {
            ClearSavedPm4ObjectMatch(objectMatch);
        }

        ImGui.Separator();
        ImGui.Text($"Top matches ({shownCandidateCount}/{objectMatch.CandidateCount})");

        if (shownCandidateCount == 0)
        {
            ImGui.TextDisabled("No placement candidates are available for this PM4 object.");
            return;
        }

        for (int candidateIndex = 0; candidateIndex < shownCandidateCount; candidateIndex++)
        {
            Pm4ObjectMatchCandidate candidate = objectMatch.Candidates[candidateIndex];
            bool isSaved = savedSelection != null
                && string.Equals(savedSelection.PlacementKind, candidate.Kind, StringComparison.OrdinalIgnoreCase)
                && savedSelection.PlacementUniqueId == candidate.UniqueId
                && string.Equals(savedSelection.ModelPath, candidate.ModelPath, StringComparison.OrdinalIgnoreCase);

            ImGui.PushID($"{idSuffix}_{candidateIndex}");
            if (isSaved)
                ImGui.TextColored(new Vector4(0.95f, 0.85f, 0.35f, 1f), $"{candidateIndex + 1}. {candidate.Kind.ToUpperInvariant()} {candidate.ModelName}  [saved]");
            else
                ImGui.TextWrapped($"{candidateIndex + 1}. {candidate.Kind.ToUpperInvariant()} {candidate.ModelName}");

            ImGui.TextDisabled($"{candidate.EvidenceSource} | tile {candidate.TileX},{candidate.TileY} | anchor {candidate.AnchorPlanarGap:F1} | planar {candidate.PlanarGap:F1} | center {candidate.CenterDistance:F1}");
            if (!compact)
                ImGui.TextDisabled(candidate.ModelPath);

            if (ImGui.SmallButton("Frame"))
            {
                _selectedPm4ObjectMatchCandidateIndex = candidateIndex;
                FocusCameraOnBounds(candidate.WorldBoundsMin, candidate.WorldBoundsMax);
            }

            ImGui.SameLine();
            if (ImGui.SmallButton("Save"))
            {
                _selectedPm4ObjectMatchCandidateIndex = candidateIndex;
                SavePm4ObjectMatchSelection(objectMatch, candidate);
            }

            if (!compact)
            {
                ImGui.SameLine();
                if (ImGui.SmallButton("Frame Pair"))
                {
                    _selectedPm4ObjectMatchCandidateIndex = candidateIndex;
                    SelectPm4ObjectMatchObject(objectMatch, frameCamera: false);
                    FocusCameraOnBounds(Vector3.Min(objectMatch.BoundsMin, candidate.WorldBoundsMin), Vector3.Max(objectMatch.BoundsMax, candidate.WorldBoundsMax));
                }
            }

            ImGui.PopID();
            if (candidateIndex + 1 < shownCandidateCount)
                ImGui.Separator();
        }
    }

    private void SelectPm4ObjectMatchObject(Pm4ObjectMatchObject objectMatch, bool frameCamera)
    {
        if (_worldScene == null)
            return;

        if (_worldScene.SelectPm4Object((objectMatch.TileX, objectMatch.TileY, objectMatch.Ck24, objectMatch.ObjectPartId)))
        {
            OpenPm4Workbench(Pm4WorkbenchTab.Selection);
            if (frameCamera)
                FocusCameraOnBounds(objectMatch.BoundsMin, objectMatch.BoundsMax);

            _statusMessage = $"Selected PM4 object CK24=0x{objectMatch.Ck24:X6} part={objectMatch.ObjectPartId}.";
        }
        else
        {
            _statusMessage = $"PM4 object CK24=0x{objectMatch.Ck24:X6} part={objectMatch.ObjectPartId} is no longer available.";
        }
    }

    private void SavePm4ObjectMatchSelection(Pm4ObjectMatchObject objectMatch, Pm4ObjectMatchCandidate candidate)
    {
        string mapName = _terrainManager?.MapName ?? _worldScene?.Terrain.MapName ?? string.Empty;
        if (string.IsNullOrWhiteSpace(mapName))
        {
            _statusMessage = "Cannot save PM4 object match: map name is unavailable.";
            return;
        }

        string key = BuildSavedPm4ObjectMatchKey(mapName, objectMatch.TileX, objectMatch.TileY, objectMatch.Ck24, objectMatch.ObjectPartId);
        _savedPm4ObjectMatches[key] = new SavedPm4ObjectMatchSelection
        {
            MapName = mapName,
            TileX = objectMatch.TileX,
            TileY = objectMatch.TileY,
            Ck24 = objectMatch.Ck24,
            ObjectPartId = objectMatch.ObjectPartId,
            PlacementKind = candidate.Kind,
            PlacementUniqueId = candidate.UniqueId,
            PlacementTileX = candidate.TileX,
            PlacementTileY = candidate.TileY,
            ModelName = candidate.ModelName,
            ModelPath = candidate.ModelPath,
            EvidenceSource = candidate.EvidenceSource,
        };

        SaveViewerSettings();
        _statusMessage = $"Saved PM4 object match: CK24=0x{objectMatch.Ck24:X6} part={objectMatch.ObjectPartId} -> {candidate.Kind} uid={candidate.UniqueId}.";
    }

    private void ClearSavedPm4ObjectMatch(Pm4ObjectMatchObject objectMatch)
    {
        string mapName = _terrainManager?.MapName ?? _worldScene?.Terrain.MapName ?? string.Empty;
        string key = BuildSavedPm4ObjectMatchKey(mapName, objectMatch.TileX, objectMatch.TileY, objectMatch.Ck24, objectMatch.ObjectPartId);
        if (_savedPm4ObjectMatches.Remove(key))
        {
            SaveViewerSettings();
            _statusMessage = $"Cleared saved PM4 object match for CK24=0x{objectMatch.Ck24:X6} part={objectMatch.ObjectPartId}.";
        }
    }

    private bool TryGetSavedPm4ObjectMatch(Pm4ObjectMatchObject objectMatch, out SavedPm4ObjectMatchSelection? selection)
    {
        string mapName = _terrainManager?.MapName ?? _worldScene?.Terrain.MapName ?? string.Empty;
        string key = BuildSavedPm4ObjectMatchKey(mapName, objectMatch.TileX, objectMatch.TileY, objectMatch.Ck24, objectMatch.ObjectPartId);
        if (_savedPm4ObjectMatches.TryGetValue(key, out SavedPm4ObjectMatchSelection? savedSelection))
        {
            selection = savedSelection;
            return true;
        }

        selection = null;
        return false;
    }

    private static string BuildSavedPm4ObjectMatchKey(string mapName, int tileX, int tileY, uint ck24, int objectPartId)
    {
        return $"{mapName.Trim().ToLowerInvariant()}|{tileX}|{tileY}|{ck24:X6}|{objectPartId}";
    }

    private void SelectPm4CorrelationMatch(Pm4WmoCorrelationMatch match, bool frameCamera)
    {
        if (_worldScene == null)
            return;

        if (_worldScene.SelectPm4Object((match.TileX, match.TileY, match.Ck24, match.ObjectPartId)))
        {
            OpenPm4Workbench(Pm4WorkbenchTab.Selection);
            if (frameCamera)
                FocusCameraOnBounds(match.BoundsMin, match.BoundsMax);

            _statusMessage = $"Selected PM4 candidate CK24=0x{match.Ck24:X6} part={match.ObjectPartId} from correlation panel.";
        }
        else
        {
            _statusMessage = $"PM4 candidate CK24=0x{match.Ck24:X6} part={match.ObjectPartId} is no longer available.";
        }
    }

    private void AlignPm4CorrelationMatchToPlacement(Pm4WmoCorrelationPlacement placement, Pm4WmoCorrelationMatch match, bool includeZ)
    {
        if (_worldScene == null)
            return;

        if (!_worldScene.SelectPm4Object((match.TileX, match.TileY, match.Ck24, match.ObjectPartId)))
        {
            _statusMessage = $"PM4 candidate CK24=0x{match.Ck24:X6} part={match.ObjectPartId} is no longer available.";
            return;
        }

        if (!_worldScene.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo debugInfo))
        {
            _statusMessage = "PM4 snap failed: selected object debug info is unavailable.";
            return;
        }

        Vector3 placementCenter = (placement.WorldBoundsMin + placement.WorldBoundsMax) * 0.5f;
        Vector3 delta = placementCenter - debugInfo.Center;
        if (!includeZ)
            delta.Z = 0f;

        _worldScene.SelectedPm4ObjectTranslation += delta;
        InvalidatePm4DerivedReports();
        _showPm4AlignmentWindow = true;

        string axes = includeZ ? "XYZ" : "XY";
        _statusMessage =
            $"Snapped PM4 CK24=0x{match.Ck24:X6} part={match.ObjectPartId} to WMO center ({axes}) by ({delta.X:F2}, {delta.Y:F2}, {delta.Z:F2}).";
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

    private void ApplySavedPm4AlignmentToScene()
    {
        if (_worldScene == null)
            return;

        _worldScene.Pm4OverlayTranslation = _pm4SavedOverlayTranslation;
        _worldScene.Pm4OverlayRotationDegrees = _pm4SavedOverlayRotationDegrees;
        _worldScene.Pm4OverlayScale = _pm4SavedOverlayScale;
        InvalidatePm4DerivedReports();
    }

    private static Vector3 NormalizeRotationDegrees(Vector3 rotation)
    {
        return new Vector3(
            NormalizeDegrees(rotation.X),
            NormalizeDegrees(rotation.Y),
            NormalizeDegrees(rotation.Z));
    }

    private static float NormalizeDegrees(float value)
    {
        float wrapped = value % 360f;
        if (wrapped < -180f)
            wrapped += 360f;
        else if (wrapped > 180f)
            wrapped -= 360f;
        return wrapped;
    }

    private static float ToggleWindingComponent(float value)
    {
        float magnitude = MathF.Abs(value);
        if (magnitude < 0.0001f)
            magnitude = 1f;

        return value < 0f ? magnitude : -magnitude;
    }

    private static string GetPm4ColorModeLabel(Pm4OverlayColorMode mode)
    {
        return mode switch
        {
            Pm4OverlayColorMode.Ck24Type => "CK24 Type",
            Pm4OverlayColorMode.Ck24ObjectId => "CK24 ObjectId",
            Pm4OverlayColorMode.Ck24Key => "CK24 Key",
            Pm4OverlayColorMode.Tile => "Tile",
            Pm4OverlayColorMode.MshdRegionId => "MSHD RegionId",
            Pm4OverlayColorMode.GroupKey => "Group Key",
            Pm4OverlayColorMode.AttributeMask => "Attribute Mask",
            Pm4OverlayColorMode.Height => "Height",
            Pm4OverlayColorMode.TypeFlags => "TypeFlags",
            Pm4OverlayColorMode.Ck24TypeVsTypeFlags => "CK24Type vs TypeFlags",
            _ => mode.ToString(),
        };
    }

    private void DrawPm4ColorLegend(string idSuffix = "")
    {
        if (_worldScene == null || !_worldScene.ShowPm4Overlay)
            return;

        Pm4ColorLegendInfo legend = _worldScene.GetPm4ColorLegend();
        if (!ImGui.CollapsingHeader($"PM4 Color Legend##{idSuffix}", ImGuiTreeNodeFlags.DefaultOpen))
            return;

        if (!string.IsNullOrWhiteSpace(legend.Description))
            ImGui.TextDisabled(legend.Description);

        if (ImGui.Button($"Export PM4 LLM Bundle##{idSuffix}"))
            ExportPm4LlmEvidenceBundle();

        if (legend.Entries.Count == 0)
        {
            ImGui.TextDisabled("No loaded PM4 objects for the current legend mode.");
            return;
        }

        for (int i = 0; i < legend.Entries.Count; i++)
        {
            Pm4ColorLegendEntry entry = legend.Entries[i];
            ImGui.ColorButton(
                $"##Pm4LegendColor{idSuffix}_{i}",
                new Vector4(entry.Color, 1f),
                ImGuiColorEditFlags.NoTooltip | ImGuiColorEditFlags.NoDragDrop,
                new Vector2(14f, 14f));
            ImGui.SameLine();
            if (entry.IsSelected)
                ImGui.TextColored(new Vector4(1f, 1f, 0.35f, 1f), $"{entry.Label}  [{entry.ObjectCount}]  selected");
            else if (legend.IsContinuous)
                ImGui.TextUnformatted(entry.Label);
            else
                ImGui.TextUnformatted($"{entry.Label}  [{entry.ObjectCount}]");
        }

        if (legend.IsTruncated)
            ImGui.TextDisabled($"Showing {legend.Entries.Count} of {legend.TotalEntryCount} legend entries.");
    }

    private void DrawSelectedPm4RegionSummary(string idSuffix)
    {
        if (_worldScene == null || !_worldScene.TryGetSelectedPm4RegionInfo(out Pm4SelectedObjectRegionInfo regionInfo))
            return;

        if (!ImGui.CollapsingHeader($"Selected MSHD Region##{idSuffix}", ImGuiTreeNodeFlags.DefaultOpen))
            return;

        ImGui.TextDisabled("Visible peers from the current camera-window PM4 overlay that share the selected object's MSHD.Field04 region id.");
        ImGui.TextDisabled($"Region {regionInfo.RegionId} | objects={regionInfo.VisibleObjectCount} tiles={regionInfo.VisibleTileCount} unique CK24={regionInfo.UniqueCk24Count} unique MSLK={regionInfo.UniqueLinkGroupCount} unique MscnRef={regionInfo.UniqueMscnRefCount}");
        ImGui.TextDisabled($"Same CK24={regionInfo.SameCk24Count} same MSLK={regionInfo.SameLinkGroupCount} same MscnRef={regionInfo.SameMscnRefCount} avg surfaces={regionInfo.AverageSurfaceCount:F1} avg center Z={regionInfo.AverageCenterHeight:F1}");
        ImGui.TextDisabled($"Type mix: {FormatPm4TypeBuckets(regionInfo.TypeBuckets)}");

        if (ImGui.Button($"Collect Visible Region##{idSuffix}"))
            AddPm4VisibleRegionToCollection(regionInfo.RegionId);

        ImGui.SameLine();
        if (ImGui.Button($"Export PM4 LLM Bundle##Region{idSuffix}"))
            ExportPm4LlmEvidenceBundle();

        if (ImGui.BeginChild($"Pm4RegionPeers##{idSuffix}", new Vector2(0f, 190f), true))
        {
            for (int index = 0; index < regionInfo.Peers.Count; index++)
            {
                Pm4RegionPeerSummary peer = regionInfo.Peers[index];
                string label = $"{index + 1}. tile=({peer.ObjectKey.tileX},{peer.ObjectKey.tileY}) CK24=0x{peer.ObjectKey.ck24:X6} part={peer.ObjectKey.objectPart} type=0x{peer.Ck24Type:X2} surf={peer.SurfaceCount}";
                if (peer.IsSelected)
                    ImGui.TextColored(new Vector4(1f, 0.95f, 0.35f, 1f), $"{label}  [selected]");
                else
                    ImGui.TextUnformatted(label);

                ImGui.TextDisabled(
                    $"objId={peer.Ck24ObjectId} mslk=0x{peer.LinkGroupObjectId:X8} mscnRef={peer.DominantMscnRefIndex} center=({peer.Center.X:F1}, {peer.Center.Y:F1}, {peer.Center.Z:F1}) {FormatPm4PeerFlags(peer)}");

                ImGui.PushID($"Pm4RegionPeer{idSuffix}_{index}");
                if (!peer.IsSelected && ImGui.SmallButton("Select"))
                    SelectPm4GraphPart(peer.ObjectKey, frameCamera: false);

                if (!peer.IsSelected)
                    ImGui.SameLine();

                if (ImGui.SmallButton("Frame"))
                    SelectPm4GraphPart(peer.ObjectKey, frameCamera: true);

                ImGui.SameLine();
                if (ImGui.SmallButton("Collect"))
                    TogglePm4ObjectCollectionMembership(peer.ObjectKey, reportStatus: true, removeIfPresent: false);

                ImGui.PopID();
                if (index + 1 < regionInfo.Peers.Count)
                    ImGui.Separator();
            }
        }

        ImGui.EndChild();
    }

    private int AddPm4VisibleRegionToCollection(uint regionId)
    {
        if (_worldScene == null)
            return 0;

        int added = AddPm4ObjectsToCollection(_worldScene.GetVisiblePm4ObjectsForRegion(regionId));
        _statusMessage = added > 0
            ? $"Added {added} visible PM4 parts from MSHD region {regionId} to the collection."
            : $"All visible PM4 parts from MSHD region {regionId} were already in the collection.";
        SyncPm4CollectionHighlight();
        return added;
    }

    private void ExportPm4LlmEvidenceBundle()
    {
        if (_worldScene == null)
            return;

        Directory.CreateDirectory(ExportDir);
        string? picked = ShowFolderDialogSTA(
            "Choose a folder for the PM4 LLM evidence bundle",
            ExportDir,
            showNewFolderButton: true);

        if (string.IsNullOrWhiteSpace(picked))
            return;

        try
        {
            string mapName = _terrainManager?.MapName ?? _worldScene.Terrain.MapName ?? "map";
            string bundleDirectory = Path.Combine(
                picked,
                $"pm4_llm_{SanitizeProjectPathSegment(mapName)}_{DateTime.Now:yyyyMMdd_HHmmss}");
            Directory.CreateDirectory(bundleDirectory);

            Pm4VisibleOverlaySummaryInfo visibleSummary = _worldScene.GetPm4VisibleOverlaySummary();
            Pm4ObjectDebugInfo selectedDebugInfo = default;
            bool hasSelectedObject = _worldScene.SelectedPm4ObjectKey.HasValue
                && _worldScene.TryGetSelectedPm4ObjectDebugInfo(out selectedDebugInfo);
            bool hasSelectedRegion = _worldScene.TryGetSelectedPm4RegionInfo(out Pm4SelectedObjectRegionInfo selectedRegionInfo);

            string jsonPath = Path.Combine(bundleDirectory, "pm4_llm_bundle.json");
            string markdownPath = Path.Combine(bundleDirectory, "pm4_llm_bundle.md");
            string visibleRegionsSvgPath = Path.Combine(bundleDirectory, "pm4_visible_regions.svg");
            string selectedRegionSvgPath = Path.Combine(bundleDirectory, "pm4_selected_region.svg");

            string json = JsonSerializer.Serialize(
                BuildJsonSafePm4LlmBundle(
                    visibleSummary,
                    hasSelectedObject ? selectedDebugInfo : null,
                    hasSelectedRegion ? selectedRegionInfo : null),
                new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(jsonPath, json, Encoding.UTF8);
            File.WriteAllText(markdownPath, BuildPm4LlmBundleMarkdown(visibleSummary, hasSelectedObject ? selectedDebugInfo : null, hasSelectedRegion ? selectedRegionInfo : null), Encoding.UTF8);
            File.WriteAllText(visibleRegionsSvgPath, BuildPm4VisibleRegionsSvg(visibleSummary), Encoding.UTF8);
            if (hasSelectedRegion)
                File.WriteAllText(selectedRegionSvgPath, BuildPm4SelectedRegionSvg(selectedRegionInfo), Encoding.UTF8);

            _statusMessage = hasSelectedRegion
                ? $"Exported PM4 LLM bundle to {bundleDirectory} (JSON, Markdown, visible-regions SVG, selected-region SVG)."
                : $"Exported PM4 LLM bundle to {bundleDirectory} (JSON, Markdown, visible-regions SVG).";
        }
        catch (Exception ex)
        {
            _statusMessage = $"PM4 LLM bundle export failed: {ex.Message}";
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 LLM Bundle] Export failed: {ex}");
        }
    }

    private object BuildJsonSafePm4LlmBundle(
        Pm4VisibleOverlaySummaryInfo visibleSummary,
        Pm4ObjectDebugInfo? selectedDebugInfo,
        Pm4SelectedObjectRegionInfo? selectedRegionInfo)
    {
        Pm4ColorLegendInfo legend = _worldScene!.GetPm4ColorLegend(12);
        string mapName = _terrainManager?.MapName ?? _worldScene.Terrain.MapName ?? string.Empty;

        return new
        {
            generatedAtUtc = DateTime.UtcNow,
            mapName,
            pm4Status = _worldScene.Pm4Status,
            pm4VisibleObjectCount = _worldScene.Pm4VisibleObjectCount,
            pm4ObjectCount = _worldScene.Pm4ObjectCount,
            pm4LoadedFiles = _worldScene.Pm4LoadedFiles,
            pm4TotalFiles = _worldScene.Pm4TotalFiles,
            colorMode = _worldScene.Pm4ColorMode.ToString(),
            colorModeLabel = GetPm4ColorModeLabel(_worldScene.Pm4ColorMode),
            legend = new
            {
                description = legend.Description,
                totalEntryCount = legend.TotalEntryCount,
                shownEntryCount = legend.Entries.Count,
                entries = legend.Entries.Select(entry => new
                {
                    label = entry.Label,
                    objectCount = entry.ObjectCount,
                    isSelected = entry.IsSelected
                }).ToList()
            },
            visibleOverlay = new
            {
                objectCount = visibleSummary.VisibleObjectCount,
                tileCount = visibleSummary.VisibleTileCount,
                regionCount = visibleSummary.RegionCount,
                selectedRegionId = visibleSummary.SelectedRegionId,
                topRegions = visibleSummary.Regions.Select(region => new
                {
                    regionId = region.RegionId,
                    objectCount = region.ObjectCount,
                    tileCount = region.TileCount,
                    uniqueCk24Count = region.UniqueCk24Count,
                    uniqueLinkGroupCount = region.UniqueLinkGroupCount,
                    averageCenterHeight = region.AverageCenterHeight,
                    isSelectedRegion = region.IsSelectedRegion,
                    typeBuckets = region.TypeBuckets.Select(static bucket => new
                    {
                        ck24Type = bucket.Ck24Type,
                        objectCount = bucket.ObjectCount
                    }).ToList()
                }).ToList()
            },
            selectedObject = selectedDebugInfo.HasValue && _worldScene.SelectedPm4ObjectKey.HasValue
                ? new
                {
                    tileX = _worldScene.SelectedPm4ObjectKey.Value.tileX,
                    tileY = _worldScene.SelectedPm4ObjectKey.Value.tileY,
                    ck24 = selectedDebugInfo.Value.Ck24,
                    ck24Type = selectedDebugInfo.Value.Ck24Type,
                    ck24ObjectId = selectedDebugInfo.Value.Ck24ObjectId,
                    objectPartId = _worldScene.SelectedPm4ObjectKey.Value.objectPart,
                    mshd = new
                    {
                        field00 = selectedDebugInfo.Value.MshdField00,
                        regionId = selectedDebugInfo.Value.MshdRegionId,
                        field08 = selectedDebugInfo.Value.MshdField08
                    },
                    linkGroupObjectId = selectedDebugInfo.Value.LinkGroupObjectId,
                    dominantMscnRefIndex = selectedDebugInfo.Value.DominantMscnRefIndex,
                    linkedPositionRefCount = selectedDebugInfo.Value.LinkedPositionRefCount,
                    surfaceCount = selectedDebugInfo.Value.SurfaceCount,
                    dominantGroupKey = selectedDebugInfo.Value.DominantGroupKey,
                    dominantAttributeMask = selectedDebugInfo.Value.DominantAttributeMask,
                    averageSurfaceHeight = JsonFiniteOrNull(selectedDebugInfo.Value.AverageSurfaceHeight),
                    center = VectorToArray(selectedDebugInfo.Value.Center),
                    boundsMin = VectorToArray(selectedDebugInfo.Value.BoundsMin),
                    boundsMax = VectorToArray(selectedDebugInfo.Value.BoundsMax)
                }
                : null,
            selectedRegion = selectedRegionInfo.HasValue
                ? new
                {
                    regionId = selectedRegionInfo.Value.RegionId,
                    visibleObjectCount = selectedRegionInfo.Value.VisibleObjectCount,
                    visibleTileCount = selectedRegionInfo.Value.VisibleTileCount,
                    uniqueCk24Count = selectedRegionInfo.Value.UniqueCk24Count,
                    uniqueLinkGroupCount = selectedRegionInfo.Value.UniqueLinkGroupCount,
                    uniqueMscnRefCount = selectedRegionInfo.Value.UniqueMscnRefCount,
                    sameCk24Count = selectedRegionInfo.Value.SameCk24Count,
                    sameLinkGroupCount = selectedRegionInfo.Value.SameLinkGroupCount,
                    sameMscnRefCount = selectedRegionInfo.Value.SameMscnRefCount,
                    averageSurfaceCount = selectedRegionInfo.Value.AverageSurfaceCount,
                    averageCenterHeight = selectedRegionInfo.Value.AverageCenterHeight,
                    typeBuckets = selectedRegionInfo.Value.TypeBuckets.Select(static bucket => new
                    {
                        ck24Type = bucket.Ck24Type,
                        objectCount = bucket.ObjectCount
                    }).ToList(),
                    peers = selectedRegionInfo.Value.Peers.Select(peer => new
                    {
                        tileX = peer.ObjectKey.tileX,
                        tileY = peer.ObjectKey.tileY,
                        ck24 = peer.ObjectKey.ck24,
                        objectPartId = peer.ObjectKey.objectPart,
                        ck24Type = peer.Ck24Type,
                        ck24ObjectId = peer.Ck24ObjectId,
                        surfaceCount = peer.SurfaceCount,
                        linkGroupObjectId = peer.LinkGroupObjectId,
                        dominantMscnRefIndex = peer.DominantMscnRefIndex,
                        center = VectorToArray(peer.Center),
                        isSelected = peer.IsSelected,
                        sameCk24 = peer.SameCk24,
                        sameLinkGroup = peer.SameLinkGroup,
                        sameMscnRefIndex = peer.SameMscnRefIndex
                    }).ToList()
                }
                : null
        };
    }

    private string BuildPm4LlmBundleMarkdown(
        Pm4VisibleOverlaySummaryInfo visibleSummary,
        Pm4ObjectDebugInfo? selectedDebugInfo,
        Pm4SelectedObjectRegionInfo? selectedRegionInfo)
    {
        string mapName = _terrainManager?.MapName ?? _worldScene!.Terrain.MapName ?? "map";
        var builder = new StringBuilder();
        builder.AppendLine("# PM4 Visible Overlay LLM Bundle");
        builder.AppendLine();
        builder.AppendLine($"- Generated: `{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC`");
        builder.AppendLine($"- Map: `{mapName}`");
        builder.AppendLine($"- PM4 status: `{_worldScene!.Pm4Status}`");
        builder.AppendLine($"- Color mode: `{_worldScene.Pm4ColorMode}` ({GetPm4ColorModeLabel(_worldScene.Pm4ColorMode)})");
        builder.AppendLine($"- Visible overlay objects: `{visibleSummary.VisibleObjectCount}` across `{visibleSummary.VisibleTileCount}` visible tiles");
        builder.AppendLine($"- Visible MSHD regions: `{visibleSummary.RegionCount}`");
        builder.AppendLine();
        builder.AppendLine("## Top Visible Regions");
        builder.AppendLine();

        foreach (Pm4VisibleRegionSummary region in visibleSummary.Regions)
        {
            string selectedSuffix = region.IsSelectedRegion ? " [selected region]" : string.Empty;
            builder.AppendLine($"- Region `{region.RegionId}`: `{region.ObjectCount}` objects across `{region.TileCount}` tiles, `{region.UniqueCk24Count}` unique CK24, `{region.UniqueLinkGroupCount}` unique MSLK groups, avg center Z `{region.AverageCenterHeight:F1}`{selectedSuffix}. Types: {FormatPm4TypeBuckets(region.TypeBuckets)}");
        }

        if (selectedDebugInfo.HasValue && _worldScene.SelectedPm4ObjectKey.HasValue)
        {
            builder.AppendLine();
            builder.AppendLine("## Selected Object");
            builder.AppendLine();
            builder.AppendLine($"- tile=(`{_worldScene.SelectedPm4ObjectKey.Value.tileX}`, `{_worldScene.SelectedPm4ObjectKey.Value.tileY}`) ck24=`0x{selectedDebugInfo.Value.Ck24:X6}` part=`{_worldScene.SelectedPm4ObjectKey.Value.objectPart}` type=`0x{selectedDebugInfo.Value.Ck24Type:X2}` objId=`{selectedDebugInfo.Value.Ck24ObjectId}`");
            builder.AppendLine($"- MSHD: field00=`{selectedDebugInfo.Value.MshdField00}` region=`{selectedDebugInfo.Value.MshdRegionId}` field08=`{selectedDebugInfo.Value.MshdField08}`");
            builder.AppendLine($"- MSLK group=`0x{selectedDebugInfo.Value.LinkGroupObjectId:X8}` MscnRef=`{selectedDebugInfo.Value.DominantMscnRefIndex}` linked refs=`{selectedDebugInfo.Value.LinkedPositionRefCount}` surfaces=`{selectedDebugInfo.Value.SurfaceCount}`");
            builder.AppendLine($"- center=(`{selectedDebugInfo.Value.Center.X:F1}`, `{selectedDebugInfo.Value.Center.Y:F1}`, `{selectedDebugInfo.Value.Center.Z:F1}`)");
        }

        if (selectedRegionInfo.HasValue)
        {
            builder.AppendLine();
            builder.AppendLine("## Selected Region Peers");
            builder.AppendLine();
            builder.AppendLine($"- Region `{selectedRegionInfo.Value.RegionId}` contains `{selectedRegionInfo.Value.VisibleObjectCount}` visible PM4 parts across `{selectedRegionInfo.Value.VisibleTileCount}` tiles.");
            builder.AppendLine($"- Same CK24 as selection: `{selectedRegionInfo.Value.SameCk24Count}`. Same MSLK group: `{selectedRegionInfo.Value.SameLinkGroupCount}`. Same MscnRef index: `{selectedRegionInfo.Value.SameMscnRefCount}`.");
            builder.AppendLine($"- Type mix: {FormatPm4TypeBuckets(selectedRegionInfo.Value.TypeBuckets)}");
            builder.AppendLine();
            foreach (Pm4RegionPeerSummary peer in selectedRegionInfo.Value.Peers)
            {
                builder.AppendLine($"- tile=(`{peer.ObjectKey.tileX}`, `{peer.ObjectKey.tileY}`) ck24=`0x{peer.ObjectKey.ck24:X6}` part=`{peer.ObjectKey.objectPart}` type=`0x{peer.Ck24Type:X2}` surf=`{peer.SurfaceCount}` center=(`{peer.Center.X:F1}`, `{peer.Center.Y:F1}`, `{peer.Center.Z:F1}`) {FormatPm4PeerFlags(peer)}");
            }
        }

        builder.AppendLine();
        builder.AppendLine("## Files");
        builder.AppendLine();
        builder.AppendLine("- `pm4_llm_bundle.json`: machine-readable summary of the visible overlay and current selection.");
        builder.AppendLine("- `pm4_visible_regions.svg`: bar-chart style infographic for the top visible MSHD regions.");
        if (selectedRegionInfo.HasValue)
            builder.AppendLine("- `pm4_selected_region.svg`: selected-region peer sheet for quick visual review.");

        return builder.ToString();
    }

    private string BuildPm4VisibleRegionsSvg(Pm4VisibleOverlaySummaryInfo visibleSummary)
    {
        int width = 1200;
        int left = 240;
        int top = 80;
        int rowHeight = 32;
        int chartWidth = 860;
        int rowCount = Math.Max(1, visibleSummary.Regions.Count);
        int height = top + 70 + (rowCount * rowHeight);
        int maxCount = Math.Max(1, visibleSummary.Regions.Count > 0 ? visibleSummary.Regions.Max(static region => region.ObjectCount) : 1);
        var builder = new StringBuilder();
        builder.AppendLine($"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">""");
        builder.AppendLine("""<rect width="100%" height="100%" fill="#10141a" />""");
        builder.AppendLine("""<text x="32" y="38" fill="#f5f7fa" font-family="Consolas, 'Courier New', monospace" font-size="28">PM4 Visible Regions</text>""");
        builder.AppendLine($"""<text x="32" y="64" fill="#aab6c3" font-family="Consolas, 'Courier New', monospace" font-size="16">visible objects={visibleSummary.VisibleObjectCount} tiles={visibleSummary.VisibleTileCount} regions={visibleSummary.RegionCount}</text>""");

        for (int index = 0; index < visibleSummary.Regions.Count; index++)
        {
            Pm4VisibleRegionSummary region = visibleSummary.Regions[index];
            int y = top + (index * rowHeight);
            int barWidth = (int)MathF.Round(chartWidth * (region.ObjectCount / (float)maxCount));
            string color = Pm4ColorToHex(Pm4ColorFromSeed(region.RegionId));
            string labelColor = region.IsSelectedRegion ? "#ffe36a" : "#f5f7fa";
            builder.AppendLine($"""<text x="32" y="{y + 20}" fill="{labelColor}" font-family="Consolas, 'Courier New', monospace" font-size="15">region {region.RegionId}</text>""");
            builder.AppendLine($"""<rect x="{left}" y="{y}" width="{Math.Max(1, barWidth)}" height="18" fill="{color}" rx="4" ry="4" />""");
            builder.AppendLine($"""<text x="{left + Math.Max(8, barWidth + 10)}" y="{y + 14}" fill="#d8e0e8" font-family="Consolas, 'Courier New', monospace" font-size="13">{EscapeSvgText($"{region.ObjectCount} objs | {region.TileCount} tiles | {FormatPm4TypeBuckets(region.TypeBuckets)}")}</text>""");
        }

        builder.AppendLine("</svg>");
        return builder.ToString();
    }

    private string BuildPm4SelectedRegionSvg(Pm4SelectedObjectRegionInfo regionInfo)
    {
        int width = 1280;
        int top = 92;
        int rowHeight = 34;
        int rowCount = Math.Max(1, regionInfo.Peers.Count);
        int height = top + 80 + (rowCount * rowHeight);
        string accent = Pm4ColorToHex(Pm4ColorFromSeed(regionInfo.RegionId));
        var builder = new StringBuilder();
        builder.AppendLine($"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">""");
        builder.AppendLine("""<rect width="100%" height="100%" fill="#0f1318" />""");
        builder.AppendLine($"""<text x="32" y="38" fill="#f5f7fa" font-family="Consolas, 'Courier New', monospace" font-size="28">Selected PM4 Region {regionInfo.RegionId}</text>""");
        builder.AppendLine($"""<text x="32" y="64" fill="#aab6c3" font-family="Consolas, 'Courier New', monospace" font-size="16">visible objects={regionInfo.VisibleObjectCount} tiles={regionInfo.VisibleTileCount} sameCK24={regionInfo.SameCk24Count} sameMSLK={regionInfo.SameLinkGroupCount} sameMscnRef={regionInfo.SameMscnRefCount}</text>""");
        builder.AppendLine($"""<rect x="32" y="76" width="1216" height="2" fill="{accent}" />""");

        for (int index = 0; index < regionInfo.Peers.Count; index++)
        {
            Pm4RegionPeerSummary peer = regionInfo.Peers[index];
            int y = top + (index * rowHeight);
            string rowFill = peer.IsSelected ? "#1e2b38" : "#151b23";
            string textFill = peer.IsSelected ? "#ffe36a" : "#f5f7fa";
            builder.AppendLine($"""<rect x="24" y="{y - 18}" width="1232" height="26" fill="{rowFill}" rx="4" ry="4" />""");
            builder.AppendLine($"""<rect x="24" y="{y - 18}" width="6" height="26" fill="{accent}" rx="3" ry="3" />""");
            builder.AppendLine($"""<text x="40" y="{y}" fill="{textFill}" font-family="Consolas, 'Courier New', monospace" font-size="14">{EscapeSvgText($"tile ({peer.ObjectKey.tileX},{peer.ObjectKey.tileY}) ck24 0x{peer.ObjectKey.ck24:X6} part {peer.ObjectKey.objectPart} type 0x{peer.Ck24Type:X2} surf {peer.SurfaceCount}")}</text>""");
            builder.AppendLine($"""<text x="700" y="{y}" fill="#b8c4cf" font-family="Consolas, 'Courier New', monospace" font-size="13">{EscapeSvgText($"mslk 0x{peer.LinkGroupObjectId:X8} mdos {peer.DominantMscnRefIndex} center ({peer.Center.X:F1}, {peer.Center.Y:F1}, {peer.Center.Z:F1}) {FormatPm4PeerFlags(peer)}")}</text>""");
        }

        builder.AppendLine("</svg>");
        return builder.ToString();
    }

    private static string FormatPm4TypeBuckets(IReadOnlyList<Pm4VisibleTypeBucket> buckets)
    {
        if (buckets.Count == 0)
            return "none";

        return string.Join(", ", buckets.Select(static bucket => $"0x{bucket.Ck24Type:X2} x{bucket.ObjectCount}"));
    }

    private static string FormatPm4PeerFlags(Pm4RegionPeerSummary peer)
    {
        var flags = new List<string>(4);
        if (peer.IsSelected)
            flags.Add("selected");
        if (peer.SameCk24)
            flags.Add("same-ck24");
        if (peer.SameLinkGroup)
            flags.Add("same-mslk");
        if (peer.SameMscnRefIndex)
            flags.Add("same-mdos");

        return flags.Count == 0
            ? "shared=none"
            : $"shared={string.Join("/", flags)}";
    }

    private static Vector3 Pm4ColorFromSeed(uint seed)
    {
        uint golden = seed * 2654435761u;
        float hue = (golden & 0x00FFFFFF) / 16777215f;
        return Pm4HsvToRgb(hue, 0.75f, 0.95f);
    }

    private static Vector3 Pm4HsvToRgb(float h, float s, float v)
    {
        h = h - MathF.Floor(h);
        float c = v * s;
        float x = c * (1f - MathF.Abs((h * 6f) % 2f - 1f));
        float m = v - c;

        float r;
        float g;
        float b;
        int sector = (int)(h * 6f);
        switch (sector)
        {
            case 0:
                r = c; g = x; b = 0f;
                break;
            case 1:
                r = x; g = c; b = 0f;
                break;
            case 2:
                r = 0f; g = c; b = x;
                break;
            case 3:
                r = 0f; g = x; b = c;
                break;
            case 4:
                r = x; g = 0f; b = c;
                break;
            default:
                r = c; g = 0f; b = x;
                break;
        }

        return new Vector3(r + m, g + m, b + m);
    }

    private static string Pm4ColorToHex(Vector3 color)
    {
        int r = (int)Math.Clamp(MathF.Round(color.X * 255f), 0f, 255f);
        int g = (int)Math.Clamp(MathF.Round(color.Y * 255f), 0f, 255f);
        int b = (int)Math.Clamp(MathF.Round(color.Z * 255f), 0f, 255f);
        return $"#{r:X2}{g:X2}{b:X2}";
    }

    private static string EscapeSvgText(string value)
    {
        return value
            .Replace("&", "&amp;", StringComparison.Ordinal)
            .Replace("<", "&lt;", StringComparison.Ordinal)
            .Replace(">", "&gt;", StringComparison.Ordinal)
            .Replace("\"", "&quot;", StringComparison.Ordinal);
    }

    private void DrawSelectedPm4ObjectGraph(string idSuffix = "")
    {
        if (_worldScene == null || !_worldScene.TryGetSelectedPm4ObjectGraphInfo(out Pm4SelectedObjectGraphInfo graph))
            return;

        if (!ImGui.CollapsingHeader($"PM4 Graph##{idSuffix}", ImGuiTreeNodeFlags.DefaultOpen))
            return;

        ImGui.TextDisabled("Derived from the current overlay build: CK24 root, MSLK-linked groups, optional MscnRef split, then connectivity parts.");
        ImGui.TextDisabled("part/ObjectPartId is a viewer-generated split id from that build, not a raw PM4 field.");
        ImGui.TextDisabled("Treat this as viewer structure, not a claim that PM4 stores matching raw graph nodes.");
        ImGui.TextDisabled($"Split flags: MscnRef={graph.SplitByMscnRef} Connectivity={graph.SplitByConnectivity}");
        ImGui.TextDisabled($"Tiles={graph.TileCount} LinkGroups={graph.LinkGroupCount} MscnRefGroups={graph.MscnRefGroupCount} Parts={graph.PartCount}");
        ImGui.TextDisabled($"Surfaces={graph.SurfaceCount} Indices={graph.TotalIndexCount} AttrMasks={graph.AttributeMaskCount} GroupKeys={graph.GroupKeyCount}");
        ImGui.TextDisabled("Click a part row to reselect it. Use Frame to move the camera to that exact part.");
        ImGui.TextDisabled("Use the graph Collect buttons as the primary PM4 multi-select path; viewport PM4 picking is not reliable enough.");

        if (ImGui.Button($"Export Graph JSON##{idSuffix}"))
            ExportSelectedPm4GraphJson(graph);
        ImGui.SameLine();
        if (ImGui.Button($"Add Part##{idSuffix}"))
            AddSelectedPm4ObjectToCollection();
        ImGui.SameLine();
        if (ImGui.Button($"Add Merged Group##{idSuffix}"))
            AddSelectedPm4GraphGroupToCollection(graph);
        ImGui.SameLine();
        if (ImGui.Button($"Export Collection JSON##{idSuffix}"))
            ExportPm4ObjectCollectionJson();
        ImGui.SameLine();
        if (ImGui.Button($"Clear Collection##{idSuffix}"))
            ClearPm4ObjectCollection();

        DrawPm4ObjectCollectionSummary(idSuffix);

        ImGuiTreeNodeFlags rootFlags = ImGuiTreeNodeFlags.DefaultOpen;
        if (ImGui.TreeNodeEx($"CK24 0x{graph.Ck24:X6} type=0x{graph.Ck24Type:X2} obj={graph.Ck24ObjectId}##Pm4GraphRoot{idSuffix}", rootFlags))
        {
            for (int linkIndex = 0; linkIndex < graph.LinkGroups.Count; linkIndex++)
            {
                Pm4SelectedObjectGraphLinkNode linkGroup = graph.LinkGroups[linkIndex];
                string linkSummary = $"MSLK 0x{linkGroup.LinkGroupObjectId:X8} parts={linkGroup.PartCount} surfaces={linkGroup.SurfaceCount} indices={linkGroup.TotalIndexCount} linkedMPRL={linkGroup.LinkedPositionRefCount}";
                if (ImGui.TreeNodeEx($"{linkSummary}##Pm4GraphLink{idSuffix}_{linkIndex}", ImGuiTreeNodeFlags.DefaultOpen))
                {
                    if (ImGui.SmallButton($"Collect Link##Pm4GraphCollectLink{idSuffix}_{linkIndex}"))
                        AddPm4LinkGroupToCollection(graph, linkGroup);

                    if (linkGroup.LinkedPositionRefSummary.TotalCount > 0)
                    {
                        ImGui.TextDisabled(
                            $"MPRL normal={linkGroup.LinkedPositionRefSummary.NormalCount} term={linkGroup.LinkedPositionRefSummary.TerminatorCount} floors={linkGroup.LinkedPositionRefSummary.FloorMin}..{linkGroup.LinkedPositionRefSummary.FloorMax}");
                    }

                    for (int mscnRefIndex = 0; mscnRefIndex < linkGroup.MscnRefGroups.Count; mscnRefIndex++)
                    {
                        Pm4SelectedObjectGraphMscnRefNode mscnRefGroup = linkGroup.MscnRefGroups[mscnRefIndex];
                        string mscnRefSummary = $"MscnRef {mscnRefGroup.MscnRefIndex} parts={mscnRefGroup.PartCount} surfaces={mscnRefGroup.SurfaceCount} indices={mscnRefGroup.TotalIndexCount} attrs={FormatPm4ByteList(mscnRefGroup.AttributeMasks)} groups={FormatPm4ByteList(mscnRefGroup.GroupKeys)}";
                        if (ImGui.TreeNodeEx($"{mscnRefSummary}##Pm4GraphMscnRef{idSuffix}_{linkIndex}_{mscnRefIndex}", ImGuiTreeNodeFlags.DefaultOpen))
                        {
                            if (ImGui.SmallButton($"Collect MscnRef##Pm4GraphCollectMscnRef{idSuffix}_{linkIndex}_{mscnRefIndex}"))
                                AddPm4MscnRefGroupToCollection(graph, mscnRefGroup);

                            for (int partIndex = 0; partIndex < mscnRefGroup.Parts.Count; partIndex++)
                            {
                                Pm4SelectedObjectGraphPartNode part = mscnRefGroup.Parts[partIndex];
                                ImGuiTreeNodeFlags partFlags = ImGuiTreeNodeFlags.Leaf | ImGuiTreeNodeFlags.NoTreePushOnOpen;
                                if (part.IsSelected)
                                    partFlags |= ImGuiTreeNodeFlags.Selected;

                                string partSummary = $"part={part.ObjectPartId} tile=({part.TileX},{part.TileY}) surfaces={part.SurfaceCount} indices={part.TotalIndexCount} lines={part.LineCount} tris={part.TriangleCount} group=0x{part.DominantGroupKey:X2} attr=0x{part.DominantAttributeMask:X2}";
                                ImGui.TreeNodeEx($"{partSummary}##Pm4GraphPart{idSuffix}_{linkIndex}_{mscnRefIndex}_{partIndex}", partFlags);
                                if (ImGui.IsItemClicked())
                                    SelectPm4GraphPart((part.TileX, part.TileY, graph.Ck24, part.ObjectPartId), frameCamera: false);
                                ImGui.SameLine();
                                if (ImGui.SmallButton($"Frame##Pm4GraphPartFrame{idSuffix}_{linkIndex}_{mscnRefIndex}_{partIndex}"))
                                    SelectPm4GraphPart((part.TileX, part.TileY, graph.Ck24, part.ObjectPartId), frameCamera: true);
                                ImGui.SameLine();
                                if (ImGui.SmallButton($"Collect##Pm4GraphPartCollect{idSuffix}_{linkIndex}_{mscnRefIndex}_{partIndex}"))
                                    TogglePm4ObjectCollectionMembership((part.TileX, part.TileY, graph.Ck24, part.ObjectPartId), reportStatus: true);
                            }

                            ImGui.TreePop();
                        }
                    }

                    ImGui.TreePop();
                }
            }

            ImGui.TreePop();
        }
    }

    private static string FormatPm4ByteList(IReadOnlyList<byte> values)
    {
        if (values.Count == 0)
            return "-";

        return string.Join(", ", values.Select(static value => $"0x{value:X2}"));
    }

    private void SelectPm4GraphPart((int tileX, int tileY, uint ck24, int objectPart) objectKey, bool frameCamera)
    {
        if (_worldScene == null)
            return;

        if (!_worldScene.SelectPm4Object(objectKey))
        {
            _statusMessage = $"PM4 graph part CK24=0x{objectKey.ck24:X6} part={objectKey.objectPart} is no longer available.";
            return;
        }

        OpenPm4Workbench(Pm4WorkbenchTab.Selection);

        if (frameCamera)
        {
            if (_worldScene.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo debugInfo))
                FocusCameraOnBounds(debugInfo.BoundsMin, debugInfo.BoundsMax);
        }

        _statusMessage = frameCamera
            ? $"Selected and framed PM4 graph part CK24=0x{objectKey.ck24:X6} part={objectKey.objectPart}."
            : $"Selected PM4 graph part CK24=0x{objectKey.ck24:X6} part={objectKey.objectPart}.";
    }

    private void ExportSelectedPm4GraphJson(Pm4SelectedObjectGraphInfo graph)
    {
        string defaultName = $"pm4_graph_ck24_{graph.Ck24:X6}_part_{graph.SelectedObjectPartId:D4}_{DateTime.Now:yyyyMMdd_HHmmss}.json";
        string? picked = ShowSaveFileDialogSTA(
            "Save Selected PM4 Graph JSON",
            "JSON Files (*.json)|*.json|All Files (*.*)|*.*",
            ExportDir,
            defaultName);

        if (string.IsNullOrWhiteSpace(picked))
            return;

        try
        {
            string json = JsonSerializer.Serialize(BuildJsonSafePm4Graph(graph), new JsonSerializerOptions
            {
                WriteIndented = true
            });
            File.WriteAllText(picked, json, Encoding.UTF8);
            _statusMessage = $"Exported selected PM4 graph JSON: {picked}";
        }
        catch (Exception ex)
        {
            _statusMessage = $"PM4 graph export failed: {ex.Message}";
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 Graph] JSON export failed: {ex}");
        }
    }

    private void AddSelectedPm4ObjectToCollection()
    {
        if (_worldScene == null || !_worldScene.SelectedPm4ObjectKey.HasValue)
            return;

        TogglePm4ObjectCollectionMembership(_worldScene.SelectedPm4ObjectKey.Value, reportStatus: true, removeIfPresent: false);
    }

    private void AddSelectedPm4GraphGroupToCollection(Pm4SelectedObjectGraphInfo graph)
    {
        var keys = graph.LinkGroups
            .SelectMany(static linkGroup => linkGroup.MscnRefGroups)
            .SelectMany(static mscnRefGroup => mscnRefGroup.Parts)
            .Select(part => (part.TileX, part.TileY, graph.Ck24, part.ObjectPartId));

        int added = AddPm4ObjectsToCollection(keys);
        _statusMessage = added > 0
            ? $"Added {added} PM4 parts from the merged group to the collection."
            : "All parts in the merged group were already in the PM4 collection.";
        SyncPm4CollectionHighlight();
    }

    private void AddPm4LinkGroupToCollection(Pm4SelectedObjectGraphInfo graph, Pm4SelectedObjectGraphLinkNode linkGroup)
    {
        var keys = linkGroup.MscnRefGroups
            .SelectMany(static mscnRefGroup => mscnRefGroup.Parts)
            .Select(part => (part.TileX, part.TileY, graph.Ck24, part.ObjectPartId));

        int added = AddPm4ObjectsToCollection(keys);
        _statusMessage = added > 0
            ? $"Added {added} PM4 parts from MSLK 0x{linkGroup.LinkGroupObjectId:X8} to the collection."
            : $"All PM4 parts from MSLK 0x{linkGroup.LinkGroupObjectId:X8} were already in the collection.";
        SyncPm4CollectionHighlight();
    }

    private void AddPm4MscnRefGroupToCollection(Pm4SelectedObjectGraphInfo graph, Pm4SelectedObjectGraphMscnRefNode mscnRefGroup)
    {
        var keys = mscnRefGroup.Parts
            .Select(part => (part.TileX, part.TileY, graph.Ck24, part.ObjectPartId));

        int added = AddPm4ObjectsToCollection(keys);
        _statusMessage = added > 0
            ? $"Added {added} PM4 parts from MscnRef {mscnRefGroup.MscnRefIndex} to the collection."
            : $"All PM4 parts from MscnRef {mscnRefGroup.MscnRefIndex} were already in the collection.";
        SyncPm4CollectionHighlight();
    }

    private int AddPm4ObjectsToCollection(IEnumerable<(int tileX, int tileY, uint ck24, int objectPart)> keys)
    {
        int added = 0;
        foreach (var key in keys)
        {
            if (_pm4ObjectCollection.Contains(key))
                continue;

            _pm4ObjectCollection.Add(key);
            added++;
        }

        if (added > 0)
            SyncPm4CollectionHighlight();

        return added;
    }

    private bool TogglePm4ObjectCollectionMembership(
        (int tileX, int tileY, uint ck24, int objectPart) key,
        bool reportStatus,
        bool removeIfPresent = true)
    {
        int existingIndex = _pm4ObjectCollection.IndexOf(key);
        if (existingIndex >= 0)
        {
            if (removeIfPresent)
            {
                _pm4ObjectCollection.RemoveAt(existingIndex);
                SyncPm4CollectionHighlight();
                if (reportStatus)
                    _statusMessage = $"Removed PM4 CK24=0x{key.ck24:X6} part={key.objectPart} from the collection.";
                return false;
            }

            if (reportStatus)
                _statusMessage = $"PM4 CK24=0x{key.ck24:X6} part={key.objectPart} is already in the collection.";
            return false;
        }

        _pm4ObjectCollection.Add(key);
        SyncPm4CollectionHighlight();
        if (reportStatus)
            _statusMessage = $"Added PM4 CK24=0x{key.ck24:X6} part={key.objectPart} to the collection.";
        return true;
    }

    private void ClearPm4ObjectCollection()
    {
        _pm4ObjectCollection.Clear();
        SyncPm4CollectionHighlight();
        _statusMessage = "Cleared PM4 object collection.";
    }

    private void SyncPm4CollectionHighlight()
    {
        _worldScene?.SetHighlightedPm4Objects(_pm4ObjectCollection);
    }

    private void DrawPm4ObjectCollectionSummary(string idSuffix)
    {
        PruneMissingPm4CollectionObjects();

        if (!ImGui.CollapsingHeader($"PM4 Collection##{idSuffix}", ImGuiTreeNodeFlags.DefaultOpen))
            return;

        ImGui.TextDisabled($"Parts in collection: {_pm4ObjectCollection.Count}");
        ImGui.TextDisabled("Use this to compare one family against duplicated placements or overlapping copies.");
        ImGui.TextDisabled("Shift+LMB PM4 add is best-effort only; use graph Collect buttons when scene overlap is ambiguous.");

        if (_pm4ObjectCollection.Count == 0)
        {
            ImGui.TextDisabled("No PM4 parts collected yet.");
            return;
        }

        if (ImGui.BeginChild($"Pm4CollectionList##{idSuffix}", new Vector2(0f, 140f), true))
        {
            for (int index = 0; index < _pm4ObjectCollection.Count; index++)
            {
                var key = _pm4ObjectCollection[index];
                bool selected = _worldScene != null
                    && _worldScene.SelectedPm4ObjectKey.HasValue
                    && _worldScene.SelectedPm4ObjectKey.Value == key;
                uint? regionId = _worldScene != null
                    && _worldScene.TryGetPm4ObjectDebugInfo(key, out Pm4ObjectDebugInfo debugInfo)
                    ? debugInfo.MshdRegionId
                    : null;
                string label = regionId.HasValue
                    ? $"{index + 1}. CK24 0x{key.ck24:X6} part={key.objectPart} tile=({key.tileX},{key.tileY}) region={regionId.Value}"
                    : $"{index + 1}. CK24 0x{key.ck24:X6} part={key.objectPart} tile=({key.tileX},{key.tileY})";

                ImGui.PushID($"Pm4CollectionItem{idSuffix}_{index}");
                if (selected)
                    ImGui.TextColored(new Vector4(1f, 0.95f, 0.35f, 1f), label);
                else
                    ImGui.TextUnformatted(label);

                ImGui.SameLine();
                if (ImGui.SmallButton("Select"))
                    SelectPm4GraphPart(key, frameCamera: false);

                ImGui.SameLine();
                if (ImGui.SmallButton("Frame"))
                    SelectPm4GraphPart(key, frameCamera: true);

                ImGui.SameLine();
                if (ImGui.SmallButton("Remove"))
                {
                    _pm4ObjectCollection.RemoveAt(index);
                    SyncPm4CollectionHighlight();
                    _statusMessage = $"Removed PM4 CK24=0x{key.ck24:X6} part={key.objectPart} from the collection.";
                    ImGui.PopID();
                    break;
                }

                ImGui.PopID();
            }
        }
        ImGui.EndChild();
    }

    private void PruneMissingPm4CollectionObjects()
    {
        if (_worldScene == null)
            return;

        for (int index = _pm4ObjectCollection.Count - 1; index >= 0; index--)
        {
            if (!_worldScene.TryGetPm4ObjectDebugInfo(_pm4ObjectCollection[index], out _))
                _pm4ObjectCollection.RemoveAt(index);
        }

        SyncPm4CollectionHighlight();
    }

    private void ExportPm4ObjectCollectionJson()
    {
        if (_worldScene == null)
            return;

        PruneMissingPm4CollectionObjects();
        if (_pm4ObjectCollection.Count == 0)
        {
            _statusMessage = "PM4 collection export skipped: no collected parts.";
            return;
        }

        string mapName = _terrainManager?.MapName ?? _worldScene.Terrain.MapName ?? "map";
        string defaultName = $"pm4_collection_{mapName}_{DateTime.Now:yyyyMMdd_HHmmss}.json";
        string? picked = ShowSaveFileDialogSTA(
            "Save PM4 Collection JSON",
            "JSON Files (*.json)|*.json|All Files (*.*)|*.*",
            ExportDir,
            defaultName);

        if (string.IsNullOrWhiteSpace(picked))
            return;

        try
        {
            string json = JsonSerializer.Serialize(BuildJsonSafePm4Collection(), new JsonSerializerOptions
            {
                WriteIndented = true
            });
            File.WriteAllText(picked, json, Encoding.UTF8);
            _statusMessage = $"Exported PM4 collection JSON: {picked}";
        }
        catch (Exception ex)
        {
            _statusMessage = $"PM4 collection export failed: {ex.Message}";
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 Collection] JSON export failed: {ex}");
        }
    }

    private object BuildJsonSafePm4Collection()
    {
        if (_worldScene == null)
        {
            return new
            {
                generatedAtUtc = DateTime.UtcNow,
                objectCount = 0,
                objects = Array.Empty<object>()
            };
        }

        var entries = new List<Pm4CollectionExportEntry>(_pm4ObjectCollection.Count);
        foreach (var key in _pm4ObjectCollection)
        {
            if (!_worldScene.TryGetPm4ObjectDebugInfo(key, out Pm4ObjectDebugInfo debugInfo))
                continue;

            _worldScene.TryGetPm4ObjectGroupKey(key, out var mergedGroupKey);
            Vector3 size = debugInfo.BoundsMax - debugInfo.BoundsMin;
            string signature = BuildPm4CollectionSignature(debugInfo, size);
            entries.Add(new Pm4CollectionExportEntry(key, mergedGroupKey, debugInfo, size, signature));
        }

        var signatureGroups = entries
            .GroupBy(static entry => entry.Signature)
            .OrderByDescending(static group => group.Count())
            .ThenBy(static group => group.Key, StringComparer.Ordinal)
            .Select(group => new
            {
                signature = group.Key,
                count = group.Count(),
                ck24 = group.First().DebugInfo.Ck24,
                linkGroupObjectId = group.First().DebugInfo.LinkGroupObjectId,
                members = group.Select(static entry => new
                {
                    tileX = entry.Key.tileX,
                    tileY = entry.Key.tileY,
                    objectPartId = entry.Key.objectPart,
                    center = VectorToArray(entry.DebugInfo.Center)
                }).ToList()
            })
            .ToList();

        var regionGroups = entries
            .GroupBy(static entry => entry.DebugInfo.MshdRegionId)
            .OrderByDescending(static group => group.Count())
            .ThenBy(static group => group.Key)
            .Select(group => new
            {
                regionId = group.Key,
                count = group.Count(),
                tileCount = group.Select(static entry => (entry.Key.tileX, entry.Key.tileY)).Distinct().Count(),
                members = group.Select(static entry => new
                {
                    tileX = entry.Key.tileX,
                    tileY = entry.Key.tileY,
                    ck24 = entry.DebugInfo.Ck24,
                    objectPartId = entry.Key.objectPart,
                    center = VectorToArray(entry.DebugInfo.Center)
                }).ToList()
            })
            .ToList();

        Dictionary<(int tileX, int tileY, uint ck24, int objectPart), Pm4CollectionDuplicateMetrics> duplicateMetrics = BuildPm4CollectionDuplicateMetrics(entries);
        var stackClusters = BuildPm4CollectionStackClusters(entries, duplicateMetrics);
        string mapName = _terrainManager?.MapName ?? _worldScene.Terrain.MapName ?? string.Empty;

        return new
        {
            generatedAtUtc = DateTime.UtcNow,
            mapName,
            objectCount = entries.Count,
            currentSelection = _worldScene.SelectedPm4ObjectKey.HasValue
                ? new
                {
                    tileX = _worldScene.SelectedPm4ObjectKey.Value.tileX,
                    tileY = _worldScene.SelectedPm4ObjectKey.Value.tileY,
                    ck24 = _worldScene.SelectedPm4ObjectKey.Value.ck24,
                    objectPartId = _worldScene.SelectedPm4ObjectKey.Value.objectPart,
                }
                : null,
            signatureGroupCount = signatureGroups.Count,
            regionGroupCount = regionGroups.Count,
            stackClusterCount = stackClusters.Count,
            signatureGroups,
            regionGroups,
            stackClusters,
            objects = entries.Select(entry => new
            {
                tileX = entry.Key.tileX,
                tileY = entry.Key.tileY,
                ck24 = entry.DebugInfo.Ck24,
                ck24Type = entry.DebugInfo.Ck24Type,
                ck24ObjectId = entry.DebugInfo.Ck24ObjectId,
                objectPartId = entry.Key.objectPart,
                mergedGroupKey = new
                {
                    tileX = entry.GroupKey.tileX,
                    tileY = entry.GroupKey.tileY,
                    ck24 = entry.GroupKey.ck24,
                },
                signature = entry.Signature,
                sameSignatureCount = duplicateMetrics[entry.Key].SameSignatureCount,
                overlapClusterSize = duplicateMetrics[entry.Key].OverlapClusterSize,
                nearestSameSignatureDistance = JsonFiniteOrNull(duplicateMetrics[entry.Key].NearestSameSignatureDistance),
                likelyDuplicateScore = duplicateMetrics[entry.Key].LikelyDuplicateScore,
                linkGroupObjectId = entry.DebugInfo.LinkGroupObjectId,
                linkedPositionRefCount = entry.DebugInfo.LinkedPositionRefCount,
                linkedPositionRefSummary = new
                {
                    totalCount = entry.DebugInfo.LinkedPositionRefSummary.TotalCount,
                    normalCount = entry.DebugInfo.LinkedPositionRefSummary.NormalCount,
                    terminatorCount = entry.DebugInfo.LinkedPositionRefSummary.TerminatorCount,
                    floorMin = entry.DebugInfo.LinkedPositionRefSummary.FloorMin,
                    floorMax = entry.DebugInfo.LinkedPositionRefSummary.FloorMax,
                    headingMinDegrees = JsonFiniteOrNull(entry.DebugInfo.LinkedPositionRefSummary.HeadingMinDegrees),
                    headingMaxDegrees = JsonFiniteOrNull(entry.DebugInfo.LinkedPositionRefSummary.HeadingMaxDegrees),
                    headingMeanDegrees = JsonFiniteOrNull(entry.DebugInfo.LinkedPositionRefSummary.HeadingMeanDegrees)
                },
                mshd = new
                {
                    field00 = entry.DebugInfo.MshdField00,
                    regionId = entry.DebugInfo.MshdRegionId,
                    field08 = entry.DebugInfo.MshdField08
                },
                surfaceCount = entry.DebugInfo.SurfaceCount,
                dominantGroupKey = entry.DebugInfo.DominantGroupKey,
                dominantAttributeMask = entry.DebugInfo.DominantAttributeMask,
                dominantMscnRefIndex = entry.DebugInfo.DominantMscnRefIndex,
                averageSurfaceHeight = JsonFiniteOrNull(entry.DebugInfo.AverageSurfaceHeight),
                boundsMin = VectorToArray(entry.DebugInfo.BoundsMin),
                boundsMax = VectorToArray(entry.DebugInfo.BoundsMax),
                boundsSize = VectorToArray(entry.BoundsSize),
                center = VectorToArray(entry.DebugInfo.Center),
                nearestPositionRefDistance = JsonFiniteOrNull(entry.DebugInfo.NearestPositionRefDistance),
                planar = new
                {
                    swapAxes = entry.DebugInfo.SwapPlanarAxes,
                    invertU = entry.DebugInfo.InvertU,
                    invertV = entry.DebugInfo.InvertV,
                    windingFlip = entry.DebugInfo.InvertsWinding
                }
            }).ToList()
        };
    }

    private static Dictionary<(int tileX, int tileY, uint ck24, int objectPart), Pm4CollectionDuplicateMetrics> BuildPm4CollectionDuplicateMetrics(IReadOnlyList<Pm4CollectionExportEntry> entries)
    {
        const float centerTolerance = 2f;
        const float sizeTolerance = 0.5f;
        var metrics = new Dictionary<(int tileX, int tileY, uint ck24, int objectPart), Pm4CollectionDuplicateMetrics>(entries.Count);

        foreach (var signatureGroup in entries.GroupBy(static entry => entry.Signature))
        {
            List<Pm4CollectionExportEntry> groupEntries = signatureGroup.ToList();
            foreach (Pm4CollectionExportEntry entry in groupEntries)
            {
                float nearestSameSignatureDistance = float.PositiveInfinity;
                int overlapClusterSize = 1;

                for (int i = 0; i < groupEntries.Count; i++)
                {
                    Pm4CollectionExportEntry candidate = groupEntries[i];
                    if (candidate.Key == entry.Key)
                        continue;

                    float distance = Vector3.Distance(entry.DebugInfo.Center, candidate.DebugInfo.Center);
                    if (distance < nearestSameSignatureDistance)
                        nearestSameSignatureDistance = distance;

                    Vector3 sizeDelta = Vector3.Abs(entry.BoundsSize - candidate.BoundsSize);
                    if (distance <= centerTolerance
                        && sizeDelta.X <= sizeTolerance
                        && sizeDelta.Y <= sizeTolerance
                        && sizeDelta.Z <= sizeTolerance)
                    {
                        overlapClusterSize++;
                    }
                }

                if (!float.IsFinite(nearestSameSignatureDistance))
                    nearestSameSignatureDistance = float.NaN;

                int sameSignatureCount = groupEntries.Count;
                float score = sameSignatureCount <= 1
                    ? 0f
                    : MathF.Min(1f,
                        (overlapClusterSize - 1) * 0.45f
                        + (sameSignatureCount - 1) * 0.15f
                        + (float.IsNaN(nearestSameSignatureDistance)
                            ? 0f
                            : MathF.Max(0f, 1f - MathF.Min(nearestSameSignatureDistance, 12f) / 12f) * 0.40f));

                metrics[entry.Key] = new Pm4CollectionDuplicateMetrics(
                    sameSignatureCount,
                    overlapClusterSize,
                    nearestSameSignatureDistance,
                    MathF.Round(score, 3));
            }
        }

        return metrics;
    }

    private static List<object> BuildPm4CollectionStackClusters(
        IReadOnlyList<Pm4CollectionExportEntry> entries,
        IReadOnlyDictionary<(int tileX, int tileY, uint ck24, int objectPart), Pm4CollectionDuplicateMetrics> duplicateMetrics)
    {
        const float centerTolerance = 2f;
        const float sizeTolerance = 0.5f;
        var clusters = new List<object>();

        foreach (var signatureGroup in entries.GroupBy(static entry => entry.Signature))
        {
            List<Pm4CollectionExportEntry> remaining = signatureGroup.ToList();
            while (remaining.Count > 0)
            {
                Pm4CollectionExportEntry seed = remaining[0];
                remaining.RemoveAt(0);

                var cluster = new List<Pm4CollectionExportEntry> { seed };
                for (int index = remaining.Count - 1; index >= 0; index--)
                {
                    Pm4CollectionExportEntry candidate = remaining[index];
                    if (Vector3.Distance(seed.DebugInfo.Center, candidate.DebugInfo.Center) > centerTolerance)
                        continue;

                    Vector3 sizeDelta = Vector3.Abs(seed.BoundsSize - candidate.BoundsSize);
                    if (sizeDelta.X > sizeTolerance || sizeDelta.Y > sizeTolerance || sizeDelta.Z > sizeTolerance)
                        continue;

                    cluster.Add(candidate);
                    remaining.RemoveAt(index);
                }

                if (cluster.Count < 2)
                    continue;

                Vector3 centroid = Vector3.Zero;
                foreach (Pm4CollectionExportEntry entry in cluster)
                    centroid += entry.DebugInfo.Center;
                centroid /= cluster.Count;

                clusters.Add(new
                {
                    signature = seed.Signature,
                    count = cluster.Count,
                    likelyDuplicateScore = cluster.Max(entry => duplicateMetrics[entry.Key].LikelyDuplicateScore),
                    centroid = VectorToArray(centroid),
                    members = cluster.Select(static entry => new
                    {
                        tileX = entry.Key.tileX,
                        tileY = entry.Key.tileY,
                        ck24 = entry.DebugInfo.Ck24,
                        objectPartId = entry.Key.objectPart,
                        center = VectorToArray(entry.DebugInfo.Center)
                    }).ToList()
                });
            }
        }

        return clusters;
    }

    private static string BuildPm4CollectionSignature(Pm4ObjectDebugInfo debugInfo, Vector3 boundsSize)
    {
        return FormattableString.Invariant($"ck24=0x{debugInfo.Ck24:X6}|mslk=0x{debugInfo.LinkGroupObjectId:X8}|surf={debugInfo.SurfaceCount}|g=0x{debugInfo.DominantGroupKey:X2}|a=0x{debugInfo.DominantAttributeMask:X2}|mscnRef={debugInfo.DominantMscnRefIndex}|size=({boundsSize.X:F2},{boundsSize.Y:F2},{boundsSize.Z:F2})");
    }

    private static float[] VectorToArray(Vector3 value) => new[] { value.X, value.Y, value.Z };

    private readonly record struct Pm4CollectionExportEntry(
        (int tileX, int tileY, uint ck24, int objectPart) Key,
        (int tileX, int tileY, uint ck24) GroupKey,
        Pm4ObjectDebugInfo DebugInfo,
        Vector3 BoundsSize,
        string Signature);

    private readonly record struct Pm4CollectionDuplicateMetrics(
        int SameSignatureCount,
        int OverlapClusterSize,
        float NearestSameSignatureDistance,
        float LikelyDuplicateScore);

    private static object BuildJsonSafePm4Graph(Pm4SelectedObjectGraphInfo graph)
    {
        return new
        {
            selectedTileX = graph.SelectedTileX,
            selectedTileY = graph.SelectedTileY,
            ck24 = graph.Ck24,
            ck24Type = graph.Ck24Type,
            ck24ObjectId = graph.Ck24ObjectId,
            selectedObjectPartId = graph.SelectedObjectPartId,
            splitByMscnRef = graph.SplitByMscnRef,
            splitByConnectivity = graph.SplitByConnectivity,
            tileCount = graph.TileCount,
            linkGroupCount = graph.LinkGroupCount,
            mscnRefGroupCount = graph.MscnRefGroupCount,
            partCount = graph.PartCount,
            surfaceCount = graph.SurfaceCount,
            totalIndexCount = graph.TotalIndexCount,
            attributeMaskCount = graph.AttributeMaskCount,
            groupKeyCount = graph.GroupKeyCount,
            linkGroups = graph.LinkGroups.Select(static linkGroup => new
            {
                linkGroupObjectId = linkGroup.LinkGroupObjectId,
                partCount = linkGroup.PartCount,
                surfaceCount = linkGroup.SurfaceCount,
                totalIndexCount = linkGroup.TotalIndexCount,
                linkedPositionRefCount = linkGroup.LinkedPositionRefCount,
                linkedPositionRefSummary = BuildJsonSafeLinkedPositionRefSummary(linkGroup.LinkedPositionRefSummary),
                mscnRefIndices = linkGroup.MscnRefIndices,
                attributeMasks = linkGroup.AttributeMasks,
                groupKeys = linkGroup.GroupKeys,
                mscnRefGroups = linkGroup.MscnRefGroups.Select(static mscnRefGroup => new
                {
                    mscnRefIndex = mscnRefGroup.MscnRefIndex,
                    partCount = mscnRefGroup.PartCount,
                    surfaceCount = mscnRefGroup.SurfaceCount,
                    totalIndexCount = mscnRefGroup.TotalIndexCount,
                    attributeMasks = mscnRefGroup.AttributeMasks,
                    groupKeys = mscnRefGroup.GroupKeys,
                    parts = mscnRefGroup.Parts.Select(static part => new
                    {
                        tileX = part.TileX,
                        tileY = part.TileY,
                        objectPartId = part.ObjectPartId,
                        surfaceCount = part.SurfaceCount,
                        totalIndexCount = part.TotalIndexCount,
                        lineCount = part.LineCount,
                        triangleCount = part.TriangleCount,
                        dominantGroupKey = part.DominantGroupKey,
                        dominantAttributeMask = part.DominantAttributeMask,
                        dominantMscnRefIndex = part.DominantMscnRefIndex,
                        isSelected = part.IsSelected,
                    }).ToList(),
                }).ToList(),
            }).ToList(),
            typeBuckets = graph.TypeBuckets.Select(static bucket => new
            {
                ck24Type = bucket.Ck24Type,
                typeLabel = bucket.TypeLabel,
                linkGroupCount = bucket.LinkGroupCount,
                surfaceCount = bucket.SurfaceCount,
                linkGroups = bucket.LinkGroups.Select(static linkGroup => new
                {
                    linkGroupObjectId = linkGroup.LinkGroupObjectId,
                    partCount = linkGroup.PartCount,
                    surfaceCount = linkGroup.SurfaceCount,
                    totalIndexCount = linkGroup.TotalIndexCount,
                    linkedPositionRefCount = linkGroup.LinkedPositionRefCount,
                    linkedPositionRefSummary = BuildJsonSafeLinkedPositionRefSummary(linkGroup.LinkedPositionRefSummary),
                    mscnRefIndices = linkGroup.MscnRefIndices,
                    attributeMasks = linkGroup.AttributeMasks,
                    groupKeys = linkGroup.GroupKeys,
                }).ToList(),
            }).ToList(),
        };
    }

    private static object BuildJsonSafeLinkedPositionRefSummary(Pm4LinkedPositionRefSummary summary)
    {
        return new
        {
            totalCount = summary.TotalCount,
            normalCount = summary.NormalCount,
            terminatorCount = summary.TerminatorCount,
            floorMin = summary.FloorMin,
            floorMax = summary.FloorMax,
            headingMinDegrees = JsonFiniteOrNull(summary.HeadingMinDegrees),
            headingMaxDegrees = JsonFiniteOrNull(summary.HeadingMaxDegrees),
            headingMeanDegrees = JsonFiniteOrNull(summary.HeadingMeanDegrees),
            hasNormalHeadings = summary.HasNormalHeadings,
        };
    }

    private static float? JsonFiniteOrNull(float value)
    {
        return float.IsFinite(value) ? value : null;
    }

    private void DrawPm4InfoPanelContent()
    {
        ImGui.PushTextWrapPos(0f);
        if (_worldScene == null || !_worldScene.HasSelectedPm4Object)
        {
            ImGui.TextDisabled("Select a PM4 object to inspect.");
            ImGui.Spacing();
            if (ImGui.Button("Export Visible PM4 Report"))
                ExportPm4OverlayReport();
            ImGui.SameLine();
            if (ImGui.Button("Export PM4 LLM Bundle"))
                ExportPm4LlmEvidenceBundle();
            ImGui.PopTextWrapPos();
            return;
        }

        var key = _worldScene.SelectedPm4ObjectKey.Value;
        ImGui.Text($"Tile ({key.tileX}, {key.tileY})  CK24 0x{key.ck24:X6}  part {key.objectPart}");

        if (_worldScene.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo d))
        {
            ImGui.Separator();
            ImGui.Text($"Type: 0x{d.Ck24Type:X2}  ObjId: {d.Ck24ObjectId}");
            ImGui.Text($"MSHD: F00={d.MshdField00}  Region={d.MshdRegionId}  F08={d.MshdField08}");
            ImGui.Text($"MSLK: group=0x{d.LinkGroupObjectId:X8}  refs={d.LinkedPositionRefCount}");
            ImGui.Text($"GroupKey: 0x{d.DominantGroupKey:X2}  Attr: 0x{d.DominantAttributeMask:X2}  MscnRef: {d.DominantMscnRefIndex}");

            if (d.DistinctTypeFlags != 0)
            {
                var tf = new List<string>();
                for (int bit = 1; bit < 32; bit++)
                    if ((d.DistinctTypeFlags & (1u << bit)) != 0)
                        tf.Add(bit switch { 0x03 => "m2-top", 0x10 => "floor-int", 0x12 => "ext-solid", _ => $"0x{bit:X2}" });
                byte gk = d.DominantGroupKey;
                bool match = (d.DistinctTypeFlags & (1u << gk)) != 0;
                string gkl = gk switch { 0x03 => "m2-surf", 0x10 => "floor-int", 0x12 => "ext-solid", 0x13 => "portal-int", _ => $"0x{gk:X2}" };
                ImGui.Text($"GroupKey={gkl}  TypeFlags: {string.Join(" ", tf)}  {(match ? "MATCH" : "MISMATCH")}");
            }

            if (_worldScene.TryGetSelectedPm4ObjectResearchInfo(out var ri) && ri.MslkRawEntries.Count > 0)
            {
                ImGui.Separator();
                ImGui.TextDisabled($"MSLK entries ({ri.MslkRawEntries.Count}):");
                foreach (string line in ri.MslkRawEntries.Take(5))
                    ImGui.TextUnformatted(line);
                if (ri.MslkRawEntries.Count > 5)
                    ImGui.TextDisabled($"+ {ri.MslkRawEntries.Count - 5} more");
            }
        }

        ImGui.Separator();
        ImGui.Spacing();
        if (ImGui.Button("Export Visible PM4 Report"))
            ExportPm4OverlayReport();
        ImGui.SameLine();
        if (ImGui.Button("Export PM4 LLM Bundle"))
            ExportPm4LlmEvidenceBundle();
        ImGui.SameLine();
        if (ImGui.Button("Dump PM4 JSON"))
            ExportPm4ObjectsJson();

        ImGui.Separator();
        DrawPm4SceneGraph();

        ImGui.PopTextWrapPos();
    }

    private void DrawPm4SceneGraph()
    {
        if (_worldScene == null) return;
        if (!ImGui.CollapsingHeader("Scene Graph"))
            return;

        System.Diagnostics.Stopwatch graphSw = WoWViewer.Logging.Pm4Profiling.Enabled
            ? System.Diagnostics.Stopwatch.StartNew() : null;
        long graphStartTicks = graphSw?.ElapsedTicks ?? 0;

        // Region(tile) → GroupKey(surface type) → ObjectId → count
        var regionTiles = new Dictionary<uint, HashSet<(int, int)>>();
        var byRegion = new Dictionary<uint, Dictionary<byte, Dictionary<ushort, int>>>();

        int walkedObjectCount = 0;
        foreach (var (tx, ty, ck24, objId, region, mslk, gk, part) in _worldScene.GetPm4ObjectHierarchy())
        {
            walkedObjectCount++;
            if (!regionTiles.TryGetValue(region, out var rt)) regionTiles[region] = rt = new();
            rt.Add((tx, ty));
            ushort oid = (ushort)(ck24 & 0xFFFF);
            if (!byRegion.TryGetValue(region, out var br)) byRegion[region] = br = new();
            if (!br.TryGetValue(gk, out var bg)) br[gk] = bg = new();
            bg.TryGetValue(oid, out int e);
            bg[oid] = e + 1;
        }

        int total = 0;
        foreach (var (rid, br) in byRegion.OrderBy(static r => r.Key))
        {
            int rSum = br.Sum(static b => b.Value.Sum(static o => o.Value));
            string ts = regionTiles.TryGetValue(rid, out var rts) && rts.Count > 0
                ? string.Join(", ", rts.OrderBy(static t => t).Select(static t => $"{t.Item1}_{t.Item2}"))
                : "?";
            if (!ImGui.TreeNodeEx($"##R{rid}", ImGuiTreeNodeFlags.None, $"Region {rid} (tile {ts})  [{rSum} surfaces]"))
                continue;

            foreach (var (gk, bg) in br.OrderBy(static b => b.Key))
            {
                int gSum = bg.Sum(static o => o.Value);
                string gkLbl = gk switch
                {
                    0x03 => $"GroupKey 0x03 (M2 surf)",
                    0x10 => $"GroupKey 0x10 (interior floor)",
                    0x12 => $"GroupKey 0x12 (exterior solid)",
                    0x13 => $"GroupKey 0x13 (portal int)",
                    _ => $"GroupKey 0x{gk:X2}",
                };
                if (!ImGui.TreeNodeEx($"##G{rid}_{gk}", ImGuiTreeNodeFlags.None, $"{gkLbl}  [{gSum} surfaces]"))
                { total += gSum; continue; }

                foreach (var (oid, cnt) in bg.OrderByDescending(static o => o.Value))
                {
                    ImGui.TextDisabled($"  ObjectId 0x{oid:X4}  [{cnt} surfaces]");
                    if (ImGui.IsItemClicked(ImGuiMouseButton.Left))
                        _worldScene.SelectPm4ObjectGroupKey(rid, (ushort)oid);
                    total++;
                }
                ImGui.TreePop();
            }
            ImGui.TreePop();
        }
        if (total == 0)
            ImGui.TextDisabled("No PM4 objects loaded.");

        if (graphSw != null)
        {
            graphSw.Stop();
            double elapsedMs = (graphSw.ElapsedTicks - graphStartTicks) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
            WorldScene.Pm4ProfilingAccumulator.RecordGraphBuild(elapsedMs, walkedObjectCount, byRegion.Count);
        }
    }

    private void DrawPm4SceneGraphPanelContent()
    {
        if (_worldScene == null)
        {
            ImGui.TextDisabled("No world scene loaded.");
            return;
        }

        if (!_worldScene.TryGetSelectedPm4ObjectGraphInfo(out Pm4SelectedObjectGraphInfo graph))
        {
            ImGui.TextDisabled("Select a PM4 object to view its scene graph.");
            return;
        }

        var typeBuckets = graph.TypeBuckets;
        if (typeBuckets.Count == 0)
        {
            ImGui.TextDisabled("No type buckets in scene graph.");
            return;
        }

        ImGui.PushTextWrapPos();
        foreach (var bucket in typeBuckets)
        {
            string bucketLabel = bucket.TypeLabel;
            if (ImGui.TreeNodeEx($"##TB_{bucket.Ck24Type:X2}",
                    ImGuiTreeNodeFlags.DefaultOpen,
                    $"0x{bucket.Ck24Type:X2} ({bucket.TypeLabel}) [{bucket.LinkGroupCount} groups, {bucket.SurfaceCount} surfaces]"))
            {
                foreach (var linkGroup in bucket.LinkGroups)
                {
                    if (ImGui.TreeNodeEx($"##LG_{bucket.Ck24Type:X2}_{linkGroup.LinkGroupObjectId}",
                            ImGuiTreeNodeFlags.None,
                            $"LinkGroup {linkGroup.LinkGroupObjectId} [{linkGroup.MscnRefGroups.Count} mscnRefs, {linkGroup.SurfaceCount} surfaces]"))
                    {
                        foreach (var mscnRefGroup in linkGroup.MscnRefGroups)
                        {
                            if (ImGui.TreeNodeEx($"##MR_{mscnRefGroup.MscnRefIndex}",
                                    ImGuiTreeNodeFlags.None,
                                    $"MscnRef {mscnRefGroup.MscnRefIndex} [{mscnRefGroup.PartCount} parts, {mscnRefGroup.SurfaceCount} surfaces]"))
                            {
                                foreach (var part in mscnRefGroup.Parts)
                                {
                                    string selected = part.IsSelected ? " [SELECTED]" : "";
                                    ImGui.TextDisabled($"  Part {part.ObjectPartId} @ {part.TileX}_{part.TileY}{selected}  ({part.SurfaceCount} surfs, {part.TotalIndexCount} idx)");
                                }
                                ImGui.TreePop();
                            }
                        }
                        ImGui.TreePop();
                    }
                }
                ImGui.TreePop();
            }
        }
        ImGui.PopTextWrapPos();
    }

    private void ExportPm4OverlayReport()
    {
        if (_worldScene == null) return;

        string timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
        string outputDir = Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "..", "output", "tmp");
        Directory.CreateDirectory(outputDir);
        string mdPath = Path.Combine(outputDir, $"pm4_overlay_{timestamp}.md");

        using var sw = new StreamWriter(mdPath);
        sw.WriteLine("# PM4 Overlay Report");
        sw.WriteLine();
        sw.WriteLine($"- Generated: `{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC`");
        sw.WriteLine($"- Objects: `{_worldScene.Pm4VisibleObjectCount}` loaded, `{_worldScene.Pm4LoadedFiles}` files");
        sw.WriteLine();

        // Legend
        Pm4ColorLegendInfo legend = _worldScene.GetPm4ColorLegend();
        sw.WriteLine($"Color mode: `{_worldScene.Pm4ColorMode}` — {legend.Description}");
        sw.WriteLine();
        if (legend.Entries.Count > 0)
        {
            sw.WriteLine("| Label | Count |");
            sw.WriteLine("|-------|-------|");
            foreach (var entry in legend.Entries)
                sw.WriteLine($"| {entry.Label} | {entry.ObjectCount} |");
            sw.WriteLine();
        }

        // Regions from overlay summary
        Pm4VisibleOverlaySummaryInfo summary = _worldScene.GetPm4VisibleOverlaySummary(10, 4);
        if (summary.Regions.Count > 0)
        {
            sw.WriteLine("## MSHD Regions");
            sw.WriteLine();
            sw.WriteLine("| Region | Objects | Tiles | CK24 | MSLK | Avg Z |");
            sw.WriteLine("|--------|---------|-------|------|------|-------|");
            foreach (var r in summary.Regions)
            {
                string marker = r.IsSelectedRegion ? " ← selected" : "";
                sw.WriteLine($"| {r.RegionId} | {r.ObjectCount} | {r.TileCount} | {r.UniqueCk24Count} | {r.UniqueLinkGroupCount} | {r.AverageCenterHeight:F1}{marker} |");
            }
            sw.WriteLine();
        }

        // Selected object
        if (_worldScene.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo debug))
        {
            sw.WriteLine("## Selected Object");
            sw.WriteLine();
            sw.WriteLine($"- Tile: `({debug.TileX}, {debug.TileY})`");
            sw.WriteLine($"- CK24: `0x{debug.Ck24:X6}` type=`0x{debug.Ck24Type:X2}` objId=`{debug.Ck24ObjectId}`");
            sw.WriteLine($"- MSHD: F00=`{debug.MshdField00}` region=`{debug.MshdRegionId}` F08=`{debug.MshdField08}`");
            sw.WriteLine($"- MSLK group=`0x{debug.LinkGroupObjectId:X8}` MscnRef=`{debug.DominantMscnRefIndex}` linked refs=`{debug.LinkedPositionRefCount}`");
            sw.WriteLine($"- Surfaces=`{debug.SurfaceCount}` group=`0x{debug.DominantGroupKey:X2}` attr=`0x{debug.DominantAttributeMask:X2}` avgH=`{debug.AverageSurfaceHeight:F2}`");
            sw.WriteLine($"- Center: `({debug.Center.X:F2}, {debug.Center.Y:F2}, {debug.Center.Z:F2})`");
            sw.WriteLine($"- Bounds: `({debug.BoundsMin.X:F2},{debug.BoundsMin.Y:F2},{debug.BoundsMin.Z:F2})` .. `({debug.BoundsMax.X:F2},{debug.BoundsMax.Y:F2},{debug.BoundsMax.Z:F2})`");
            sw.WriteLine();

            if (_worldScene.TryGetSelectedPm4ObjectResearchInfo(out Pm4SelectedObjectResearchInfo rinfo))
            {
                if (rinfo.MshdRawFields != null)
                    sw.WriteLine($"- {rinfo.MshdRawFields}");
                if (rinfo.MslkRawEntries.Count > 0)
                {
                    sw.WriteLine();
                    sw.WriteLine("### MSLK Entries");
                    sw.WriteLine();
                    foreach (string mslkLine in rinfo.MslkRawEntries)
                        sw.WriteLine($"  {mslkLine}");
                    sw.WriteLine();
                }
            }
        }

        _statusMessage = $"Wrote PM4 overlay report: {mdPath}";
    }
}
