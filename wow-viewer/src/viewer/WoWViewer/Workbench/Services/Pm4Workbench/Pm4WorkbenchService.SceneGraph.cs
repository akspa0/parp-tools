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

// Pm4WorkbenchService: info panel, scene graph, MSLK linking summary, full-scene outliner, selected-object graph, overlay report.
// Pm4WorkbenchService: members moved from ViewerApp_Pm4Utilities.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
// Original file note: Partial class containing PM4 alignment and viewer utility windows.
internal sealed partial class Pm4WorkbenchService
{

    internal void DrawPm4InfoPanelContent()
    {
        ImGui.PushTextWrapPos(0f);
        if (_worldScene == null || !_worldScene.Pm4Overlay.HasSelectedPm4Object)
        {
            ImGui.TextDisabled("Select a PM4 object to inspect.");
            ImGui.Spacing();
            if (ImGui.Button("Open Data I/O"))
                OpenWorkbenchTab(WorkbenchTab.Editor, 2); // Spec 231 D1: exports live on the Data I/O page
            ImGui.PopTextWrapPos();
            return;
        }

        var key = _worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value;
        ImGui.Text($"Tile ({key.tileX}, {key.tileY})  CK24 0x{key.ck24:X6}  part {key.objectPart}");

        if (_worldScene.Pm4Overlay.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo d))
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

            if (_worldScene.Pm4Overlay.TryGetSelectedPm4ObjectResearchInfo(out var ri) && ri.MslkRawEntries.Count > 0)
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
        if (ImGui.Button("Open Data I/O"))
            OpenWorkbenchTab(WorkbenchTab.Editor, 2); // Spec 231 D1: exports live on the Data I/O page

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
        foreach (var (tx, ty, ck24, objId, region, mslk, gk, part) in _worldScene.Pm4Overlay.GetPm4ObjectHierarchy())
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
                        _worldScene.Pm4Overlay.SelectPm4ObjectGroupKey(rid, (ushort)oid);
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
            Pm4OverlayScene.Pm4ProfilingAccumulator.RecordGraphBuild(elapsedMs, walkedObjectCount, byRegion.Count);
        }
    }

    private enum Pm4SceneGraphMode
    {
        FullScene,
        SelectedObject,
    }

    private Pm4SceneGraphMode _pm4SceneGraphMode = Pm4SceneGraphMode.FullScene;

    internal void DrawPm4SceneGraphPanelContent()
    {
        if (_worldScene == null)
        {
            ImGui.TextDisabled("No world scene loaded.");
            return;
        }

        // Mode toggle
        string[] modeLabels = { "Full Scene", "Selected Object" };
        int modeIndex = (int)_pm4SceneGraphMode;
        if (ImGui.BeginCombo("##Pm4SceneGraphMode", modeLabels[modeIndex]))
        {
            for (int i = 0; i < modeLabels.Length; i++)
            {
                bool selected = i == modeIndex;
                if (ImGui.Selectable(modeLabels[i], selected))
                    _pm4SceneGraphMode = (Pm4SceneGraphMode)i;
                if (selected)
                    ImGui.SetItemDefaultFocus();
            }
            ImGui.EndCombo();
        }
        ImGui.SameLine();
        ImGui.TextDisabled("View mode");

        ImGui.Separator();

        if (_pm4SceneGraphMode == Pm4SceneGraphMode.FullScene)
        {
            DrawPm4MslkLinkingSummary();
            ImGui.Separator();
            DrawPm4FullSceneOutliner();
        }
        else
        {
            DrawPm4SelectedObjectGraph();
        }
    }

    /// <summary>
    /// Full scene outliner — like Blender's outliner. Shows ALL PM4 objects
    /// organized hierarchically by tile, CK24, part, MSLK group, and MscnRef.
    /// Click any item to select it and frame the camera on it.
    /// </summary>
    /// <summary>
    /// Draws a summary of MSLK linking data across all loaded PM4 files.
    /// Shows anchor-only vs path-window link counts and component coverage.
    /// </summary>
    private void DrawPm4MslkLinkingSummary()
    {
        if (_worldScene == null) return;

        if (!ImGui.CollapsingHeader("MSLK Linking Summary", ImGuiTreeNodeFlags.DefaultOpen))
            return;

        Pm4MslkLinkingStats stats = _worldScene.Pm4Overlay.GetPm4MslkLinkingStats();

        if (stats.TotalFiles == 0)
        {
            ImGui.TextDisabled("No PM4 files loaded.");
            return;
        }

        int totalChecked = stats.AnchorOnlyLinks + stats.PathWindowLinks;
        double anchorPct = totalChecked > 0 ? 100.0 * stats.AnchorOnlyLinks / totalChecked : 0;
        double pathPct = totalChecked > 0 ? 100.0 * stats.PathWindowLinks / totalChecked : 0;
        double linkedPct = stats.TotalComponents > 0 ? 100.0 * stats.ComponentsWithLinks / stats.TotalComponents : 0;
        double unlinkedPct = stats.TotalComponents > 0 ? 100.0 * stats.ComponentsWithoutLinks / stats.TotalComponents : 0;

        ImGui.TextDisabled($"Files: {stats.TotalFiles}  MSLK entries: {stats.TotalMslkEntries}");
        ImGui.Separator();
        ImGui.Text($"Anchor-only links: {stats.AnchorOnlyLinks} ({anchorPct:F1}%)");
        ImGui.TextDisabled("MspiFirstIndex < 0 — prior art reads these as doodad placements carrying anchors");
        ImGui.Text($"Path-window links: {stats.PathWindowLinks} ({pathPct:F1}%)");
        ImGui.TextDisabled("MspiFirstIndex >= 0 — these connect surfaces via MSPI/MSPV path vertices");
        ImGui.Separator();
        ImGui.Text($"Components with MSLK links: {stats.ComponentsWithLinks} ({linkedPct:F1}%)");
        ImGui.Text($"Components without links: {stats.ComponentsWithoutLinks} ({unlinkedPct:F1}%)");
        ImGui.TextDisabled($"RefIndex mismatches: {stats.RefIndexMismatches} (MSLK.RefIndex → MSUR bounds failures)");

        if (ImGui.IsItemHovered())
        {
            ImGui.BeginTooltip();
            ImGui.Text("MSLK.RefIndex → MSUR is 99.64% resolved (1,268,782 fits / 4,553 misses)");
            ImGui.Text("but that is a bounds test, not a semantic one.");
            ImGui.Text("The 4,553 misses are RefIndex values that don't index a valid MSUR entry.");
            ImGui.EndTooltip();
        }

        ImGui.Separator();
        ImGui.TextColored(new Vector4(0.8f, 0.9f, 1f, 1f), "Research leads:");
        ImGui.BulletText("Anchor-only links (53% of 1.27M links) — next place to look for doodad identity");
        ImGui.BulletText("MSLK.RefIndex semantics — 4,553 entries don't fit MSUR (gap in bounds test)");
        ImGui.BulletText("MSLK.TypeFlags — 10 distinct values, 19 subtypes (0x03=M2, 0x10=floor, 0x12=wall)");
    }

    private void DrawPm4FullSceneOutliner()
    {
        if (_worldScene == null) return;

        // Collect all objects grouped by tile, then by CK24, then by part
        // Use a flat list with unnamed tuples to match GetPm4TileObjectSummaries return type
        // Indices: 0=tx, 1=ty, 2=ck24, 3=part, 4=ck24ObjectId, 5=mshdRegionId, 6=linkGroupObjectId,
        //          7=groupKey, 8=attributeMask, 9=mscnRefIndex, 10=surfaceCount, 11=totalIndexCount,
        //          12=avgHeight, 13=boundsMin, 14=boundsMax, 15=linkedPositionRefCount
        var allObjects = new List<(int tx, int ty, uint ck24, int part, uint ck24ObjectId, uint mshdRegionId, uint linkGroupObjectId, byte groupKey, byte attributeMask, uint mscnRefIndex, int surfaceCount, int totalIndexCount, float avgHeight, Vector3 boundsMin, Vector3 boundsMax, int linkedPositionRefCount)>();
        foreach (var (tileKey, objects) in _worldScene.Pm4Overlay.GetPm4TileObjectSummaries())
        {
            foreach (var o in objects)
            {
                allObjects.Add((tileKey.tileX, tileKey.tileY, o.Item1, o.Item2, o.Item3,
                    o.Item4, o.Item5, o.Item6, o.Item7,
                    o.Item8, o.Item9, o.Item10, o.Item11,
                    o.Item12, o.Item13, o.Item14));
            }
        }

        if (allObjects.Count == 0)
        {
            ImGui.TextDisabled("No PM4 objects loaded. Enable PM4 Overlay to populate the scene graph.");
            return;
        }

        // Group by tile
        var byTile = allObjects.GroupBy(static e => (e.tx, e.ty)).OrderBy(static g => g.Key.ty).ThenBy(static g => g.Key.tx).ToList();

        ImGui.TextDisabled($"Total: {allObjects.Count} objects across {byTile.Count} tiles");

        // Search filter
        ImGui.SetNextItemWidth(ImGui.GetContentRegionAvail().X);
        ImGui.InputText("##Pm4SceneFilter", ref _pm4SceneFilter, 256);
        bool hasFilter = !string.IsNullOrWhiteSpace(_pm4SceneFilter);
        string filterLower = hasFilter ? _pm4SceneFilter.ToLowerInvariant() : "";

        ImGui.Separator();

        // Build tree: Tile → CK24 → Part
        foreach (var tileGroup in byTile)
        {
            var (tx, ty) = tileGroup.Key;
            var tileObjects = tileGroup.ToList();

            // Group by CK24
            var byCk24 = tileObjects
                .GroupBy(static e => e.ck24)
                .OrderBy(static g => g.Key)
                .ToList();

            int tileSurfaceCount = tileObjects.Sum(static e => e.surfaceCount);
            int tileCk24Count = byCk24.Count;

            string tileLabel = $"Tile ({tx}, {ty})  [{tileCk24Count} CK24 groups, {tileSurfaceCount} surfaces]";
            if (hasFilter && !tileLabel.ToLowerInvariant().Contains(filterLower))
            {
                bool childMatch = byCk24.Any(g => Ck24MatchesFilter(g.Key, g.ToList(), filterLower));
                if (!childMatch) continue;
            }

            ImGuiTreeNodeFlags tileFlags = ImGuiTreeNodeFlags.DefaultOpen;
            if (!ImGui.TreeNodeEx($"##Tile_{tx}_{ty}", tileFlags, tileLabel))
                continue;

            foreach (var ck24Group in byCk24)
            {
                uint ck24 = ck24Group.Key;
                var ck24Objects = ck24Group.ToList();
                var first = ck24Objects[0];
                byte ck24Type = (byte)((ck24 >> 16) & 0xFF);
                ushort ck24ObjId = (ushort)(ck24 & 0xFFFF);
                uint regionId = first.mshdRegionId;
                uint linkGroupId = first.linkGroupObjectId;
                int ck24SurfaceCount = ck24Objects.Sum(static e => e.surfaceCount);
                int partCount = ck24Objects.Count;

                string typeLabel = GetCk24TypeLabel(ck24Type);
                string ck24Label = $"CK24 0x{ck24:X6}  Type=0x{ck24Type:X2} ({typeLabel})  ObjId={ck24ObjId}  [{partCount} parts, {ck24SurfaceCount} surfaces]  Region={regionId}  MSLK=0x{linkGroupId:X8}";

                if (hasFilter && !ck24Label.ToLowerInvariant().Contains(filterLower))
                    continue;

                bool isCk24Selected = _worldScene.Pm4Overlay.SelectedPm4ObjectKey.HasValue
                    && _worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value.tileX == tx
                    && _worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value.tileY == ty
                    && _worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value.ck24 == ck24;

                ImGuiTreeNodeFlags ck24Flags = isCk24Selected
                    ? ImGuiTreeNodeFlags.DefaultOpen
                    : ImGuiTreeNodeFlags.None;

                bool ck24Open = ImGui.TreeNodeEx($"##CK24_{tx}_{ty}_{ck24}", ck24Flags, ck24Label);

                if (ImGui.IsItemClicked(ImGuiMouseButton.Right))
                    ImGui.OpenPopup($"##CK24Ctx_{tx}_{ty}_{ck24}");

                if (ImGui.BeginPopup($"##CK24Ctx_{tx}_{ty}_{ck24}"))
                {
                    if (ImGui.Selectable("Select All Parts"))
                    {
                        var f = ck24Objects[0];
                        _worldScene.Pm4Overlay.SelectPm4ObjectByKey(f.tx, f.ty, f.ck24, f.part);
                        if (_worldScene.Pm4Overlay.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo dbg))
                            FocusCameraOnBounds(dbg.BoundsMin, dbg.BoundsMax);
                    }
                    if (ImGui.Selectable("Frame All Parts"))
                    {
                        Vector3 bmin = new(float.MaxValue), bmax = new(float.MinValue);
                        foreach (var o in ck24Objects)
                        {
                            bmin = Vector3.Min(bmin, o.boundsMin);
                            bmax = Vector3.Max(bmax, o.boundsMax);
                        }
                        FocusCameraOnBounds(bmin, bmax);
                    }
                    ImGui.EndPopup();
                }

                if (!ck24Open) continue;

                // Show MSLK linking info at CK24 level
                uint distinctLinkGroupId = ck24Objects.Select(static e => e.linkGroupObjectId).Distinct().SingleOrDefault();
                int totalLinkedRefs = ck24Objects.Sum(static e => e.linkedPositionRefCount);
                if (totalLinkedRefs > 0 || distinctLinkGroupId != 0)
                {
                    ImGui.TextDisabled($"  MSLK Group=0x{distinctLinkGroupId:X8}  Linked MPRL refs={totalLinkedRefs}");
                }

                // Show parts
                foreach (var o in ck24Objects.OrderBy(static e => e.part))
                {
                    bool isPartSelected = _worldScene.Pm4Overlay.SelectedPm4ObjectKey.HasValue
                        && _worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value.tileX == tx
                        && _worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value.tileY == ty
                        && _worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value.ck24 == ck24
                        && _worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value.objectPart == o.part;

                    string partLabel = $"  Part {o.part}  ({o.surfaceCount} surfs, {o.totalIndexCount} idx)  "
                        + $"GK=0x{o.groupKey:X2}  Attr=0x{o.attributeMask:X2}  "
                        + $"MscnRef={o.mscnRefIndex}  H={o.avgHeight:F1}"
                        + (isPartSelected ? "  [SELECTED]" : "");

                    if (hasFilter && !partLabel.ToLowerInvariant().Contains(filterLower))
                        continue;

                    if (ImGui.Selectable($"##Part_{tx}_{ty}_{ck24}_{o.part}", isPartSelected, ImGuiSelectableFlags.None, new Vector2(ImGui.GetContentRegionAvail().X, 0)))
                    {
                        _worldScene.Pm4Overlay.SelectPm4ObjectByKey(tx, ty, ck24, o.part);
                        if (_worldScene.Pm4Overlay.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo dbg))
                            FocusCameraOnBounds(dbg.BoundsMin, dbg.BoundsMax);
                    }

                    if (ImGui.IsItemHovered())
                    {
                        ImGui.BeginTooltip();
                        ImGui.Text($"CK24: 0x{ck24:X6}  Type: 0x{ck24Type:X2}  ObjId: {ck24ObjId}");
                        ImGui.Text($"MSLK Group: 0x{linkGroupId:X8}");
                        ImGui.Text($"Linked MPRL refs: {o.linkedPositionRefCount}");
                        ImGui.Text($"MSHD: F00={first.ck24ObjectId}  Region={regionId}");
                        ImGui.Text($"GroupKey: 0x{o.groupKey:X2}  Attr: 0x{o.attributeMask:X2}  MscnRef: {o.mscnRefIndex}");
                        ImGui.Text($"Surfaces: {o.surfaceCount}  Indices: {o.totalIndexCount}");
                        ImGui.Text($"Bounds: ({o.boundsMin.X:F1}, {o.boundsMin.Y:F1}, {o.boundsMin.Z:F1}) - ({o.boundsMax.X:F1}, {o.boundsMax.Y:F1}, {o.boundsMax.Z:F1})");
                        ImGui.EndTooltip();
                    }
                }

                ImGui.TreePop();
            }
            ImGui.TreePop();
        }
    }

    private void DrawPm4SelectedObjectGraph()
    {
        if (_worldScene == null) return;

        if (!_worldScene.Pm4Overlay.TryGetSelectedPm4ObjectGraphInfo(out Pm4SelectedObjectGraphInfo graph))
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

        // Summary header
        ImGui.Text($"CK24 0x{graph.Ck24:X6}  Type=0x{graph.Ck24Type:X2}  ObjId={graph.Ck24ObjectId}");
        ImGui.TextDisabled($"Tile ({graph.SelectedTileX}, {graph.SelectedTileY})  Part {graph.SelectedObjectPartId}");
        ImGui.TextDisabled($"{graph.LinkGroupCount} link groups, {graph.MscnRefGroupCount} mscnRef groups, {graph.PartCount} parts, {graph.SurfaceCount} surfaces, {graph.TotalIndexCount} indices");
        ImGui.Separator();

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
                            $"LinkGroup 0x{linkGroup.LinkGroupObjectId:X8} [{linkGroup.MscnRefGroups.Count} mscnRefs, {linkGroup.SurfaceCount} surfaces]"))
                    {
                        // Show linked position ref summary
                        if (linkGroup.LinkedPositionRefCount > 0)
                        {
                            var lprs = linkGroup.LinkedPositionRefSummary;
                            ImGui.TextDisabled($"  MPRL refs: {lprs.TotalCount} (normal={lprs.NormalCount}, term={lprs.TerminatorCount})  floor={lprs.FloorMin}-{lprs.FloorMax}  heading={lprs.HeadingMeanDegrees:F1}°");
                        }

                        foreach (var mscnRefGroup in linkGroup.MscnRefGroups)
                        {
                            if (ImGui.TreeNodeEx($"##MR_{mscnRefGroup.MscnRefIndex}",
                                    ImGuiTreeNodeFlags.None,
                                    $"MscnRef {mscnRefGroup.MscnRefIndex} [{mscnRefGroup.PartCount} parts, {mscnRefGroup.SurfaceCount} surfaces]"))
                            {
                                foreach (var part in mscnRefGroup.Parts)
                                {
                                    string selected = part.IsSelected ? " [SELECTED]" : "";
                                    string partLabel = $"Part {part.ObjectPartId} @ {part.TileX}_{part.TileY}{selected}  ({part.SurfaceCount} surfs, {part.TotalIndexCount} idx)";
                                    if (ImGui.Selectable($"##Part_{part.TileX}_{part.TileY}_{part.ObjectPartId}", part.IsSelected))
                                    {
                                        _worldScene.Pm4Overlay.SelectPm4ObjectByKey(part.TileX, part.TileY, graph.Ck24, part.ObjectPartId);
                                    }
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

    private static string GetCk24TypeLabel(byte ck24Type)
    {
        return ck24Type switch
        {
            0x00 => "doodad/decor",
            0x40 => "WMO roof",
            0x41 => "WMO wall",
            0x42 => "WMO structure",
            0x43 => "WMO detail",
            0x44 => "WMO column",
            0x45 => "WMO arch",
            0x46 => "WMO stair",
            0x47 => "WMO platform",
            0x48 => "WMO bridge",
            0x80 => "terrain/ground",
            0x81 => "terrain cliff",
            0x82 => "terrain slope",
            _ => $"0x{ck24Type:X2}"
        };
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
        sw.WriteLine($"- Objects: `{_worldScene.Pm4Overlay.Pm4VisibleObjectCount}` loaded, `{_worldScene.Pm4Overlay.Pm4LoadedFiles}` files");
        sw.WriteLine();

        // Legend
        Pm4ColorLegendInfo legend = _worldScene.Pm4Overlay.GetPm4ColorLegend();
        sw.WriteLine($"Color mode: `{_worldScene.Pm4Overlay.Pm4ColorMode}` — {legend.Description}");
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
        Pm4VisibleOverlaySummaryInfo summary = _worldScene.Pm4Overlay.GetPm4VisibleOverlaySummary(10, 4);
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
        if (_worldScene.Pm4Overlay.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo debug))
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

            if (_worldScene.Pm4Overlay.TryGetSelectedPm4ObjectResearchInfo(out Pm4SelectedObjectResearchInfo rinfo))
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

    private static bool Ck24MatchesFilter(uint ck24, List<(int tx, int ty, uint ck24, int part, uint ck24ObjectId, uint mshdRegionId, uint linkGroupObjectId, byte groupKey, byte attributeMask, uint mscnRefIndex, int surfaceCount, int totalIndexCount, float avgHeight, Vector3 boundsMin, Vector3 boundsMax, int linkedPositionRefCount)> ck24Objects, string filterLower)
    {
        var first = ck24Objects[0];
        byte ck24Type = (byte)((ck24 >> 16) & 0xFF);
        ushort ck24ObjId = (ushort)(ck24 & 0xFFFF);
        string ck24Label = $"CK24 0x{ck24:X6}  Type=0x{ck24Type:X2}  ObjId={ck24ObjId}  Region={first.mshdRegionId}  MSLK=0x{first.linkGroupObjectId:X8}";
        if (ck24Label.ToLowerInvariant().Contains(filterLower))
            return true;

        foreach (var o in ck24Objects)
        {
            string partLabel = $"Part {o.part}  ({o.surfaceCount} surfs)  GK=0x{o.groupKey:X2}  Attr=0x{o.attributeMask:X2}  MscnRef={o.mscnRefIndex}";
            if (partLabel.ToLowerInvariant().Contains(filterLower))
                return true;
        }

        return false;
    }
}
