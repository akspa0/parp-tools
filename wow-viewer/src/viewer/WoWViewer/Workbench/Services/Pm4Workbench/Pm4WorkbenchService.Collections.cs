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

// Pm4WorkbenchService: colour legend, region summaries, LLM evidence bundle, SVG output, object collections and their JSON.
// Pm4WorkbenchService: members moved from ViewerApp_Pm4Utilities.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
// Original file note: Partial class containing PM4 alignment and viewer utility windows.
internal sealed partial class Pm4WorkbenchService
{

    /// <summary>MSUR._0x00 values seen corpus-wide, with what each was measured to separate.</summary>
    private static readonly (byte Value, string Label)[] Pm4SurfaceClassLabels =
    [
        (0x03, "0x03  doodad surfaces"),
        (0x10, "0x10  lowest floors"),
        (0x11, "0x11  floors"),
        (0x12, "0x12  floors (most common)"),
        (0x13, "0x13  floors (upper)"),
        (0x14, "0x14  highest / angled"),
        (0x15, "0x15  lowest (rare)"),
    ];

    private static readonly Dictionary<byte, string> Pm4SurfaceClassTooltips = new()
    {
        [0x03] = "184,356 surfaces. 100.0% carry NO placement height - this is the doodad marker, and a second identifier for the M2 population independent of the Z field.",
        [0x10] = "87,243 surfaces. Sits lowest inside its object (0.291 of its Z extent). 95.3% Z-dominant, 92.9% up-facing.",
        [0x11] = "13,493 surfaces. 0.402 height in object, 95.2% Z-dominant.",
        [0x12] = "161,359 surfaces, the most common placed class. 0.474 height in object, 91.8% Z-dominant.",
        [0x13] = "71,293 surfaces. 0.427 height in object, 93.3% Z-dominant.",
        [0x14] = "105 surfaces. Highest in its object (0.624) and the ONLY class that is mostly not Z-dominant (38.1%) - the closest thing to a non-floor here.",
        [0x15] = "243 surfaces. Lowest of all (0.161) and almost perfectly flat: 99.2% Z-dominant, 99.6% up-facing.",
    };

    private static string GetPm4ColorModeLabel(Pm4OverlayColorMode mode)
    {
        return mode switch
        {
            Pm4OverlayColorMode.PlacementZ => "Placement Z (MSUR._0x1C as float)",
            Pm4OverlayColorMode.Population => "Population: placed vs doodad",
            Pm4OverlayColorMode.Tile => "Tile",
            Pm4OverlayColorMode.MshdRegionId => "MSHD RegionId (partial)",
            Pm4OverlayColorMode.SurfaceCount => "Surface count",
            Pm4OverlayColorMode.GroupKey => "MSUR._0x00 (unmeasured)",
            Pm4OverlayColorMode.Height => "Height",
            Pm4OverlayColorMode.TypeFlags => "MSLK._0x00 TypeFlags (partial)",
            _ => mode.ToString(),
        };
    }

    private void DrawPm4ColorLegend(string idSuffix = "")
    {
        if (_worldScene == null || !_worldScene.Pm4Overlay.ShowPm4Overlay)
            return;

        Pm4ColorLegendInfo legend = _worldScene.Pm4Overlay.GetPm4ColorLegend();
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
        if (_worldScene == null || !_worldScene.Pm4Overlay.TryGetSelectedPm4RegionInfo(out Pm4SelectedObjectRegionInfo regionInfo))
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

        int added = AddPm4ObjectsToCollection(_worldScene.Pm4Overlay.GetVisiblePm4ObjectsForRegion(regionId));
        _statusMessage = added > 0
            ? $"Added {added} visible PM4 parts from MSHD region {regionId} to the collection."
            : $"All visible PM4 parts from MSHD region {regionId} were already in the collection.";
        SyncPm4CollectionHighlight();
        return added;
    }

    internal void ExportPm4LlmEvidenceBundle()
    {
        if (_worldScene == null)
            return;

        Directory.CreateDirectory(ExportDir);
        ImGuiPathPicker.Instance.Open(
            "Choose a folder for the PM4 LLM evidence bundle",
            pickFolder: true,
            initialPath: ExportDir,
            filterExtension: null,
            picked =>
            {
                if (string.IsNullOrWhiteSpace(picked))
                    return;

                try
                {
                    string mapName = _terrainManager?.MapName ?? _worldScene.Terrain.MapName ?? "map";
                    string bundleDirectory = Path.Combine(
                        picked,
                        $"pm4_llm_{ProjectOutputService.SanitizeProjectPathSegment(mapName)}_{DateTime.Now:yyyyMMdd_HHmmss}");
                    Directory.CreateDirectory(bundleDirectory);

                    Pm4VisibleOverlaySummaryInfo visibleSummary = _worldScene.Pm4Overlay.GetPm4VisibleOverlaySummary();
                    Pm4ObjectDebugInfo selectedDebugInfo = default;
                    bool hasSelectedObject = _worldScene.Pm4Overlay.SelectedPm4ObjectKey.HasValue
                        && _worldScene.Pm4Overlay.TryGetSelectedPm4ObjectDebugInfo(out selectedDebugInfo);
                    bool hasSelectedRegion = _worldScene.Pm4Overlay.TryGetSelectedPm4RegionInfo(out Pm4SelectedObjectRegionInfo selectedRegionInfo);

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
            });
    }

    private object BuildJsonSafePm4LlmBundle(
        Pm4VisibleOverlaySummaryInfo visibleSummary,
        Pm4ObjectDebugInfo? selectedDebugInfo,
        Pm4SelectedObjectRegionInfo? selectedRegionInfo)
    {
        Pm4ColorLegendInfo legend = _worldScene!.Pm4Overlay.GetPm4ColorLegend(12);
        string mapName = _terrainManager?.MapName ?? _worldScene.Terrain.MapName ?? string.Empty;

        return new
        {
            generatedAtUtc = DateTime.UtcNow,
            mapName,
            pm4Status = _worldScene.Pm4Overlay.Pm4Status,
            pm4VisibleObjectCount = _worldScene.Pm4Overlay.Pm4VisibleObjectCount,
            pm4ObjectCount = _worldScene.Pm4Overlay.Pm4ObjectCount,
            pm4LoadedFiles = _worldScene.Pm4Overlay.Pm4LoadedFiles,
            pm4TotalFiles = _worldScene.Pm4Overlay.Pm4TotalFiles,
            colorMode = _worldScene.Pm4Overlay.Pm4ColorMode.ToString(),
            colorModeLabel = GetPm4ColorModeLabel(_worldScene.Pm4Overlay.Pm4ColorMode),
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
            selectedObject = selectedDebugInfo.HasValue && _worldScene.Pm4Overlay.SelectedPm4ObjectKey.HasValue
                ? new
                {
                    tileX = _worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value.tileX,
                    tileY = _worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value.tileY,
                    ck24 = selectedDebugInfo.Value.Ck24,
                    ck24Type = selectedDebugInfo.Value.Ck24Type,
                    ck24ObjectId = selectedDebugInfo.Value.Ck24ObjectId,
                    objectPartId = _worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value.objectPart,
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
        builder.AppendLine($"- PM4 status: `{_worldScene!.Pm4Overlay.Pm4Status}`");
        builder.AppendLine($"- Color mode: `{_worldScene.Pm4Overlay.Pm4ColorMode}` ({GetPm4ColorModeLabel(_worldScene.Pm4Overlay.Pm4ColorMode)})");
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

        if (selectedDebugInfo.HasValue && _worldScene.Pm4Overlay.SelectedPm4ObjectKey.HasValue)
        {
            builder.AppendLine();
            builder.AppendLine("## Selected Object");
            builder.AppendLine();
            builder.AppendLine($"- tile=(`{_worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value.tileX}`, `{_worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value.tileY}`) ck24=`0x{selectedDebugInfo.Value.Ck24:X6}` part=`{_worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value.objectPart}` type=`0x{selectedDebugInfo.Value.Ck24Type:X2}` objId=`{selectedDebugInfo.Value.Ck24ObjectId}`");
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

        if (!_worldScene.Pm4Overlay.SelectPm4Object(objectKey))
        {
            _statusMessage = $"PM4 graph part CK24=0x{objectKey.ck24:X6} part={objectKey.objectPart} is no longer available.";
            return;
        }

        OpenPm4Workbench(Pm4WorkbenchTab.Selection);

        if (frameCamera)
        {
            if (_worldScene.Pm4Overlay.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo debugInfo))
                FocusCameraOnBounds(debugInfo.BoundsMin, debugInfo.BoundsMax);
        }

        _statusMessage = frameCamera
            ? $"Selected and framed PM4 graph part CK24=0x{objectKey.ck24:X6} part={objectKey.objectPart}."
            : $"Selected PM4 graph part CK24=0x{objectKey.ck24:X6} part={objectKey.objectPart}.";
    }

    private void ExportSelectedPm4GraphJson(Pm4SelectedObjectGraphInfo graph)
    {
        string defaultName = $"pm4_graph_ck24_{graph.Ck24:X6}_part_{graph.SelectedObjectPartId:D4}_{DateTime.Now:yyyyMMdd_HHmmss}.json";
        ImGuiPathPicker.Instance.Open(
            "Save Selected PM4 Graph JSON",
            ImGuiPathPickerMode.SaveFile,
            ExportDir,
            ".json",
            picked =>
            {
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
            },
            defaultName);
    }

    private void AddSelectedPm4ObjectToCollection()
    {
        if (_worldScene == null || !_worldScene.Pm4Overlay.SelectedPm4ObjectKey.HasValue)
            return;

        TogglePm4ObjectCollectionMembership(_worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value, reportStatus: true, removeIfPresent: false);
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

    internal bool TogglePm4ObjectCollectionMembership(
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
        _worldScene?.Pm4Overlay.SetHighlightedPm4Objects(_pm4ObjectCollection);
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
                    && _worldScene.Pm4Overlay.SelectedPm4ObjectKey.HasValue
                    && _worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value == key;
                uint? regionId = _worldScene != null
                    && _worldScene.Pm4Overlay.TryGetPm4ObjectDebugInfo(key, out Pm4ObjectDebugInfo debugInfo)
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
            if (!_worldScene.Pm4Overlay.TryGetPm4ObjectDebugInfo(_pm4ObjectCollection[index], out _))
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
        ImGuiPathPicker.Instance.Open(
            "Save PM4 Collection JSON",
            ImGuiPathPickerMode.SaveFile,
            ExportDir,
            ".json",
            picked =>
            {
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
            },
            defaultName);
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
            if (!_worldScene.Pm4Overlay.TryGetPm4ObjectDebugInfo(key, out Pm4ObjectDebugInfo debugInfo))
                continue;

            _worldScene.Pm4Overlay.TryGetPm4ObjectGroupKey(key, out var mergedGroupKey);
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
            currentSelection = _worldScene.Pm4Overlay.SelectedPm4ObjectKey.HasValue
                ? new
                {
                    tileX = _worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value.tileX,
                    tileY = _worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value.tileY,
                    ck24 = _worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value.ck24,
                    objectPartId = _worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value.objectPart,
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
}
