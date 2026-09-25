using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Visibility;

namespace WoWViewer;

public partial class ViewerApp
{
    [Flags]
    internal enum McnkOverlayFlags
    {
        None = 0,
        HasShadows = 0x1,
        Impassable = 0x2,
        River = 0x4,
        Ocean = 0x8,
        HasMagma = 0x10,
        HasSlime = 0x20,
        HasMccv = 0x40,
        HasBakedShadows = 0x20000,
    }

    internal static readonly string[] WorldObjectVisibilityProfileLabels =
    [
        "Quality",
        "Balanced",
        "Performance",
    ];

    internal enum VisualInvestigationMode
    {
        Auto,
        Adt,
        Wmo,
        M2,
        Pm4,
    }

    private VisualInvestigationMode _visualInvestigationMode = VisualInvestigationMode.Auto;
    private bool _showMcnkFlagOverlay;
    private bool _showMcnkWeakCorners = true;
    private McnkOverlayFlags _mcnkOverlayFlags = McnkOverlayFlags.Impassable;
    private int _lastMcnkOverlayChunkCount;
    private int _lastMcnkWeakCornerCount;
    private readonly int[] _litFocusTrackOrder =
    {
        LitLoader.TrackDirectColor,
        LitLoader.TrackAmbientColor,
        LitLoader.TrackSkyTop,
        LitLoader.TrackSkyHorizon,
        LitLoader.TrackFogColor,
    };

    private void DrawVisualInvestigationToolbox(bool showWorldObjectRangeControls)
    {
        ImGui.Text("Investigation Toolbox");
        ImGui.TextDisabled("Switch hover and inspector focus by visual target.");

        DrawVisualInvestigationModeButton(VisualInvestigationMode.Auto, "◎ Auto", "Follow the current hovered visual target.");
        ImGui.SameLine();
        DrawVisualInvestigationModeButton(VisualInvestigationMode.Adt, "▦ ADT", "Inspect terrain chunks, layers, alpha, and assigned MTEX textures.");
        ImGui.SameLine();
        DrawVisualInvestigationModeButton(VisualInvestigationMode.Wmo, "▣ WMO", "Limit hover inspection to WMO placements.");
        ImGui.SameLine();
        DrawVisualInvestigationModeButton(VisualInvestigationMode.M2, "◇ M2", "Limit hover inspection to MDX/M2 doodad placements.");

        ImGui.TextDisabled($"Current target: {GetVisualInvestigationModeLabel(_visualInvestigationMode)}");

        if (!showWorldObjectRangeControls || _worldScene == null)
            return;

        float objectRangeMultiplier = _worldScene.ObjectStreamingRangeMultiplier;
        if (ImGui.SliderFloat("Object Stream Range", ref objectRangeMultiplier, 0.25f, 4.00f, "%.2fx"))
            _worldScene.ObjectStreamingRangeMultiplier = objectRangeMultiplier;

        bool useHierarchicalSceneTraversal = _worldScene.UseHierarchicalSceneTraversal;
        if (ImGui.Checkbox("Use ADT Scene Graph", ref useHierarchicalSceneTraversal))
            _worldScene.UseHierarchicalSceneTraversal = useHierarchicalSceneTraversal;

        int visibilityProfileIndex = (int)_worldScene.ObjectVisibilityProfile;
        if (ImGui.Combo("Object Detail", ref visibilityProfileIndex, WorldObjectVisibilityProfileLabels, WorldObjectVisibilityProfileLabels.Length))
            _worldScene.ObjectVisibilityProfile = (WorldObjectVisibilityProfile)visibilityProfileIndex;

        bool showWmos = _worldScene.WmosVisible;
        if (ImGui.Checkbox("Show WMOs", ref showWmos))
            _worldScene.WmosVisible = showWmos;
        ImGui.SameLine();
        bool showDoodads = _worldScene.DoodadsVisible;
        if (ImGui.Checkbox("Show Doodads", ref showDoodads))
            _worldScene.DoodadsVisible = showDoodads;

        WorldRenderFrameStats stats = _worldScene.LastRenderFrameStats;
        double wmoObjectCostMs = stats.WmoVisibility.DurationMs
            + stats.WmoSubmission.DurationMs
            + stats.WmoTransparentSubmission.DurationMs;
        double mdxObjectCostMs = stats.MdxVisibility.DurationMs + stats.MdxOpaqueSubmission.DurationMs
            + stats.MdxTransparentSort.DurationMs + stats.MdxTransparentSubmission.DurationMs;
        string hotspot = wmoObjectCostMs >= mdxObjectCostMs
            ? $"WMO scene-pass extraction is still the larger measured object-side cost ({wmoObjectCostMs:0.00} ms)."
            : $"MDX visibility/submission is currently larger ({mdxObjectCostMs:0.00} ms).";
        ImGui.TextDisabled(hotspot);
        ImGui.TextDisabled($"ADT graph: {(_worldScene.IsHierarchicalSceneTraversalActive ? "active" : "inactive")}  roots={_worldScene.SceneGraphResidentAdtCount}");
        ImGui.TextDisabled("Visibility admission and queued object loads use this multiplier. Default is 0.50x.");
        ImGui.TextDisabled("Balanced/Performance also use FOV-aware projected-size culling and stop queueing tiny off-view assets.");
        ImGui.TextDisabled("MDX 'batched' counts in the stats are shared-shader submissions, not true GPU instancing.");

        ImGui.Separator();
        DrawHoveredAssetInspectionContent();
    }

    private void DrawVisualInvestigationModeButton(VisualInvestigationMode mode, string label, string tooltip)
    {
        bool selected = _visualInvestigationMode == mode;
        if (selected)
        {
            ImGui.PushStyleColor(ImGuiCol.Button, new Vector4(0.20f, 0.42f, 0.64f, 1f));
            ImGui.PushStyleColor(ImGuiCol.ButtonHovered, new Vector4(0.24f, 0.48f, 0.72f, 1f));
            ImGui.PushStyleColor(ImGuiCol.ButtonActive, new Vector4(0.16f, 0.36f, 0.56f, 1f));
        }

        if (ImGui.Button(label))
            _visualInvestigationMode = mode;

        if (ImGui.IsItemHovered())
            ImGui.SetTooltip(tooltip);

        if (selected)
            ImGui.PopStyleColor(3);
    }

    private static string GetVisualInvestigationModeLabel(VisualInvestigationMode mode)
    {
        return mode switch
        {
            VisualInvestigationMode.Adt => "ADT chunk",
            VisualInvestigationMode.Wmo => "WMO",
            VisualInvestigationMode.M2 => "M2/MDX",
            _ => "Auto",
        };
    }

    private void DrawHoveredAssetInspectionContent()
    {
        if (_worldScene == null)
            return;

        ImGui.Text("Hovered Asset");

        if (_visualInvestigationMode == VisualInvestigationMode.Adt)
        {
            ImGui.TextDisabled("Switch to Auto, WMO, or M2/MDX mode to inspect hovered asset paths.");
            return;
        }

        if (_worldScene.HoveredAssetInfo is not HoveredAssetInfo info || !ShouldShowHoveredAssetInfoForInvestigation(info))
        {
            ImGui.TextDisabled("Hover a WMO or MDX/M2 placement to inspect its exact asset path.");
            return;
        }

        ImGui.TextDisabled($"Kind: {info.AssetKind}");
        ImGui.TextWrapped(info.DisplayName);

        if (!string.IsNullOrWhiteSpace(info.SourcePath))
            DrawAssetPathActions("Asset Path", info.SourcePath, "HoveredInvestigationAsset");

        if (!string.IsNullOrWhiteSpace(info.DetailLine))
            ImGui.TextDisabled(info.DetailLine);

        ImGui.TextDisabled($"World: ({info.WorldPosition.X:F1}, {info.WorldPosition.Y:F1}, {info.WorldPosition.Z:F1})");

        if (info.HasSceneObject)
        {
            if (ImGui.SmallButton("Inspect Hovered In Scene"))
            {
                if (TryInspectHoveredSceneAssetInSelection())
                    _statusMessage = $"Selected hovered {info.AssetKind}: {info.DisplayName}";
            }
        }
        else
        {
            ImGui.TextDisabled("This hovered target is not a selectable scene object.");
        }
    }

    private void DrawTerrainChunkInvestigationPanel(bool defaultOpen)
    {
        if (_terrainManager == null && _vlmTerrainManager == null)
            return;

        ImGuiTreeNodeFlags flags = defaultOpen ? ImGuiTreeNodeFlags.DefaultOpen : ImGuiTreeNodeFlags.None;
        if (!ImGui.CollapsingHeader("ADT Chunk Investigation", flags))
            return;

        DrawTerrainChunkInvestigationContent();
    }

    // Spec 223: all chunk detail and overlay controls have one Inspector owner.
    private void DrawTerrainChunkInvestigationContent()
    {
        if (_useTabUi)
        {
            ImGui.TextDisabled("Chunk details and flag overlays are in Inspector > Context.");
            if (ImGui.Button("Open Terrain Inspector##Investigation"))
                OpenWorkbenchTab(WoWViewer.Workbench.WorkbenchTab.Inspect);
        }
        else
        {
            DrawUnifiedInspectorContent();
        }
    }

    private void DrawMcnkExplorerContent() => DrawTerrainChunkInvestigationContent();
    private void DrawWlLiquidInvestigationPanel(bool defaultOpen)
    {
        if (_worldScene == null)
            return;

        ImGuiTreeNodeFlags flags = defaultOpen ? ImGuiTreeNodeFlags.DefaultOpen : ImGuiTreeNodeFlags.None;
        if (!ImGui.CollapsingHeader("WL Liquid Investigation", flags))
            return;

        bool showWlLiquids = _worldScene.ShowWlLiquids;
        if (ImGui.Checkbox("Show WL Liquids", ref showWlLiquids))
            _worldScene.ShowWlLiquids = showWlLiquids;

        if (!_worldScene.WlLoadAttempted)
        {
            ImGui.TextDisabled("WLW/WLM/WLQ/WLL files are lazy-loaded for the current map.");
            if (ImGui.Button("Load WL Liquids"))
                _worldScene.ShowWlLiquids = true;
            return;
        }

        WlLiquidLoader? loader = _worldScene.WlLoader;
        LiquidRenderer? liquidRenderer = _terrainManager?.LiquidRenderer;
        if (loader == null || !loader.HasData)
        {
            ImGui.TextDisabled("No WL loose liquid files were found for this map.");
            return;
        }

        var ts = WlLiquidLoader.TransformSettings;
        int visibleCount = loader.Bodies.Count;
        if (liquidRenderer != null)
        {
            visibleCount = 0;
            for (int i = 0; i < loader.Bodies.Count; i++)
            {
                if (liquidRenderer.IsWlBodyVisible(loader.Bodies[i].BodyKey))
                    visibleCount++;
            }
        }

        ImGui.TextDisabled($"Files: {loader.SourceFileCount}  Bodies: {loader.Bodies.Count}  Blocks: {loader.TotalBlockCount}");
        ImGui.TextDisabled($"Visible: {visibleCount}/{loader.Bodies.Count}  Mode: {GetWlLiquidGroupingModeLabel(ts.GroupingMode)}");

        if (liquidRenderer != null)
        {
            bool showSelectedWiremesh = liquidRenderer.ShowSelectedWlWireframeOverlay;
            if (ImGui.Checkbox("Show Selected WL Wiremesh Overlay", ref showSelectedWiremesh))
                liquidRenderer.ShowSelectedWlWireframeOverlay = showSelectedWiremesh;
        }

        if (IsWlListIsolationActive)
        {
            ImGui.TextDisabled("List is isolated to the selected WL body from the last scene pick.");
            ImGui.SameLine();
            if (ImGui.SmallButton("Clear WL List Isolation"))
                _wlLayerListIsolationEnabled = false;
        }

        WlLiquidLoader.WlBodyGroupingMode groupingMode = ts.GroupingMode;
        if (ImGui.BeginCombo("WL Grouping", GetWlLiquidGroupingModeLabel(groupingMode)))
        {
            foreach (WlLiquidLoader.WlBodyGroupingMode option in Enum.GetValues<WlLiquidLoader.WlBodyGroupingMode>())
            {
                bool isSelected = option == groupingMode;
                if (ImGui.Selectable(GetWlLiquidGroupingModeLabel(option), isSelected))
                    ts.GroupingMode = option;
                if (isSelected)
                    ImGui.SetItemDefaultFocus();
            }

            ImGui.EndCombo();
        }

        float planeHeightTolerance = ts.PlaneHeightTolerance;
        if (ts.GroupingMode != WlLiquidLoader.WlBodyGroupingMode.PlaneWelded)
            ImGui.BeginDisabled();
        if (ImGui.SliderFloat("WL Plane Weld Tolerance", ref planeHeightTolerance, 0.05f, 4.00f, "%.2f"))
            ts.PlaneHeightTolerance = planeHeightTolerance;
        if (ts.GroupingMode != WlLiquidLoader.WlBodyGroupingMode.PlaneWelded)
            ImGui.EndDisabled();

        if (ImGui.Button("Apply + Reload WL"))
            _worldScene.ReloadWlLiquids();

        if (liquidRenderer != null)
        {
            ImGui.SameLine();
            if (ImGui.SmallButton("Show All WL"))
                liquidRenderer.SetAllWlBodiesVisible(true);

            ImGui.SameLine();
            if (ImGui.SmallButton("Hide All WL"))
                liquidRenderer.SetAllWlBodiesVisible(false);

            ImGui.SameLine();
            bool hasSelectedBody = !string.IsNullOrWhiteSpace(_wlLayerSelectedBodyKey);
            if (!hasSelectedBody)
                ImGui.BeginDisabled();
            if (ImGui.SmallButton("Solo Selected WL"))
            {
                liquidRenderer.SetAllWlBodiesVisible(false);
                liquidRenderer.SetWlBodyVisible(_wlLayerSelectedBodyKey, true);
            }
            if (!hasSelectedBody)
                ImGui.EndDisabled();
        }

        ImGui.TextDisabled("Plane-welded mode is heuristic: neighboring WL blocks stay together only when their shared edge heights match within tolerance.");
        ImGui.Separator();

        if (TryGetFocusedWlLiquidBody(out WlLiquidBody? focusedBody, out bool usingHoveredBody) && focusedBody != null)
        {
            ImGui.TextDisabled(usingHoveredBody ? "Target: hovered WL body" : "Target: selected WL body");
            ImGui.Text($"{focusedBody.Name}");
            ImGui.TextDisabled(focusedBody.SourcePath);
            ImGui.TextDisabled($"{focusedBody.FileType}  {focusedBody.GroupLabel}  blocks={focusedBody.BlockCount}  verts={focusedBody.Vertices.Length}");
            ImGui.TextDisabled($"Height range: {focusedBody.MinHeight:F2} .. {focusedBody.MaxHeight:F2}  avg={focusedBody.AverageHeight:F2}");
            ImGui.TextDisabled($"Coord range: X {focusedBody.CoordMinX:F1}..{focusedBody.CoordMaxX:F1}  Y {focusedBody.CoordMinY:F1}..{focusedBody.CoordMaxY:F1}");
            ImGui.TextDisabled($"Metadata patterns: {focusedBody.MetadataPatternCount}  non-zero words/block: {focusedBody.MetadataNonZeroMin}..{focusedBody.MetadataNonZeroMax}");

            if (ImGui.SmallButton("Copy WL Summary"))
                CopyTextToClipboard(BuildWlLiquidSummary(focusedBody), "WL liquid summary");
        }
        else
        {
            ImGui.TextDisabled("Hover or select a WL body to inspect its grouped block data.");
        }

        ImGui.Separator();
        if (ImGui.BeginTable("##wl_investigation", 6, ImGuiTableFlags.BordersInnerV | ImGuiTableFlags.RowBg | ImGuiTableFlags.SizingStretchProp | ImGuiTableFlags.ScrollY, new Vector2(0f, 240f)))
        {
            ImGui.TableSetupColumn("V", ImGuiTableColumnFlags.WidthFixed, 24f);
            ImGui.TableSetupColumn("Type", ImGuiTableColumnFlags.WidthFixed, 46f);
            ImGui.TableSetupColumn("Group", ImGuiTableColumnFlags.WidthFixed, 78f);
            ImGui.TableSetupColumn("Blocks", ImGuiTableColumnFlags.WidthFixed, 54f);
            ImGui.TableSetupColumn("Z", ImGuiTableColumnFlags.WidthFixed, 110f);
            ImGui.TableSetupColumn("Body", ImGuiTableColumnFlags.WidthStretch);
            ImGui.TableHeadersRow();

            for (int i = 0; i < loader.Bodies.Count; i++)
            {
                WlLiquidBody body = loader.Bodies[i];
                if (!ShouldIncludeWlBodyInUiList(body))
                    continue;

                ImGui.TableNextRow();

                ImGui.TableSetColumnIndex(0);
                bool visible = liquidRenderer == null || liquidRenderer.IsWlBodyVisible(body.BodyKey);
                if (liquidRenderer == null)
                    ImGui.BeginDisabled();
                if (ImGui.Checkbox($"##wl_invest_vis_{i}", ref visible) && liquidRenderer != null)
                    liquidRenderer.SetWlBodyVisible(body.BodyKey, visible);
                if (liquidRenderer == null)
                    ImGui.EndDisabled();

                ImGui.TableSetColumnIndex(1);
                ImGui.TextUnformatted(body.FileType.ToString());

                ImGui.TableSetColumnIndex(2);
                ImGui.TextUnformatted(body.GroupLabel);

                ImGui.TableSetColumnIndex(3);
                ImGui.Text($"{body.BlockCount}");

                ImGui.TableSetColumnIndex(4);
                ImGui.Text($"{body.MinHeight:F1}..{body.MaxHeight:F1}");

                ImGui.TableSetColumnIndex(5);
                bool isSelected = string.Equals(_wlLayerSelectedBodyKey, body.BodyKey, StringComparison.OrdinalIgnoreCase);
                if (ImGui.Selectable($"{body.Name}##wl_invest_body_{i}", isSelected, ImGuiSelectableFlags.SpanAllColumns))
                    SetSelectedWlLiquidBody(body, isolateInList: false, focusInspectWorkspace: false);

                if (isSelected && _wlPendingScrollToSelectedBody)
                {
                    ImGui.SetScrollHereY(0.20f);
                    _wlPendingScrollToSelectedBody = false;
                }

                if (ImGui.IsItemHovered())
                {
                    ImGui.BeginTooltip();
                    ImGui.TextUnformatted(body.SourcePath);
                    ImGui.TextDisabled($"{body.FileType}  {body.GroupLabel}");
                    ImGui.TextDisabled($"Blocks={body.BlockCount}  Verts={body.Vertices.Length}");
                    ImGui.TextDisabled($"Coord X {body.CoordMinX:F1}..{body.CoordMaxX:F1}  Y {body.CoordMinY:F1}..{body.CoordMaxY:F1}");
                    ImGui.EndTooltip();
                }
            }

            ImGui.EndTable();
        }
    }

    private void DrawLitInvestigationPanel(bool defaultOpen)
    {
        if (_worldScene == null)
            return;

        ImGuiTreeNodeFlags flags = defaultOpen ? ImGuiTreeNodeFlags.DefaultOpen : ImGuiTreeNodeFlags.None;
        if (!ImGui.CollapsingHeader("LIT Lighting Investigation", flags))
            return;

        bool showLitLights = _worldScene.ShowLitLights;
        if (ImGui.Checkbox("Show LIT Overlay", ref showLitLights))
            _worldScene.ShowLitLights = showLitLights;

        bool showAreaRegions = _worldScene.ShowAreaRegionOverlay;
        if (ImGui.Checkbox("Show Area Boundaries", ref showAreaRegions))
            _worldScene.ShowAreaRegionOverlay = showAreaRegions;
        if (_worldScene.AreaOverlayResidentChunkCount > 0)
        {
            ImGui.SameLine();
            ImGui.TextDisabled($"{_worldScene.AreaOverlayRegions.Count} groups / {_worldScene.AreaOverlayResidentChunkCount} resident chunks");
        }
        if (_worldScene.AreaOverlayUnresolvedChunkCount > 0)
        {
            ImGui.TextColored(
                new Vector4(1f, 0.72f, 0.35f, 1f),
                $"Unresolved area chunks: {_worldScene.AreaOverlayUnresolvedChunkCount} (no guessed labels)");
        }

        ImGui.SameLine();
        bool useLitFogOverride = _worldScene.UseLitFogOverride;
        if (ImGui.Checkbox("Use LIT Lighting Override", ref useLitFogOverride))
            _worldScene.UseLitFogOverride = useLitFogOverride;

        if (_worldScene.LitAutoFallbackActive)
            ImGui.TextDisabled($"Automatic LIT fallback: {_worldScene.LitAutoFallbackReason}");

        IReadOnlyList<string> sourcePaths = _worldScene.AvailableLitSourcePaths;
        string currentSourcePath = _worldScene.SelectedLitSourcePath
            ?? _worldScene.LitLoader?.SourcePath
            ?? string.Empty;
        if (sourcePaths.Count > 1)
        {
            if (ImGui.BeginCombo("LIT Source", currentSourcePath))
            {
                for (int i = 0; i < sourcePaths.Count; i++)
                {
                    string sourcePath = sourcePaths[i];
                    bool isSelectedSource = string.Equals(
                        currentSourcePath,
                        sourcePath,
                        StringComparison.OrdinalIgnoreCase);
                    if (ImGui.Selectable(sourcePath, isSelectedSource))
                    {
                        _worldScene.ReloadLit(sourcePath);
                        ImGui.EndCombo();
                        return;
                    }

                    if (isSelectedSource)
                        ImGui.SetItemDefaultFocus();
                }

                ImGui.EndCombo();
            }
        }

        if (!_worldScene.LitLoadAttempted)
        {
            ImGui.TextDisabled("All .lit profiles directly inside World\\<map> or World\\Maps\\<map> are discovered and lazy-loaded when present.");
            if (ImGui.Button("Load LIT Lighting"))
                _worldScene.ShowLitLights = true;
            return;
        }

        LitLoader? loader = _worldScene.LitLoader;
        if (loader == null || !loader.HasData)
        {
            ImGui.TextDisabled(_worldScene.LitStatus);
            return;
        }

        ImGui.TextDisabled($"Path: {loader.SourcePath}");
        ImGui.TextDisabled($"Version: 0x{loader.Version:X8}  RawCount: {loader.RawLightCount}  Parsed lights: {loader.Lights.Count}");
        ImGui.TextDisabled("Runtime LIT uses clear group 0. Global diffuse/ambient/sky/fog are applied together; local-zone coordinates and sky-band placement remain diagnostic-only.");

        LitLoader.LitLightingSample? litSample = _worldScene.LastLitSample;
        if (litSample != null)
        {
            ImGui.Separator();
            ImGui.TextDisabled($"Camera sample: {litSample.DominantLightName}  weight={litSample.DominantWeight:F2}  time={litSample.TimeOfDay:F0}/2880");
            DrawLitColorSwatch("sample_direct", "Direct", litSample.DirectColor);
            ImGui.SameLine();
            DrawLitColorSwatch("sample_ambient", "Ambient", litSample.AmbientColor);
            ImGui.SameLine();
            DrawLitColorSwatch("sample_fog", "Fog", litSample.FogColor);
            ImGui.TextDisabled($"Fog start/end: {litSample.FogStart:F1} / {litSample.FogEnd:F1}  scaler={litSample.FogStartScalar:F3}");

            if (ImGui.SmallButton("Copy LIT Sample Summary"))
                CopyTextToClipboard(BuildLitSampleSummary(loader, litSample), "LIT lighting summary");
        }

        int selectedIndex = _worldScene.SelectedLitLightIndex;
        if (selectedIndex < 0 || selectedIndex >= loader.Lights.Count)
            selectedIndex = 0;

        ImGui.Separator();
        if (ImGui.BeginTable("##lit_investigation", 5, ImGuiTableFlags.BordersInnerV | ImGuiTableFlags.RowBg | ImGuiTableFlags.SizingStretchProp | ImGuiTableFlags.ScrollY, new Vector2(0f, 220f)))
        {
            ImGui.TableSetupColumn("Sel", ImGuiTableColumnFlags.WidthFixed, 28f);
            ImGui.TableSetupColumn("Type", ImGuiTableColumnFlags.WidthFixed, 56f);
            ImGui.TableSetupColumn("Radius", ImGuiTableColumnFlags.WidthFixed, 72f);
            ImGui.TableSetupColumn("Position", ImGuiTableColumnFlags.WidthFixed, 210f);
            ImGui.TableSetupColumn("Light", ImGuiTableColumnFlags.WidthStretch);
            ImGui.TableHeadersRow();

            for (int i = 0; i < loader.Lights.Count; i++)
            {
                LitLoader.LitLight light = loader.Lights[i];
                ImGui.TableNextRow();

                ImGui.TableSetColumnIndex(0);
                bool isSelected = selectedIndex == i;
                if (ImGui.Selectable($"##lit_sel_{i}", isSelected, ImGuiSelectableFlags.SpanAllColumns))
                    _worldScene.SelectedLitLightIndex = i;

                ImGui.TableSetColumnIndex(1);
                ImGui.TextUnformatted(light.IsDefaultLight ? "Default" : "Local");

                ImGui.TableSetColumnIndex(2);
                ImGui.Text(light.Radius > 0f ? $"{light.Radius:F1}" : "-");

                ImGui.TableSetColumnIndex(3);
                ImGui.Text(light.HasMeaningfulPosition
                    ? $"({light.Position.X:F1}, {light.Position.Y:F1}, {light.Position.Z:F1})"
                    : "(none)");

                ImGui.TableSetColumnIndex(4);
                if (ImGui.Selectable($"{light.DisplayName}##lit_label_{i}", isSelected, ImGuiSelectableFlags.SpanAllColumns))
                    _worldScene.SelectedLitLightIndex = i;

                if (ImGui.IsItemHovered())
                {
                    ImGui.BeginTooltip();
                    ImGui.TextUnformatted(light.DisplayName);
                    ImGui.TextDisabled($"Chunk=({light.ChunkX}, {light.ChunkY}) chunkRadius={light.ChunkRadius}");
                    ImGui.TextDisabled($"Radius raw={light.RadiusRaw:F1} normalized={light.Radius:F1}");
                    ImGui.TextDisabled($"Dropoff raw={light.DropoffRaw:F1} normalized={light.Dropoff:F1}");
                    ImGui.EndTooltip();
                }
            }

            ImGui.EndTable();
        }

        selectedIndex = Math.Clamp(_worldScene.SelectedLitLightIndex >= 0 ? _worldScene.SelectedLitLightIndex : selectedIndex, 0, loader.Lights.Count - 1);
        LitLoader.LitLight selectedLight = loader.Lights[selectedIndex];
        DrawLitLightDetails(selectedLight, litSample?.TimeOfDay ?? (_terrainManager?.Lighting.GameTime ?? 0f) * 2880f);
    }

    private bool ShouldShowHoveredAssetInfoForInvestigation(HoveredAssetInfo info)
    {
        return _visualInvestigationMode switch
        {
            VisualInvestigationMode.Wmo => info.SceneObjectType == ObjectType.Wmo || string.Equals(info.AssetKind, "WMO", StringComparison.OrdinalIgnoreCase),
            VisualInvestigationMode.M2 => info.SceneObjectType == ObjectType.Mdx || string.Equals(info.AssetKind, "MDX", StringComparison.OrdinalIgnoreCase),
            VisualInvestigationMode.Adt => false,
            VisualInvestigationMode.Pm4 => info.Pm4ObjectKey.HasValue,
            _ => true,
        };
    }

    private bool TryGetTerrainChunkInspectionTarget(bool preferHoveredChunk, out TerrainRenderer.TerrainChunkInfo info, out bool usingHoveredChunk)
    {
        info = default;
        usingHoveredChunk = false;

        TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (renderer == null)
            return false;

        if (preferHoveredChunk && _terrainQuery.TryPickTerrainChunkUnderMouse(renderer, out info))
        {
            usingHoveredChunk = true;
            return true;
        }

        TerrainRenderer.TerrainChunkInfo? cameraChunk = renderer.GetChunkInfoAt(_camera.Position.X, _camera.Position.Y);
        if (cameraChunk.HasValue)
        {
            info = cameraChunk.Value;
            return true;
        }

        return false;
    }

    private static string ResolveTerrainTextureName(IReadOnlyList<string>? tileTextures, int textureIndex)
    {
        if (tileTextures == null || textureIndex < 0 || textureIndex >= tileTextures.Count)
            return "<missing>";

        string texturePath = tileTextures[textureIndex];
        return string.IsNullOrWhiteSpace(texturePath)
            ? "<empty>"
            : texturePath;
    }

    private static string BuildTerrainChunkTextureSummary(
        TerrainRenderer.TerrainChunkInfo chunkInfo,
        TerrainChunkData chunkData,
        IReadOnlyList<string>? tileTextures)
    {
        var builder = new System.Text.StringBuilder();
        builder.AppendLine($"Tile ({chunkInfo.TileY}, {chunkInfo.TileX}) Chunk ({chunkInfo.ChunkX}, {chunkInfo.ChunkY})");
        builder.AppendLine($"AreaId={chunkData.AreaId} Layers={chunkData.Layers.Length} Holes=0x{chunkData.HoleMask:X4} AlphaMaps={chunkData.AlphaMaps.Count}");
        builder.AppendLine($"McnkFlags=0x{(uint)chunkData.McnkFlags:X8} ({DescribeMcnkFlags(chunkData.McnkFlags)})");

        for (int layerIndex = 0; layerIndex < chunkData.Layers.Length; layerIndex++)
        {
            TerrainLayer layer = chunkData.Layers[layerIndex];
            bool hasAlpha = layerIndex > 0 && chunkData.AlphaMaps.ContainsKey(layerIndex);
            builder.Append("L").Append(layerIndex)
                .Append(": tex#").Append(layer.TextureIndex)
                .Append(" ").Append(ResolveTerrainTextureName(tileTextures, layer.TextureIndex))
                .Append(" flags=0x").Append(layer.Flags.ToString("X8"))
                .Append(" effect=").Append(layer.EffectId)
                .Append(" alpha=").Append(hasAlpha ? "yes" : "no")
                .AppendLine();
        }

        return builder.ToString().TrimEnd();
    }

    private void DrawMcnkFlagOverlayControls()
    {
        if (!WoWViewer.UI.SharedUiWidgets.DrawSectionHeader("MCNK Flag Overlay",
            "Highlight resident terrain chunks by raw MCNK header flags. Diagonal weak corners require the overlay and Impassable flag."))
            return;

        bool showOverlay = _showMcnkFlagOverlay;
        if (ImGui.Checkbox("Show MCNK Flag Overlay", ref showOverlay))
            _showMcnkFlagOverlay = showOverlay;

        bool showWeakCorners = _showMcnkWeakCorners;
        if (ImGui.Checkbox("Highlight Diagonal Weak Corners", ref showWeakCorners))
            _showMcnkWeakCorners = showWeakCorners;
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Marks loaded 2x2 chunk blocks where impassable chunks only touch diagonally and leave the shared corner exposed.");

        if (ImGui.BeginTable("##McnkFlagChoices", 2, ImGuiTableFlags.SizingStretchSame))
        {
            foreach (var (flag, label) in new[]
            {
                (McnkOverlayFlags.Impassable, "Impassable"), (McnkOverlayFlags.River, "River"),
                (McnkOverlayFlags.Ocean, "Ocean"), (McnkOverlayFlags.HasMagma, "Magma"),
                (McnkOverlayFlags.HasSlime, "Slime"), (McnkOverlayFlags.HasShadows, "Shadows"),
                (McnkOverlayFlags.HasMccv, "MCCV"), (McnkOverlayFlags.HasBakedShadows, "BakedShadows"),
            })
            {
                ImGui.TableNextColumn();
                DrawMcnkFlagToggle(flag, label);
            }
            ImGui.EndTable();
        }

        ImGui.TextDisabled($"Loaded flagged chunks: {_lastMcnkOverlayChunkCount}  Weak corners: {_lastMcnkWeakCornerCount}");
    }

    private void DrawMcnkFlagToggle(McnkOverlayFlags flag, string label)
    {
        bool enabled = (_mcnkOverlayFlags & flag) != 0;
        if (ImGui.Checkbox(label, ref enabled))
        {
            if (enabled)
                _mcnkOverlayFlags |= flag;
            else
                _mcnkOverlayFlags &= ~flag;
        }
    }

    internal static string DescribeMcnkFlags(int rawFlags)
    {
        McnkOverlayFlags flags = (McnkOverlayFlags)(uint)rawFlags;
        if (flags == McnkOverlayFlags.None)
            return "none";

        var labels = new List<string>();
        AppendMcnkFlagLabel(labels, flags, McnkOverlayFlags.HasShadows, "HasShadows");
        AppendMcnkFlagLabel(labels, flags, McnkOverlayFlags.Impassable, "Impassable");
        AppendMcnkFlagLabel(labels, flags, McnkOverlayFlags.River, "River");
        AppendMcnkFlagLabel(labels, flags, McnkOverlayFlags.Ocean, "Ocean");
        AppendMcnkFlagLabel(labels, flags, McnkOverlayFlags.HasMagma, "Magma");
        AppendMcnkFlagLabel(labels, flags, McnkOverlayFlags.HasSlime, "Slime");
        AppendMcnkFlagLabel(labels, flags, McnkOverlayFlags.HasMccv, "HasMccv");
        AppendMcnkFlagLabel(labels, flags, McnkOverlayFlags.HasBakedShadows, "HasBakedShadows");

        uint unknownMask = (uint)rawFlags & ~(uint)(McnkOverlayFlags.HasShadows
            | McnkOverlayFlags.Impassable
            | McnkOverlayFlags.River
            | McnkOverlayFlags.Ocean
            | McnkOverlayFlags.HasMagma
            | McnkOverlayFlags.HasSlime
            | McnkOverlayFlags.HasMccv
            | McnkOverlayFlags.HasBakedShadows);
        if (unknownMask != 0)
            labels.Add($"Unknown=0x{unknownMask:X8}");

        return string.Join(", ", labels);
    }

    private static void AppendMcnkFlagLabel(List<string> labels, McnkOverlayFlags flags, McnkOverlayFlags flag, string label)
    {
        if ((flags & flag) != 0)
            labels.Add(label);
    }

    private bool TryGetFocusedWlLiquidBody(out WlLiquidBody? body, out bool usingHoveredBody)
    {
        body = null;
        usingHoveredBody = false;

        if (_worldScene?.WlLoader == null || !_worldScene.WlLoader.HasData)
            return false;

        if (_worldScene.HoveredAssetInfo is HoveredAssetInfo hoveredInfo
            && string.Equals(hoveredInfo.AssetKind, "WL liquid", StringComparison.OrdinalIgnoreCase)
            && TryResolveHoveredWlLiquidBody(hoveredInfo, out body))
        {
            usingHoveredBody = true;
            return true;
        }

        return TryFindWlLiquidBodyByKey(_wlLayerSelectedBodyKey, out body);
    }

    private bool TryFindWlLiquidBodyByKey(string bodyKey, out WlLiquidBody? body)
    {
        body = null;
        if (_worldScene?.WlLoader == null || string.IsNullOrWhiteSpace(bodyKey))
            return false;

        for (int i = 0; i < _worldScene.WlLoader.Bodies.Count; i++)
        {
            WlLiquidBody candidate = _worldScene.WlLoader.Bodies[i];
            if (!string.Equals(candidate.BodyKey, bodyKey, StringComparison.OrdinalIgnoreCase))
                continue;

            body = candidate;
            return true;
        }

        return false;
    }

    private bool TryFindWlLiquidBody(string sourcePath, string displayName, out WlLiquidBody? body)
    {
        body = null;
        if (_worldScene?.WlLoader == null)
            return false;

        for (int i = 0; i < _worldScene.WlLoader.Bodies.Count; i++)
        {
            WlLiquidBody candidate = _worldScene.WlLoader.Bodies[i];
            if (!string.Equals(candidate.SourcePath, sourcePath, StringComparison.OrdinalIgnoreCase))
                continue;
            if (!string.Equals(candidate.Name, displayName, StringComparison.OrdinalIgnoreCase))
                continue;

            body = candidate;
            return true;
        }

        return false;
    }

    private bool TryResolveHoveredWlLiquidBody(HoveredAssetInfo hoveredInfo, out WlLiquidBody? body)
    {
        body = null;
        if (!string.Equals(hoveredInfo.AssetKind, "WL liquid", StringComparison.OrdinalIgnoreCase))
            return false;

        if (!string.IsNullOrWhiteSpace(hoveredInfo.WlBodyKey)
            && TryFindWlLiquidBodyByKey(hoveredInfo.WlBodyKey, out body))
        {
            return true;
        }

        return TryFindWlLiquidBody(hoveredInfo.SourcePath, hoveredInfo.DisplayName, out body);
    }

    private int FindWlLiquidBodyIndex(string bodyKey)
    {
        if (_worldScene?.WlLoader == null || string.IsNullOrWhiteSpace(bodyKey))
            return -1;

        for (int i = 0; i < _worldScene.WlLoader.Bodies.Count; i++)
        {
            if (string.Equals(_worldScene.WlLoader.Bodies[i].BodyKey, bodyKey, StringComparison.OrdinalIgnoreCase))
                return i;
        }

        return -1;
    }

    private bool IsWlListIsolationActive
        => _wlLayerListIsolationEnabled && !string.IsNullOrWhiteSpace(_wlLayerSelectedBodyKey);

    private bool ShouldIncludeWlBodyInUiList(WlLiquidBody body)
    {
        if (!IsWlListIsolationActive)
            return true;

        return string.Equals(body.BodyKey, _wlLayerSelectedBodyKey, StringComparison.OrdinalIgnoreCase);
    }

    private void SetSelectedWlLiquidBody(WlLiquidBody body, bool isolateInList, bool focusInspectWorkspace, string? statusMessage = null)
    {
        _wlLayerSelectedBodyKey = body.BodyKey;
        if (isolateInList)
            _wlLayerListIsolationEnabled = true;

        _wlPendingScrollToSelectedBody = true;
        _selectedObjectType = "WL liquid";
        _selectedObjectIndex = FindWlLiquidBodyIndex(body.BodyKey);
        _selectedObjectInfo = BuildWlLiquidSummary(body);

        if (_terrainManager?.LiquidRenderer != null)
            _terrainManager.LiquidRenderer.SelectedWlBodyKey = body.BodyKey;

        if (focusInspectWorkspace)
            SetEditorWorkspaceTask(EditorWorkspaceTask.Inspect);

        if (!string.IsNullOrWhiteSpace(statusMessage))
            _statusMessage = statusMessage;
    }

    private void ClearSelectedWlLiquidBody(bool clearListIsolation)
    {
        _wlLayerSelectedBodyKey = string.Empty;
        _wlPendingScrollToSelectedBody = false;
        if (clearListIsolation)
            _wlLayerListIsolationEnabled = false;

        if (_terrainManager?.LiquidRenderer != null)
            _terrainManager.LiquidRenderer.SelectedWlBodyKey = null;

        if (string.Equals(_selectedObjectType, "WL liquid", StringComparison.OrdinalIgnoreCase))
        {
            _selectedObjectIndex = -1;
            _selectedObjectType = string.Empty;
            _selectedObjectInfo = string.Empty;
        }
    }

    internal static string GetWlLiquidGroupingModeLabel(WlLiquidLoader.WlBodyGroupingMode mode)
    {
        return mode switch
        {
            WlLiquidLoader.WlBodyGroupingMode.FileWelded => "File welded",
            WlLiquidLoader.WlBodyGroupingMode.PlaneWelded => "Plane welded",
            WlLiquidLoader.WlBodyGroupingMode.BlockUnwelded => "Block unwelded",
            _ => "Unknown",
        };
    }

    private static string BuildWlLiquidSummary(WlLiquidBody body)
    {
        var builder = new System.Text.StringBuilder();
        builder.AppendLine(body.Name);
        builder.AppendLine(body.SourcePath);
        builder.Append("FileType=").Append(body.FileType)
            .Append(" Group=").Append(body.GroupLabel)
            .Append(" Blocks=").Append(body.BlockCount)
            .Append(" Verts=").Append(body.Vertices.Length)
            .AppendLine();
        builder.Append("Height=").Append(body.MinHeight.ToString("F3"))
            .Append("..").Append(body.MaxHeight.ToString("F3"))
            .Append(" Avg=").Append(body.AverageHeight.ToString("F3"))
            .AppendLine();
        builder.Append("CoordX=").Append(body.CoordMinX.ToString("F2"))
            .Append("..").Append(body.CoordMaxX.ToString("F2"))
            .Append(" CoordY=").Append(body.CoordMinY.ToString("F2"))
            .Append("..").Append(body.CoordMaxY.ToString("F2"))
            .AppendLine();
        builder.Append("MetadataPatterns=").Append(body.MetadataPatternCount)
            .Append(" NonZeroWordsPerBlock=").Append(body.MetadataNonZeroMin)
            .Append("..").Append(body.MetadataNonZeroMax)
            .AppendLine();
        builder.Append("SourceBlocks=").Append(string.Join(",", body.SourceBlockIndices));
        return builder.ToString().TrimEnd();
    }

    private void DrawLitLightDetails(LitLoader.LitLight light, float timeOfDay)
    {
        ImGui.Separator();
        ImGui.Text(light.DisplayName);
        ImGui.TextDisabled($"Chunk=({light.ChunkX}, {light.ChunkY}) chunkRadius={light.ChunkRadius}  Groups={light.Groups.Count}");
        ImGui.TextDisabled(light.HasMeaningfulPosition
            ? $"Renderer: ({light.Position.X:F2}, {light.Position.Y:F2}, {light.Position.Z:F2})  WoW: ({light.GamePosition.X:F2}, {light.GamePosition.Y:F2}, {light.GamePosition.Z:F2})"
            : "Position: none");
        ImGui.TextDisabled($"Radius={light.Radius:F2} (raw {light.RadiusRaw:F2})  Dropoff={light.Dropoff:F2} (raw {light.DropoffRaw:F2})");

        if (light.Groups.Count == 0)
        {
            ImGui.TextDisabled("No light-data groups were parsed for this entry.");
            return;
        }

        LitLoader.LitGroup group = light.Groups[0];
        ImGui.TextDisabled($"Inspecting group 0 at time {timeOfDay:F0}/2880.");
        for (int i = 0; i < _litFocusTrackOrder.Length; i++)
        {
            int trackIndex = _litFocusTrackOrder[i];
            if (!group.TryEvaluateTrack(trackIndex, timeOfDay, out Vector3 color))
                continue;

            DrawLitColorSwatch($"detail_track_{trackIndex}", GetLitTrackLabel(trackIndex), color);
            if ((i % 3) != 2 && i + 1 < _litFocusTrackOrder.Length)
                ImGui.SameLine();
        }

        ImGui.TextDisabled($"Fog end={group.EvaluateFogEnd(timeOfDay):F1}  Fog start scalar={group.EvaluateFogStartScaler(timeOfDay):F3}");
    }

    private static void DrawLitColorSwatch(string id, string label, Vector3 color)
    {
        ImGui.ColorButton($"##{id}", new Vector4(color, 1f), ImGuiColorEditFlags.NoTooltip | ImGuiColorEditFlags.NoDragDrop, new Vector2(18f, 18f));
        ImGui.SameLine();
        ImGui.TextUnformatted(label);
    }

    private static string GetLitTrackLabel(int trackIndex)
    {
        return trackIndex switch
        {
            LitLoader.TrackDirectColor => "Direct",
            LitLoader.TrackAmbientColor => "Ambient",
            LitLoader.TrackSkyTop => "Sky Top",
            LitLoader.TrackSkyHorizon => "Sky Horizon",
            LitLoader.TrackFogColor => "Fog",
            _ => $"Track {trackIndex}",
        };
    }

    private static string BuildLitSampleSummary(LitLoader loader, LitLoader.LitLightingSample sample)
    {
        var builder = new System.Text.StringBuilder();
        builder.AppendLine(loader.SourcePath ?? "(unknown)");
        builder.AppendLine($"Version=0x{loader.Version:X8} RawCount={loader.RawLightCount} Parsed={loader.Lights.Count}");
        builder.AppendLine($"Dominant={sample.DominantLightName} Weight={sample.DominantWeight:F3} Time={sample.TimeOfDay:F0}/2880");
        builder.AppendLine($"Direct={sample.DirectColor.X:F3},{sample.DirectColor.Y:F3},{sample.DirectColor.Z:F3}");
        builder.AppendLine($"Ambient={sample.AmbientColor.X:F3},{sample.AmbientColor.Y:F3},{sample.AmbientColor.Z:F3}");
        builder.AppendLine($"Fog={sample.FogColor.X:F3},{sample.FogColor.Y:F3},{sample.FogColor.Z:F3}");
        builder.AppendLine($"FogStart={sample.FogStart:F3} FogEnd={sample.FogEnd:F3} FogStartScalar={sample.FogStartScalar:F3}");
        builder.Append($"SkyTop={sample.SkyTopColor.X:F3},{sample.SkyTopColor.Y:F3},{sample.SkyTopColor.Z:F3} ");
        builder.Append($"SkyHorizon={sample.SkyHorizonColor.X:F3},{sample.SkyHorizonColor.Y:F3},{sample.SkyHorizonColor.Z:F3}");
        return builder.ToString();
    }
}
