using System.Numerics;
using ImGuiNET;
using MdxViewer.Terrain;
using WowViewer.Core.Runtime.World;

namespace MdxViewer;

public partial class ViewerApp
{
    private enum VisualInvestigationMode
    {
        Auto,
        Adt,
        Wmo,
        M2,
    }

    private VisualInvestigationMode _visualInvestigationMode = VisualInvestigationMode.Auto;

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
        if (ImGui.SliderFloat("Object Stream Range", ref objectRangeMultiplier, 1.00f, 4.00f, "%.2fx"))
            _worldScene.ObjectStreamingRangeMultiplier = objectRangeMultiplier;

        WorldRenderFrameStats stats = _worldScene.LastRenderFrameStats;
        double wmoObjectCostMs = stats.WmoVisibility.DurationMs + stats.WmoSubmission.DurationMs;
        double mdxObjectCostMs = stats.MdxVisibility.DurationMs + stats.MdxOpaqueSubmission.DurationMs
            + stats.MdxTransparentSort.DurationMs + stats.MdxTransparentSubmission.DurationMs;
        string hotspot = wmoObjectCostMs >= mdxObjectCostMs
            ? $"WMO scene-pass extraction is still the larger measured object-side cost ({wmoObjectCostMs:0.00} ms)."
            : $"MDX visibility/submission is currently larger ({mdxObjectCostMs:0.00} ms).";
        ImGui.TextDisabled(hotspot);
        ImGui.TextDisabled("Visibility admission and queued object loads use this multiplier. Default is 2.00x.");
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

    private void DrawTerrainChunkInvestigationPanel(bool defaultOpen)
    {
        if (_terrainManager == null && _vlmTerrainManager == null)
            return;

        ImGuiTreeNodeFlags flags = defaultOpen ? ImGuiTreeNodeFlags.DefaultOpen : ImGuiTreeNodeFlags.None;
        if (!ImGui.CollapsingHeader("ADT Chunk Investigation", flags))
            return;

        if (!TryGetTerrainChunkInspectionTarget(preferHoveredChunk: true, out TerrainRenderer.TerrainChunkInfo chunkInfo, out bool usingHoveredChunk))
        {
            ImGui.TextDisabled("Hover a loaded terrain chunk, or move the camera onto one, to inspect its visual data.");
            return;
        }

        if (!TryResolveTerrainChunkInspectionData(chunkInfo, out TerrainChunkData? chunkData, out IReadOnlyList<string>? tileTextures) || chunkData == null)
        {
            ImGui.TextDisabled("Chunk metadata is not available in the current tile cache.");
            return;
        }

        ImGui.TextDisabled(usingHoveredChunk ? "Target: hovered chunk" : "Target: camera chunk");
        ImGui.Text($"Tile ({chunkInfo.TileX}, {chunkInfo.TileY})  Chunk ({chunkInfo.ChunkX}, {chunkInfo.ChunkY})");
        ImGui.TextDisabled($"AreaId: {chunkData.AreaId}  Layers: {chunkData.Layers.Length}  Holes: 0x{chunkData.HoleMask:X4}");
        ImGui.TextDisabled($"Alpha maps: {chunkData.AlphaMaps.Count}  Shadow: {(chunkData.ShadowMap != null ? "yes" : "no")}  MCCV: {(chunkData.MccvColors != null ? "yes" : "no")}");
        ImGui.TextDisabled($"World origin: ({chunkData.WorldPosition.X:F1}, {chunkData.WorldPosition.Y:F1}, {chunkData.WorldPosition.Z:F1})");

        string summary = BuildTerrainChunkTextureSummary(chunkInfo, chunkData, tileTextures);
        if (ImGui.SmallButton("Copy Chunk Texture Summary"))
            CopyTextToClipboard(summary, "chunk texture summary");

        ImGui.Separator();
        for (int layerIndex = 0; layerIndex < chunkData.Layers.Length; layerIndex++)
        {
            TerrainLayer layer = chunkData.Layers[layerIndex];
            string textureName = ResolveTerrainTextureName(tileTextures, layer.TextureIndex);
            bool hasAlpha = layerIndex > 0 && chunkData.AlphaMaps.ContainsKey(layerIndex);
            string baseLayerLabel = layerIndex == 0 ? "base" : "blend";
            ImGui.BulletText($"L{layerIndex} [{baseLayerLabel}] tex#{layer.TextureIndex}: {textureName}");
            ImGui.TextDisabled($"flags=0x{layer.Flags:X8} effect={layer.EffectId} alpha={(hasAlpha ? "yes" : "no")}");
        }
    }

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
                    _wlLayerSelectedBodyKey = body.BodyKey;

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

    private bool TryDrawTerrainChunkHoverOverlay()
    {
        if (_worldScene != null && !_worldScene.ShowHoveredAssetTooltips)
            return false;

        TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (renderer == null)
            return false;

        if (ImGui.GetIO().WantCaptureMouse)
            return false;

        if (!TryGetSceneViewportRect(out float vpX, out float vpY, out float vpW, out float vpH))
            return false;

        if (_lastMouseX < vpX || _lastMouseX > vpX + vpW || _lastMouseY < vpY || _lastMouseY > vpY + vpH)
            return false;

        if (!TryPickTerrainChunkUnderMouse(renderer, out TerrainRenderer.TerrainChunkInfo chunkInfo))
            return false;

        if (!TryResolveTerrainChunkInspectionData(chunkInfo, out TerrainChunkData? chunkData, out IReadOnlyList<string>? tileTextures) || chunkData == null)
            return false;

        Vector2 displaySize = ImGui.GetIO().DisplaySize;
        Vector2 overlayPos = new(
            MathF.Min(_lastMouseX + 18f, MathF.Max(8f, displaySize.X - 390f)),
            MathF.Min(_lastMouseY + 18f, MathF.Max(8f, displaySize.Y - 290f)));

        ImGui.SetNextWindowPos(overlayPos, ImGuiCond.Always);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(16f, 13f));
        ImGui.PushStyleVar(ImGuiStyleVar.WindowBorderSize, 2f);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowRounding, 4f);
        ImGui.PushStyleColor(ImGuiCol.WindowBg, new Vector4(0.04f, 0.05f, 0.09f, 0.985f));
        ImGui.PushStyleColor(ImGuiCol.Border, new Vector4(0.52f, 0.88f, 0.54f, 0.98f));
        ImGui.PushStyleColor(ImGuiCol.Separator, new Vector4(0.42f, 0.74f, 0.44f, 0.82f));

        ImGuiWindowFlags flags = ImGuiWindowFlags.NoDecoration
            | ImGuiWindowFlags.AlwaysAutoResize
            | ImGuiWindowFlags.NoDocking
            | ImGuiWindowFlags.NoSavedSettings
            | ImGuiWindowFlags.NoFocusOnAppearing
            | ImGuiWindowFlags.NoNav
            | ImGuiWindowFlags.NoMove
            | ImGuiWindowFlags.NoInputs;

        if (!ImGui.Begin("##TerrainChunkHoverOverlay", flags))
        {
            ImGui.End();
            ImGui.PopStyleColor(3);
            ImGui.PopStyleVar(3);
            return false;
        }

        ImGui.SetWindowFontScale(1.22f);
        ImGui.TextColored(new Vector4(0.78f, 0.96f, 0.80f, 1.0f), $"ADT {chunkInfo.TileX},{chunkInfo.TileY}:{chunkInfo.ChunkX},{chunkInfo.ChunkY}");
        ImGui.SetWindowFontScale(1.0f);
        ImGui.TextColored(new Vector4(0.60f, 0.88f, 0.62f, 1.0f), "ADT chunk");
        ImGui.TextColored(new Vector4(0.86f, 0.88f, 0.94f, 1.0f), $"Layers: {chunkData.Layers.Length}  AreaId: {chunkData.AreaId}");
        ImGui.TextColored(new Vector4(0.72f, 0.78f, 0.90f, 1.0f), $"World: ({chunkData.WorldPosition.X:F1}, {chunkData.WorldPosition.Y:F1}, {chunkData.WorldPosition.Z:F1})");
        ImGui.Separator();

        for (int layerIndex = 0; layerIndex < chunkData.Layers.Length; layerIndex++)
        {
            TerrainLayer layer = chunkData.Layers[layerIndex];
            string textureName = ResolveTerrainTextureName(tileTextures, layer.TextureIndex);
            bool hasAlpha = layerIndex > 0 && chunkData.AlphaMaps.ContainsKey(layerIndex);
            ImGui.TextColored(new Vector4(0.92f, 0.96f, 1.0f, 1.0f), $"L{layerIndex}: {textureName}");
            ImGui.TextDisabled($"tex#{layer.TextureIndex} flags=0x{layer.Flags:X8} alpha={(hasAlpha ? "yes" : "no")}");
        }

        ImGui.Separator();
        ImGui.TextColored(new Vector4(0.86f, 0.92f, 0.76f, 1.0f), "Use Inspect or Objects workspace for full chunk details.");
        ImGui.End();
        ImGui.PopStyleColor(3);
        ImGui.PopStyleVar(3);
        return true;
    }

    private bool ShouldShowHoveredAssetInfoForInvestigation(HoveredAssetInfo info)
    {
        return _visualInvestigationMode switch
        {
            VisualInvestigationMode.Wmo => info.SceneObjectType == ObjectType.Wmo || string.Equals(info.AssetKind, "WMO", StringComparison.OrdinalIgnoreCase),
            VisualInvestigationMode.M2 => info.SceneObjectType == ObjectType.Mdx || string.Equals(info.AssetKind, "MDX", StringComparison.OrdinalIgnoreCase),
            VisualInvestigationMode.Adt => false,
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

        if (preferHoveredChunk && TryPickTerrainChunkUnderMouse(renderer, out info))
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

    private bool TryResolveTerrainChunkInspectionData(
        TerrainRenderer.TerrainChunkInfo chunkInfo,
        out TerrainChunkData? chunkData,
        out IReadOnlyList<string>? tileTextures)
    {
        chunkData = null;
        tileTextures = null;

        if (_terrainManager != null)
        {
            TileLoadResult result = _terrainManager.GetOrLoadTileLoadResult(chunkInfo.TileX, chunkInfo.TileY);
            chunkData = FindChunkData(result, chunkInfo.ChunkX, chunkInfo.ChunkY);
            _terrainManager.Adapter.TileTextures.TryGetValue((chunkInfo.TileX, chunkInfo.TileY), out List<string>? textures);
            tileTextures = textures;
            return chunkData != null;
        }

        if (_vlmTerrainManager != null && _vlmTerrainManager.TryGetTileLoadResult(chunkInfo.TileX, chunkInfo.TileY, out TileLoadResult resultVlm))
        {
            chunkData = FindChunkData(resultVlm, chunkInfo.ChunkX, chunkInfo.ChunkY);
            _vlmTerrainManager.Loader.TileTextures.TryGetValue((chunkInfo.TileX, chunkInfo.TileY), out List<string>? textures);
            tileTextures = textures;
            return chunkData != null;
        }

        return false;
    }

    private static TerrainChunkData? FindChunkData(TileLoadResult result, int chunkX, int chunkY)
    {
        for (int index = 0; index < result.Chunks.Count; index++)
        {
            TerrainChunkData chunk = result.Chunks[index];
            if (chunk.ChunkX == chunkX && chunk.ChunkY == chunkY)
                return chunk;
        }

        return null;
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
        builder.AppendLine($"Tile ({chunkInfo.TileX}, {chunkInfo.TileY}) Chunk ({chunkInfo.ChunkX}, {chunkInfo.ChunkY})");
        builder.AppendLine($"AreaId={chunkData.AreaId} Layers={chunkData.Layers.Length} Holes=0x{chunkData.HoleMask:X4} AlphaMaps={chunkData.AlphaMaps.Count}");

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

    private bool TryGetFocusedWlLiquidBody(out WlLiquidBody? body, out bool usingHoveredBody)
    {
        body = null;
        usingHoveredBody = false;

        if (_worldScene?.WlLoader == null || !_worldScene.WlLoader.HasData)
            return false;

        if (_worldScene.HoveredAssetInfo is HoveredAssetInfo hoveredInfo
            && string.Equals(hoveredInfo.AssetKind, "WL liquid", StringComparison.OrdinalIgnoreCase)
            && TryFindWlLiquidBody(hoveredInfo.SourcePath, hoveredInfo.DisplayName, out body))
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

    private static string GetWlLiquidGroupingModeLabel(WlLiquidLoader.WlBodyGroupingMode mode)
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
}