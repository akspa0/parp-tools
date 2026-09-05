using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using WowViewer.Core.Runtime.World.Inspection;
using WowViewer.Core.Wmo;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WoWViewer.Workbench;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;

namespace WoWViewer;

public partial class ViewerApp
{
    /// <summary>
    /// Builds the UI-agnostic <see cref="InspectorContent"/> payload describing the current selection
    /// or active camera/terrain context (Spec 223 Phase 1: US2a, US3).
    /// </summary>
    private InspectorContent BuildInspectorContent()
    {
        var builder = new InspectorContentBuilder();

        // 1. PM4 Selection
        if (_worldScene?.HasSelectedPm4Object == true && _worldScene.SelectedPm4ObjectKey is { } pm4Key)
        {
            builder.ObjectType = "PM4";
            if (_worldScene.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo debug))
            {
                float z = BitConverter.UInt32BitsToSingle(debug.Ck24 << 8);
                builder.Headline = debug.Ck24 == 0
                    ? $"PM4 Object ({pm4Key.tileX}, {pm4Key.tileY}) part {pm4Key.objectPart}"
                    : $"PM4 Object ({pm4Key.tileX}, {pm4Key.tileY}) part {pm4Key.objectPart} (Z {z:F3} yd)";

                var idSection = builder.AddSection("PM4 Object Identity");
                idSection.Row("Tile", $"({pm4Key.tileX}, {pm4Key.tileY})", isImportant: true);
                idSection.Row("Viewer Part", $"{pm4Key.objectPart}");
                idSection.Row("Placement Z", debug.Ck24 == 0 ? "none" : $"{z:F3} yd", isImportant: true);
                idSection.Row("Surface Class", $"0x{debug.DominantGroupKey:X2}", isImportant: true);
                idSection.Row("MSHD Region", $"{debug.MshdRegionId}");
                idSection.Row("Dominant MSCN Ref", $"{debug.DominantMscnRefIndex}");
                idSection.Row("MSLK Adjacency Records", $"{debug.DominantAttributeMask}");
                idSection.Row("MSLK Group", $"0x{debug.LinkGroupObjectId:X8}");
                idSection.Row("MPRL Linked Position Refs", $"{debug.LinkedPositionRefCount}");
                idSection.Row("Raw MSUR._0x1C", $"0x{debug.Ck24:X6}");
            }
            else
            {
                builder.Headline = $"PM4 Object ({pm4Key.tileX}, {pm4Key.tileY}) part {pm4Key.objectPart}";
                builder.AddSection("PM4 Object Identity")
                    .Row("Tile", $"({pm4Key.tileX}, {pm4Key.tileY})", isImportant: true)
                    .Row("Viewer Part", $"{pm4Key.objectPart}")
                    .Row("Raw MSUR._0x1C", $"0x{pm4Key.ck24:X6}");
            }

            builder.AddSection("PM4 Actions")
                .Action("clear_pm4_selection", "Clear PM4 Selection")
                .Action("open_pm4_workbench", "Open in PM4 Tools")
                .Action("export_pm4_json", "Export PM4 JSON")
                .Action("export_pm4_obj", "Export PM4 OBJ")
                .Action("export_pm4_llm", "Export PM4 LLM Bundle");

            AppendTerrainChunkInspection(builder);
            return builder.Build();
        }

        // 2. WL* Liquid Selection
        if (string.Equals(_selectedObjectType, "WL liquid", StringComparison.OrdinalIgnoreCase)
            && TryFindWlLiquidBodyByKey(_wlLayerSelectedBodyKey, out WlLiquidBody? wlBody)
            && wlBody != null)
        {
            builder.ObjectType = "WL";
            builder.Headline = $"WL Liquid: {wlBody.Name}";

            var idSection = builder.AddSection("WL Liquid Identity");
            idSection.Row("Body Name", wlBody.Name, isImportant: true);
            idSection.Row("Source Path", wlBody.SourcePath);
            idSection.Row("File Type", wlBody.FileType.ToString(), isImportant: true);
            idSection.Row("Group", wlBody.GroupLabel);
            idSection.Row("Blocks", $"{wlBody.BlockCount}");
            idSection.Row("Vertices", $"{wlBody.Vertices.Length}");
            idSection.Row("Height Range", $"{wlBody.MinHeight:F2} .. {wlBody.MaxHeight:F2} yd", isImportant: true);
            idSection.Row("Coordinate Range X", $"{wlBody.CoordMinX:F1} .. {wlBody.CoordMaxX:F1}");
            idSection.Row("Coordinate Range Y", $"{wlBody.CoordMinY:F1} .. {wlBody.CoordMaxY:F1}");

            builder.AddSection("WL Actions")
                .Action("copy_asset_path", "Copy Source Path")
                .Action("clear_wl_selection", "Clear Selection")
                .Action("toggle_wl_liquids", "Toggle WL Liquids Visibility");

            AppendTerrainChunkInspection(builder);
            return builder.Build();
        }

        // 3. Placed World Object Selection
        if (_worldScene?.SelectedInstance.HasValue == true)
        {
            ObjectInstance inst = _worldScene.SelectedInstance.Value;
            ObjectType objType = _worldScene.SelectedObjectType;

            switch (objType)
            {
                case ObjectType.Wmo:
                    BuildWmoInspector(builder, inst);
                    break;
                case ObjectType.WmoDoodad:
                    BuildWmoDoodadInspector(builder, inst);
                    break;
                case ObjectType.Mdx:
                default:
                    BuildMdxInspector(builder, inst);
                    break;
            }

            AppendTerrainChunkInspection(builder);
            return builder.Build();
        }

        // 4. Standalone Model (no world object selected, but a standalone model is open)
        if (_renderer != null)
        {
            BuildStandaloneModelInspector(builder);
            return builder.Build();
        }

        // 5. ADT / MCNK Chunk (Camera or Hovered)
        if (AppendTerrainChunkInspection(builder, primary: true))
        {
            return builder.Build();
        }

        return builder.Build();
    }

    private void BuildWmoInspector(InspectorContentBuilder builder, ObjectInstance inst)
    {
        builder.ObjectType = "WMO";
        builder.Headline = $"WMO '{Path.GetFileName(inst.ModelPath)}' (Index {inst.PlacementEntryIndex})";

        var placement = builder.AddSection("Placement");
        placement.Row("Asset", inst.ModelPath, isImportant: true);
        float wowX = WoWConstants.MapOrigin - inst.PlacementPosition.Y;
        float wowY = WoWConstants.MapOrigin - inst.PlacementPosition.X;
        float wowZ = inst.PlacementPosition.Z;
        placement.Row("World Position", $"({inst.PlacementPosition.X:F2}, {inst.PlacementPosition.Y:F2}, {inst.PlacementPosition.Z:F2})");
        placement.Row("WoW Coordinates", $"({wowX:F2}, {wowY:F2}, {wowZ:F2})");
        placement.Row("Rotation", $"({inst.PlacementRotation.X:F1}°, {inst.PlacementRotation.Y:F1}°, {inst.PlacementRotation.Z:F1}°)");
        placement.Row("Scale", $"{inst.PlacementScale:F3}");
        placement.Row("UniqueId", $"{inst.UniqueId}");
        placement.Row("Bounds", $"Min ({inst.BoundsMin.X:F1}, {inst.BoundsMin.Y:F1}, {inst.BoundsMin.Z:F1}) Max ({inst.BoundsMax.X:F1}, {inst.BoundsMax.Y:F1}, {inst.BoundsMax.Z:F1})");

        string normKey = WorldAssetManager.NormalizeKey(inst.ModelPath);
        WmoRenderer? wmo = _worldScene?.Assets.GetWmo(normKey);
        if (wmo != null)
        {
            var geo = builder.AddSection("WMO Geometry & Doodads");
            geo.Row("Group Count", $"{wmo.GroupRenderCount}", isImportant: true);
            geo.Row("Doodad Definitions", $"{wmo.DoodadDefCount}");
            geo.Row("Doodad Instances", $"{wmo.DoodadInstanceCount}");
            geo.Row("Active Doodad Set", $"{wmo.ActiveDoodadSet} ({wmo.GetDoodadSetName(wmo.ActiveDoodadSet)})", isImportant: true);
            geo.Row("Total Doodad Sets", $"{wmo.DoodadSetCount}");

            if (wmo.DoodadSetCount > 0)
            {
                var setsSection = builder.AddSection("Doodad Sets (Live Switch)");
                for (int s = 0; s < wmo.DoodadSetCount; s++)
                {
                    string setName = wmo.GetDoodadSetName(s);
                    wmo.TryGetDoodadSetRange(s, out _, out int startIdx, out int count);
                    bool isActive = s == wmo.ActiveDoodadSet;
                    string status = isActive ? "[ACTIVE]" : "";
                    setsSection.Row($"Set {s}: {setName}", $"Start: {startIdx}, Count: {count} {status}", isActive);
                    if (!isActive)
                    {
                        setsSection.Action("switch_wmo_doodad_set", $"Activate Set {s}: {setName}", s);
                    }
                }
            }
        }

        builder.AddSection("WMO Actions")
            .Action("frame_selection", "Frame in Camera")
            .Action("copy_asset_path", "Copy Asset Path");
    }

    private void BuildWmoDoodadInspector(InspectorContentBuilder builder, ObjectInstance inst)
    {
        builder.ObjectType = "WMO Doodad";
        builder.Headline = $"WMO Doodad '{Path.GetFileName(inst.ModelPath)}' (Index {inst.PlacementEntryIndex})";

        var placement = builder.AddSection("Placement");
        placement.Row("Asset", inst.ModelPath, isImportant: true);
        float wowX = WoWConstants.MapOrigin - inst.PlacementPosition.Y;
        float wowY = WoWConstants.MapOrigin - inst.PlacementPosition.X;
        float wowZ = inst.PlacementPosition.Z;
        placement.Row("World Position", $"({inst.PlacementPosition.X:F2}, {inst.PlacementPosition.Y:F2}, {inst.PlacementPosition.Z:F2})");
        placement.Row("WoW Coordinates", $"({wowX:F2}, {wowY:F2}, {wowZ:F2})");
        placement.Row("Rotation", $"({inst.PlacementRotation.X:F1}°, {inst.PlacementRotation.Y:F1}°, {inst.PlacementRotation.Z:F1}°)");
        placement.Row("Scale", $"{inst.PlacementScale:F3}");
        if (inst.SelectionBoundsResolved)
        {
            placement.Row("Local Bounds", $"Min ({inst.LocalBoundsMin.X:F2}, {inst.LocalBoundsMin.Y:F2}, {inst.LocalBoundsMin.Z:F2}) Max ({inst.LocalBoundsMax.X:F2}, {inst.LocalBoundsMax.Y:F2}, {inst.LocalBoundsMax.Z:F2})");
        }

        var parentSection = builder.AddSection("Parent WMO & Def");
        parentSection.Row("MODD Table Index", $"{inst.PlacementEntryIndex}", isImportant: true);

        if (_worldScene?.TryGetSelectedWmoDoodad(out WmoDoodadInfo doodadInfo, out _, out ObjectInstance parentWmo) == true)
        {
            parentSection.Row("Parent WMO Model", parentWmo.ModelPath, isImportant: true);
            parentSection.Row("Parent WMO Index", $"{_worldScene.SelectedWmoParentIndex}");

            string normKey = WorldAssetManager.NormalizeKey(parentWmo.ModelPath);
            WmoRenderer? wmoRenderer = _worldScene.Assets.GetWmo(normKey);
            if (wmoRenderer != null)
            {
                string defName = wmoRenderer.GetDoodadDefName(doodadInfo.DoodadDefIndex);
                if (!string.IsNullOrEmpty(defName))
                    parentSection.Row("MODN Def Name", defName);

                if (wmoRenderer.TryGetDoodadDef(doodadInfo.DoodadDefIndex, out var doodadDef))
                {
                    uint color = doodadDef.Color;
                    byte a = (byte)((color >> 24) & 0xFF);
                    byte r = (byte)((color >> 16) & 0xFF);
                    byte g = (byte)((color >> 8) & 0xFF);
                    byte b = (byte)(color & 0xFF);
                    parentSection.Row("Color (BGRA)", $"#{r:X2}{g:X2}{b:X2}{a:X2}");
                }

                var groups = wmoRenderer.GetRenderGroupsForDoodadDef(doodadInfo.DoodadDefIndex);
                if (groups.Count > 0)
                {
                    parentSection.Row("Referenced Groups", string.Join(", ", groups.Select(g => $"[{g}] {wmoRenderer.GetRenderGroupName(g)}")));
                }
            }
        }

        builder.AddSection("Doodad Actions")
            .Action("frame_selection", "Frame Doodad")
            .Action("copy_asset_path", "Copy Asset Path");
    }

    private void BuildMdxInspector(InspectorContentBuilder builder, ObjectInstance inst)
    {
        string kind = inst.ModelPath.EndsWith(".m2", StringComparison.OrdinalIgnoreCase) ? "M2" : "MDX";
        builder.ObjectType = kind;
        builder.Headline = $"{kind} '{Path.GetFileName(inst.ModelPath)}' (Index {inst.PlacementEntryIndex})";

        var placement = builder.AddSection("Placement");
        placement.Row("Asset", inst.ModelPath, isImportant: true);
        float wowX = WoWConstants.MapOrigin - inst.PlacementPosition.Y;
        float wowY = WoWConstants.MapOrigin - inst.PlacementPosition.X;
        float wowZ = inst.PlacementPosition.Z;
        placement.Row("World Position", $"({inst.PlacementPosition.X:F2}, {inst.PlacementPosition.Y:F2}, {inst.PlacementPosition.Z:F2})");
        placement.Row("WoW Coordinates", $"({wowX:F2}, {wowY:F2}, {wowZ:F2})");
        placement.Row("Rotation", $"({inst.PlacementRotation.X:F1}°, {inst.PlacementRotation.Y:F1}°, {inst.PlacementRotation.Z:F1}°)");
        placement.Row("Scale", $"{inst.PlacementScale:F3}");
        placement.Row("UniqueId", $"{inst.UniqueId}");
        placement.Row("Bounds", $"Min ({inst.BoundsMin.X:F1}, {inst.BoundsMin.Y:F1}, {inst.BoundsMin.Z:F1}) Max ({inst.BoundsMax.X:F1}, {inst.BoundsMax.Y:F1}, {inst.BoundsMax.Z:F1})");

        string normKey = WorldAssetManager.NormalizeKey(inst.ModelPath);
        IModelRenderer? modelRenderer = _worldScene?.Assets.GetMdx(normKey);
        if (modelRenderer != null)
        {
            var animSection = builder.AddSection("Sequences & Mesh");
            int seqCount = modelRenderer.Animator?.Sequences.Count ?? 0;
            int currSeq = modelRenderer.Animator?.CurrentSequence ?? 0;
            bool isPlaying = modelRenderer.Animator?.IsPlaying ?? false;
            animSection.Row("Sequence Count", $"{seqCount}", isImportant: true);
            animSection.Row("Current Sequence", $"{currSeq} ({(isPlaying ? "Playing" : "Stopped")})");
            animSection.Row("Sub-Objects / Geosets", $"{modelRenderer.SubObjectCount}");
        }

        builder.AddSection("Model Actions")
            .Action("frame_selection", "Frame in Camera")
            .Action("copy_asset_path", "Copy Asset Path");
    }

    private void BuildStandaloneModelInspector(InspectorContentBuilder builder)
    {
        string modelName = Path.GetFileName(_loadedFilePath ?? "Model");
        if (_renderer is WmoRenderer standaloneWmo)
        {
            builder.ObjectType = "WMO";
            builder.Headline = $"Standalone WMO: {modelName}";

            var info = builder.AddSection("WMO Information");
            info.Row("Asset Path", _loadedFilePath ?? "Unknown", isImportant: true);
            info.Row("Groups", $"{standaloneWmo.GroupRenderCount}");
            info.Row("Doodad Definitions", $"{standaloneWmo.DoodadDefCount}");
            info.Row("Active Doodad Set", $"{standaloneWmo.ActiveDoodadSet} ({standaloneWmo.GetDoodadSetName(standaloneWmo.ActiveDoodadSet)})", isImportant: true);
            info.Row("Total Sets", $"{standaloneWmo.DoodadSetCount}");

            if (standaloneWmo.DoodadSetCount > 0)
            {
                var setsSection = builder.AddSection("Doodad Sets (Live Switch)");
                for (int s = 0; s < standaloneWmo.DoodadSetCount; s++)
                {
                    string setName = standaloneWmo.GetDoodadSetName(s);
                    standaloneWmo.TryGetDoodadSetRange(s, out _, out int startIdx, out int count);
                    bool isActive = s == standaloneWmo.ActiveDoodadSet;
                    string status = isActive ? "[ACTIVE]" : "";
                    setsSection.Row($"Set {s}: {setName}", $"Start: {startIdx}, Count: {count} {status}", isActive);
                    if (!isActive)
                    {
                        setsSection.Action("switch_standalone_wmo_doodad_set", $"Activate Set {s}: {setName}", s);
                    }
                }
            }
        }
        else if (_renderer is IModelRenderer standaloneModel)
        {
            string kind = (_loadedFilePath ?? "").EndsWith(".m2", StringComparison.OrdinalIgnoreCase) ? "M2" : "MDX";
            builder.ObjectType = kind;
            builder.Headline = $"Standalone {kind}: {modelName}";

            var info = builder.AddSection("Model Information");
            info.Row("Asset Path", _loadedFilePath ?? "Unknown", isImportant: true);
            int seqCount = standaloneModel.Animator?.Sequences.Count ?? 0;
            int currSeq = standaloneModel.Animator?.CurrentSequence ?? 0;
            bool isPlaying = standaloneModel.Animator?.IsPlaying ?? false;
            info.Row("Sequences", $"{seqCount}", isImportant: true);
            info.Row("Current Sequence", $"{currSeq} ({(isPlaying ? "Playing" : "Stopped")})");
            info.Row("Sub-Objects / Geosets", $"{standaloneModel.SubObjectCount}");
        }

        builder.AddSection("Actions")
            .Action("frame_model", "Frame Model")
            .Action("copy_asset_path", "Copy Model Path");
    }

    private bool AppendTerrainChunkInspection(InspectorContentBuilder builder, bool primary = false)
    {
        if (_terrainManager == null && _vlmTerrainManager == null)
            return false;

        if (!TryGetTerrainChunkInspectionTarget(preferHoveredChunk: true, out TerrainRenderer.TerrainChunkInfo chunkInfo, out bool usingHoveredChunk))
            return false;

        if (!TryResolveTerrainChunkInspectionData(chunkInfo, out TerrainChunkData? chunkData, out IReadOnlyList<string>? tileTextures) || chunkData == null)
            return false;

        if (primary)
        {
            builder.ObjectType = "ADT";
            builder.Headline = $"ADT Chunk ({chunkInfo.TileY}, {chunkInfo.TileX}) MCNK ({chunkInfo.ChunkX}, {chunkInfo.ChunkY})";
        }

        string sectionTitle = primary ? "Chunk Metadata" : "ADT Terrain Context";
        var meta = builder.AddSection(sectionTitle);
        meta.Row("Target Source", usingHoveredChunk ? "Hovered Cursor Chunk" : "Camera Chunk", isImportant: true);
        meta.Row("Tile (X, Y)", $"({chunkInfo.TileX}, {chunkInfo.TileY})");
        meta.Row("MCNK (X, Y)", $"({chunkInfo.ChunkX}, {chunkInfo.ChunkY})");
        meta.Row("Area ID", $"{chunkData.AreaId}", isImportant: true);
        meta.Row("MCNK Flags", $"0x{(uint)chunkData.McnkFlags:X8} ({DescribeMcnkFlags(chunkData.McnkFlags)})");
        meta.Row("Holes Mask", $"0x{chunkData.HoleMask:X4}");
        meta.Row("World Position", $"({chunkData.WorldPosition.X:F2}, {chunkData.WorldPosition.Y:F2}, {chunkData.WorldPosition.Z:F2})");
        meta.Row("Layers", $"{chunkData.Layers.Length}");
        meta.Row("Alpha Maps", $"{chunkData.AlphaMaps.Count}");
        meta.Row("Shadow Map", chunkData.ShadowMap != null ? "Present (64x64)" : "None");
        meta.Row("MCCV Vertex Colors", chunkData.MccvColors != null ? "Present (145 verts)" : "None");

        if (chunkData.Layers.Length > 0)
        {
            var layersSection = builder.AddSection(primary ? "Texture Layers (Stratigraphy)" : "Terrain Texture Layers");
            for (int i = 0; i < chunkData.Layers.Length; i++)
            {
                var layer = chunkData.Layers[i];
                string texName = ResolveTerrainTextureName(tileTextures, layer.TextureIndex);
                bool hasAlpha = i > 0 && chunkData.AlphaMaps.ContainsKey(i);
                layersSection.Row($"Layer {i}", $"{texName} (tex#{layer.TextureIndex}, flags=0x{layer.Flags:X8}, alpha={(hasAlpha ? "yes" : "no")})");
            }
        }

        builder.AddSection(primary ? "Chunk Actions" : "Terrain Actions")
            .Action("copy_chunk_texture_summary", "Copy Texture Summary to Clipboard");

        return true;
    }

    private void HandleInspectorAction(InspectorAction action)
    {
        switch (action.Id)
        {
            case "switch_wmo_doodad_set":
                if (_worldScene?.SelectedInstance.HasValue == true && _worldScene.SelectedObjectType == ObjectType.Wmo)
                {
                    ObjectInstance inst = _worldScene.SelectedInstance.Value;
                    string normKey = WorldAssetManager.NormalizeKey(inst.ModelPath);
                    WmoRenderer? wmo = _worldScene.Assets.GetWmo(normKey);
                    if (wmo != null)
                    {
                        int targetSet = WmoDoodadSetResolver.ResolveActiveSetIndex(action.Argument, wmo.DoodadSetCount);
                        wmo.SetActiveDoodadSet(targetSet);
                        _statusMessage = $"Switched WMO '{Path.GetFileName(inst.ModelPath)}' doodad set to {targetSet} ({wmo.GetDoodadSetName(targetSet)}).";
                    }
                }
                break;

            case "switch_standalone_wmo_doodad_set":
                if (_renderer is WmoRenderer standaloneWmo)
                {
                    int targetSet = WmoDoodadSetResolver.ResolveActiveSetIndex(action.Argument, standaloneWmo.DoodadSetCount);
                    standaloneWmo.SetActiveDoodadSet(targetSet);
                    _statusMessage = $"Switched standalone WMO doodad set to {targetSet} ({standaloneWmo.GetDoodadSetName(targetSet)}).";
                }
                break;

            case "frame_selection":
                if (_worldScene?.SelectedInstance.HasValue == true)
                {
                    ObjectInstance inst = _worldScene.SelectedInstance.Value;
                    if (inst.BoundsResolved)
                        FrameBounds(inst.BoundsMin, inst.BoundsMax, mdxMirrorX: false);
                    else
                        FramePoint(inst.PlacementPosition, radius: 5f);
                }
                break;

            case "frame_model":
                FrameCurrentModel();
                break;

            case "copy_asset_path":
                if (_worldScene?.SelectedInstance.HasValue == true)
                    CopyTextToClipboard(_worldScene.SelectedInstance.Value.ModelPath, "Asset Path");
                else if (string.Equals(_selectedObjectType, "WL liquid", StringComparison.OrdinalIgnoreCase)
                    && TryFindWlLiquidBodyByKey(_wlLayerSelectedBodyKey, out var wl) && wl != null)
                    CopyTextToClipboard(wl.SourcePath, "WL Source Path");
                else if (!string.IsNullOrEmpty(_loadedFilePath))
                    CopyTextToClipboard(_loadedFilePath, "Model Path");
                break;

            case "copy_chunk_texture_summary":
                if (TryGetTerrainChunkInspectionTarget(preferHoveredChunk: true, out var chunkInfo, out _)
                    && TryResolveTerrainChunkInspectionData(chunkInfo, out var chunkData, out var tileTextures)
                    && chunkData != null)
                {
                    string summary = BuildTerrainChunkTextureSummary(chunkInfo, chunkData, tileTextures);
                    CopyTextToClipboard(summary, "Chunk Texture Summary");
                }
                break;

            case "clear_pm4_selection":
                _worldScene?.ClearPm4ObjectSelection();
                break;

            case "open_pm4_workbench":
                OpenPm4Workbench(Pm4WorkbenchTab.Selection);
                break;

            case "export_pm4_json":
                ExportPm4ObjectsJson();
                break;

            case "export_pm4_obj":
                ExportPm4ObjectsObjSet();
                break;

            case "export_pm4_llm":
                ExportPm4LlmEvidenceBundle();
                break;

            case "clear_wl_selection":
                ClearSelectedWlLiquidBody(clearListIsolation: true);
                break;

            case "toggle_wl_liquids":
                if (_worldScene != null)
                    _worldScene.ShowWlLiquids = !_worldScene.ShowWlLiquids;
                break;
        }
    }
}
