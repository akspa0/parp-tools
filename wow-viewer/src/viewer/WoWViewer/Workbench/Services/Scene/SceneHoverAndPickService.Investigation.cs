using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Visibility;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// SceneHoverAndPickService: members moved from ViewerApp_Investigation.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class SceneHoverAndPickService
{

    private bool TryDrawTerrainChunkHoverOverlay()
    {
        if (_worldScene != null && !_worldScene.ShowHoveredAssetTooltips)
            return false;

        TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (renderer == null)
            return false;

        if (IsSceneMouseCaptureBlocked(_lastMouseX, _lastMouseY))
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
        ImGui.TextColored(new Vector4(0.95f, 0.78f, 0.62f, 1.0f), $"MCNK: 0x{(uint)chunkData.McnkFlags:X8}  {DescribeMcnkFlags(chunkData.McnkFlags)}");
        ImGui.TextColored(new Vector4(0.72f, 0.78f, 0.90f, 1.0f), $"World: ({chunkData.WorldPosition.X:F1}, {chunkData.WorldPosition.Y:F1}, {chunkData.WorldPosition.Z:F1})");
        ImGui.Separator();

        ImGui.TextColored(new Vector4(0.86f, 0.92f, 0.76f, 1.0f), "Left-click or use Inspector for full chunk details.");
        ImGui.End();
        ImGui.PopStyleColor(3);
        ImGui.PopStyleVar(3);
        return true;
    }

    /// <summary>
    /// Whether the hover overlay may show PM4 asset-match candidates.
    /// </summary>
    /// <remarks>
    /// Gated to the explicit PM4 investigation mode. These candidates come from the geometric
    /// fingerprint matcher, whose measured precision is P@1 = 1.3% (specs 046/065), so surfacing
    /// them on every hover put a mostly-wrong answer in front of the user constantly. They stay
    /// available when PM4 identity is what is being investigated, and are silent otherwise.
    /// Note that object identity now has a far better route than fingerprinting - MSUR._0x1C is the
    /// producing placement's Z, which resolves the asset outright (see `pm4 object-library`).
    /// </remarks>
    private bool ShouldShowHoveredPm4MatchCandidates()
        => _visualInvestigationMode == VisualInvestigationMode.Pm4;

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
}
