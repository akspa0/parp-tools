using System.Numerics;
using ImGuiNET;
using WoWViewer.Terrain;
using WoWViewer.Terrain.Vlm;

namespace WoWViewer;

/// <summary>
/// Utilities &gt; Perf (tab and floating window): frame timing, hitch attribution, submission counters and
/// the whole-scene render switches (opaque batching, GPU instancing, world doodad animation).
/// </summary>
internal sealed partial class PerfPanelService
{
    private readonly IViewerAppHost _host;

    internal PerfPanelService(IViewerAppHost host)
    {
        _host = host;
    }

    private ref bool _showPerfWindow => ref _host.ShowPerfWindow;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private ref WorldScene? _worldScene => ref _host.WorldScene;

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
}
