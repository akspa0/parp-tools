using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.IO.Mdx;
using WoWViewer.DataSources;
using WoWViewer.Export;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Catalog;
using WoWViewer.Capture;
using WoWViewer.Population;
using WoWViewer.Terrain;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using static WoWViewer.ViewerApp;
using static WoWViewer.InvestigationService;

namespace WoWViewer;

/// <summary>
/// Terrain queries and editor overlays: chunk picking under the mouse, terrain raycasts, height sampling, loaded-chunk lookup, scene far plane, and the editor chunk overlays.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed partial class TerrainQueryService
{
    private readonly IViewerAppHost _host;

    internal TerrainQueryService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref Camera _camera => ref _host.Camera;
    private M2CameraPathDocument _cameraPath => _host.CameraPath;
    private ref (int tileX, int tileY, int chunkX, int chunkY)? _chunkClipboardCopiedKey => ref _host.ChunkClipboardCopiedKey;
    private ref (int tileX, int tileY, int chunkX, int chunkY)? _chunkClipboardLockedTargetKey => ref _host.ChunkClipboardLockedTargetKey;
    private ref bool _chunkClipboardShowOverlay => ref _host.ChunkClipboardShowOverlay;
    private ref Terrain.BoundingBoxRenderer? _editorOverlayBb => ref _host.EditorOverlayBb;
    private ref float _fovDegrees => ref _host.FovDegrees;
    private ref GL _gl => ref _host.Gl;
    private ref int _lastMcnkOverlayChunkCount => ref _host.LastMcnkOverlayChunkCount;
    private ref int _lastMcnkWeakCornerCount => ref _host.LastMcnkWeakCornerCount;
    private ref McnkOverlayFlags _mcnkOverlayFlags => ref _host.McnkOverlayFlags;
    private HashSet<(int tileX, int tileY, int chunkX, int chunkY)> _selectedChunks => _host.SelectedChunks;
    private ShellLayoutService _shellLayout => _host.ShellLayout;
    private ref bool _showCameraPathOverlay => ref _host.ShowCameraPathOverlay;
    private ref bool _showMcnkFlagOverlay => ref _host.ShowMcnkFlagOverlay;
    private ref bool _showMcnkWeakCorners => ref _host.ShowMcnkWeakCorners;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;

    private const float MinTerrainFarPlane = 1f;
    // Keep the WDL horizon visible well past the LIT/DBC fog endpoint.  FogEnd
    // remains the full-detail/visibility authority; this is projection room for
    // the low-detail WDL replacement terrain, not a second fog range.
    private const float TerrainFarPlanePadding = 2500f;
    private const float MaxTerrainFarPlane = MaxTerrainFogDistance + TerrainFarPlanePadding;

    internal void DrawEditorOverlays(Matrix4x4 view, Matrix4x4 proj)
    {
        var renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        bool drawCameraPathOverlay = _showCameraPathOverlay && _cameraPath.Keyframes.Count > 0;
        if (renderer == null && !drawCameraPathOverlay)
            return;

        bool drawChunkClipboardOverlay = renderer != null && _chunkClipboardShowOverlay
            && (_selectedChunks.Count > 0 || _chunkClipboardLockedTargetKey != null || _chunkClipboardCopiedKey != null);
        bool drawMcnkOverlay = renderer != null && ShouldDrawMcnkFlagOverlay(renderer);
        if (!drawChunkClipboardOverlay && !drawMcnkOverlay && !drawCameraPathOverlay)
            return;

        _editorOverlayBb ??= new Terrain.BoundingBoxRenderer(_gl);

        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.DepthMask(false);

        float overlayTime = (float)(System.Diagnostics.Stopwatch.GetTimestamp() / (double)System.Diagnostics.Stopwatch.Frequency);

        if (drawMcnkOverlay)
        {
            _gl.Enable(EnableCap.Blend);
            _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            _editorOverlayBb.BeginSolidBatch();
            _editorOverlayBb.BeginBatch();
            BatchMcnkFlagOverlayGeometry(_editorOverlayBb);
            _editorOverlayBb.FlushSolidBatch(view, proj);
            _gl.Disable(EnableCap.Blend);
        }

        if (drawChunkClipboardOverlay)
        {
            if (!drawMcnkOverlay)
                _editorOverlayBb.BeginBatch();

            if (_selectedChunks.Count > 0)
            {
                foreach (var (tx, ty, cx, cy) in _selectedChunks)
                {
                    if (renderer.TryGetChunkInfo(tx, ty, cx, cy, out var sel))
                        _editorOverlayBb.BatchBoxMinMax(sel.BoundsMin, sel.BoundsMax, new Vector3(0f, 1f, 1f));
                }
            }

            if (_chunkClipboardLockedTargetKey is { } locked && renderer.TryGetChunkInfo(locked.tileX, locked.tileY, locked.chunkX, locked.chunkY, out var lockedInfo))
                _editorOverlayBb.BatchHighlightedBoxMinMax(
                    lockedInfo.BoundsMin,
                    lockedInfo.BoundsMax,
                    overlayTime,
                    new Vector3(1f, 1f, 1f),
                    new Vector3(1f, 0.8f, 0.1f),
                    new Vector3(0.1f, 0.9f, 1f));

            if (_chunkClipboardCopiedKey is (int copiedTx, int copiedTy, int copiedCx, int copiedCy) copied && renderer.TryGetChunkInfo(copiedTx, copiedTy, copiedCx, copiedCy, out var copiedInfo))
                _editorOverlayBb.BatchBoxMinMax(copiedInfo.BoundsMin, copiedInfo.BoundsMax, new Vector3(1f, 1f, 0f));
        }

        if (drawCameraPathOverlay)
        {
            if (!drawMcnkOverlay && !drawChunkClipboardOverlay)
                _editorOverlayBb.BeginBatch();
            DrawCameraPathOverlay(_editorOverlayBb);
        }

        _editorOverlayBb.FlushBatch(view, proj);

        _gl.DepthMask(true);
    }

    internal bool TryPickTerrainChunkUnderMouse(TerrainRenderer renderer, out TerrainRenderer.TerrainChunkInfo info)
    {
        info = default;

        if (!_shellLayout.TryGetSceneViewportRect(out float vpX, out float vpY, out float vpW, out float vpH))
            return false;

        var mouse = ImGui.GetMousePos();
        float mouseX = mouse.X;
        float mouseY = mouse.Y;
        if (mouseX < vpX || mouseX > vpX + vpW || mouseY < vpY || mouseY > vpY + vpH)
            return false;

        float aspect = vpW / Math.Max(vpH, 1f);
        var view = _camera.GetViewMatrix();
        float farPlane = GetSceneFarPlane();
        var proj = Matrix4x4.CreatePerspectiveFieldOfView(_fovDegrees * MathF.PI / 180f, aspect, 0.1f, farPlane);

        float localX = mouseX - vpX;
        float localY = mouseY - vpY;
        float ndcX = (localX / vpW) * 2f - 1f;
        float ndcY = 1f - (localY / vpH) * 2f;

        var (rayOrigin, rayDir) = WorldScene.ScreenToRay(ndcX, ndcY, view, proj);
        return TryRaycastTerrain(renderer, rayOrigin, rayDir, farPlane, out info);
    }

    internal bool TryRaycastTerrain(TerrainRenderer renderer, Vector3 rayOrigin, Vector3 rayDir, float maxDistance, out TerrainRenderer.TerrainChunkInfo info)
    {
        return TryRaycastTerrain(renderer, rayOrigin, rayDir, maxDistance, out info, out _);
    }

    internal bool TryRaycastTerrain(TerrainRenderer renderer, Vector3 rayOrigin, Vector3 rayDir, float maxDistance, out TerrainRenderer.TerrainChunkInfo info, out Vector3 hitPoint)
    {
        info = default;
        hitPoint = default;

        const float step = 16f;
        int maxSteps = (int)MathF.Ceiling(maxDistance / step);
        maxSteps = Math.Clamp(maxSteps, 16, 1024);

        float prevT = 0f;
        float prevD = float.NaN;

        for (int i = 0; i <= maxSteps; i++)
        {
            float t = i * step;
            var p = rayOrigin + rayDir * t;

            if (!TrySampleTerrainHeightLoaded(renderer, p.X, p.Y, out float height, out var curInfo))
                continue;

            float d = p.Z - height;
            if (!float.IsNaN(prevD))
            {
                if (prevD > 0f && d <= 0f)
                {
                    float a = prevT;
                    float b = t;
                    TerrainRenderer.TerrainChunkInfo best = curInfo;
                    for (int it = 0; it < 10; it++)
                    {
                        float m = (a + b) * 0.5f;
                        var pm = rayOrigin + rayDir * m;
                        if (!TrySampleTerrainHeightLoaded(renderer, pm.X, pm.Y, out float hm, out var mi))
                        {
                            a = m;
                            continue;
                        }

                        best = mi;
                        float dm = pm.Z - hm;
                        if (dm > 0f)
                            a = m;
                        else
                            b = m;
                    }

                    float hitDistance = (a + b) * 0.5f;
                    hitPoint = rayOrigin + rayDir * hitDistance;
                    info = best;
                    return true;
                }
            }

            prevT = t;
            prevD = d;
        }

        return false;
    }

    internal float GetSceneFarPlane()
    {
        if (_terrainManager != null)
            return ComputeSceneFarPlane(_terrainManager.Lighting.FogEnd);

        if (_vlmTerrainManager != null)
            return ComputeSceneFarPlane(_vlmTerrainManager.Lighting.FogEnd);

        return 10000f;
    }

    internal static float ComputeSceneFarPlane(float fogEnd)
    {
        float safeFogEnd = float.IsFinite(fogEnd) && fogEnd > 0f ? fogEnd : 1500f;
        return Math.Clamp(safeFogEnd + TerrainFarPlanePadding, MinTerrainFarPlane, MaxTerrainFarPlane);
    }

    internal bool TrySampleTerrainHeightLoaded(TerrainRenderer renderer, float worldX, float worldY, out float height, out TerrainRenderer.TerrainChunkInfo info)
    {
        height = 0f;
        info = default;

        var ci = renderer.GetChunkInfoAt(worldX, worldY);
        if (!ci.HasValue)
            return false;

        info = ci.Value;
        if (!TryGetChunkDataLoadedOnly(info.TileX, info.TileY, info.ChunkX, info.ChunkY, out var chunk))
            return false;

        float localX = chunk.WorldPosition.Y - worldY;
        float localY = chunk.WorldPosition.X - worldX;
        localX = Math.Clamp(localX, 0f, WoWConstants.ChunkSize);
        localY = Math.Clamp(localY, 0f, WoWConstants.ChunkSize);

        height = TerrainChunkMath.SampleHeightOuterGrid(chunk, localX, localY);
        return true;
    }

    private bool TryGetChunkDataLoadedOnly(int tileX, int tileY, int chunkX, int chunkY, out Terrain.TerrainChunkData chunk)
    {
        chunk = new Terrain.TerrainChunkData();

        List<Terrain.TerrainChunkData>? chunks = null;
        if (_terrainManager != null)
        {
            if (!_terrainManager.TryGetTileLoadResult(tileX, tileY, out var tile))
                return false;
            chunks = tile.Chunks;
        }
        else if (_vlmTerrainManager != null)
        {
            if (!_vlmTerrainManager.TryGetTileLoadResult(tileX, tileY, out var tile))
                return false;
            chunks = tile.Chunks;
        }

        if (chunks == null || chunks.Count == 0)
            return false;

        var found = chunks.FirstOrDefault(c => c != null && c.ChunkX == chunkX && c.ChunkY == chunkY);
        if (found == null || found.Heights == null || found.Heights.Length < 145)
            return false;

        chunk = found;
        return true;
    }
}
