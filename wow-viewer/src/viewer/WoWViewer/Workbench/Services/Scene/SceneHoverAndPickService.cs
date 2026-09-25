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
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.M2;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WowViewer.Core.IO.Maps;
using WoWViewer.Terrain.Vlm;
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
/// Scene hover and pick: mouse object picking, click-selection candidates, hovered-asset info, wireframe reveal, 3D cursor and the hover/PM4-match overlays.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed partial class SceneHoverAndPickService
{
    private readonly IViewerAppHost _host;

    internal SceneHoverAndPickService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref Camera _camera => ref _host.Camera;
    private ref EditorWorkspaceTask _editorWorkspaceTask => ref _host.EditorWorkspaceTask;
    private ref float _fovDegrees => ref _host.FovDegrees;
    private ref GL _gl => ref _host.Gl;
    private ref Pm4ObjectMatchObject? _hoveredPm4ObjectMatch => ref _host.HoveredPm4ObjectMatch;
    private ref int _hoveredPm4ObjectMatchCacheMaxMatches => ref _host.HoveredPm4ObjectMatchCacheMaxMatches;
    private ref (int tileX, int tileY, uint ck24, int objectPart)? _hoveredPm4ObjectMatchKey => ref _host.HoveredPm4ObjectMatchKey;
    private ref float _lastMouseX => ref _host.LastMouseX;
    private ref float _lastMouseY => ref _host.LastMouseY;
    private ref int _pm4ObjectMatchMaxMatchesPerObject => ref _host.Pm4ObjectMatchMaxMatchesPerObject;
    private ref SceneClusterSelector3D? _sceneClusterSelector3D => ref _host.SceneClusterSelector3D;
    private ref SceneCursorRenderer? _sceneCursorRenderer => ref _host.SceneCursorRenderer;
    private ref int _selectedObjectIndex => ref _host.SelectedObjectIndex;
    private ref string _selectedObjectInfo => ref _host.SelectedObjectInfo;
    private ref string _selectedObjectType => ref _host.SelectedObjectType;
    private ref string _statusMessage => ref _host.StatusMessage;
    private TaxiAndAreaPoiSelectionService _taxiAndAreaPoi => _host.TaxiAndAreaPoi;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref VisualInvestigationMode _visualInvestigationMode => ref _host.VisualInvestigationMode;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private ref WorkspaceMode _workspaceMode => ref _host.WorkspaceMode;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private bool CanSceneConsumeMouse(float x, float y) => _host.CanSceneConsumeMouse(x, y);
    private void ClearSelectedWlLiquidBody(bool clearListIsolation) => _host.ClearSelectedWlLiquidBody(clearListIsolation);
    private float GetSceneFarPlane() => _host.GetSceneFarPlane();
    private bool IsSceneMouseCaptureBlocked(float x, float y) => _host.IsSceneMouseCaptureBlocked(x, y);
    private void RefreshSelectedWorldObjectInfo() => _host.RefreshSelectedWorldObjectInfo();
    private void SelectTerrainChunkFromClick(TerrainRenderer.TerrainChunkInfo info) => _host.SelectTerrainChunkFromClick(info);
    private void SetSelectedWlLiquidBody(WlLiquidBody body, bool isolateInList, bool focusInspectWorkspace, string? statusMessage = null) => _host.SetSelectedWlLiquidBody(body, isolateInList, focusInspectWorkspace, statusMessage);
    private bool ShouldShowHoveredAssetInfoForInvestigation(HoveredAssetInfo info) => _host.ShouldShowHoveredAssetInfoForInvestigation(info);
    private bool TogglePm4ObjectCollectionMembership((int tileX, int tileY, uint ck24, int objectPart) key, bool reportStatus, bool removeIfPresent = true) => _host.TogglePm4ObjectCollectionMembership(key, reportStatus, removeIfPresent);
    private bool TryFindWlLiquidBodyByKey(string bodyKey, out WlLiquidBody? body) => _host.TryFindWlLiquidBodyByKey(bodyKey, out body);
    private bool TryGetSceneViewportRect(out float x, out float y, out float width, out float height) => _host.TryGetSceneViewportRect(out x, out y, out width, out height);
    private bool TryPickTerrainChunkUnderMouse(TerrainRenderer renderer, out TerrainRenderer.TerrainChunkInfo info) => _host.TryPickTerrainChunkUnderMouse(renderer, out info);
    private bool TryRaycastTerrain(TerrainRenderer renderer, Vector3 rayOrigin, Vector3 rayDir, float maxDistance, out TerrainRenderer.TerrainChunkInfo info) => _host.TryRaycastTerrain(renderer, rayOrigin, rayDir, maxDistance, out info);
    private bool TryRaycastTerrain(TerrainRenderer renderer, Vector3 rayOrigin, Vector3 rayDir, float maxDistance, out TerrainRenderer.TerrainChunkInfo info, out Vector3 hitPoint) => _host.TryRaycastTerrain(renderer, rayOrigin, rayDir, maxDistance, out info, out hitPoint);
    private bool TryResolveHoveredWlLiquidBody(HoveredAssetInfo hoveredInfo, out WlLiquidBody? body) => _host.TryResolveHoveredWlLiquidBody(hoveredInfo, out body);


    internal void PickObjectAtMouse(float mouseX, float mouseY, bool addPm4ToCollection = false)
    {
        if (_worldScene == null) return;

        System.Diagnostics.Stopwatch clickSw = WoWViewer.Logging.Pm4Profiling.Enabled
            ? System.Diagnostics.Stopwatch.StartNew() : null;
        long clickPickStartTicks = 0;
        long clickSelectionStartTicks = 0;
        double clickPickMs = 0;
        double clickSelectionMs = 0;

        try
        {
            if (!TryGetSceneViewportRect(out float vpX, out float vpY, out float vpW, out float vpH))
                return;

            if (mouseX < vpX || mouseX > vpX + vpW || mouseY < vpY || mouseY > vpY + vpH)
                return;

            float aspect = vpW / Math.Max(vpH, 1f);
            var view = _camera.GetViewMatrix();
            float farPlane = GetSceneFarPlane();
            var proj = Matrix4x4.CreatePerspectiveFieldOfView(_fovDegrees * MathF.PI / 180f, aspect, 0.1f, farPlane);

            // Convert viewport-local mouse coords to NDC (-1..1)
            float localX = mouseX - vpX;
            float localY = mouseY - vpY;
            float ndcX = (localX / vpW) * 2f - 1f;
            float ndcY = 1f - (localY / vpH) * 2f; // flip Y

            var (rayOrigin, rayDir) = WorldScene.ScreenToRay(ndcX, ndcY, view, proj);
            var hoveredPm4Key = _worldScene.Pm4Overlay.ShowPm4Overlay ? _worldScene.HoveredAssetInfo?.Pm4ObjectKey : null;

            if (addPm4ToCollection)
            {
                // Only the Shift+LMB collection branch needs the ray PM4 pick;
                // the normal-click path picks PM4 inside TryHandleSceneClickSelection
                // (a duplicate outer pick here doubled the per-click cost on dense maps).
                if (clickSw != null) clickPickStartTicks = clickSw.ElapsedTicks;
                _worldScene.Pm4Overlay.TryPickPm4ObjectByRay(rayOrigin, rayDir, out var pm4HitKey, out var _, out _);
                if (clickSw != null) clickPickMs = (clickSw.ElapsedTicks - clickPickStartTicks) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;

                ClearPendingClickSelection();
                _worldScene.ClearTaxiSelection();
                _worldScene.ClearSelection();
                _taxiAndAreaPoi.ClearSelectedAreaPoiInfo();

                var collectionPm4Key = hoveredPm4Key ?? pm4HitKey;
                if (collectionPm4Key.HasValue && _worldScene.Pm4Overlay.SelectPm4Object(collectionPm4Key.Value))
                {
                    TogglePm4ObjectCollectionMembership(collectionPm4Key.Value, reportStatus: true);
                    UpdateSelectedPm4ObjectInfo(collectionPm4Key);
                }
                else
                {
                    _statusMessage = "Shift+LMB PM4 add failed: no PM4 object was hit under the cursor. Use the PM4 graph Collect buttons when overlaps are dense.";
                }

                return;
            }

            if (clickSw != null) clickSelectionStartTicks = clickSw.ElapsedTicks;
            bool handledBySelection = TryHandleSceneClickSelection(mouseX, mouseY, localX, localY, vpW, vpH, view, proj, rayOrigin, rayDir);
            if (clickSw != null) clickSelectionMs = (clickSw.ElapsedTicks - clickSelectionStartTicks) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
            if (handledBySelection)
                return;

            ClearPendingClickSelection();
            ClearSelectedWlLiquidBody(clearListIsolation: true);
            _worldScene.ClearSelection();
            _worldScene.ClearTaxiSelection();
            _worldScene.Pm4Overlay.ClearPm4ObjectSelection();
            _taxiAndAreaPoi.ClearSelectedAreaPoiInfo();
            _selectedObjectIndex = -1;
            _selectedObjectType = "";
            _selectedObjectInfo = "";
        }
        finally
        {
            if (clickSw != null)
            {
                clickSw.Stop();
                double totalMs = clickSw.ElapsedMilliseconds;
                if (totalMs >= 50.0)
                {
                    ViewerLog.Info(ViewerLog.Category.Terrain,
                        $"[PM4-PROFILE] PickObjectAtMouse: total={totalMs:0.0}ms pick={clickPickMs:0.0}ms selection={clickSelectionMs:0.0}ms shift={addPm4ToCollection}");
                }
            }
        }
    }

    private void UpdateSelectedPm4ObjectInfo((int tileX, int tileY, uint ck24, int objectPart)? pm4ObjectKey)
    {
        if (_worldScene == null)
            return;

        _selectedObjectType = "PM4";

        if (_worldScene.Pm4Overlay.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo debugInfo))
        {
            string nearestRef = float.IsNaN(debugInfo.NearestPositionRefDistance)
                ? "n/a"
                : $"{debugInfo.NearestPositionRefDistance:F2}";

            // Identity first, raw slices last. MSUR._0x1C is the producing placement's Z read as a
            // float (see docs/wowdev-wiki/pm4-pd4-draft.md), so that height plus the tile is what
            // actually names this object. "CK24" is the top three bytes of that float and its
            // "type" byte is the float's exponent band; both stay below for cross-referencing older
            // reports, not as identity.
            float selectedPlacementZ = BitConverter.UInt32BitsToSingle(debugInfo.Ck24 << 8);

            _selectedObjectInfo =
                $"PM4 Object\n" +
                $"Identity: {(debugInfo.Ck24 == 0 ? "NO PLACEMENT HEIGHT - unattributed; population is mostly M2 doodad collision" : $"placement Z {selectedPlacementZ:F3}")} on tile ({debugInfo.TileX}, {debugInfo.TileY})\n" +
                $"Region: {debugInfo.MshdRegionId}\n" +
                $"Raw MSUR._0x1C slice: 0x{debugInfo.Ck24:X6}  (exponent band 0x{debugInfo.Ck24Type:X2}, NOT a type)\n" +
                $"Viewer Part: {debugInfo.ObjectPartId} - assigned during the current overlay build after viewer-side splitting; not a raw PM4 field\n" +
                $"MSLK Group: 0x{debugInfo.LinkGroupObjectId:X8}\n" +
                $"Linked MPRL refs: {debugInfo.LinkedPositionRefCount}\n" +
                $"Surfaces: {debugInfo.SurfaceCount}\n" +
                $"GroupKey: 0x{debugInfo.DominantGroupKey:X2}  AttrMask: 0x{debugInfo.DominantAttributeMask:X2}  MscnRef: {debugInfo.DominantMscnRefIndex}\n" +
                $"Planar: swap={debugInfo.SwapPlanarAxes} invertU={debugInfo.InvertU} invertV={debugInfo.InvertV} windingFlip={debugInfo.InvertsWinding}\n" +
                $"Center: ({debugInfo.Center.X:F1}, {debugInfo.Center.Y:F1}, {debugInfo.Center.Z:F1})\n" +
                $"Nearest MPRL: {nearestRef}\n" +
                $"Offset: ({_worldScene.Pm4Overlay.SelectedPm4ObjectTranslation.X:F2}, {_worldScene.Pm4Overlay.SelectedPm4ObjectTranslation.Y:F2}, {_worldScene.Pm4Overlay.SelectedPm4ObjectTranslation.Z:F2})";
            return;
        }

        if (!pm4ObjectKey.HasValue)
            return;

        var selectedPm4 = pm4ObjectKey.Value;
        _selectedObjectInfo =
            $"PM4 Object\n" +
            $"Tile: ({selectedPm4.tileX}, {selectedPm4.tileY})\n" +
            $"CK24: 0x{selectedPm4.ck24:X6} (viewerPart={selectedPm4.objectPart})\n" +
            $"Viewer Part: assigned during the current overlay build; not a raw PM4 field\n" +
            $"Offset: ({_worldScene.Pm4Overlay.SelectedPm4ObjectTranslation.X:F2}, {_worldScene.Pm4Overlay.SelectedPm4ObjectTranslation.Y:F2}, {_worldScene.Pm4Overlay.SelectedPm4ObjectTranslation.Z:F2})";
    }

    internal void UpdateWorldSceneWireframeReveal(Matrix4x4 view, Matrix4x4 proj)
    {
        if (_worldScene == null || !_worldScene.WireframeRevealEnabled)
            return;

        if (IsSceneMouseCaptureBlocked(_lastMouseX, _lastMouseY) || !TryGetSceneViewportRect(out float vpX, out float vpY, out float vpW, out float vpH))
        {
            _worldScene.ClearWireframeReveal();
            return;
        }

        if (_lastMouseX < vpX || _lastMouseX > vpX + vpW || _lastMouseY < vpY || _lastMouseY > vpY + vpH)
        {
            _worldScene.ClearWireframeReveal();
            return;
        }

        float localX = _lastMouseX - vpX;
        float localY = _lastMouseY - vpY;
        _worldScene.UpdateWireframeReveal(view, proj, localX, localY, vpW, vpH);
    }

    internal void UpdateWorldSceneHoveredAssetInfo(Matrix4x4 view, Matrix4x4 proj)
    {
        if (_worldScene == null)
            return;

        if (IsSceneMouseCaptureBlocked(_lastMouseX, _lastMouseY) || !TryGetSceneViewportRect(out float vpX, out float vpY, out float vpW, out float vpH))
        {
            _worldScene.ClearHoveredAssetInfo();
            return;
        }

        if (_lastMouseX < vpX || _lastMouseX > vpX + vpW || _lastMouseY < vpY || _lastMouseY > vpY + vpH)
        {
            _worldScene.ClearHoveredAssetInfo();
            return;
        }

        float localX = _lastMouseX - vpX;
        float localY = _lastMouseY - vpY;
        _worldScene.UpdateHoveredAssetInfo(view, proj, localX, localY, vpW, vpH);

        // Operator feedback 2026-09-06: the mouse picked objects many tiles away
        // THROUGH the ground. The hover picker tests object distance but never
        // terrain occlusion, so an object behind a hill was still hovered. If
        // terrain is hit first along the same ray, the hover is invalid.
        // WL bodies are source-data inspection targets, not scene objects. A composed layer can
        // legitimately put terrain in front of their original bounds, but that must not erase the
        // WL hover identity that the click inspector consumes. Keep terrain occlusion for actual
        // placed scene objects (WMO/M2/PM4) only.
        if (_worldScene.HoveredAssetInfo is { IsPreciseRayHit: true } hovered
            && !string.Equals(hovered.AssetKind, "WL liquid", StringComparison.OrdinalIgnoreCase))
        {
            float ndcX = (localX / MathF.Max(vpW, 1f)) * 2f - 1f;
            float ndcY = 1f - (localY / MathF.Max(vpH, 1f)) * 2f;
            var (rayOrigin, rayDir) = WorldScene.ScreenToRay(ndcX, ndcY, view, proj);
            TerrainRenderer? occlusionRenderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
            if (occlusionRenderer != null
                && TryRaycastTerrain(occlusionRenderer, rayOrigin, rayDir, GetSceneFarPlane(), out _, out Vector3 terrainHit))
            {
                float terrainDistance = Vector3.Distance(rayOrigin, terrainHit);
                float objectDistance = Vector3.Distance(rayOrigin, hovered.WorldPosition);
                if (objectDistance > terrainDistance + 1f)
                    _worldScene.ClearHoveredAssetInfo();
            }
        }
    }

    /// <summary>
    /// World-space pass for the in-scene cluster selection rings. Runs with the rest of the 3D
    /// scene so the rings occlude correctly against terrain and objects.
    /// </summary>
    internal void RenderSceneClusterSelector3D(Matrix4x4 proj)
    {
        if (_sceneClusterSelector3D != null && _sceneClusterSelector3D.IsActive)
        {
            _sceneClusterSelector3D.RenderWorld3D(_camera, proj);
        }
    }

    /// <summary>
    /// Overlay pass for the 3D scene cursor.
    /// </summary>
    /// <remarks>
    /// Called after <c>_imGui.Render()</c>, never with the 3D scene. ImGui draws in its own pass on
    /// top of whatever the 3D pass produced, so a cursor drawn during the 3D pass is painted over by
    /// every panel, menu, popup and hover card - and since the hardware cursor is hidden while this
    /// cursor is active, the pointer simply vanishes under the UI. Depth state cannot help; only
    /// draw order can. The caller is responsible for setting the scene viewport before this and
    /// restoring the full framebuffer viewport after.
    ///
    /// ImGui's Silk.NET controller restores the GL state it found, but this pass runs on whatever
    /// it left, so the state the cursor depends on is set explicitly below rather than assumed.
    /// </remarks>
    internal void RenderSceneCursor(
        Matrix4x4 view,
        Matrix4x4 proj,
        float vpX,
        float vpY,
        float vpW,
        float vpH)
    {
        if (_sceneCursorRenderer == null || _sceneCursorRenderer.Style == CursorStyle.ClassicOSArrow)
            return;

        if (!CanSceneConsumeMouse(_lastMouseX, _lastMouseY))
            return;

        float localX = _lastMouseX - vpX;
        float localY = _lastMouseY - vpY;
        float ndcX = (localX / vpW) * 2f - 1f;
        float ndcY = 1f - (localY / vpH) * 2f;

        var (rayOrigin, rayDir) = WorldScene.ScreenToRay(ndcX, ndcY, view, proj);

        float? hitDistance = null;
        if (_worldScene?.HoveredAssetInfo is HoveredAssetInfo hoverInfo && hoverInfo.IsPreciseRayHit)
        {
            hitDistance = (hoverInfo.WorldPosition - rayOrigin).Length();
            _sceneCursorRenderer.State = (hoverInfo.AssetKind.Contains("NPC", StringComparison.OrdinalIgnoreCase)
                || hoverInfo.DisplayName.Contains("Creature", StringComparison.OrdinalIgnoreCase))
                ? SceneCursorState.Speak
                : SceneCursorState.Interact;
        }
        else
        {
            TerrainRenderer? terrainRenderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
            if (terrainRenderer != null && TryRaycastTerrain(terrainRenderer, rayOrigin, rayDir, GetSceneFarPlane(), out _, out Vector3 hitPoint))
            {
                hitDistance = (hitPoint - rayOrigin).Length();
            }

            _sceneCursorRenderer.State = _workspaceMode == WorkspaceMode.Editor
                && (_editorWorkspaceTask == EditorWorkspaceTask.Terrain || _editorWorkspaceTask == EditorWorkspaceTask.Objects)
                ? SceneCursorState.CastGlow
                : SceneCursorState.Pointer;
        }

        // ImGui leaves scissor test enabled and clipped to its last draw command; anything left
        // clipped here would silently discard the cursor. Blending must be on for the cursor's
        // alpha, and face culling off because the billboard can present either winding.
        _gl.Disable(EnableCap.ScissorTest);
        _gl.Disable(EnableCap.CullFace);
        _gl.Enable(EnableCap.Blend);
        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
        _gl.DepthMask(true);

        _sceneCursorRenderer.Render(_camera, proj, rayOrigin, rayDir, hitDistance, _fovDegrees, 0.1f);
    }

    /// <summary>
    /// Coloured text that is NOT run through printf formatting.
    /// </summary>
    /// <remarks>
    /// <c>ImGui.Text</c> and <c>ImGui.TextColored</c> treat their argument as a format string, so a
    /// '%' arriving from data - an asset path, a percentage in a detail line - is read as a
    /// conversion specifier and prints garbage pulled off the stack. A tooltip reading
    /// "95.135345743157f" where "95.1%" was written is exactly that. Any string that comes from data
    /// rather than from a literal must go through here.
    /// </remarks>
    internal static void TextColoredUnformatted(Vector4 color, string text)
    {
        ImGui.PushStyleColor(ImGuiCol.Text, color);
        ImGui.TextUnformatted(text ?? string.Empty);
        ImGui.PopStyleColor();
    }

    internal void DrawSceneHoverAssetOverlay()
    {
        if (_visualInvestigationMode == VisualInvestigationMode.Adt)
        {
            TryDrawTerrainChunkHoverOverlay();
            return;
        }

        if (_sceneCursorRenderer != null && _sceneCursorRenderer.Style != CursorStyle.ClassicOSArrow)
            return;

        if (_sceneClusterSelector3D != null && _sceneClusterSelector3D.IsActive)
            return;

        if (_worldScene != null && !_worldScene.ShowHoveredAssetTooltips)
            return;

        if (_worldScene?.HoveredAssetInfo is not HoveredAssetInfo info)
            return;

        if (!info.IsPreciseRayHit || !ShouldShowHoveredAssetInfoForInvestigation(info))
            return;

        if (!TryGetSceneViewportRect(out float vpX, out float vpY, out float vpW, out float vpH))
            return;

        if (_lastMouseX < vpX || _lastMouseX > vpX + vpW || _lastMouseY < vpY || _lastMouseY > vpY + vpH)
            return;

        Vector2 displaySize = ImGui.GetIO().DisplaySize;
        Vector2 overlayPos = new(
            MathF.Min(_lastMouseX + 18f, MathF.Max(8f, displaySize.X - 390f)),
            MathF.Min(_lastMouseY + 18f, MathF.Max(8f, displaySize.Y - 290f)));

        ImGui.SetNextWindowPos(overlayPos, ImGuiCond.Always);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(16f, 13f));
        ImGui.PushStyleVar(ImGuiStyleVar.WindowBorderSize, 2f);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowRounding, 4f);
        ImGui.PushStyleColor(ImGuiCol.WindowBg, new Vector4(0.04f, 0.05f, 0.09f, 0.985f));
        ImGui.PushStyleColor(ImGuiCol.Border, new Vector4(0.95f, 0.79f, 0.28f, 0.98f));
        ImGui.PushStyleColor(ImGuiCol.Separator, new Vector4(0.88f, 0.73f, 0.22f, 0.82f));

        ImGuiWindowFlags flags = ImGuiWindowFlags.NoDecoration
            | ImGuiWindowFlags.AlwaysAutoResize
            | ImGuiWindowFlags.NoDocking
            | ImGuiWindowFlags.NoSavedSettings
            | ImGuiWindowFlags.NoFocusOnAppearing
            | ImGuiWindowFlags.NoNav
            | ImGuiWindowFlags.NoMove
            | ImGuiWindowFlags.NoInputs;

        if (!ImGui.Begin("##SceneHoverAssetOverlay", flags))
        {
            ImGui.End();
            ImGui.PopStyleColor(3);
            ImGui.PopStyleVar(3);
            return;
        }

        ImGui.SetWindowFontScale(1.22f);
        TextColoredUnformatted(GetHoveredAssetTitleColor(info), info.DisplayName);
        ImGui.SetWindowFontScale(1.0f);
        TextColoredUnformatted(new Vector4(1.0f, 0.91f, 0.56f, 1.0f), info.AssetKind);

        if (!string.IsNullOrWhiteSpace(info.SourcePath))
        {
            ImGui.PushTextWrapPos(ImGui.GetCursorPosX() + 340f);
            TextColoredUnformatted(new Vector4(0.54f, 0.84f, 0.52f, 1.0f), info.SourcePath);
            ImGui.PopTextWrapPos();
        }

        if (!string.IsNullOrWhiteSpace(info.ParentSourcePath))
        {
            ImGui.PushTextWrapPos(ImGui.GetCursorPosX() + 340f);
            TextColoredUnformatted(new Vector4(0.62f, 0.72f, 0.86f, 1.0f), $"Parent WMO: {info.ParentSourcePath}");
            ImGui.PopTextWrapPos();
        }

        if (!string.IsNullOrWhiteSpace(info.DetailLine))
            TextColoredUnformatted(new Vector4(0.86f, 0.88f, 0.94f, 1.0f), info.DetailLine);

        ImGui.Separator();

        ImGui.TextColored(new Vector4(0.72f, 0.78f, 0.90f, 1.0f), $"World: ({info.WorldPosition.X:F1}, {info.WorldPosition.Y:F1}, {info.WorldPosition.Z:F1})");

        if (info.Pm4ObjectKey.HasValue && ShouldShowHoveredPm4MatchCandidates())
            DrawHoveredPm4MatchCandidates(info.Pm4ObjectKey.Value);

        ImGui.Separator();
        ImGui.TextColored(new Vector4(1.0f, 0.85f, 0.38f, 1.0f), "Left-click to view in Inspector");

        if (info.AdditionalHitCount > 0)
        {
            string suffix = info.AdditionalHitCount == 1 ? string.Empty : "s";
            ImGui.TextColored(new Vector4(0.78f, 0.73f, 0.57f, 1.0f), $"+{info.AdditionalHitCount} more nearby asset hit{suffix}");
        }

        ImGui.SetWindowFontScale(1.0f);
        ImGui.End();
        ImGui.PopStyleColor(3);
        ImGui.PopStyleVar(3);
    }

    private static Vector4 GetHoveredAssetTitleColor(HoveredAssetInfo info)
    {
        return info.AssetKind switch
        {
            "PM4" => new Vector4(1.0f, 0.82f, 0.32f, 1.0f),
            "WMO" => new Vector4(0.78f, 0.92f, 1.0f, 1.0f),
            "WMO Doodad" => new Vector4(0.96f, 0.84f, 0.58f, 1.0f),
            "WL liquid" => new Vector4(0.60f, 0.88f, 1.0f, 1.0f),
            _ => new Vector4(0.92f, 0.96f, 1.0f, 1.0f)
        };
    }

    private bool TryGetHoveredPm4ObjectMatch((int tileX, int tileY, uint ck24, int objectPart) objectKey, out Pm4ObjectMatchObject objectMatch)
    {
        objectMatch = null!;

        int maxMatches = Math.Max(3, Math.Min(5, _pm4ObjectMatchMaxMatchesPerObject));
        if (_hoveredPm4ObjectMatch != null
            && _hoveredPm4ObjectMatchKey.HasValue
            && _hoveredPm4ObjectMatchKey.Value == objectKey
            && _hoveredPm4ObjectMatchCacheMaxMatches == maxMatches)
        {
            objectMatch = _hoveredPm4ObjectMatch;
            return true;
        }

        if (_worldScene == null || !_worldScene.Pm4Overlay.TryBuildPm4ObjectMatch(objectKey, maxMatches, out Pm4ObjectMatchObject hoveredMatch))
            return false;

        _hoveredPm4ObjectMatch = hoveredMatch;
        _hoveredPm4ObjectMatchKey = objectKey;
        _hoveredPm4ObjectMatchCacheMaxMatches = maxMatches;
        objectMatch = hoveredMatch;
        return true;
    }

    private void DrawHoveredPm4MatchCandidates((int tileX, int tileY, uint ck24, int objectPart) objectKey)
    {
        ImGui.Separator();
        ImGui.TextColored(new Vector4(1.0f, 0.90f, 0.52f, 1.0f), "Likely matches");

        if (!TryGetHoveredPm4ObjectMatch(objectKey, out Pm4ObjectMatchObject objectMatch))
        {
            ImGui.TextColored(new Vector4(0.74f, 0.78f, 0.86f, 1.0f), "No PM4 match preview available for this hovered part.");
            return;
        }

        if (objectMatch.Candidates.Count == 0)
        {
            ImGui.TextColored(new Vector4(0.74f, 0.78f, 0.86f, 1.0f), "No nearby WMO or M2 placement candidates were found.");
            return;
        }

        int shownCount = Math.Min(3, objectMatch.Candidates.Count);
        for (int i = 0; i < shownCount; i++)
        {
            Pm4ObjectMatchCandidate candidate = objectMatch.Candidates[i];
            ImGui.PushID($"HoverPm4Candidate_{i}");
            ImGui.TextColored(new Vector4(0.80f, 0.96f, 0.82f, 1.0f), $"{i + 1}. {candidate.Kind}  gap={candidate.PlanarGap:F1}");

            if (!string.IsNullOrWhiteSpace(candidate.ModelName))
                ImGui.TextColored(new Vector4(0.92f, 0.94f, 0.98f, 1.0f), candidate.ModelName);

            ImGui.TextColored(
                new Vector4(0.72f, 0.78f, 0.90f, 1.0f),
                $"{candidate.EvidenceSource}  vertical={candidate.VerticalGap:F1}  overlap={candidate.PlanarOverlapRatio:P0}");
            ImGui.PopID();
        }

        if (objectMatch.Candidates.Count > shownCount)
            ImGui.TextColored(new Vector4(0.78f, 0.73f, 0.57f, 1.0f), $"+{objectMatch.Candidates.Count - shownCount} more in the inspector");
    }
}
