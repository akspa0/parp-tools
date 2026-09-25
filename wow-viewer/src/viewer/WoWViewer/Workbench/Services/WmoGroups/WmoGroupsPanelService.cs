using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

/// <summary>
/// Standalone WMO group overlays: group highlighting, labels and group overlay drawing.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed partial class WmoGroupsPanelService
{
    private readonly IViewerAppHost _host;

    internal WmoGroupsPanelService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge: see WmoGroupsPanelService.Host.cs.

    private const float StandaloneWmoGroupPickPadding = 2.0f;
    private const float StandaloneWmoHighlightedLabelScale = 1.8f;

    internal void DrawStandaloneWmoGroupOverlay(WmoRenderer wmoRenderer, Matrix4x4 view, Matrix4x4 proj,
        float viewportX, float viewportY, float viewportWidth, float viewportHeight)
    {
        NormalizeStandaloneWmoGroupSelection(wmoRenderer);
        _hoveredStandaloneWmoGroupIndex = -1;

        if (wmoRenderer.GroupRenderCount == 0)
            return;

        bool canInteract = _shellLayout.CanSceneConsumeMouse(_lastMouseX, _lastMouseY)
            && _lastMouseX >= viewportX && _lastMouseX <= viewportX + viewportWidth
            && _lastMouseY >= viewportY && _lastMouseY <= viewportY + viewportHeight;

        int hoveredGroup = -1;
        float hoveredDistance = float.MaxValue;

        if (canInteract)
        {
            float localX = _lastMouseX - viewportX;
            float localY = _lastMouseY - viewportY;
            float ndcX = (localX / MathF.Max(viewportWidth, 1f)) * 2f - 1f;
            float ndcY = 1f - (localY / MathF.Max(viewportHeight, 1f)) * 2f;
            var (rayOrigin, rayDir) = WorldScene.ScreenToRay(ndcX, ndcY, view, proj);

            for (int renderGroupIndex = 0; renderGroupIndex < wmoRenderer.GroupRenderCount; renderGroupIndex++)
            {
                bool effectiveVisible = wmoRenderer.GetRenderGroupEffectiveVisible(renderGroupIndex);
                bool isSelected = renderGroupIndex == _selectedStandaloneWmoGroupIndex;
                if (!_standaloneWmoOverlayIncludeHiddenGroups && !effectiveVisible && !isSelected)
                    continue;

                wmoRenderer.GetRenderGroupBounds(renderGroupIndex, out Vector3 boundsMin, out Vector3 boundsMax);
                float hitDistance = SceneViewportMath.RayAabbIntersect(rayOrigin, rayDir,
                    boundsMin - new Vector3(StandaloneWmoGroupPickPadding),
                    boundsMax + new Vector3(StandaloneWmoGroupPickPadding));
                if (hitDistance >= 0f && hitDistance < hoveredDistance)
                {
                    hoveredDistance = hitDistance;
                    hoveredGroup = renderGroupIndex;
                }
            }
        }

        if (_standaloneWmoGroupOverlayEnabled)
        {
            _editorOverlayBb ??= new BoundingBoxRenderer(_gl);
            _editorOverlayBb.BeginBatch();

            for (int renderGroupIndex = 0; renderGroupIndex < wmoRenderer.GroupRenderCount; renderGroupIndex++)
            {
                bool manualVisible = wmoRenderer.GetRenderGroupManualVisible(renderGroupIndex);
                bool runtimeVisible = wmoRenderer.GetRenderGroupRuntimeVisible(renderGroupIndex);
                bool effectiveVisible = manualVisible && runtimeVisible;
                bool isSelected = renderGroupIndex == _selectedStandaloneWmoGroupIndex;
                bool isHovered = renderGroupIndex == hoveredGroup;

                if (!_standaloneWmoOverlayIncludeHiddenGroups && !effectiveVisible && !isSelected)
                    continue;

                Vector3 color = wmoRenderer.GetRenderGroupDebugColor(renderGroupIndex);
                if (!manualVisible)
                    color *= 0.3f;
                else if (!runtimeVisible)
                    color = Vector3.Lerp(color, new Vector3(0.35f, 0.35f, 0.35f), 0.55f);

                wmoRenderer.GetRenderGroupBounds(renderGroupIndex, out Vector3 boundsMin, out Vector3 boundsMax);
                if (isSelected || isHovered)
                {
                    _editorOverlayBb.BatchHighlightedBoxMinMax(
                        boundsMin,
                        boundsMax,
                        (float)ImGui.GetTime(),
                        color,
                        Vector3.Min(Vector3.One, color + new Vector3(0.35f, 0.25f, 0.15f)),
                        Vector3.Min(Vector3.One, color + new Vector3(0.10f, 0.35f, 0.35f)));
                }
                else
                {
                    _editorOverlayBb.BatchBoxMinMax(boundsMin, boundsMax, color);
                }
            }

            _editorOverlayBb.FlushBatch(view, proj);
        }

        DrawStandaloneHighlightedWmoGroupLabels(wmoRenderer, viewportX, viewportY, viewportWidth, viewportHeight, view, proj);

        _hoveredStandaloneWmoGroupIndex = hoveredGroup;

        if (canInteract)
        {
            ImGuiIOPtr io = ImGui.GetIO();
            if (ImGui.IsMouseClicked(ImGuiMouseButton.Left))
            {
                if (hoveredGroup >= 0)
                {
                    if (io.KeyShift)
                    {
                        ToggleStandaloneWmoGroupHighlight(hoveredGroup);
                        _selectedStandaloneWmoGroupIndex = hoveredGroup;
                    }
                    else
                    {
                        _selectedStandaloneWmoGroupIndex = hoveredGroup;
                    }
                }
                else
                {
                    _selectedStandaloneWmoGroupIndex = -1;
                }
            }
        }
    }

    private void DrawStandaloneHighlightedWmoGroupLabels(WmoRenderer wmoRenderer,
        float viewportX, float viewportY, float viewportWidth, float viewportHeight, Matrix4x4 view, Matrix4x4 proj)
    {
        if (!_standaloneWmoGroupLabelsAllEnabled &&
            _highlightedStandaloneWmoGroupIndices.Count == 0 &&
            _selectedStandaloneWmoGroupIndex < 0)
            return;

        var drawList = ImGui.GetForegroundDrawList();
        var labelIndices = _standaloneWmoGroupLabelsAllEnabled
            ? Enumerable.Range(0, wmoRenderer.GroupRenderCount).ToList()
            : _highlightedStandaloneWmoGroupIndices.OrderBy(index => index).ToList();
        if (_selectedStandaloneWmoGroupIndex >= 0 && !labelIndices.Contains(_selectedStandaloneWmoGroupIndex))
            labelIndices.Add(_selectedStandaloneWmoGroupIndex);

        foreach (int renderGroupIndex in labelIndices)
        {
            if (renderGroupIndex < 0 || renderGroupIndex >= wmoRenderer.GroupRenderCount)
                continue;

            bool effectiveVisible = wmoRenderer.GetRenderGroupEffectiveVisible(renderGroupIndex);
            if (!_standaloneWmoOverlayIncludeHiddenGroups && !effectiveVisible)
                continue;

            if (!SceneViewportMath.TryProjectWorldToViewport(
                    wmoRenderer.GetRenderGroupCenter(renderGroupIndex),
                    view,
                    proj,
                    viewportWidth,
                    viewportHeight,
                    out Vector2 localProjected))
            {
                continue;
            }

            string label = wmoRenderer.GetRenderGroupName(renderGroupIndex);
            float fontSize = ImGui.GetFontSize() * StandaloneWmoHighlightedLabelScale;
            Vector2 textSize = ImGui.CalcTextSize(label) * StandaloneWmoHighlightedLabelScale;
            Vector2 textPos = new(
                viewportX + localProjected.X - textSize.X * 0.5f,
                viewportY + localProjected.Y - textSize.Y - 18f);
            Vector2 rectMin = textPos - new Vector2(12f, 8f);
            Vector2 rectMax = textPos + textSize + new Vector2(12f, 8f);

            Vector3 groupColor = wmoRenderer.GetRenderGroupDebugColor(renderGroupIndex);
            bool manualVisible = wmoRenderer.GetRenderGroupManualVisible(renderGroupIndex);
            bool runtimeVisible = wmoRenderer.GetRenderGroupRuntimeVisible(renderGroupIndex);
            bool isSelected = renderGroupIndex == _selectedStandaloneWmoGroupIndex;

            Vector4 bgColor = new(groupColor * 0.45f + new Vector3(0.18f, 0.18f, 0.18f), 0.92f);
            if (!manualVisible)
                bgColor = new Vector4(0.22f, 0.12f, 0.12f, 0.88f);
            else if (!runtimeVisible)
                bgColor = new Vector4(0.16f, 0.16f, 0.20f, 0.82f);

            Vector4 borderColor = isSelected
                ? new Vector4(1f, 0.96f, 0.74f, 1f)
                : new Vector4(groupColor, 0.98f);

            drawList.AddRectFilled(rectMin, rectMax, ImGui.ColorConvertFloat4ToU32(bgColor), 6f);
            drawList.AddRect(rectMin, rectMax, ImGui.ColorConvertFloat4ToU32(borderColor), 6f, ImDrawFlags.None, isSelected ? 3f : 2f);
            drawList.AddText(ImGui.GetFont(), fontSize, textPos, ImGui.ColorConvertFloat4ToU32(effectiveVisible
                ? new Vector4(0.98f, 0.99f, 0.99f, 1f)
                : new Vector4(0.84f, 0.84f, 0.84f, 1f)), label);
        }
    }

    internal void NormalizeStandaloneWmoGroupSelection(WmoRenderer wmoRenderer)
    {
        if (_selectedStandaloneWmoGroupIndex >= wmoRenderer.GroupRenderCount)
            _selectedStandaloneWmoGroupIndex = -1;

        if (_hoveredStandaloneWmoGroupIndex >= wmoRenderer.GroupRenderCount)
            _hoveredStandaloneWmoGroupIndex = -1;

        _highlightedStandaloneWmoGroupIndices.RemoveWhere(index => index < 0 || index >= wmoRenderer.GroupRenderCount);
    }

    internal void ToggleStandaloneWmoGroupHighlight(int renderGroupIndex)
    {
        if (renderGroupIndex < 0)
            return;

        if (!_highlightedStandaloneWmoGroupIndices.Add(renderGroupIndex))
            _highlightedStandaloneWmoGroupIndices.Remove(renderGroupIndex);
    }
}
