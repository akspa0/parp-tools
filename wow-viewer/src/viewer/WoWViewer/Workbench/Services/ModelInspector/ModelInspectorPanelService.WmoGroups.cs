using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// ModelInspectorPanelService: members moved from ViewerApp_WmoGroups.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class ModelInspectorPanelService
{

    private void DrawStandaloneWmoGroupControls(WmoRenderer wmoRenderer)
    {
        NormalizeStandaloneWmoGroupSelection(wmoRenderer);

        ImGui.Text("WMO Group Overlay:");
        ImGui.Checkbox("Show group boxes", ref _standaloneWmoGroupOverlayEnabled);
        ImGui.Checkbox("Show all group names", ref _standaloneWmoGroupLabelsAllEnabled);
        ImGui.Checkbox("Include hidden groups", ref _standaloneWmoOverlayIncludeHiddenGroups);
        ImGui.TextDisabled($"Rendered groups: {wmoRenderer.GroupRenderCount}  Doodad defs: {wmoRenderer.DoodadDefCount}");

        int inspectionGroup = _selectedStandaloneWmoGroupIndex >= 0
            ? _selectedStandaloneWmoGroupIndex
            : _hoveredStandaloneWmoGroupIndex;

        if (_hoveredStandaloneWmoGroupIndex >= 0 && _hoveredStandaloneWmoGroupIndex != _selectedStandaloneWmoGroupIndex)
        {
            ImGui.TextColored(new Vector4(0.82f, 0.94f, 0.86f, 1f), $"Hover: {wmoRenderer.GetRenderGroupName(_hoveredStandaloneWmoGroupIndex)}");
            if (ImGui.SmallButton("Select Hovered Group"))
                _selectedStandaloneWmoGroupIndex = _hoveredStandaloneWmoGroupIndex;
        }

        if (inspectionGroup < 0)
        {
            ImGui.TextDisabled("Click a group box to inspect it. Group names can also be controlled from the bottom bar.");
            ImGui.TextDisabled("Left click: select  Shift+click: pin big label");
            return;
        }

        string groupName = wmoRenderer.GetRenderGroupName(inspectionGroup);
        bool manualVisible = wmoRenderer.GetRenderGroupManualVisible(inspectionGroup);
        bool runtimeVisible = wmoRenderer.GetRenderGroupRuntimeVisible(inspectionGroup);
        bool effectiveVisible = wmoRenderer.GetRenderGroupEffectiveVisible(inspectionGroup);
        Vector3 groupColor = wmoRenderer.GetRenderGroupDebugColor(inspectionGroup);
        bool labelHighlighted = _highlightedStandaloneWmoGroupIndices.Contains(inspectionGroup);
        wmoRenderer.GetRenderGroupBounds(inspectionGroup, out Vector3 boundsMin, out Vector3 boundsMax);

        ImGui.ColorButton("##SelectedWmoGroupColor", new Vector4(groupColor, 1f), ImGuiColorEditFlags.NoTooltip, new Vector2(18f, 18f));
        ImGui.SameLine();
        ImGui.TextWrapped(groupName);
        ImGui.TextDisabled($"Manual={manualVisible}  Runtime={runtimeVisible}  Effective={effectiveVisible}");
        ImGui.TextDisabled($"Bounds: ({boundsMin.X:F1}, {boundsMin.Y:F1}, {boundsMin.Z:F1}) -> ({boundsMax.X:F1}, {boundsMax.Y:F1}, {boundsMax.Z:F1})");

        if (ImGui.SmallButton(manualVisible ? "Hide Group" : "Show Group"))
            wmoRenderer.SetRenderGroupVisible(inspectionGroup, !manualVisible);

        ImGui.SameLine();
        if (ImGui.SmallButton(labelHighlighted ? "Remove Label" : "Highlight Label"))
            ToggleStandaloneWmoGroupHighlight(inspectionGroup);

        ImGui.SameLine();
        if (ImGui.SmallButton("Isolate Group"))
            wmoRenderer.IsolateRenderGroup(inspectionGroup);

        ImGui.SameLine();
        if (ImGui.SmallButton("Show All Groups"))
            wmoRenderer.SetAllRenderGroupsVisible(true);

        ImGui.SameLine();
        if (ImGui.SmallButton("Clear Labels"))
            _highlightedStandaloneWmoGroupIndices.Clear();

        ImGui.SameLine();
        if (ImGui.SmallButton("Clear Selection"))
            _selectedStandaloneWmoGroupIndex = -1;

        ImGui.SameLine();
        if (ImGui.SmallButton("Frame Group"))
            FrameBounds(boundsMin, boundsMax, mdxMirrorX: false);

        int groupDoodadCount = wmoRenderer.GetDoodadCountForRenderGroup(inspectionGroup);
        if (groupDoodadCount > 0)
        {
            ImGui.SameLine();
            if (ImGui.SmallButton("Show Doodads"))
            {
                _standaloneWmoDoodadGroupFilter = inspectionGroup;
                _statusMessage = $"Filtered doodad inspector to {groupDoodadCount} doodads in group [{inspectionGroup}].";
            }
            ImGui.TextDisabled($"Doodads in this group: {groupDoodadCount}");
        }

        ImGui.TextDisabled("The selected group gets a big label immediately. Highlighted groups keep big labels pinned.");
    }
}
