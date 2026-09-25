using System.Numerics;
using System.Diagnostics;
using System.Text.Json;
using ImGuiNET;
using WoWViewer.DataSources;
using WoWViewer.Workbench;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Visibility;
using WoWViewer.Population;
using WoWViewer.UI;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;

namespace WoWViewer;

/// <summary>
/// Partial class containing the large sidebar and inspector UI blocks.
/// </summary>
public partial class ViewerApp
{
    void IViewerAppHost.DrawWorkspaceBarsPanelContent() => _viewerChrome.DrawWorkspaceBarsPanelContent();
    private UtilitiesBottomTab? _pendingQuickUtilityPage;
    private UtilitiesBottomTab? _legacyUtilityPage;
    ref UtilitiesBottomTab? IViewerAppHost.LegacyUtilityPage => ref _legacyUtilityPage;
    private bool _legacyUtilityScrollPending;
    void IViewerAppHost.DrawUnifiedInspectorContent() => _workbenchPanels.DrawUnifiedInspectorContent();
    void IViewerAppHost.DrawDockedShellPanelsForLane(ShellPanelLane lane, float sidebarHeight) => _workbenchPanels.DrawDockedShellPanelsForLane(lane, sidebarHeight);
    void IViewerAppHost.DrawFixedSidebarWidthControl(string label, ref float width, bool isLeftSidebar, float displayWidth, string tooltip) => _viewerChrome.DrawFixedSidebarWidthControl(label, ref width, isLeftSidebar, displayWidth, tooltip);


    internal static float GetUniformListRowHeight()
    {
        return MathF.Max(ImGui.GetTextLineHeightWithSpacing(), ImGui.GetFrameHeightWithSpacing());
    }

    internal static void GetVisibleListRange(int itemCount, float rowHeight, out int startIndex, out int endIndex)
    {
        if (itemCount <= 0)
        {
            startIndex = 0;
            endIndex = 0;
            return;
        }

        float safeRowHeight = MathF.Max(1f, rowHeight);
        float scrollY = ImGui.GetScrollY();
        float windowHeight = ImGui.GetWindowHeight();
        const int overscan = 4;

        startIndex = Math.Max((int)MathF.Floor(scrollY / safeRowHeight) - overscan, 0);
        endIndex = Math.Min((int)MathF.Ceiling((scrollY + windowHeight) / safeRowHeight) + overscan, itemCount);
        if (endIndex < startIndex)
            endIndex = startIndex;
    }
    void IViewerAppHost.DrawTerrainControlsAdjustmentWeakSignalContent() => _terrainControlsPanel.DrawTerrainControlsAdjustmentWeakSignalContent();
    void IViewerAppHost.DrawTemporalStratigraphySubTab() => _terrainControlsPanel.DrawTemporalStratigraphySubTab();
    void IViewerAppHost.OpenWorkbenchTab(UtilitiesBottomTab tab) => _workbenchPanels.OpenWorkbenchTab(tab);
    void IViewerAppHost.OpenWorkbenchTab(ToolsBottomTab tab) => _workbenchPanels.OpenWorkbenchTab(tab);
    void IViewerAppHost.OpenWorkbenchTab(WorldBottomTab tab) => _workbenchPanels.OpenWorkbenchTab(tab);
    void IViewerAppHost.OpenWorkbenchTab(ModelBottomTab tab) => _workbenchPanels.OpenWorkbenchTab(tab);
    void IViewerAppHost.OpenWorkbenchTab(WorkbenchTab topTab, int bottomIndex) => _workbenchPanels.OpenWorkbenchTab(topTab, bottomIndex);
}
