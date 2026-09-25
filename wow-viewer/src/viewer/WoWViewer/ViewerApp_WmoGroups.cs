using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;

namespace WoWViewer;

public partial class ViewerApp
{
    void IViewerAppHost.NormalizeStandaloneWmoGroupSelection(WmoRenderer wmoRenderer) => _wmoGroupsPanel.NormalizeStandaloneWmoGroupSelection(wmoRenderer);
    void IViewerAppHost.ToggleStandaloneWmoGroupHighlight(int renderGroupIndex) => _wmoGroupsPanel.ToggleStandaloneWmoGroupHighlight(renderGroupIndex);

}
