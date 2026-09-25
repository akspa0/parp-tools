using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Visibility;

namespace WoWViewer;

public partial class ViewerApp
{
    void IViewerAppHost.DrawVisualInvestigationModeButton(InvestigationService.VisualInvestigationMode mode, string label, string tooltip) => _investigation.DrawVisualInvestigationModeButton(mode, label, tooltip);
}
