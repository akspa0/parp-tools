using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WoWViewer.Workbench;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// ViewerChromeService: members moved from ViewerApp_Workspaces.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class ViewerChromeService
{

    private void DrawWorkspaceToolbarControls()
    {
        ImGui.TextDisabled("Unified workspace");
        ImGui.TextWrapped(GetWorkspaceTargetSummary());
        ImGui.TextDisabled(GetWorkspaceSaveStatusSummary());
    }
}
