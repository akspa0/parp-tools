using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WoWViewer.Workbench;

namespace WoWViewer;

/// <summary>
/// Partial class containing viewer/editor workspace shell helpers.
/// </summary>
public partial class ViewerApp
{
    void IViewerAppHost.SetWorkspaceMode(WorkspaceMode mode) => _workspaces.SetWorkspaceMode(mode);
    bool IViewerAppHost.IsEditorTaskAvailable(EditorWorkspaceTask task) => _workspaces.IsEditorTaskAvailable(task);
    string IViewerAppHost.GetWorkspaceTargetSummary() => _workspaces.GetWorkspaceTargetSummary();
    string IViewerAppHost.GetWorkspaceSaveStatusSummary() => _workspaces.GetWorkspaceSaveStatusSummary();
}
