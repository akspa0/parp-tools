using WoWViewer.UI;

namespace WoWViewer.Workbench.Pages;

/// <summary>
/// Spec 231 Editor page 1: Placement & Objects. Phase 1 hosts the legacy
/// Tasks & Workspace, 3D Object Library, and Population content as sections;
/// consolidation of the scattered transform/correlation panels happens in Phase 3.
/// </summary>
public sealed class EditorPlacementObjectsPage
{
    private readonly ViewerAppContext _context;

    public EditorPlacementObjectsPage(ViewerAppContext context) => _context = context;

    public void Draw()
    {
        if (SharedUiWidgets.SectionHeader(
                "Tasks & Workspace",
                defaultOpen: true,
                id: "EditorPlacementTasks"))
        {
            _context.Host.DrawTasksAndWorkspace();
        }

        if (SharedUiWidgets.SectionHeader(
                "3D Object Library",
                defaultOpen: true,
                id: "EditorPlacementLibrary"))
        {
            _context.Host.DrawObjectLibrary();
        }

        if (SharedUiWidgets.SectionHeader(
                "Population",
                defaultOpen: true,
                id: "EditorPlacementPopulation"))
        {
            _context.Host.DrawPopulation();
        }
    }
}
