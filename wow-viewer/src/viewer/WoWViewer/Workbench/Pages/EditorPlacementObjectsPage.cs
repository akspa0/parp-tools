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
        // Spec 231 T031/visible de-clutter: the primary task surface stays open;
        // every other section is collapsed by default so the page does not stack
        // three large panels into one wall of content.
        if (SharedUiWidgets.SectionHeader(
                "Tasks & Workspace",
                defaultOpen: true,
                id: "EditorPlacementTasks"))
        {
            _context.Host.DrawTasksAndWorkspace();
        }

        if (SharedUiWidgets.SectionHeader(
                "PM4 Placement Tools",
                "Overlay, selection (transform / match / reconcile), and correlation tools for the selected PM4 object. The correlation tab draws the single surviving correlation page (Spec 231 D2).",
                defaultOpen: false,
                id: "EditorPlacementPm4Tools"))
        {
            _context.Host.DrawPm4Workbench();
        }

        if (SharedUiWidgets.SectionHeader(
                "3D Object Library",
                defaultOpen: false,
                id: "EditorPlacementLibrary"))
        {
            _context.Host.DrawObjectLibrary();
        }

        if (SharedUiWidgets.SectionHeader(
                "Population",
                defaultOpen: false,
                id: "EditorPlacementPopulation"))
        {
            _context.Host.DrawPopulation();
        }
    }
}
