using WoWViewer.UI;

namespace WoWViewer.Workbench.Pages;

/// <summary>
/// Placement & Objects: archaeology tasks, the Rosetta object library, and population.
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
                "Archaeology Tasks",
                defaultOpen: true,
                id: "EditorPlacementTasks"))
        {
            _context.Host.DrawTasksAndWorkspace();
        }

        if (SharedUiWidgets.SectionHeader(
                "Rosetta Object Library",
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
