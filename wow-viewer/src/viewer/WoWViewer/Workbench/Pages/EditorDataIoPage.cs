using WoWViewer.UI;

namespace WoWViewer.Workbench.Pages;

/// <summary>
/// Data I/O: new-map creation and archaeology imports. PM4 exports belong to Viewer inspection.
/// </summary>
public sealed class EditorDataIoPage
{
    private readonly ViewerAppContext _context;

    public EditorDataIoPage(ViewerAppContext context) => _context = context;

    public void Draw()
    {
        if (SharedUiWidgets.SectionHeader(
                "New Map Creator",
                "Create a new templated procedural or flat map (Spec 234 US3) and load it directly into the viewer for editing.",
                defaultOpen: true,
                id: "EditorDataIoNewMapCreator"))
        {
            _context.Host.DrawNewMapCreator();
        }

        if (SharedUiWidgets.SectionHeader(
                "Archaeology Imports",
                defaultOpen: true,
                id: "EditorDataIoImportsExports"))
        {
            _context.Host.DrawImportsAndExports();
        }
    }
}
