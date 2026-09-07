using WoWViewer.UI;

namespace WoWViewer.Workbench.Pages;

/// <summary>
/// Spec 231 Editor page 3: Data I/O. Phase 1 hosts the legacy Imports & Exports
/// content; the export-button consolidation (D1) and background-execution audit (D6)
/// land in Phase 2.
/// </summary>
public sealed class EditorDataIoPage
{
    private readonly ViewerAppContext _context;

    public EditorDataIoPage(ViewerAppContext context) => _context = context;

    public void Draw()
    {
        if (SharedUiWidgets.SectionHeader(
                "Imports & Exports",
                defaultOpen: true,
                id: "EditorDataIoImportsExports"))
        {
            _context.Host.DrawImportsAndExports();
        }
    }
}
