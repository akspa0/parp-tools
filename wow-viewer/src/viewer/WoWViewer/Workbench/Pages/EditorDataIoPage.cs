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
                "PM4 Exports",
                "Single authoritative home for the PM4 export commands (Spec 231 D1): objects JSON dump, OBJ set, LLM evidence bundle, visible overlay report, and PM4/WMO correlation JSON.",
                defaultOpen: false,
                id: "EditorDataIoPm4Exports"))
        {
            _context.Host.DrawPm4Exports();
        }

        if (SharedUiWidgets.SectionHeader(
                "Imports & Exports",
                defaultOpen: true,
                id: "EditorDataIoImportsExports"))
        {
            _context.Host.DrawImportsAndExports();
        }
    }
}
