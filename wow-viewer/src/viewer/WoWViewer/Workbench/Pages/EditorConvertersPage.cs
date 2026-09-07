using WoWViewer.UI;

namespace WoWViewer.Workbench.Pages;

/// <summary>
/// Spec 231 Editor page 4: Converters. Hosts the converter sub-tab content;
/// the Spec 221 regression-harness surfaces join it in Phase 4.
/// </summary>
public sealed class EditorConvertersPage
{
    private readonly ViewerAppContext _context;

    public EditorConvertersPage(ViewerAppContext context) => _context = context;

    public void Draw()
    {
        if (SharedUiWidgets.SectionHeader(
                "Converters",
                defaultOpen: true,
                id: "EditorConvertersContent"))
        {
            _context.Host.DrawConverters();
        }
    }
}
