using WoWViewer.UI;

namespace WoWViewer.Workbench.Pages;

/// <summary>
/// Spec 231 Editor page 4: Converters. Hosts the converter sub-tab content;
/// the Spec 221 regression-harness surfaces join it in Phase 4.
/// </summary>
public sealed class EditorConvertersPage
{
    private readonly ViewerAppContext _context;
    private readonly Action? _drawConverters;

    public EditorConvertersPage(ViewerAppContext context, Action? drawConverters = null)
    {
        _context = context;
        _drawConverters = drawConverters;
    }

    public void Draw()
    {
        if (SharedUiWidgets.SectionHeader(
                "Converters",
                defaultOpen: true,
                id: "EditorConvertersContent"))
        {
            if (_drawConverters != null) _drawConverters();
            else _context.Host.DrawConverters();
        }
    }
}
