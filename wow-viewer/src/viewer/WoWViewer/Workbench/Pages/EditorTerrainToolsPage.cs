using WoWViewer.UI;

namespace WoWViewer.Workbench.Pages;

/// <summary>
/// Spec 231 Editor page 2: Terrain Tools. Phase 1 hosts the legacy Terrain Lab
/// content; the Clipboard + Save dedupe (D3) lands in Phase 4.
/// </summary>
public sealed class EditorTerrainToolsPage
{
    private readonly ViewerAppContext _context;

    public EditorTerrainToolsPage(ViewerAppContext context) => _context = context;

    public void Draw()
    {
        if (SharedUiWidgets.SectionHeader(
                "Terrain Lab",
                defaultOpen: true,
                id: "EditorTerrainToolsLab"))
        {
            _context.Host.DrawTerrainLab();
        }
    }
}
