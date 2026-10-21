using WoWViewer.UI;

namespace WoWViewer.Workbench.Pages;

/// <summary>
/// Spec 231 Editor page 2: Terrain Tools. Phase 1 hosts the legacy Terrain Lab
/// content; the Clipboard + Save dedupe (D3) lands in Phase 4.
/// </summary>
public sealed class EditorTerrainToolsPage
{
    private readonly ViewerAppContext _context;
    private readonly Action? _drawFileOperations;

    public EditorTerrainToolsPage(ViewerAppContext context, Action? drawFileOperations = null)
    {
        _context = context;
        _drawFileOperations = drawFileOperations;
    }

    public void Draw()
    {
        // Parent injects MainMenuBarService.DrawTerrainFileOperationsContent. No shell type or
        // broad host dependency is introduced into this destination page.
        _drawFileOperations?.Invoke();
        if (SharedUiWidgets.SectionHeader(
                "Terrain Lab",
                defaultOpen: true,
                id: "EditorTerrainToolsLab"))
        {
            _context.Host.DrawTerrainLab();
        }
    }
}
