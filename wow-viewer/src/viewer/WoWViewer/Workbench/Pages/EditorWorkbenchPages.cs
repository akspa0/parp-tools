namespace WoWViewer.Workbench.Pages;

/// <summary>
/// Aggregate of the four Spec 231 Editor workbench pages. The shell keeps a single
/// field of this type and delegates — the Spec 228 extraction pattern (one field +
/// delegation, no per-feature god-class members).
/// </summary>
public sealed class EditorWorkbenchPages
{
    /// <summary>Number of Editor pages in the Spec 231 IA.</summary>
    public const int PageCount = 4;

    private readonly EditorPlacementObjectsPage _placementObjects;
    private readonly EditorTerrainToolsPage _terrainTools;
    private readonly EditorDataIoPage _dataIo;
    private readonly EditorConvertersPage _converters;

    public EditorWorkbenchPages(ViewerAppContext context)
    {
        _placementObjects = new EditorPlacementObjectsPage(context);
        _terrainTools = new EditorTerrainToolsPage(context);
        _dataIo = new EditorDataIoPage(context);
        _converters = new EditorConvertersPage(context);
    }

    /// <summary>
    /// Maps a pre-231 saved or menu-passed Editor bottom-tab index onto the 4-page IA.
    /// Old order: Tasks & Workspace / Converters / 3D Object Library / Imports & Exports /
    /// Terrain Lab / Population. New order: Placement & Objects / Terrain Tools / Data I/O /
    /// Converters.
    /// </summary>
    public static int MigrateLegacyEditorPageIndex(int legacyIndex) => legacyIndex switch
    {
        1 => 3, // Converters
        2 => 0, // 3D Object Library → Placement & Objects
        3 => 2, // Imports & Exports → Data I/O
        4 => 1, // Terrain Lab → Terrain Tools
        5 => 0, // Population → Placement & Objects
        _ => Math.Clamp(legacyIndex, 0, PageCount - 1),
    };

    public void Draw(int pageIndex)
    {
        switch (Math.Clamp(pageIndex, 0, PageCount - 1))
        {
            case 0:
                _placementObjects.Draw();
                break;
            case 1:
                _terrainTools.Draw();
                break;
            case 2:
                _dataIo.Draw();
                break;
            case 3:
                _converters.Draw();
                break;
        }
    }
}
