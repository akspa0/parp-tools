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

    public EditorWorkbenchPages(ViewerAppContext context, Action? terrainFileOperations = null, Action? converters = null)
    {
        _placementObjects = new EditorPlacementObjectsPage(context);
        _terrainTools = new EditorTerrainToolsPage(context, terrainFileOperations);
        _dataIo = new EditorDataIoPage(context);
        _converters = new EditorConvertersPage(context, converters);
    }


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
