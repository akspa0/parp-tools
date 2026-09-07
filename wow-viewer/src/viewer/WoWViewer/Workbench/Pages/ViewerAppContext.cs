namespace WoWViewer.Workbench.Pages;

/// <summary>
/// Narrow draw-contract the Spec 231 Editor workbench pages call back into.
/// Implemented explicitly by the <c>ViewerApp</c> shell partial; pages never reach
/// into god-class state beyond these delegated draws (Spec 228 extraction pattern).
/// </summary>
public interface IEditorPageHost
{
    void DrawTasksAndWorkspace();
    void DrawObjectLibrary();
    void DrawPopulation();
    void DrawTerrainLab();
    void DrawImportsAndExports();
    void DrawConverters();

    /// <summary>
    /// Spec 231 D1: single authoritative draw site for the PM4 export command
    /// set (JSON dump, OBJ set, LLM bundle, visible report, correlation JSON).
    /// All PM4 panels link here instead of drawing their own export buttons.
    /// </summary>
    void DrawPm4Exports();

    /// <summary>
    /// Spec 231 T031: the PM4 workbench inspector (overlay/selection/correlation
    /// tabs — transform, match, reconcile, collection, and graph tools) hosted as
    /// a Placement & Objects section. The correlation tab draws the single
    /// surviving correlation page (D2).
    /// </summary>
    void DrawPm4Workbench();
}

/// <summary>
/// Narrow context constructor-injected into the Editor workbench pages
/// (Spec 231 T010). Grows by adding explicit dependencies only — never the
/// god-class instance itself.
/// </summary>
public sealed class ViewerAppContext
{
    public ViewerAppContext(IEditorPageHost host) => Host = host;

    public IEditorPageHost Host { get; }
}
