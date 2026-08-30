using WowViewer.Core.Editor.Eras;
using WowViewer.Core.Editor.Operations;
using WowViewer.Core.IO.Terrain;

namespace WowViewer.Core.Editor.Plugins;

/// <summary>
/// Editor plugin for browsing the terrain brush & paste catalog, adjusting interactive stamping parameters,
/// and launching templated procedural map generation.
/// </summary>
public sealed class TerrainTemplateEditorPlugin : IEditorPlugin
{
    public const string PluginId = "editor.terrain.template_brush";

    public EditorPluginDescriptor Descriptor { get; } = new(
        PluginId,
        "Terrain Brushes & Templates",
        "Catalog browser, interactive brush stamping, and templated procedural map generator.",
        EditorBuildEraRange.All);

    public TerrainBrushLibrary Library { get; }
    public TerrainBrushPaste? SelectedPaste { get; set; }
    public TerrainStampOptions CurrentStampOptions { get; set; } = new();
    public TerrainMapTemplate CurrentMapTemplate { get; set; } = new();

    public string SelectedCategory { get; set; } = "All";
    public string SearchQuery { get; set; } = string.Empty;

    public TerrainTemplateEditorPlugin(TerrainBrushLibrary? library = null)
    {
        Library = library ?? CuratedTerrainBrushLibrary.Instance;
        SelectedPaste = Library.AllPastes.FirstOrDefault();
    }

    public EditorPluginAvailability GetAvailability(EditorBuildVersion build)
        => EditorPluginAvailability.Available;

    public void OnActivated(EditorPluginContext context)
    {
        context.Log.Info("Terrain Brushes & Templates plugin activated.");
    }

    public void OnDeactivated(EditorPluginContext context)
    {
        context.Log.Info("Terrain Brushes & Templates plugin deactivated.");
    }

    public void OnUpdate(EditorPluginContext context)
    {
    }

    public void OnDraw(EditorPluginContext context)
    {
        context.Log.Trace("Terrain Brushes & Templates plugin drawing surface.");
    }

    public void OnReset(EditorPluginContext context)
    {
        SelectedCategory = "All";
        SearchQuery = string.Empty;
        SelectedPaste = Library.AllPastes.FirstOrDefault();
        CurrentStampOptions = new TerrainStampOptions();
        CurrentMapTemplate = new TerrainMapTemplate();
        context.Log.Info("Terrain Brushes & Templates plugin reset.");
    }

    public void OnDisposed(EditorPluginContext context)
    {
        context.Log.Info("Terrain Brushes & Templates plugin disposed.");
    }
}
