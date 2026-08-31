using WowViewer.Core.Editor.Eras;
using WowViewer.Core.Editor.Operations;

namespace WowViewer.Core.Editor.Plugins;

/// <summary>
/// Editor plugin for unified multi-tile and sub-tile chunk selection, transposition,
/// rotation, and placement shifting.
/// </summary>
public sealed class ChunkManipulatorEditorPlugin : IEditorPlugin
{
    public const string PluginId = "chunk.manipulator";

    public EditorPluginDescriptor Descriptor { get; } = new(
        PluginId,
        "Chunk Manipulator",
        "Multi-tile & sub-cell terrain chunk selection, copy/cut, rotation, and transposition.",
        EditorBuildEraRange.All);

    public ChunkSelectionRegion Selection { get; } = new();
    public ChunkTranspositionOptions Options { get; } = new();
    public ChunkTranspositionPayload? Clipboard { get; set; }
    public GlobalChunkCoordinate? TargetOrigin { get; set; }
    public string Status { get; set; } = "Ready";

    public bool ShowOverheadOverlay { get; set; } = true;
    public bool Show3dSelectionBox { get; set; } = true;

    public EditorPluginAvailability GetAvailability(EditorBuildVersion build)
        => EditorPluginAvailability.Available;

    public void OnActivated(EditorPluginContext context)
    {
        context.Log.Info("Chunk Manipulator plugin activated.");
    }

    public void OnDeactivated(EditorPluginContext context)
    {
        context.Log.Info("Chunk Manipulator plugin deactivated.");
    }

    public void OnUpdate(EditorPluginContext context)
    {
    }

    public void OnDraw(EditorPluginContext context)
    {
        context.Log.Trace("Chunk Manipulator plugin drawing surface.");
    }

    public void OnReset(EditorPluginContext context)
    {
        Selection.Clear();
        Clipboard = null;
        Status = "Ready";
        context.Log.Info("Chunk Manipulator plugin reset.");
    }

    public void OnDisposed(EditorPluginContext context)
    {
        Selection.Clear();
        Clipboard = null;
        context.Log.Info("Chunk Manipulator plugin disposed.");
    }
}
